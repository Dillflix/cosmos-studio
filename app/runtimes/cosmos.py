from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..prompting import parse_size
from .base import GenerationResult, ProgressCallback

logger = logging.getLogger("cosmos-studio.runtime.cosmos")


class CosmosRuntime:
    family = "cosmos"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe: Any | None = None
        self.loaded_model: str | None = None
        self.scheduler_config: dict[str, Any] | None = None

    def _patch_allocator(self) -> None:
        if not self.settings.skip_allocator_warmup:
            return
        import diffusers.models.model_loading_utils as loading
        import diffusers.models.modeling_utils as modeling

        def skip(*_args: Any, **_kwargs: Any) -> None:
            return None

        loading._caching_allocator_warmup = skip
        modeling._caching_allocator_warmup = skip

    @staticmethod
    def _loader_device_map(device: str) -> str:
        if device == "cuda" or device.startswith("cuda:"):
            return "cuda"
        if device == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported Cosmos device {device!r}")

    @staticmethod
    def _denoise_progress(step_index: int, total_steps: int) -> float:
        """Map a completed Diffusers step onto the job progress interval."""
        completed = min(max(step_index + 1, 0), max(total_steps, 1))
        return 0.15 + (0.68 * completed / max(total_steps, 1))

    def _ensure_loaded(self, mode: str, progress: ProgressCallback) -> None:
        import torch
        from diffusers import Cosmos3OmniPipeline

        model = (
            self.settings.cosmos_image_model
            if mode == "cosmos_image"
            else self.settings.cosmos_video_model
        )
        if self.pipe is not None and self.loaded_model == model:
            return
        self.unload()
        self._patch_allocator()
        self._verify_device(torch, self.settings.cosmos_device)
        device_index = (
            int(self.settings.cosmos_device.split(":", 1)[1])
            if ":" in self.settings.cosmos_device
            else 0
        )
        # Diffusers accepts the strategy name "cuda", not an indexed device
        # such as "cuda:0". Select the indexed PyTorch device first so the
        # strategy resolves to the configured GPU.
        torch.cuda.set_device(device_index)
        actual_name = torch.cuda.get_device_name(device_index)
        expected_name = self.settings.expected_aux_name
        if expected_name and expected_name.lower() not in actual_name.lower():
            raise RuntimeError(
                f"Cosmos device guard expected {expected_name!r} on "
                f"{self.settings.cosmos_device}, found {actual_name!r}"
            )
        progress(0.05, f"Loading Cosmos checkpoint {model}")
        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map=self._loader_device_map(self.settings.cosmos_device),
            low_cpu_mem_usage=True,
            enable_safety_checker=self.settings.cosmos_safety_checker,
        )
        if hasattr(self.pipe.transformer, "set_attention_backend"):
            self.pipe.transformer.set_attention_backend("native")
        self.scheduler_config = dict(self.pipe.scheduler.config)
        self.loaded_model = model

    @staticmethod
    def _verify_device(torch: Any, device: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm is unavailable through torch.cuda")
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Configured device {device} is unavailable; found "
                f"{torch.cuda.device_count()} device(s)"
            )

    def generate(
        self,
        job: dict[str, Any],
        output_dir: Path,
        progress: ProgressCallback,
    ) -> GenerationResult:
        import torch
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from diffusers.utils import encode_video, export_to_video, load_image, load_video
        from PIL import Image

        self._ensure_loaded(job["mode"], progress)
        assert self.pipe is not None and self.scheduler_config is not None
        params = job["params"]
        width, height = parse_size(params.get("size", "1280x720"))
        flow_shift = float(params.get("flow_shift", 3.0))
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.scheduler_config, flow_shift=flow_shift, use_karras_sigmas=False
        )
        total_steps = int(params.get("num_inference_steps", 35))

        def on_step_end(
            _pipeline: Any,
            step_index: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            completed = step_index + 1
            progress(
                self._denoise_progress(step_index, total_steps),
                f"Cosmos denoising step {completed}/{total_steps}",
            )
            return callback_kwargs

        kwargs: dict[str, Any] = {
            "prompt": params.get("prepared_prompt", job["prompt"]),
            "negative_prompt": params.get("negative_prompt") or None,
            "num_frames": (
                1
                if job["mode"] == "cosmos_image"
                else int(params.get("num_frames", 189))
            ),
            "height": height,
            "width": width,
            "num_inference_steps": total_steps,
            "guidance_scale": float(params.get("guidance_scale", 4.0)),
            "generator": torch.Generator(
                device=self.settings.cosmos_device
            ).manual_seed(job["seed"]),
            "output_type": "pil",
            "callback_on_step_end": on_step_end,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        if job["mode"] != "cosmos_image":
            kwargs["fps"] = float(params.get("fps", 24))
            kwargs["enable_sound"] = bool(params.get("enable_sound", False))
        inputs = job["inputs"]
        if job["mode"] == "cosmos_image_video":
            kwargs["image"] = load_image(inputs["image"]["path"])
        elif job["mode"] == "cosmos_video_continue":
            kwargs["video"] = load_video(inputs["source_video"]["path"])
            kwargs["condition_frame_indexes_vision"] = params.get(
                "condition_frame_indexes_vision", [0, 1]
            )
            kwargs["condition_video_keep"] = params.get(
                "condition_video_keep", "first"
            )
        progress(0.15, f"Cosmos denoising step 0/{total_steps}")
        started = time.perf_counter()
        vae = self.pipe.vae
        original_decode = vae.decode
        had_instance_decode = "decode" in vae.__dict__
        previous_instance_decode = vae.__dict__.get("decode")
        decode_seconds = 0.0
        decode_calls = 0

        def timed_decode(*args: Any, **decode_kwargs: Any) -> Any:
            nonlocal decode_calls, decode_seconds
            progress(0.86, "Cosmos VAE decoding output")
            decode_started = time.perf_counter()
            decoded = original_decode(*args, **decode_kwargs)
            torch.cuda.synchronize(self.settings.cosmos_device)
            call_seconds = time.perf_counter() - decode_started
            decode_calls += 1
            decode_seconds += call_seconds
            logger.info(
                "Cosmos VAE decode %d completed in %.3f seconds",
                decode_calls,
                call_seconds,
            )
            progress(0.93, f"Cosmos VAE decode complete ({call_seconds:.1f}s)")
            return decoded

        vae.decode = timed_decode
        try:
            with torch.inference_mode():
                result = self.pipe(**kwargs)
        finally:
            if had_instance_decode:
                vae.decode = previous_instance_decode
            else:
                delattr(vae, "decode")
        torch.cuda.synchronize(self.settings.cosmos_device)
        elapsed = time.perf_counter() - started
        progress(0.94, "Cosmos encoding output file")
        if job["mode"] == "cosmos_image":
            image: Any = result.video
            while isinstance(image, (list, tuple)):
                image = image[0]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Cosmos returned unexpected image type {type(image)!r}")
            output_format = str(params.get("output_format", "png")).lower()
            extension = "jpg" if output_format == "jpeg" else output_format
            path = output_dir / f"{job['id']}.{extension}"
            save_kwargs = {"quality": int(params.get("quality", 90))}
            if output_format == "png":
                image.save(path, format="PNG")
                media_type = "image/png"
            elif output_format == "jpeg":
                image.convert("RGB").save(path, format="JPEG", **save_kwargs)
                media_type = "image/jpeg"
            elif output_format == "webp":
                image.save(path, format="WEBP", **save_kwargs)
                media_type = "image/webp"
            else:
                raise ValueError("output_format must be png, jpeg, or webp")
        else:
            path = output_dir / f"{job['id']}.mp4"
            fps = float(params.get("fps", 24))
            if kwargs.get("enable_sound") and getattr(result, "sound", None) is not None:
                sample_rate = self.pipe.sound_tokenizer.config.sampling_rate
                encode_video(
                    result.video,
                    fps=fps,
                    audio=result.sound,
                    audio_sample_rate=sample_rate,
                    output_path=str(path),
                )
            else:
                export_to_video(result.video, str(path), fps=fps, macro_block_size=1)
            media_type = "video/mp4"
        progress(0.96, "Cosmos output encoded")
        return GenerationResult(
            path=path,
            media_type=media_type,
            metadata={
                "backend": "diffusers",
                "runtime": "Cosmos3OmniPipeline",
                "model": self.loaded_model,
                "device": self.settings.cosmos_device,
                "generation_seconds": round(elapsed, 3),
                "vae_decode_seconds": round(decode_seconds, 3),
                "vae_decode_calls": decode_calls,
                "width": width,
                "height": height,
                "seed": job["seed"],
            },
        )

    def unload(self) -> None:
        self.pipe = None
        self.loaded_model = None
        self.scheduler_config = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
