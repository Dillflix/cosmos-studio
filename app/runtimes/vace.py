from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..prompting import parse_size
from .base import GenerationResult, ProgressCallback


class VaceRuntime:
    family = "vace"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe: Any | None = None

    @staticmethod
    def _device_index(device: str) -> int:
        return int(device.split(":", 1)[1]) if ":" in device else 0

    def _verify_named_device(self, torch: Any, device: str, expected: str) -> None:
        index = self._device_index(device)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Configured device {device} is unavailable")
        actual = torch.cuda.get_device_name(index)
        if expected and expected.lower() not in actual.lower():
            raise RuntimeError(
                f"Device guard rejected {device}: expected name containing "
                f"{expected!r}, found {actual!r}. Correct the device indices or "
                "expected-name settings; do not disable the guard by accident."
            )

    def _ensure_loaded(self, progress: ProgressCallback) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("ROCm is unavailable through torch.cuda")
        progress(0.05, f"Loading VACE checkpoint {self.settings.vace_model}")
        vae = AutoencoderKLWan.from_pretrained(
            self.settings.vace_model, subfolder="vae", torch_dtype=torch.float32
        )
        if self.settings.device_profile == "vace_split":
            if not self.settings.vace_transformer_file:
                raise RuntimeError(
                    "vace_split requires COSMOS_STUDIO_VACE_TRANSFORMER_FILE"
                )
            if self.settings.aux_device == self.settings.denoiser_device:
                raise RuntimeError("vace_split requires two distinct devices")
            self._verify_named_device(
                torch, self.settings.aux_device, self.settings.expected_aux_name
            )
            self._verify_named_device(
                torch,
                self.settings.denoiser_device,
                self.settings.expected_denoiser_name,
            )
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module
            from diffusers import GGUFQuantizationConfig, WanVACETransformer3DModel

            transformer = WanVACETransformer3DModel.from_single_file(
                self.settings.vace_transformer_file,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16
                ),
                torch_dtype=torch.bfloat16,
                device_map=self.settings.denoiser_device,
            )
            self.pipe = WanVACEPipeline.from_pretrained(
                self.settings.vace_model,
                transformer=transformer,
                vae=vae,
                torch_dtype=torch.bfloat16,
            )
            self.pipe.text_encoder.to(self.settings.aux_device)
            self.pipe.vae.to(self.settings.aux_device)
            add_hook_to_module(
                self.pipe.text_encoder,
                AlignDevicesHook(execution_device=self.settings.aux_device),
            )
            add_hook_to_module(
                self.pipe.vae,
                AlignDevicesHook(execution_device=self.settings.aux_device),
            )
            add_hook_to_module(
                self.pipe.transformer,
                AlignDevicesHook(execution_device=self.settings.denoiser_device),
            )
        elif self.settings.device_profile == "amd_8060s":
            self._verify_named_device(
                torch, self.settings.vace_device, self.settings.expected_aux_name
            )
            self.pipe = WanVACEPipeline.from_pretrained(
                self.settings.vace_model,
                vae=vae,
                torch_dtype=torch.bfloat16,
            )
            self.pipe.to(self.settings.vace_device)
        else:
            raise RuntimeError(
                "COSMOS_STUDIO_DEVICE_PROFILE must be amd_8060s or vace_split"
            )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=3.0
        )

    @staticmethod
    def _normalize_frames(frames: list[Any], count: int, size: tuple[int, int]) -> list[Any]:
        if not frames:
            raise ValueError("Input video has no decodable frames")
        width, height = size
        if len(frames) == count:
            selected = frames
        elif count == 1:
            selected = [frames[0]]
        else:
            selected = [
                frames[round(index * (len(frames) - 1) / (count - 1))]
                for index in range(count)
            ]
        return [frame.resize((width, height)) for frame in selected]

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
        from diffusers.utils import export_to_video, load_image, load_video
        from PIL import Image

        self._ensure_loaded(progress)
        assert self.pipe is not None
        params = job["params"]
        width, height = parse_size(params.get("size", "832x480"))
        num_frames = int(params.get("num_frames", 81))
        if num_frames < 5 or (num_frames - 1) % 4:
            raise ValueError("VACE num_frames must be 4k+1 and at least 5")
        flow_shift = float(params.get("flow_shift", 3.0 if height <= 480 else 5.0))
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=flow_shift
        )
        device = (
            self.settings.denoiser_device
            if self.settings.device_profile == "vace_split"
            else self.settings.vace_device
        )
        kwargs: dict[str, Any] = {
            "prompt": params.get("prepared_prompt", job["prompt"]),
            "negative_prompt": params.get("negative_prompt") or None,
            "conditioning_scale": float(params.get("conditioning_scale", 1.0)),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(params.get("num_inference_steps", 30)),
            "guidance_scale": float(params.get("guidance_scale", 5.0)),
            "generator": torch.Generator(device=device).manual_seed(job["seed"]),
            "output_type": "pil",
        }
        inputs = job["inputs"]
        if job["mode"] == "vace_reference_video":
            kwargs["reference_images"] = [
                load_image(item["path"]) for item in inputs["reference_images"]
            ]
        elif job["mode"] == "vace_control_edit":
            control = load_video(inputs["control_video"]["path"])
            kwargs["video"] = self._normalize_frames(
                control, num_frames, (width, height)
            )
            mask_item = inputs.get("mask")
            if mask_item:
                suffix = Path(mask_item["path"]).suffix.lower()
                if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                    mask_frames = load_video(mask_item["path"])
                    kwargs["mask"] = self._normalize_frames(
                        mask_frames, num_frames, (width, height)
                    )
                else:
                    mask_image = load_image(mask_item["path"]).convert("L")
                    kwargs["mask"] = [mask_image.resize((width, height))] * num_frames
            else:
                kwargs["mask"] = [
                    Image.new("L", (width, height), 255)
                ] * num_frames
            if inputs.get("reference_images"):
                kwargs["reference_images"] = [
                    load_image(item["path"])
                    for item in inputs["reference_images"]
                ]
        progress(0.15, "VACE denoising")
        started = time.perf_counter()
        with torch.inference_mode():
            frames = self.pipe(**kwargs).frames[0]
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        path = output_dir / f"{job['id']}.mp4"
        export_to_video(
            frames,
            str(path),
            fps=float(params.get("fps", 16)),
            macro_block_size=1,
        )
        progress(0.96, "VACE output encoded")
        return GenerationResult(
            path=path,
            media_type="video/mp4",
            metadata={
                "backend": "diffusers",
                "runtime": "WanVACEPipeline",
                "model": self.settings.vace_model,
                "device_profile": self.settings.device_profile,
                "aux_device": self.settings.aux_device,
                "denoiser_device": device,
                "generation_seconds": round(elapsed, 3),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "seed": job["seed"],
                "split_profile_validated": False
                if self.settings.device_profile == "vace_split"
                else None,
            },
        )

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
