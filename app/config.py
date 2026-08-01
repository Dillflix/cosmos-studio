from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    host: str
    port: int
    api_key: str
    backend: str
    data_dir: Path
    db_path: Path
    upload_dir: Path
    output_dir: Path
    prompt_tmp_dir: Path
    static_dir: Path
    max_upload_mib: int
    gallery_limit: int
    worker_poll_seconds: float
    device_profile: str
    aux_device: str
    denoiser_device: str
    expected_aux_name: str
    expected_denoiser_name: str
    cosmos_image_model: str
    cosmos_video_model: str
    cosmos_device: str
    cosmos_safety_checker: bool
    skip_allocator_warmup: bool
    vace_model: str
    vace_device: str
    vace_transformer_file: str
    prompt_endpoint: str
    prompt_model: str
    prompt_token: str
    prompt_timeout_seconds: int
    prompt_cache_max: int

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = _path("COSMOS_STUDIO_DATA_DIR", "/data")
        return cls(
            app_name="Cosmos Studio",
            host=os.getenv("COSMOS_STUDIO_HOST", "0.0.0.0"),
            port=_int("COSMOS_STUDIO_PORT", 8000),
            api_key=os.getenv(
                "COSMOS_STUDIO_API_KEY", os.getenv("COSMOS_API_KEY", "")
            ),
            backend=os.getenv("COSMOS_STUDIO_BACKEND", "mock").lower(),
            data_dir=data_dir,
            db_path=_path("COSMOS_STUDIO_DB_PATH", str(data_dir / "studio.sqlite3")),
            upload_dir=_path("COSMOS_STUDIO_UPLOAD_DIR", str(data_dir / "uploads")),
            output_dir=_path("COSMOS_STUDIO_OUTPUT_DIR", "/outputs"),
            prompt_tmp_dir=_path(
                "COSMOS_STUDIO_PROMPT_TMP_DIR", str(data_dir / "prompt-tmp")
            ),
            static_dir=_path("COSMOS_STUDIO_STATIC_DIR", "/app/static"),
            max_upload_mib=_int("COSMOS_STUDIO_MAX_UPLOAD_MIB", 1024),
            gallery_limit=_int("COSMOS_STUDIO_GALLERY_LIMIT", 100),
            worker_poll_seconds=float(
                os.getenv("COSMOS_STUDIO_WORKER_POLL_SECONDS", "1.0")
            ),
            device_profile=os.getenv(
                "COSMOS_STUDIO_DEVICE_PROFILE", "amd_8060s"
            ).lower(),
            aux_device=os.getenv("COSMOS_STUDIO_AUX_DEVICE", "cuda:0"),
            denoiser_device=os.getenv(
                "COSMOS_STUDIO_DENOISER_DEVICE", "cuda:1"
            ),
            expected_aux_name=os.getenv(
                "COSMOS_STUDIO_EXPECTED_AUX_NAME", "Radeon 8060S"
            ),
            expected_denoiser_name=os.getenv(
                "COSMOS_STUDIO_EXPECTED_DENOISER_NAME", "RX 7900 XT"
            ),
            cosmos_image_model=os.getenv(
                "COSMOS_STUDIO_COSMOS_IMAGE_MODEL",
                os.getenv(
                    "COSMOS_MODEL_PATH", "/models/Cosmos3-Super-Text2Image-nf4"
                ),
            ),
            cosmos_video_model=os.getenv(
                "COSMOS_STUDIO_COSMOS_VIDEO_MODEL", "/models/Cosmos3-Super"
            ),
            cosmos_device=os.getenv("COSMOS_STUDIO_COSMOS_DEVICE", "cuda:0"),
            cosmos_safety_checker=_bool(
                "COSMOS_STUDIO_COSMOS_SAFETY_CHECKER", True
            ),
            skip_allocator_warmup=_bool(
                "COSMOS_STUDIO_SKIP_ALLOCATOR_WARMUP", True
            ),
            vace_model=os.getenv(
                "COSMOS_STUDIO_VACE_MODEL",
                "/models/Wan2.1-VACE-14B-diffusers",
            ),
            vace_device=os.getenv("COSMOS_STUDIO_VACE_DEVICE", "cuda:0"),
            vace_transformer_file=os.getenv(
                "COSMOS_STUDIO_VACE_TRANSFORMER_FILE", ""
            ),
            prompt_endpoint=os.getenv("PROMPT_UPSAMPLER_ENDPOINT_URL", ""),
            prompt_model=os.getenv("PROMPT_UPSAMPLER_MODEL_NAME", ""),
            prompt_token=os.getenv(
                "PROMPT_UPSAMPLER_API_TOKEN", "not-required"
            ),
            prompt_timeout_seconds=_int(
                "PROMPT_UPSAMPLER_TIMEOUT_SECONDS", 300
            ),
            prompt_cache_max=_int("PROMPT_CACHE_MAX", 128),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.output_dir,
            self.prompt_tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_config(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "authentication_enabled": bool(self.api_key),
            "device_profile": self.device_profile,
            "max_upload_mib": self.max_upload_mib,
            "modes": [
                "cosmos_image",
                "cosmos_text_video",
                "cosmos_image_video",
                "cosmos_video_continue",
                "vace_text_video",
                "vace_reference_video",
                "vace_control_edit",
            ],
        }
