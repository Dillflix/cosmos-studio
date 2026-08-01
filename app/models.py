from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GenerationMode(StrEnum):
    COSMOS_IMAGE = "cosmos_image"
    COSMOS_TEXT_VIDEO = "cosmos_text_video"
    COSMOS_IMAGE_VIDEO = "cosmos_image_video"
    COSMOS_VIDEO_CONTINUE = "cosmos_video_continue"
    VACE_TEXT_VIDEO = "vace_text_video"
    VACE_REFERENCE_VIDEO = "vace_reference_video"
    VACE_CONTROL_EDIT = "vace_control_edit"

    @property
    def family(self) -> str:
        return "cosmos" if self.value.startswith("cosmos_") else "vace"

    @property
    def is_video(self) -> bool:
        return self is not GenerationMode.COSMOS_IMAGE


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LegacyImageRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: Any | None = None
    size: str = "1280x720"
    num_inference_steps: int = Field(default=35, ge=1, le=100)
    guidance_scale: float = Field(default=4.0, ge=0, le=30)
    flow_shift: float = Field(default=3.0, ge=0, le=30)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    enhance_prompt: bool = True
    enhancement_fallback_to_plain: bool = True
    structured_prompt: Any | None = None
    output_format: str = "png"
    quality: int = Field(default=90, ge=1, le=100)
    n: int = Field(default=1, ge=1, le=1)


class PromptEnhanceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: str = "text2image"
    size: str = "1280x720"
    use_cache: bool = True


class JobEnvelope(BaseModel):
    id: str
    mode: GenerationMode
    state: JobState
    seed: int
    queue_position: int | None = None
    status_url: str
    gallery_url: str
