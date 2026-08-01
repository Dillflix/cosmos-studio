from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from fastapi import HTTPException, UploadFile

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
SAFE_FIELD = re.compile(r"[^a-zA-Z0-9_-]+")


async def save_upload(
    upload: UploadFile,
    *,
    directory: Path,
    field: str,
    index: int = 0,
    kind: str,
    max_bytes: int,
) -> dict[str, str | int]:
    suffix = Path(upload.filename or "").suffix.lower()
    if kind == "mask":
        allowed = IMAGE_SUFFIXES | VIDEO_SUFFIXES
    else:
        allowed = IMAGE_SUFFIXES if kind == "image" else VIDEO_SUFFIXES
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"{field} must use one of: {', '.join(sorted(allowed))}",
        )
    content_type = (upload.content_type or "").lower()
    allowed_prefixes = ("image/", "video/") if kind == "mask" else (
        ("image/",) if kind == "image" else ("video/",)
    )
    if (
        content_type
        and content_type != "application/octet-stream"
        and not content_type.startswith(allowed_prefixes)
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{field} has incompatible content type {content_type}",
        )
    directory.mkdir(parents=True, exist_ok=True)
    safe_field = SAFE_FIELD.sub("-", field).strip("-") or "input"
    path = directory / f"{safe_field}-{index}{suffix}"
    total = 0
    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{field} exceeds the upload limit",
                    )
                handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return {
        "path": str(path),
        "original_name": upload.filename or path.name,
        "content_type": upload.content_type or "application/octet-stream",
        "size": total,
        "kind": kind,
    }


def validate_required_inputs(mode: str, inputs: dict[str, object]) -> None:
    required = {
        "cosmos_image_video": "image",
        "cosmos_video_continue": "source_video",
        "vace_reference_video": "reference_images",
        "vace_control_edit": "control_video",
    }
    field = required.get(mode)
    if field and not inputs.get(field):
        raise HTTPException(
            status_code=422, detail=f"Mode {mode} requires {field}"
        )


def input_paths(items: Iterable[dict[str, object]]) -> list[Path]:
    return [Path(str(item["path"])) for item in items]
