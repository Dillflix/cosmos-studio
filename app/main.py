from __future__ import annotations

import json
import logging
import secrets
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from . import __version__
from .auth import authorize
from .config import Settings
from .db import JobStore
from .models import GenerationMode, JobState, LegacyImageRequest, PromptEnhanceRequest
from .prompting import PromptEnhancer, parse_size
from .storage import save_upload, validate_required_inputs
from .worker import GenerationWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cosmos-studio")


settings = Settings.from_env()
store = JobStore(settings.db_path)
enhancer = PromptEnhancer(settings)
worker = GenerationWorker(settings, store, enhancer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    store.initialize()
    recovered = store.recover_interrupted()
    if recovered:
        logger.warning("Recovered %s interrupted job(s)", recovered)
    app.state.settings = settings
    app.state.store = store
    app.state.worker = worker
    worker.start()
    yield
    await worker.stop()


app = FastAPI(
    title="Cosmos Studio API",
    version=__version__,
    description="Durable queue for Cosmos3 generation and Wan2.1-VACE editing",
    lifespan=lifespan,
)

settings.ensure_directories()
app.mount("/media", StaticFiles(directory=settings.output_dir), name="media")
app.mount("/ui", StaticFiles(directory=settings.static_dir, html=True), name="ui")


def _public_input(value: Any) -> Any:
    if isinstance(value, list):
        return [_public_input(item) for item in value]
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if key != "path"}
    return value


def public_job(job: dict[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    public = {key: value for key, value in job.items() if key != "output_path"}
    public["inputs"] = _public_input(public.get("inputs", {}))
    if not include_metadata:
        public.pop("metadata", None)
        public.pop("params", None)
    if public["state"] == JobState.QUEUED:
        public["queue_position"] = store.queue_position(public["id"])
    else:
        public["queue_position"] = None
    public["status_url"] = f"/v1/jobs/{public['id']}"
    return public


def job_envelope(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "job_id": job["id"],
        "mode": job["mode"],
        "state": job["state"],
        "seed": job["seed"],
        "queue_position": store.queue_position(job["id"]),
        "status_url": f"/v1/jobs/{job['id']}",
        "gallery_url": "/ui/#gallery",
    }


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any, cast: type, default: Any) -> Any:
    if value is None or str(value).strip() == "":
        return default
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric value {value!r}") from exc


def validate_parameters(mode: GenerationMode, params: dict[str, Any]) -> None:
    try:
        parse_size(str(params["size"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    steps = int(params["num_inference_steps"])
    if not 1 <= steps <= 100:
        raise HTTPException(status_code=422, detail="num_inference_steps must be 1..100")
    if not 0 <= float(params["guidance_scale"]) <= 30:
        raise HTTPException(status_code=422, detail="guidance_scale must be 0..30")
    if not 0 <= float(params["flow_shift"]) <= 30:
        raise HTTPException(status_code=422, detail="flow_shift must be 0..30")
    if params.get("output_format") not in {"png", "jpeg", "webp"}:
        raise HTTPException(status_code=422, detail="output_format must be png, jpeg, or webp")
    if not 1 <= int(params.get("quality", 90)) <= 100:
        raise HTTPException(status_code=422, detail="quality must be 1..100")
    if mode.is_video:
        frames = int(params["num_frames"])
        if not 5 <= frames <= 241:
            raise HTTPException(status_code=422, detail="num_frames must be 5..241")
        if mode.family == "vace" and (frames - 1) % 4:
            raise HTTPException(status_code=422, detail="VACE num_frames must be 4k+1")


async def enqueue_form(request: Request, forced_mode: GenerationMode | None = None) -> JSONResponse:
    form = await request.form()
    try:
        mode = forced_mode or GenerationMode(str(form.get("mode", "")))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Unknown generation mode") from exc
    prompt = str(form.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")
    raw_params = form.get("params_json")
    try:
        params = json.loads(str(raw_params)) if raw_params else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="params_json must be valid JSON") from exc
    if not isinstance(params, dict):
        raise HTTPException(status_code=422, detail="params_json must be an object")
    video_default = 189 if mode.family == "cosmos" else 81
    size_default = "1280x720" if mode.family == "cosmos" else "832x480"
    params.update(
        {
            "size": str(form.get("size") or params.get("size") or size_default),
            "num_frames": _number(
                form.get("num_frames"),
                int,
                params.get("num_frames", video_default),
            ),
            "fps": _number(
                form.get("fps"),
                float,
                params.get("fps", 24 if mode.family == "cosmos" else 16),
            ),
            "num_inference_steps": _number(
                form.get("num_inference_steps"),
                int,
                params.get(
                    "num_inference_steps",
                    35 if mode.family == "cosmos" else 30,
                ),
            ),
            "guidance_scale": _number(
                form.get("guidance_scale"),
                float,
                params.get(
                    "guidance_scale",
                    4.0 if mode.family == "cosmos" else 5.0,
                ),
            ),
            "flow_shift": _number(
                form.get("flow_shift"), float, params.get("flow_shift", 3.0)
            ),
            "conditioning_scale": _number(
                form.get("conditioning_scale"),
                float,
                params.get("conditioning_scale", 1.0),
            ),
            "negative_prompt": str(
                form.get("negative_prompt") or params.get("negative_prompt") or ""
            ),
            "enhance_prompt": _bool_value(
                form.get("enhance_prompt"),
                bool(params.get("enhance_prompt", mode.family == "cosmos")),
            ),
            "enhancement_fallback_to_plain": _bool_value(
                form.get("enhancement_fallback_to_plain"), True
            ),
            "enable_sound": _bool_value(
                form.get("enable_sound"),
                bool(params.get("enable_sound", False)),
            ),
            "output_format": str(
                form.get("output_format") or params.get("output_format") or "png"
            ),
            "quality": _number(form.get("quality"), int, params.get("quality", 90)),
        }
    )
    structured = form.get("structured_prompt")
    if structured:
        try:
            params["structured_prompt"] = json.loads(str(structured))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="structured_prompt must be JSON") from exc
    validate_parameters(mode, params)
    seed_value = form.get("seed")
    seed = _number(seed_value, int, None)
    if seed is None:
        seed = secrets.randbelow(2**63 - 1)
    if not 0 <= seed <= 2**63 - 1:
        raise HTTPException(status_code=422, detail="seed is outside the supported range")
    job_id = uuid.uuid4().hex
    job_dir = settings.upload_dir / job_id
    max_bytes = settings.max_upload_mib * 1024 * 1024
    inputs: dict[str, Any] = {}
    field_specs = {
        "image": "image",
        "source_video": "video",
        "control_video": "video",
        "mask": "mask",
    }
    try:
        for field, kind in field_specs.items():
            upload = form.get(field)
            if isinstance(upload, UploadFile) and upload.filename:
                inputs[field] = await save_upload(
                    upload,
                    directory=job_dir,
                    field=field,
                    kind=kind,
                    max_bytes=max_bytes,
                )
        references = [
            item
            for item in form.getlist("reference_images")
            if isinstance(item, UploadFile) and item.filename
        ]
        if references:
            inputs["reference_images"] = [
                await save_upload(
                    item,
                    directory=job_dir,
                    field="reference",
                    index=index,
                    kind="image",
                    max_bytes=max_bytes,
                )
                for index, item in enumerate(references)
            ]
        validate_required_inputs(mode.value, inputs)
        record = {
            "id": job_id,
            "mode": mode.value,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt"),
            "params": params,
            "inputs": inputs,
            "created_at": time.time(),
        }
        store.enqueue(record)
    except Exception:
        if job_dir.exists():
            shutil.rmtree(job_dir)
        raise
    worker.notify()
    job = store.get(job_id)
    assert job is not None
    return JSONResponse(job_envelope(job), status_code=202)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui/", status_code=307)


@app.get("/health")
async def health() -> JSONResponse:
    counts = store.counts() if settings.db_path.exists() else {state.value: 0 for state in JobState}
    status = "running" if worker.current_job_id else "ready"
    payload: dict[str, Any] = {
        "status": status,
        "version": __version__,
        "backend": settings.backend,
        "device_profile": settings.device_profile,
        "current_job_id": worker.current_job_id,
        "queue": counts,
        "last_error": worker.last_error,
        "prompt_enhancement_configured": enhancer.configured,
    }
    try:
        import torch

        payload["runtime"] = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu_available": torch.cuda.is_available(),
            "devices": [
                {"index": index, "name": torch.cuda.get_device_name(index)}
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        payload["runtime"] = {"torch": None, "gpu_available": False, "devices": []}
    return JSONResponse(payload, status_code=200)


@app.get("/v1/config", dependencies=[Depends(authorize)])
async def config() -> dict[str, object]:
    return settings.public_config()


@app.get("/v1/status", dependencies=[Depends(authorize)])
async def status() -> dict[str, Any]:
    return {
        "current_job_id": worker.current_job_id,
        "queue": store.counts(),
        "last_error": worker.last_error,
        "runtime_family": worker.runtimes.current_family,
    }


@app.get("/v1/models", dependencies=[Depends(authorize)])
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "cosmos3-image",
                "path": settings.cosmos_image_model,
                "device": settings.cosmos_device,
            },
            {
                "id": "cosmos3-video",
                "path": settings.cosmos_video_model,
                "device": settings.cosmos_device,
            },
            {
                "id": "wan2.1-vace",
                "path": settings.vace_model,
                "device_profile": settings.device_profile,
            },
        ],
    }


@app.post("/v1/prompt/enhance", dependencies=[Depends(authorize)])
async def prompt_enhance(data: PromptEnhanceRequest) -> dict[str, Any]:
    try:
        started = time.perf_counter()
        structured = await enhancer.enhance(
            data.prompt, size=data.size, mode=data.mode, use_cache=data.use_cache
        )
        return {
            "prompt": data.prompt,
            "structured_prompt": structured,
            "structured_prompt_json": json.dumps(structured, ensure_ascii=False),
            "size": data.size,
            "enhancement_seconds": round(time.perf_counter() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Prompt enhancement timed out after {exc.timeout}s",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prompt enhancement failed: {exc}") from exc


@app.post("/v1/jobs", dependencies=[Depends(authorize)])
async def enqueue_job(request: Request) -> JSONResponse:
    return await enqueue_form(request)


@app.post("/v1/cosmos/images", dependencies=[Depends(authorize)])
async def cosmos_image(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.COSMOS_IMAGE)


@app.post("/v1/cosmos/videos/text", dependencies=[Depends(authorize)])
async def cosmos_text_video(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.COSMOS_TEXT_VIDEO)


@app.post("/v1/cosmos/videos/image", dependencies=[Depends(authorize)])
async def cosmos_image_video(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.COSMOS_IMAGE_VIDEO)


@app.post("/v1/cosmos/videos/continue", dependencies=[Depends(authorize)])
async def cosmos_video_continue(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.COSMOS_VIDEO_CONTINUE)


@app.post("/v1/vace/videos/text", dependencies=[Depends(authorize)])
async def vace_text_video(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.VACE_TEXT_VIDEO)


@app.post("/v1/vace/videos/reference", dependencies=[Depends(authorize)])
async def vace_reference_video(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.VACE_REFERENCE_VIDEO)


@app.post("/v1/vace/videos/edit", dependencies=[Depends(authorize)])
async def vace_control_edit(request: Request) -> JSONResponse:
    return await enqueue_form(request, GenerationMode.VACE_CONTROL_EDIT)


@app.post("/v1/images/generations", dependencies=[Depends(authorize)])
async def legacy_image_generation(data: LegacyImageRequest) -> JSONResponse:
    seed = data.seed if data.seed is not None else secrets.randbelow(2**63 - 1)
    record = {
        "id": uuid.uuid4().hex,
        "mode": GenerationMode.COSMOS_IMAGE.value,
        "seed": seed,
        "prompt": data.prompt,
        "negative_prompt": data.negative_prompt,
        "params": data.model_dump(exclude={"prompt", "seed", "n"}),
        "inputs": {},
        "created_at": time.time(),
    }
    store.enqueue(record)
    worker.notify()
    job = store.get(record["id"])
    assert job is not None
    return JSONResponse(job_envelope(job), status_code=202)


@app.get("/v1/jobs", dependencies=[Depends(authorize)])
async def list_jobs(limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    return {"data": [public_job(job, include_metadata=False) for job in store.list(limit=limit)]}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
async def get_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(job)


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    state = store.request_cancel(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "state": state}


@app.get("/v1/gallery", dependencies=[Depends(authorize)])
async def gallery(limit: int | None = None) -> dict[str, Any]:
    actual_limit = max(1, min(limit or settings.gallery_limit, 200))
    return {
        "data": [
            public_job(job)
            for job in store.list(limit=actual_limit, completed_only=True)
        ]
    }
