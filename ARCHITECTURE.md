# Cosmos Studio architecture

This design is grounded in the supplied `cosmos3.zip` (SHA-256
`0A2A8EDEB0A8D85DCC97723C0876585755FA5991A07E3CE9B12E2FF169CAC6DF`).
The archive contains a single 1,526-line FastAPI module, one 1,395-line
self-contained browser UI, a smoke test, prompt samples, and two Podman launch
scripts. The original implementation runs one Cosmos3 text-to-image pipeline,
uses an `asyncio.Lock` as an in-memory FIFO-ish gate, keeps each HTTP request
open until generation finishes, and stores gallery history only in browser
storage.

## Target shape

```text
Browser / API client
        |
        v
FastAPI control plane (CPU, always responsive)
  |-- Bearer API-key check
  |-- upload validation and managed asset storage
  |-- prompt enhancement endpoint
  |-- health, status, jobs, cancellation, gallery APIs
  `-- static responsive UI
        |
        v
SQLite job and gallery database
        |
        v
One persistent queue supervisor
  |-- assigns no seeds (the API already persisted the concrete seed)
  |-- recovers interrupted jobs after restart
  |-- polls running-job cancellation state
  `-- owns one replaceable inference subprocess
        |-- lazily loads/unloads one runtime family at a time
        |-- Cosmos3 adapter
        `-- Wan2.1-VACE adapter
        |
        v
Managed outputs + sidecar metadata
```

The control plane and queue supervisor share the always-responsive server
process. Model runtimes and their ROCm context live in a persistent child
process inside the same container. Normal jobs reuse loaded models. Cancelling
a running job terminates and replaces only that child, so a native kernel or
VAE decode cannot hold the API or subsequent queue entries hostage.

## Modes

| API mode | Runtime | Required media | Semantics |
|---|---|---|---|
| `cosmos_image` | Cosmos3 | none | Text-to-image (`num_frames=1`) |
| `cosmos_text_video` | Cosmos3 | none | Text-to-video |
| `cosmos_image_video` | Cosmos3 | image | First-frame-conditioned video |
| `cosmos_video_continue` | Cosmos3 | source video | Leading-latent-frame conditioning / continuation |
| `vace_text_video` | VACE | none | VACE text-to-video |
| `vace_reference_video` | VACE | one or more reference images | Subject/reference-guided video |
| `vace_control_edit` | VACE | control/source video; mask and references optional | Whole-video or masked controllable editing |

Cosmos video conditioning is deliberately described as continuation rather
than full editing. Diffusers documents that it anchors selected leading latent
frames, whereas VACE accepts full video, mask, and reference-image controls.

## Queue and seed contract

1. The API validates the request and copies uploads into an application-owned
   job directory.
2. If the caller omitted a seed, the API creates one before inserting the job.
3. The response is `202 Accepted` and includes the concrete seed, job ID,
   position estimate, status URL, and gallery URL.
4. The worker atomically claims one queued job. Only one GPU job runs at once.
5. Completed output and metadata are committed to SQLite and become visible in
   the browser gallery. Failed jobs retain their error and inputs for diagnosis.
6. At restart, jobs left in `running` state return to `queued` with an
   interruption note. Seeds never change during recovery or retry.

Queued jobs cancel immediately. For running jobs, the supervisor observes the
durable cancellation flag within 250 ms, terminates the inference child, waits
briefly for ROCm cleanup, and escalates to a forced kill if necessary. The job
is then marked cancelled and the next queue entry receives a fresh inference
process. This deliberately trades the loaded model cache for bounded
cancellation when a native call is stuck.

## Device placement on the target AMD host

ROCm exposes AMD devices through PyTorch's `cuda` API. Device indices must be
verified on the actual host because enumeration can change.

### Default, supported profile: `amd_8060s`

- Radeon 8060S / gfx1151 / 128 GB unified pool: complete Cosmos3 pipelines,
  complete BF16 VACE pipeline, prompt/media preparation, VAE encode/decode.
- RX 7900 XT / gfx1100 / 20 GB: unused.

This is the safe default. It avoids automatic `balanced` placement and
cross-device traffic, and it is the only profile expected to work without a
quantized VACE transformer that Diffusers can load directly.

### Opt-in experimental profile: `vace_split`

- 8060S: UMT5 tokenizer/text encoder, Wan VAE, input preprocessing and final
  decode.
- 7900 XT: only a quantized `WanVACETransformer3DModel` that has been confirmed
  to fit with inference workspace.
- Transfers: prompt embeddings and encoded conditioning latents move to the
  denoiser device; generated latents move back to the VAE device; decoded
  frames move to CPU for FFmpeg.

The profile refuses to start unless two named devices and a quantized
transformer path are configured. ComfyUI GGUF files are not assumed to be
drop-in compatible: Diffusers supports GGUF through model-class
`from_single_file`, not arbitrary ComfyUI loader objects. The split executor is
therefore marked experimental until tested against the user's exact checkpoint,
Diffusers commit, ROCm build, and device ordering.

No profile splits individual transformer blocks across devices. That would add
traffic at many block boundaries and is a poor match for this asymmetric pair.

## Runtime lifecycle

- Heavy libraries are imported lazily so the web service can expose diagnostic
  health information even when ROCm or a checkpoint is unavailable.
- A runtime family is loaded on first use and retained for subsequent jobs.
- The runtime and ROCm context exist only in the inference subprocess.
- Cancelling or crashing inference replaces that process; FastAPI and SQLite
  remain available throughout.
- Switching from Cosmos to VACE unloads the previous family, runs garbage
  collection, and clears the ROCm allocator before loading the next family.
- Model loading errors fail the claimed job but do not kill the control plane.
- `COSMOS_STUDIO_BACKEND=mock` exercises the complete queue/API/gallery path
  without model weights. It is for validation only.

## Storage and security

- SQLite is the source of truth for jobs and gallery records.
- Uploaded and generated files use server-generated names; original filenames
  are metadata only.
- Extension, content type, and size are checked before acceptance.
- `COSMOS_STUDIO_API_KEY` enables constant-time Bearer-token checks for API
  operations. `/health` remains public for container probes.
- Output media URLs are unguessable but intentionally readable without custom
  headers so `<img>` and `<video>` work in the LAN UI. Do not publish the port
  directly to the internet; use a VPN or authenticated HTTPS reverse proxy.
- The container binds `0.0.0.0:8000`; the installer includes an optional
  source-subnet-restricted firewalld rule.

## Compatibility boundaries

The original `/v1/images/generations` route remains as an asynchronous
compatibility endpoint. It now returns a job envelope instead of holding the
connection for several minutes. The original prompt enhancement and model-list
routes remain. Environment variables have unified names with documented legacy
aliases where retaining them is useful.

## References

- [Diffusers Cosmos3 pipeline documentation](https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3)
- [Diffusers Wan/VACE pipeline documentation](https://huggingface.co/docs/diffusers/api/pipelines/wan)
- [Diffusers GGUF documentation](https://huggingface.co/docs/diffusers/quantization/gguf)
- [NVIDIA Cosmos repository](https://github.com/NVIDIA/cosmos)
