# Migration plan from `cosmos3.zip`

## What is reused

- FastAPI as the control plane and static UI host.
- Bearer API-key behavior and browser key persistence.
- Cosmos structured-prompt upsampler integration, timeout, and cache intent.
- Existing image controls: size, steps, guidance, flow shift, seed, format, and
  quality.
- `/health`, `/v1/models`, `/v1/prompt/enhance`, `/v1/images/generations`,
  `/docs`, and LAN binding.
- Persistent model process, one active GPU generation at a time, output
  sidecars, live status, and browser gallery concepts.
- The Strix Halo Diffusers allocator-warmup workaround, but scoped behind a
  configuration switch rather than patched unconditionally at import time.

## What changes

1. **Monolith to modules.** Configuration, authentication, storage, database,
   queueing, prompt enhancement, runtime adapters, APIs, and UI become separate
   units.
2. **Held requests to durable jobs.** The `asyncio.Lock` request gate is
   replaced by a SQLite-backed queue and `202` job responses.
3. **Browser-only history to server gallery.** Completed outputs and metadata
   are queryable from any LAN browser and survive browser cache clearing.
4. **One model to two runtime families.** Cosmos and VACE share a worker and
   storage contract but keep model-specific argument preparation isolated.
5. **JSON-only generation to multipart jobs.** Upload-capable endpoints accept
   managed images, videos, masks, and references; the legacy image endpoint is
   retained.
6. **One visible GPU assumption to verified profiles.** Startup diagnostics
   report every ROCm device. The default profile targets the 8060S, while the
   7900 XT path is explicit and opt-in.
7. **Ad-hoc launch files to an installable project.** A Containerfile, compose
   file, environment template, installer, launcher, firewall helper, and smoke
   test live beside the application.

## Deployment sequence

1. Copy this new directory to the Fedora host; keep the original Cosmos
   directory and archive unchanged.
2. Run the installer to create state, uploads, outputs, cache, and model mount
   directories and a private environment file.
3. Set the actual model paths, GPU indices, API key, and prompt-upsampler
   service in the environment file.
4. Build the ROCm container and start in `mock` mode. Validate health, job
   queueing, cancellation, restart recovery, gallery, uploads, and LAN access.
5. Switch to `diffusers` with `amd_8060s`; validate Cosmos image first, then
   Cosmos video, then VACE 1.3B or 14B on the 8060S.
6. Validate frame normalization, memory peaks, duration, and output encoding
   for each real mode.
7. Only after the single-device VACE path passes, trial the `vace_split`
   profile with the exact quantized transformer. Record the Diffusers commit,
   GGUF quantization, device enumeration, and peak VRAM. Keep the 8060S profile
   as the rollback path.

## Validation boundary

This workspace does not provide the Fedora/ROCm host, either GPU, the model
weights, the prompt-upsampler endpoint, or long-running media inference. Local
validation can cover Python syntax, request validation, durable queue behavior,
auth, storage, the mock adapter, and static UI behavior. Real generation and
the experimental split profile require host validation.
