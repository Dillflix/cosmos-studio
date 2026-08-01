# Validation report

Validated in the local workspace:

- Original `cosmos3.zip` remained unchanged at SHA-256
  `0A2A8EDEB0A8D85DCC97723C0876585755FA5991A07E3CE9B12E2FF169CAC6DF`.
- All Python modules compile under Python 3.12.
- Ruff completed with no findings.
- SQLite queue tests cover FIFO claiming, seed persistence, restart recovery,
  queued cancellation, and the running-job cancellation flag consumed by the
  inference supervisor.
- A process-isolation test force-stops an active inference child, starts a
  replacement process, and completes the next mock generation.
- Resolution tests cover the native and practical presets, custom multiples of
  16, bounds, and invalid dimensions.
- JavaScript syntax validation passed for `static/app.js`.
- A FastAPI mock-backend integration test passed for public health, rejected
  unauthenticated API access, asynchronous image enqueue, concrete seed return,
  job completion, gallery listing, and static UI serving.
- Multipart integration passed for Cosmos image-to-video and VACE control edit,
  including required uploads, mask/reference uploads, independent random seeds,
  serial processing, and completed job status.

Not validated in this workspace:

- Building the Podman image from the user's local ROCm base image.
- Fedora device permissions, SELinux behavior, and firewalld changes.
- Radeon 8060S or RX 7900 XT enumeration and ROCm kernels.
- Cosmos3 image, video, audio, image-conditioning, or video-conditioning
  inference with real checkpoints.
- VACE decoding, mask behavior, control preprocessing, or reference guidance
  with real checkpoints.
- The experimental GGUF VACE split profile and its cross-device alignment
  hooks.
- The external prompt-upsampler endpoint and exact model-specific output.
- ROCm process-termination latency, peak memory, thermal behavior, and media
  encoding performance on the target host.

Use the staged process in `MIGRATION.md`; keep the mock backend and 8060S-only
profile as the first two host validation gates.
