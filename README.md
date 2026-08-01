# Cosmos Studio

Cosmos Studio is a unified, LAN-friendly FastAPI application for:

- Cosmos3 text-to-image
- Cosmos3 text-to-video
- Cosmos3 image-to-video
- Cosmos3 source-video-conditioned continuation
- Wan2.1-VACE text-to-video
- VACE reference-image-to-video
- VACE control-video editing with optional masks and reference images

It was migrated from the supplied `cosmos3.zip` without modifying that
archive. See [ARCHITECTURE.md](ARCHITECTURE.md) for the ZIP inventory and
device design, [MIGRATION.md](MIGRATION.md) for the staged host rollout, and
[VALIDATION.md](VALIDATION.md) for exact local checks and remaining GPU work.

## Important status

The control plane, durable queue, authentication, managed uploads, gallery,
browser UI, mock adapter, launch files, and API contracts can be validated on a
normal machine. Real inference cannot be validated here because this workspace
does not have the Fedora/ROCm host, Radeon GPUs, checkpoints, FFmpeg runtime,
or prompt-upsampler service.

The default environment deliberately uses `COSMOS_STUDIO_BACKEND=mock`. Do not
switch to `diffusers` until the mock smoke test passes on the target host.

The `vace_split` profile is experimental. Its alignment hooks and GGUF loader
must be tested with the exact VACE transformer, Diffusers 0.39.0, ROCm build,
and GPU order. The supported rollback is the complete VACE pipeline on the
8060S.

## Quick start on Fedora

The container defaults to the original prompt-capable server image,
`localhost/cosmos3-rocm-server:7.2.4`. That image contains the
`cosmos_framework.inference.prompt_upsampling` module used by prompt
enhancement; the lower-level `localhost/cosmos3-rocm:7.2.4` image does not.

For a new installation on the existing Cosmos3 Fedora host, run:

```bash
curl -fsSL https://raw.githubusercontent.com/Dillflix/cosmos-studio/main/scripts/install-fedora.sh | bash
```

The installer adds the Fedora packages it needs, clones or safely updates the
project under `~/ai`, builds and starts the rootless Podman container, creates a
random API key, waits for a healthy server, and adds a source-restricted
firewalld rule for the detected LAN. It keeps an existing private configuration
and refuses to overwrite a modified source checkout. The original local ROCm
base image, `/dev/kfd`, and `/dev/dri` must already be present.

It deliberately starts with the mock backend. To request the real Diffusers
backend during installation (only after model paths and GPU ordering are known):

```bash
curl -fsSL https://raw.githubusercontent.com/Dillflix/cosmos-studio/main/scripts/install-fedora.sh \
  | env COSMOS_STUDIO_BACKEND=diffusers bash
```

Useful overrides include `COSMOS_STUDIO_LAN_SUBNET=192.168.0.0/24`,
`COSMOS_STUDIO_PORT=8000`, `COSMOS_STUDIO_BASE_IMAGE=localhost/your-image:tag`,
and `COSMOS_STUDIO_CONFIGURE_FIREWALL=0`. The generated API key is stored with
mode `0600` at `~/.config/cosmos-studio/api-key`.

For a manual installation instead:

```bash
cd /path/to/cosmos-studio
bash scripts/install.sh
```

Edit the generated private configuration:

```bash
editor ~/.config/cosmos-studio/server.env
```

Keep `COSMOS_STUDIO_BACKEND=mock`, set a strong API key, then start:

```bash
bash scripts/run.sh
podman logs -f cosmos-studio
```

Validate queue, seed persistence, gallery output, and auth:

```bash
set -a
source ~/.config/cosmos-studio/server.env
set +a
bash scripts/smoke-test.sh
```

Open `http://HOST_LAN_IP:8000/`. The service and container both bind to
`0.0.0.0`; the host firewall still controls which clients can connect.

To allow only the local subnet through firewalld:

```bash
bash scripts/firewall-lan.sh 192.168.0.0/24
```

This helper adds a source-restricted permanent rule. Do not forward this port
from the router. Use Tailscale or an authenticated HTTPS reverse proxy for
remote access.

## Verify GPU ordering before inference

ROCm device indices are not inferred from marketing names. Verify the order
inside the built container:

```bash
podman run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add keep-groups \
  --security-opt=label=disable \
  localhost/cosmos-studio:0.1.0 \
  python3 -c 'import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])'
```

Set `COSMOS_STUDIO_COSMOS_DEVICE`, `COSMOS_STUDIO_VACE_DEVICE`, and the
expected-name guard to match. The guard intentionally fails jobs if `cuda:0`
is not the expected 8060S.

## Enable the supported real profile

After mock validation, set:

```dotenv
COSMOS_STUDIO_BACKEND=diffusers
COSMOS_STUDIO_DEVICE_PROFILE=amd_8060s
COSMOS_STUDIO_COSMOS_DEVICE=cuda:0
COSMOS_STUDIO_VACE_DEVICE=cuda:0
COSMOS_STUDIO_EXPECTED_AUX_NAME=Radeon 8060S
```

The default video checkpoint is `SanDiegoDude/Cosmos3-Super-nf4`: the complete
64B Cosmos3-Super omni model with its transformer pre-quantized to NF4. It
retains text-to-video, image-to-video, optional sound, and the full Super model
tier while reducing the loaded transformer enough for the 8060S. This is the
same embedded bitsandbytes/NF4 loading pattern used by the supplied
`Cosmos3-Super-Text2Image-nf4` reference server; it is not the smaller Nano
model. Diffusers downloads it into the persistent Hugging Face cache on first
use.

The Fedora installer migrates only the old, nonfunctional
`/models/Cosmos3-Super` placeholder. Existing real custom paths and explicitly
selected model repositories are preserved. Local overrides must point to an
actual Diffusers-format directory mounted under `/models`.

Start with Cosmos image generation using the known specialized Super NF4
checkpoint. Then test Cosmos video with the full omni Super NF4 checkpoint.
Finally test VACE on the 8060S, preferably beginning with the official 1.3B
Diffusers checkpoint before the 14B checkpoint.

The official Cosmos3 Diffusers pipeline accepts `image=` and `video=` alongside
`num_frames`, `fps`, and scheduler controls. The VACE pipeline accepts full
`video`, `mask`, and `reference_images` conditioning. The adapters follow those
interfaces from the current official documentation:

- [Cosmos3 in Diffusers](https://huggingface.co/docs/diffusers/main/api/pipelines/cosmos3)
- [Wan and VACE in Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/wan)

## Experimental VACE split profile

Only attempt this after single-device VACE succeeds:

```dotenv
COSMOS_STUDIO_DEVICE_PROFILE=vace_split
COSMOS_STUDIO_AUX_DEVICE=cuda:0
COSMOS_STUDIO_EXPECTED_AUX_NAME=Radeon 8060S
COSMOS_STUDIO_DENOISER_DEVICE=cuda:1
COSMOS_STUDIO_EXPECTED_DENOISER_NAME=RX 7900 XT
COSMOS_STUDIO_VACE_TRANSFORMER_FILE=/models/vace-transformer.gguf
```

The implementation loads the quantized `WanVACETransformer3DModel` with the
Diffusers model-class `from_single_file` path, places UMT5 and the VAE on the
8060S, places the transformer on the 7900 XT, and attaches alignment hooks at
component boundaries. It does not reuse the ComfyUI GGUF loader and does not
split transformer blocks between devices. Diffusers documents that GGUF files
are loaded through model classes rather than directly through pipeline loaders:
[Diffusers GGUF guide](https://huggingface.co/docs/diffusers/quantization/gguf).

Record peak VRAM, output correctness, and transfer behavior before treating
this profile as usable. If the transformer plus activation workspace exceeds
20 GB, keep VACE entirely on the 8060S.

## Prompt enhancement

The existing Cosmos Framework prompt-upsampling command is retained. Configure:

```dotenv
PROMPT_UPSAMPLER_ENDPOINT_URL=https://your-openai-compatible-endpoint/v1/
PROMPT_UPSAMPLER_MODEL_NAME=your-model
PROMPT_UPSAMPLER_API_TOKEN=your-token
```

Seeds are assigned and persisted before enhancement runs. A slow or failed
upsampler therefore does not change reproducibility or block the API process.
Cosmos `image2video` enhancement receives the uploaded image path. VACE uses the
same enhancer as an optional convenience, converting a structured result to a
caption; this is not a VACE-specific official prompt enhancer.

## Queue behavior

- `POST` returns `202 Accepted` as soon as uploads and the job record are safe.
- An omitted seed is generated at enqueue time and returned immediately.
- Jobs run one at a time in creation order.
- A restart puts interrupted `running` jobs back in `queued` without changing
  the seed.
- Queued jobs cancel immediately. A running job executes in an isolated model
  process; cancellation terminates that process even if ROCm or VAE decoding is
  stuck, then advances the queue with a fresh process.
- Switching between Cosmos and VACE unloads the prior runtime before loading
  the next. Grouping jobs by model family reduces reload time.

## Resolution controls

The browser groups model-native resolutions separately from balanced and fast
diagnostic presets. Both Cosmos and VACE include square, landscape, and portrait
choices at 512, 640, and 768-class sizes. Selecting **Custom width × height**
accepts dimensions from 256 through 2048; both values must be divisible by 16.
The browser shows relative pixel cost and the server repeats the same validation
for every UI and API request.

## APIs

Primary routes:

| Route | Purpose |
|---|---|
| `POST /v1/jobs` | Generic multipart enqueue |
| `POST /v1/cosmos/images` | Cosmos image job |
| `POST /v1/cosmos/videos/text` | Cosmos text-to-video job |
| `POST /v1/cosmos/videos/image` | Cosmos image-to-video job |
| `POST /v1/cosmos/videos/continue` | Cosmos video continuation job |
| `POST /v1/vace/videos/text` | VACE text-to-video job |
| `POST /v1/vace/videos/reference` | VACE reference-guided job |
| `POST /v1/vace/videos/edit` | VACE control/edit job |
| `GET /v1/jobs/{id}` | Job status and metadata |
| `DELETE /v1/jobs/{id}` | Cancel queued/running job |
| `GET /v1/gallery` | Completed outputs |
| `POST /v1/prompt/enhance` | Prompt upsampling |
| `GET /health` | Public container health and device diagnostics |

`POST /v1/images/generations` remains as an asynchronous compatibility route.
Unlike the old server, it returns a job envelope instead of holding the HTTP
request open until inference completes.

Example:

```bash
curl -H "Authorization: Bearer $COSMOS_STUDIO_API_KEY" \
  -F mode=cosmos_image \
  -F prompt='A weathered lighthouse at golden hour' \
  -F size=1280x720 \
  -F enhance_prompt=true \
  http://HOST:8000/v1/jobs
```

Image-to-video adds `-F image=@starting-frame.png`. VACE editing adds
`-F control_video=@control.mp4`, optionally `-F mask=@mask.mp4`, and repeats
`-F reference_images=@subject.png` for multiple references.

## Storage and backups

By default the launcher stores:

- SQLite and uploads: `~/ai/cosmos-studio/data`
- Outputs and metadata sidecars: `~/ai/cosmos-studio/outputs`
- Private configuration: `~/.config/cosmos-studio/server.env`

Back up the SQLite database and outputs together. Uploaded inputs are retained
for reproducibility; no automatic retention policy is enabled in this first
release.

## Local developer checks

With development dependencies installed:

```bash
python3 -m compileall -q app tests
python3 -m pytest -q
python3 -m ruff check app tests
```

For a no-GPU local run, point data/output paths at writable directories and
keep the mock backend:

```bash
COSMOS_STUDIO_BACKEND=mock \
COSMOS_STUDIO_DATA_DIR=./data \
COSMOS_STUDIO_OUTPUT_DIR=./outputs \
COSMOS_STUDIO_STATIC_DIR=./static \
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
