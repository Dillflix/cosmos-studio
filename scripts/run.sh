#!/usr/bin/env bash
set -euo pipefail

STUDIO_HOME="${COSMOS_STUDIO_HOME:-$HOME/ai/cosmos-studio}"
ENV_FILE="${COSMOS_STUDIO_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/cosmos-studio/server.env}"
MODELS_DIR="${COSMOS_STUDIO_MODELS_DIR:-$HOME/ai/cosmos3/models}"
CACHE_DIR="${COSMOS_STUDIO_CACHE_DIR:-$HOME/ai/cosmos3/hf-cache}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 1
fi
mkdir -p "$STUDIO_HOME/data" "$STUDIO_HOME/outputs" "$MODELS_DIR" "$CACHE_DIR"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

podman run \
  --detach \
  --replace \
  --name cosmos-studio \
  --restart=unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add keep-groups \
  --security-opt=label=disable \
  --security-opt=seccomp=unconfined \
  --ipc=host \
  -p "${COSMOS_STUDIO_PORT:-8000}:8000" \
  --env-file "$ENV_FILE" \
  -v "$STUDIO_HOME/data:/data:rw" \
  -v "$STUDIO_HOME/outputs:/outputs:rw" \
  -v "$MODELS_DIR:/models:rw" \
  -v "$CACHE_DIR:/models/hf-cache:rw" \
  "${COSMOS_STUDIO_IMAGE:-localhost/cosmos-studio:0.1.0}"

echo "Cosmos Studio is starting on http://0.0.0.0:${COSMOS_STUDIO_PORT:-8000}/"
echo "Follow startup: podman logs -f cosmos-studio"
