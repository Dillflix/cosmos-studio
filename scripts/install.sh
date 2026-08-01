#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_HOME="${COSMOS_STUDIO_HOME:-$HOME/ai/cosmos-studio}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cosmos-studio"
ENV_FILE="${COSMOS_STUDIO_ENV_FILE:-$CONFIG_DIR/server.env}"

mkdir -p "$STUDIO_HOME/data" "$STUDIO_HOME/outputs" "$CONFIG_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
  install -m 600 "$PROJECT_DIR/.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE"
  echo "Edit the API key, model paths, and verified GPU indices before real inference."
else
  echo "Keeping existing $ENV_FILE"
fi

podman build \
  --build-arg "BASE_IMAGE=${COSMOS_STUDIO_BASE_IMAGE:-localhost/cosmos3-rocm:7.2.4}" \
  -t "${COSMOS_STUDIO_IMAGE:-localhost/cosmos-studio:0.1.0}" \
  -f "$PROJECT_DIR/Containerfile" \
  "$PROJECT_DIR"

echo "Installation complete. Start with: $PROJECT_DIR/scripts/run.sh"
