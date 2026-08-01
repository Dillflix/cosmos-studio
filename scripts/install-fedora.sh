#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="${COSMOS_STUDIO_REPOSITORY:-https://github.com/Dillflix/cosmos-studio.git}"
readonly GIT_REF="${COSMOS_STUDIO_GIT_REF:-main}"
readonly SOURCE_DIR="${COSMOS_STUDIO_SOURCE_DIR:-$HOME/ai/cosmos-studio-src}"
readonly STUDIO_HOME="${COSMOS_STUDIO_HOME:-$HOME/ai/cosmos-studio}"
readonly CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cosmos-studio"
readonly ENV_FILE="${COSMOS_STUDIO_ENV_FILE:-$CONFIG_DIR/server.env}"
readonly API_KEY_FILE="$CONFIG_DIR/api-key"
readonly BASE_IMAGE="${COSMOS_STUDIO_BASE_IMAGE:-localhost/cosmos3-rocm-server:7.2.4}"
readonly STUDIO_IMAGE="${COSMOS_STUDIO_IMAGE:-localhost/cosmos-studio:0.1.0}"
readonly BACKEND="${COSMOS_STUDIO_BACKEND:-mock}"
readonly PORT="${COSMOS_STUDIO_PORT:-8000}"
readonly CONFIGURE_FIREWALL="${COSMOS_STUDIO_CONFIGURE_FIREWALL:-1}"
readonly REQUESTED_SUBNET="${COSMOS_STUDIO_LAN_SUBNET:-}"

log() {
  printf '\n==> %s\n' "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '\nInstallation stopped at line %s (exit %s).\n' "${BASH_LINENO[0]}" "$exit_code" >&2
  exit "$exit_code"
}
trap on_error ERR

set_env_value() {
  local key="$1"
  local value="$2"
  local escaped
  escaped="$(printf '%s' "$value" | sed 's/[&|]/\\&/g')"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${escaped}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
  fi
}

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1 | tr -d '"\r'
}

detect_lan_subnet() {
  local interface
  interface="$(ip -4 route show default | awk 'NR == 1 { print $5 }')"
  if [[ -z "$interface" ]]; then
    return 1
  fi
  ip -o -4 route show dev "$interface" proto kernel scope link \
    | awk '$1 != "default" && index($1, "/") { print $1; exit }'
}

detect_lan_ip() {
  local interface
  interface="$(ip -4 route show default | awk 'NR == 1 { print $5 }')"
  if [[ -z "$interface" ]]; then
    return 1
  fi
  ip -o -4 addr show dev "$interface" scope global \
    | awk 'NR == 1 { split($4, address, "/"); print address[1] }'
}

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  fail "Run this installer as your normal Fedora user, not as root. It uses rootless Podman."
fi

[[ -f /etc/fedora-release ]] || fail "This installer supports Fedora Linux."
command -v sudo >/dev/null || fail "sudo is required."
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) \
  || fail "COSMOS_STUDIO_PORT must be a valid TCP port."
[[ "$BACKEND" == "mock" || "$BACKEND" == "diffusers" ]] \
  || fail "COSMOS_STUDIO_BACKEND must be mock or diffusers."

log "Installing Fedora prerequisites"
sudo -v
packages=(git podman curl openssl fuse-overlayfs iproute)
if [[ "$CONFIGURE_FIREWALL" == "1" ]]; then
  packages+=(firewalld)
fi
sudo dnf install -y "${packages[@]}"

log "Checking rootless Podman and AMD device access"
podman info >/dev/null
[[ -e /dev/kfd ]] || fail "/dev/kfd is missing. Install and verify the AMD/ROCm host driver first."
[[ -d /dev/dri ]] || fail "/dev/dri is missing. Install and verify the AMD graphics driver first."
if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
  fail "$(id -un) cannot access /dev/kfd. Add this user to the device's group, log out and back in, then rerun."
fi
if ! podman image exists "$BASE_IMAGE"; then
  fail "Required local ROCm base image '$BASE_IMAGE' was not found. This project intentionally reuses the image from the original Cosmos3 installation. Set COSMOS_STUDIO_BASE_IMAGE if yours has another name."
fi

log "Installing the Cosmos Studio source"
if [[ -d "$SOURCE_DIR/.git" ]]; then
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain)" ]] \
    || fail "$SOURCE_DIR has local changes. Commit or move them before rerunning."
  git -C "$SOURCE_DIR" remote set-url origin "$REPOSITORY"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$GIT_REF"
  git -C "$SOURCE_DIR" checkout -B "$GIT_REF" FETCH_HEAD
elif [[ -e "$SOURCE_DIR" ]]; then
  fail "$SOURCE_DIR already exists but is not a Git checkout. Move it aside or set COSMOS_STUDIO_SOURCE_DIR."
else
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone --depth 1 --branch "$GIT_REF" "$REPOSITORY" "$SOURCE_DIR"
fi

log "Building the application container"
export COSMOS_STUDIO_HOME="$STUDIO_HOME"
export COSMOS_STUDIO_ENV_FILE="$ENV_FILE"
export COSMOS_STUDIO_BASE_IMAGE="$BASE_IMAGE"
export COSMOS_STUDIO_IMAGE="$STUDIO_IMAGE"
export COSMOS_STUDIO_PORT="$PORT"
bash "$SOURCE_DIR/scripts/install.sh"

log "Securing and updating the server configuration"
mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"
api_key="$(env_value COSMOS_STUDIO_API_KEY)"
if [[ -z "$api_key" || "$api_key" == "replace-with-a-long-random-value" ]]; then
  api_key="$(openssl rand -hex 32)"
fi
set_env_value COSMOS_STUDIO_API_KEY "$api_key"
set_env_value COSMOS_STUDIO_BACKEND "$BACKEND"
set_env_value COSMOS_STUDIO_PORT "$PORT"
video_model="$(env_value COSMOS_STUDIO_COSMOS_VIDEO_MODEL)"
if [[ -n "${COSMOS_STUDIO_COSMOS_VIDEO_MODEL:-}" ]]; then
  video_model="$COSMOS_STUDIO_COSMOS_VIDEO_MODEL"
elif [[ -z "$video_model" || "$video_model" == "/models/Cosmos3-Super" ]]; then
  video_model="nvidia/Cosmos3-Nano"
fi
set_env_value COSMOS_STUDIO_COSMOS_VIDEO_MODEL "$video_model"
chmod 600 "$ENV_FILE"
printf '%s\n' "$api_key" >"$API_KEY_FILE"
chmod 600 "$API_KEY_FILE"

log "Starting Cosmos Studio"
bash "$SOURCE_DIR/scripts/run.sh"

log "Waiting for the health endpoint"
healthy=0
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done
if [[ "$healthy" != "1" ]]; then
  podman logs --tail 100 cosmos-studio >&2 || true
  fail "Cosmos Studio did not become healthy within 90 seconds."
fi

firewall_summary="not changed"
if [[ "$CONFIGURE_FIREWALL" == "1" ]]; then
  log "Restricting LAN access with firewalld"
  sudo systemctl enable --now firewalld
  subnet="$REQUESTED_SUBNET"
  if [[ -z "$subnet" ]]; then
    subnet="$(detect_lan_subnet || true)"
  fi
  if [[ -n "$subnet" ]]; then
    COSMOS_STUDIO_PORT="$PORT" bash "$SOURCE_DIR/scripts/firewall-lan.sh" "$subnet"
    firewall_summary="TCP $PORT allowed only from $subnet"
  else
    firewall_summary="skipped because the LAN subnet could not be detected"
  fi
fi

lan_ip="$(detect_lan_ip || true)"
if [[ -z "$lan_ip" ]]; then
  lan_ip="HOST_LAN_IP"
fi

cat <<EOF

Cosmos Studio is installed and healthy.

  Browser:      http://${lan_ip}:${PORT}/
  Backend:      ${BACKEND}
  Firewall:     ${firewall_summary}
  Configuration: ${ENV_FILE}
  API key file:  ${API_KEY_FILE}
  API key:        ${api_key}

Follow the server log with:
  podman logs -f cosmos-studio

The default mock backend validates the full web application without loading
models. To enable real inference, verify the GPU indices and model paths in
${ENV_FILE}, set COSMOS_STUDIO_BACKEND=diffusers, and rerun:
  bash ${SOURCE_DIR}/scripts/run.sh
EOF
