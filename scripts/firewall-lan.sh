#!/usr/bin/env bash
set -euo pipefail

SUBNET="${1:-192.168.0.0/24}"
PORT="${COSMOS_STUDIO_PORT:-8000}"
ZONE="${COSMOS_STUDIO_FIREWALL_ZONE:-$(sudo firewall-cmd --get-active-zones | awk 'NR == 1 { print $1 }')}"
RULE="rule family=\"ipv4\" source address=\"$SUBNET\" port port=\"$PORT\" protocol=\"tcp\" accept"

if [[ -z "$ZONE" ]]; then
  echo "No active firewalld zone was found." >&2
  exit 1
fi

# Replace an older all-sources port rule so this really is LAN-only.
if sudo firewall-cmd --permanent --zone="$ZONE" --query-port="$PORT/tcp" >/dev/null; then
  sudo firewall-cmd --permanent --zone="$ZONE" --remove-port="$PORT/tcp"
fi
if ! sudo firewall-cmd --permanent --zone="$ZONE" --query-rich-rule="$RULE" >/dev/null; then
  sudo firewall-cmd --permanent --zone="$ZONE" --add-rich-rule="$RULE"
fi
sudo firewall-cmd --reload
sudo firewall-cmd --zone="$ZONE" --list-rich-rules

echo "Allowed TCP $PORT from $SUBNET in firewalld zone $ZONE"
