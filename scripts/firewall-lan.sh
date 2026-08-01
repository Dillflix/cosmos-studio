#!/usr/bin/env bash
set -euo pipefail

SUBNET="${1:-192.168.0.0/24}"
PORT="${COSMOS_STUDIO_PORT:-8000}"
ZONE="$(sudo firewall-cmd --get-active-zones | awk 'NR == 1 { print $1 }')"
RULE="rule family=\"ipv4\" source address=\"$SUBNET\" port port=\"$PORT\" protocol=\"tcp\" accept"

sudo firewall-cmd --permanent --zone="$ZONE" --add-rich-rule="$RULE"
sudo firewall-cmd --reload
sudo firewall-cmd --zone="$ZONE" --list-rich-rules

echo "Allowed TCP $PORT from $SUBNET in firewalld zone $ZONE"
