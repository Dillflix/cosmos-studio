#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${COSMOS_STUDIO_URL:-http://127.0.0.1:8000}"
API_KEY="${COSMOS_STUDIO_API_KEY:-}"
AUTH=()
if [[ -n "$API_KEY" ]]; then
  AUTH=(-H "Authorization: Bearer $API_KEY")
fi

curl -fsS "$BASE_URL/health" | python3 -m json.tool

RESPONSE="$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"prompt":"A weathered lighthouse at golden hour","enhance_prompt":false}' \
  "$BASE_URL/v1/images/generations")"
echo "$RESPONSE" | python3 -m json.tool
JOB_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' <<<"$RESPONSE")"

for _ in $(seq 1 30); do
  JOB="$(curl -fsS "${AUTH[@]}" "$BASE_URL/v1/jobs/$JOB_ID")"
  STATE="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' <<<"$JOB")"
  if [[ "$STATE" == "completed" ]]; then
    echo "$JOB" | python3 -m json.tool
    exit 0
  fi
  if [[ "$STATE" == "failed" || "$STATE" == "cancelled" ]]; then
    echo "$JOB" | python3 -m json.tool
    exit 1
  fi
  sleep 1
done

echo "Timed out waiting for $JOB_ID" >&2
exit 1
