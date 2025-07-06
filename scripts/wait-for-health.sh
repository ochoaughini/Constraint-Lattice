#!/usr/bin/env bash
# Wait until all specified Docker Compose services report healthy status.
# Usage: wait-for-health.sh service1 service2 ...
# Optional environment variable: TIMEOUT (default 300 seconds)

set -euo pipefail

services=("$@")

if [[ ${#services[@]} -eq 0 ]]; then
  echo "No services given" >&2
  exit 1
fi

TIMEOUT="${TIMEOUT:-300}"
INTERVAL=5
ELAPSED=0

echo "Waiting up to ${TIMEOUT}s for services to become healthy: ${services[*]}"

while true; do
  all_healthy=true
  for svc in "${services[@]}"; do
    # docker inspect returns 0 only if container exists, otherwise fallback status.
    status=$(docker inspect --format='{{.State.Health.Status}}' "${svc}" 2>/dev/null || echo "unavailable")
    if [[ "${status}" != "healthy" ]]; then
      all_healthy=false
      echo "${svc}: ${status}"
    fi
  done

  if $all_healthy; then
    echo "✅ All services are healthy."
    exit 0
  fi

  if (( ELAPSED >= TIMEOUT )); then
    echo "⏰ Timeout reached while waiting for services to become healthy." >&2
    exit 1
  fi

  sleep "${INTERVAL}"
  ELAPSED=$(( ELAPSED + INTERVAL ))
done
