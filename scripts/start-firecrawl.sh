#!/bin/bash
# start-firecrawl.sh - Start Firecrawl Docker stack with health check

set -e

COMPOSE_DIR="$(dirname "$0")/../docker/firecrawl"
API_URL="${FIRECRAWL_API_URL:-http://localhost:3002}"
MAX_WAIT=120
INTERVAL=2

cd "$COMPOSE_DIR"

if curl -sf --max-time 5 "$API_URL/v1/scrape" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","formats":["markdown"]}' > /dev/null 2>&1; then
  echo "Firecrawl is already running and healthy"
  exit 0
fi

echo "Starting Firecrawl docker stack..."
docker compose up -d

echo "Waiting for Firecrawl to be healthy (max ${MAX_WAIT}s)..."

for i in $(seq 1 $((MAX_WAIT / INTERVAL))); do
  if curl -sf --max-time 5 "$API_URL/v1/scrape" \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"url":"https://example.com","formats":["markdown"]}' > /dev/null 2>&1; then
    echo "Firecrawl is healthy and ready!"
    exit 0
  fi

  if ! docker ps --format '{{.Names}}' | grep -q firecrawl-api; then
    echo "ERROR: Firecrawl container stopped running"
    docker compose logs
    exit 1
  fi

  echo "  Waiting... ($((i * INTERVAL))s elapsed)"
  sleep $INTERVAL
done

echo "ERROR: Firecrawl did not become healthy within ${MAX_WAIT}s"
docker compose logs
exit 1