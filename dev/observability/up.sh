#!/usr/bin/env bash
# dev/observability/up.sh — single entry point for the local Agent SDK
# observability stacks.
#
# Subcommands:
#   langfuse  Bring up Langfuse + its bundled OTel collector.
#   grafana   Bring up Tempo + Prometheus + Grafana with their own
#             OTel collector.
#   both      Bring up both stacks. The Langfuse collector is scaled
#             to zero so the Grafana stack's collector owns the OTLP
#             ports and fans out to Tempo + Prometheus + Langfuse.
#   down      Tear down both stacks (idempotent — missing stack is
#             not an error).
#
# Use:
#   ./dev/observability/up.sh grafana
#   ./dev/observability/up.sh both
#   ./dev/observability/up.sh down
#
# Documentation:
#   crates/agent-sdk/docs/observability/LANGFUSE.md
#   crates/agent-sdk/docs/observability/GRAFANA.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGFUSE_COMPOSE="${SCRIPT_DIR}/langfuse/docker-compose.yml"
GRAFANA_COMPOSE="${SCRIPT_DIR}/grafana/docker-compose.yml"

# Host-exposed Langfuse OTel ingest URL. The Grafana collector reaches
# it via host.docker.internal so the two compose stacks stay on
# separate Docker networks.
LANGFUSE_OTLP_ENDPOINT_DEFAULT="http://host.docker.internal:4000/api/public/otel"

usage() {
  cat <<'USAGE'
usage: dev/observability/up.sh <command>

commands:
  langfuse   start the Langfuse + collector stack
  grafana    start the Tempo + Prometheus + Grafana stack
  both       start both stacks; the Grafana collector fans out to Langfuse
  down       stop both stacks (no-op for a stack that isn't running)

URLs (when up):
  http://localhost:4000   Langfuse UI            (otel@example.com / changeme123)
  http://localhost:3001   Grafana UI             (admin / admin)
  http://localhost:3200   Tempo HTTP
  http://localhost:9090   Prometheus
  http://localhost:4317   OTLP gRPC (collector)
  http://localhost:4318   OTLP HTTP (collector)
USAGE
}

require_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on PATH" >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "error: docker compose plugin not available" >&2
    exit 1
  fi
}

up_langfuse() {
  require_compose
  echo ">> bringing up Langfuse stack (project: agent-sdk-langfuse)"
  docker compose -f "${LANGFUSE_COMPOSE}" up -d
}

up_grafana() {
  require_compose
  echo ">> bringing up Grafana stack (project: agent-sdk-grafana)"
  docker compose -f "${GRAFANA_COMPOSE}" up -d
}

up_both() {
  require_compose
  echo ">> bringing up Langfuse stack without its bundled collector"
  # `--scale otel-collector=0` keeps the Langfuse collector dormant so
  # the Grafana stack's collector can own the OTLP host ports and fan
  # out to Langfuse via host.docker.internal:4000.
  docker compose -f "${LANGFUSE_COMPOSE}" up -d --scale otel-collector=0

  echo ">> bringing up Grafana stack with Langfuse fan-out enabled"
  LANGFUSE_OTLP_ENDPOINT="${LANGFUSE_OTLP_ENDPOINT:-${LANGFUSE_OTLP_ENDPOINT_DEFAULT}}" \
    docker compose -f "${GRAFANA_COMPOSE}" up -d
}

down_all() {
  require_compose
  echo ">> stopping Grafana stack (if running)"
  docker compose -f "${GRAFANA_COMPOSE}" down || true
  echo ">> stopping Langfuse stack (if running)"
  docker compose -f "${LANGFUSE_COMPOSE}" down || true
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    langfuse) up_langfuse ;;
    grafana)  up_grafana ;;
    both)     up_both ;;
    down)     down_all ;;
    -h|--help|help) usage ;;
    *)
      echo "error: unknown command '$1'" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
