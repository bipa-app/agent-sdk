#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$repo_root/compose.yml"
service="postgres18"
database_url="postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk"
cargo_home="${SQLX_DEV_CARGO_HOME:-${CARGO_HOME:-${TMPDIR:-/tmp}/agent-sdk-cargo-home}}"

cd "$repo_root"
mkdir -p "$cargo_home"
export CARGO_HOME="$cargo_home"

run_compose() {
  docker compose -f "$compose_file" "$@"
}

wait_for_postgres() {
  run_compose up -d "$service"
  until run_compose exec -T "$service" pg_isready -U agent_sdk -d agent_sdk >/dev/null 2>&1; do
    sleep 1
  done
}

usage() {
  cat <<'EOF'
Usage: scripts/postgres18-dev.sh <command>

Commands:
  up               Start the local Postgres 18 container
  down             Stop the local Postgres 18 container
  wait             Start Postgres 18 and wait until it is healthy
  psql             Open a psql shell against the local database
  url              Print the DATABASE_URL for the local database
  migrate          Apply the service-host Postgres migrations locally
  prepare          Apply migrations and refresh sqlx offline metadata
  test-migrations  Apply migrations and run the Postgres store tests
EOF
}

command="${1:-}"

case "$command" in
  up)
    run_compose up -d "$service"
    ;;
  down)
    run_compose down
    ;;
  wait)
    wait_for_postgres
    ;;
  psql)
    wait_for_postgres
    run_compose exec "$service" psql -U agent_sdk -d agent_sdk
    ;;
  url)
    printf '%s\n' "$database_url"
    ;;
  migrate)
    wait_for_postgres
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx database reset -y --source crates/agent-service-host/migrations/postgres
    ;;
  prepare)
    wait_for_postgres
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx database reset -y --source crates/agent-service-host/migrations/postgres
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx prepare --workspace -- -p agent-service-host --all-targets
    ;;
  test-migrations)
    wait_for_postgres
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx database reset -y --source crates/agent-service-host/migrations/postgres
    SQLX_OFFLINE=false TEST_DATABASE_URL="$database_url" DATABASE_URL="$database_url" \
      cargo test -p agent-service-host postgres::store -- --nocapture
    ;;
  *)
    usage
    exit 1
    ;;
esac
