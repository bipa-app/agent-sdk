#!/usr/bin/env bash
# Helper script for SQLite durable-core development.
# Mirrors the Postgres script (postgres18-dev.sh) for the SQLite backend.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
migrations_dir="$repo_root/crates/agent-service-host/migrations/sqlite"
db_path="$repo_root/target/sqlx-sqlite-dev.db"
database_url="sqlite:${db_path}?mode=rwc"

cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: scripts/sqlite-dev.sh <command>

Commands:
  prepare    Create a temp SQLite DB, apply migrations, and refresh the sqlx
             offline cache (.sqlx/) for the sqlite feature.  Preserves any
             existing Postgres entries in the .sqlx/ directory.
  url        Print the DATABASE_URL for the local SQLite database.
  reset      Remove the development SQLite database.
EOF
}

command="${1:-}"

case "$command" in
  prepare)
    echo "→ Removing stale database..."
    rm -f "$db_path"

    echo "→ Applying migrations..."
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx migrate run --source "$migrations_dir"

    # cargo sqlx prepare clears .sqlx/ before writing, so we must
    # back up any existing (e.g. Postgres) cache files and merge them
    # back after the SQLite prepare run.  Use a trap so a failure of
    # `cargo sqlx prepare` still restores the backup instead of
    # permanently deleting the Postgres cache entries along with the
    # temp dir.
    backup_dir=$(mktemp -d)
    restore_backup() {
      if compgen -G "$backup_dir/query-*.json" > /dev/null 2>&1; then
        mkdir -p .sqlx
        cp -n "$backup_dir"/query-*.json .sqlx/ || true
      fi
      rm -rf "$backup_dir"
    }
    trap restore_backup EXIT

    if compgen -G ".sqlx/query-*.json" > /dev/null 2>&1; then
      cp .sqlx/query-*.json "$backup_dir/"
    fi

    echo "→ Refreshing sqlx offline cache for sqlite feature..."
    SQLX_OFFLINE=false DATABASE_URL="$database_url" \
      cargo sqlx prepare --workspace -- -p agent-service-host --no-default-features --features sqlite --all-targets

    # On success, merge the backup back explicitly so the summary
    # counts reflect the restored Postgres entries.  The EXIT trap
    # will then find the backup already drained; keep it as a safety
    # net if restore_backup fails mid-way.
    restore_backup
    trap - EXIT

    # grep exits 1 when no matches exist; under `set -euo pipefail`
    # that would abort the script *after* a successful prepare and
    # report a false failure.  Fall back to 0 on no-match so the
    # summary line always prints.
    sqlite_count=$(grep -rl '"SQLite"' .sqlx/query-*.json 2>/dev/null | wc -l | tr -d ' ' || echo 0)
    pg_count=$(grep -rl '"PostgreSQL"' .sqlx/query-*.json 2>/dev/null | wc -l | tr -d ' ' || echo 0)
    echo "✓ SQLite offline cache updated (${sqlite_count} SQLite + ${pg_count} Postgres queries)."
    ;;
  url)
    printf '%s\n' "$database_url"
    ;;
  reset)
    rm -f "$db_path"
    echo "✓ Removed $db_path"
    ;;
  *)
    usage
    exit 1
    ;;
esac
