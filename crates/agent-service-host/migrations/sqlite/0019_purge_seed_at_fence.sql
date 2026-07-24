-- no-transaction
-- Purge identity becomes durable at FENCE time (SQLite dialect). Mirror of
-- postgres/0019_purge_seed_at_fence.sql. SQLite cannot alter CHECK
-- constraints in place, so the thread table is rebuilt carrying the full
-- current schema with the widened purge-record constraint.
--
-- Runs outside sqlx's transaction with foreign_keys OFF — see
-- 0016_input_injection_kind.sql for the deferred-FK constraint that
-- forbids dropping a referenced table under defer_foreign_keys.
PRAGMA foreign_keys = OFF;

BEGIN;

CREATE TABLE agent_sdk_threads_new (
    thread_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    committed_turns INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    purge_receipt_json TEXT NULL,
    CONSTRAINT agent_sdk_threads_status_check
        CHECK (status IN ('active', 'completed', 'deleting', 'deleted')),
    CONSTRAINT agent_sdk_threads_committed_turns_check
        CHECK (committed_turns >= 0),
    CONSTRAINT agent_sdk_threads_total_usage_check
        CHECK (
            total_input_tokens >= 0
            AND total_output_tokens >= 0
        ),
    CONSTRAINT agent_sdk_threads_completed_turns_check
        CHECK (status <> 'completed' OR committed_turns > 0),
    CONSTRAINT agent_sdk_threads_purge_record_check
        CHECK (
            (
                status IN ('deleting', 'deleted')
                AND purge_receipt_json IS NOT NULL
                AND json_valid(purge_receipt_json)
            )
            OR (status NOT IN ('deleting', 'deleted') AND purge_receipt_json IS NULL)
        )
);

INSERT INTO agent_sdk_threads_new (
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at,
    purge_receipt_json
)
SELECT
    thread_id, status, committed_turns,
    total_input_tokens, total_output_tokens, created_at, updated_at,
    -- Rows fenced before seeds existed have no recorded identity; adopt
    -- the fence write's own timestamp as started_at, scoped to one thread.
    CASE
        WHEN status = 'deleting' AND purge_receipt_json IS NULL THEN json_object(
            'root_thread_id', thread_id,
            'scope', 'thread',
            'started_at', updated_at
        )
        ELSE purge_receipt_json
    END
FROM agent_sdk_threads;

DROP TABLE agent_sdk_threads;
ALTER TABLE agent_sdk_threads_new RENAME TO agent_sdk_threads;

CREATE TEMP TABLE fk_rebuild_gate (violations INTEGER NOT NULL CHECK (violations = 0));
INSERT INTO fk_rebuild_gate SELECT count(*) FROM pragma_foreign_key_check;
DROP TABLE fk_rebuild_gate;

COMMIT;

PRAGMA foreign_keys = ON;
