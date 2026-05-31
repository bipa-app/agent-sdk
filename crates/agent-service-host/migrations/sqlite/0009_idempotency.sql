-- Durable at-least-once idempotency records (SQLite dialect).
-- Mirror of postgres/0009_idempotency.sql. See the Postgres variant for
-- the full rationale (ENG-8707, Phase 10 · E).
--
-- SQLite stores the fingerprint as BLOB and result_json as TEXT per the
-- SQLite dialect (no native JSONB / BYTEA types).

CREATE TABLE agent_sdk_idempotency (
    request_id  TEXT NOT NULL PRIMARY KEY,
    kind        TEXT NOT NULL,
    fingerprint BLOB NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL,

    CONSTRAINT agent_sdk_idempotency_kind_check
        CHECK (kind IN ('create_thread', 'submit_work', 'fork_thread', 'decide_confirmation'))
);
