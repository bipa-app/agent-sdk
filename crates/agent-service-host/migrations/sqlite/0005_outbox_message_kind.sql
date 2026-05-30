-- SQLite-dialect mirror of postgres/0005_outbox_message_kind.sql.
--
-- Phase 8.1 — Transactional Outbox Contract and Message Kinds.
--
-- SQLite cannot ALTER existing CHECK constraints or relax NOT NULL on
-- columns referenced by foreign keys without rebuilding the table, so
-- we use the standard "rename → create new → copy → drop old" recipe.

-- ---------------------------------------------------------------------
-- 1. Drop the relay-scan + by-thread + claimed-sweep indexes that
--    point at the existing table (SQLite drops them automatically
--    when the table is dropped, but listing them explicitly makes
--    the intent obvious).
-- ---------------------------------------------------------------------

DROP INDEX IF EXISTS agent_sdk_outbox_relay_scan_idx;
DROP INDEX IF EXISTS agent_sdk_outbox_by_thread_idx;
DROP INDEX IF EXISTS agent_sdk_outbox_claimed_sweep_idx;

-- ---------------------------------------------------------------------
-- 2. Rename the legacy table out of the way.
-- ---------------------------------------------------------------------

ALTER TABLE agent_sdk_outbox RENAME TO agent_sdk_outbox_legacy;

-- ---------------------------------------------------------------------
-- 3. Create the new table with the Phase 8.1 contract baked in.
-- ---------------------------------------------------------------------

CREATE TABLE agent_sdk_outbox (
    id              TEXT        PRIMARY KEY,
    kind            TEXT        NOT NULL,
    thread_id       TEXT        NOT NULL,
    event_id        TEXT        NULL,
    sequence        INTEGER     NULL,
    status          TEXT        NOT NULL DEFAULT 'pending',
    payload_json    TEXT        NOT NULL,
    created_at      TEXT        NOT NULL,
    next_attempt_at TEXT        NOT NULL,
    attempt_count   INTEGER     NOT NULL DEFAULT 0,
    max_attempts    INTEGER     NOT NULL DEFAULT 5,
    last_error      TEXT        NULL,
    claimed_by      TEXT        NULL,
    claimed_at      TEXT        NULL,
    delivered_at    TEXT        NULL,

    CONSTRAINT agent_sdk_outbox_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    CONSTRAINT agent_sdk_outbox_event_fk
        FOREIGN KEY (event_id) REFERENCES agent_sdk_committed_events(event_id)
        ON DELETE CASCADE,

    CONSTRAINT agent_sdk_outbox_kind_check
        CHECK (kind IN ('task_wakeup', 'thread_events_available')),

    CONSTRAINT agent_sdk_outbox_status_check
        CHECK (status IN ('pending', 'claimed', 'delivered', 'expired')),

    CONSTRAINT agent_sdk_outbox_sequence_check
        CHECK (sequence IS NULL OR sequence >= 0),

    CONSTRAINT agent_sdk_outbox_attempt_bounds_check
        CHECK (
            attempt_count >= 0
            AND max_attempts >= 1
            AND attempt_count <= max_attempts
        ),

    CONSTRAINT agent_sdk_outbox_payload_json_check
        CHECK (json_type(payload_json) = 'object'),

    CONSTRAINT agent_sdk_outbox_claim_atomicity_check
        CHECK (
            (claimed_by IS NULL AND claimed_at IS NULL)
            OR (claimed_by IS NOT NULL AND claimed_at IS NOT NULL)
        ),

    CONSTRAINT agent_sdk_outbox_delivered_check
        CHECK (
            (status = 'delivered' AND delivered_at IS NOT NULL)
            OR (status <> 'delivered' AND delivered_at IS NULL)
        ),

    CONSTRAINT agent_sdk_outbox_error_check
        CHECK (
            (status = 'expired' AND last_error IS NOT NULL)
            OR (status <> 'expired' AND last_error IS NULL)
        ),

    CONSTRAINT agent_sdk_outbox_kind_payload_check
        CHECK (
            (kind = 'thread_events_available'
                AND event_id IS NOT NULL
                AND sequence IS NOT NULL)
            OR (kind = 'task_wakeup'
                AND event_id IS NULL
                AND sequence IS NULL)
        )
);

-- ---------------------------------------------------------------------
-- 4. Backfill from the legacy table.  Every legacy row was logically
--    a thread_events_available message (the only kind the original
--    outbox layer wrote), so we hard-code the kind during the copy
--    and rewrite payload_json into the Phase 8.1 advisory shape.
-- ---------------------------------------------------------------------

INSERT INTO agent_sdk_outbox (
    id, kind, thread_id, event_id, sequence, status, payload_json,
    created_at, next_attempt_at, attempt_count, max_attempts,
    last_error, claimed_by, claimed_at, delivered_at
)
SELECT
    id, 'thread_events_available', thread_id, event_id, sequence,
    status, json_object('thread_id', thread_id, 'last_sequence', sequence),
    created_at, next_attempt_at, attempt_count,
    max_attempts, last_error, claimed_by, claimed_at, delivered_at
FROM agent_sdk_outbox_legacy;

DROP TABLE agent_sdk_outbox_legacy;

-- ---------------------------------------------------------------------
-- 5. Recreate indexes — same names, same shapes, plus the new
--    by-kind partial index introduced in Phase 8.1.
-- ---------------------------------------------------------------------

CREATE INDEX agent_sdk_outbox_relay_scan_idx
    ON agent_sdk_outbox (next_attempt_at, id)
    WHERE status = 'pending';

CREATE INDEX agent_sdk_outbox_by_thread_idx
    ON agent_sdk_outbox (thread_id, sequence);

CREATE INDEX agent_sdk_outbox_claimed_sweep_idx
    ON agent_sdk_outbox (claimed_at, id)
    WHERE status = 'claimed';

CREATE INDEX agent_sdk_outbox_relay_scan_by_kind_idx
    ON agent_sdk_outbox (kind, next_attempt_at, id)
    WHERE status = 'pending';
