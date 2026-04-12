-- SQLite-dialect mirror of postgres/0002_event_journal_outbox.sql
-- See 0001_durable_core.sql header for dialect translation rules.

-- =====================================================================
-- 1. Committed Events — append-only event journal
-- =====================================================================

CREATE TABLE agent_sdk_committed_events (
    event_id    TEXT        PRIMARY KEY,
    thread_id   TEXT        NOT NULL,
    sequence    INTEGER     NOT NULL,
    event_json  TEXT        NOT NULL,
    committed_at TEXT       NOT NULL,

    CONSTRAINT agent_sdk_committed_events_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    CONSTRAINT agent_sdk_committed_events_thread_sequence_key
        UNIQUE (thread_id, sequence),

    CONSTRAINT agent_sdk_committed_events_sequence_check
        CHECK (sequence >= 0),

    CONSTRAINT agent_sdk_committed_events_event_json_check
        CHECK (json_type(event_json) = 'object')
);

CREATE INDEX agent_sdk_committed_events_replay_idx
    ON agent_sdk_committed_events (thread_id, sequence);

CREATE INDEX agent_sdk_committed_events_by_time_idx
    ON agent_sdk_committed_events (thread_id, committed_at);


-- =====================================================================
-- 2. Transactional Outbox — relay delivery buffer
-- =====================================================================

CREATE TABLE agent_sdk_outbox (
    id              TEXT        PRIMARY KEY,
    thread_id       TEXT        NOT NULL,
    event_id        TEXT        NOT NULL,
    sequence        INTEGER     NOT NULL,
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

    CONSTRAINT agent_sdk_outbox_status_check
        CHECK (status IN ('pending', 'claimed', 'delivered', 'expired')),

    CONSTRAINT agent_sdk_outbox_sequence_check
        CHECK (sequence >= 0),

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
        )
);

CREATE INDEX agent_sdk_outbox_relay_scan_idx
    ON agent_sdk_outbox (next_attempt_at, id)
    WHERE status = 'pending';

CREATE INDEX agent_sdk_outbox_by_thread_idx
    ON agent_sdk_outbox (thread_id, sequence);

CREATE INDEX agent_sdk_outbox_claimed_sweep_idx
    ON agent_sdk_outbox (claimed_at, id)
    WHERE status = 'claimed';


-- =====================================================================
-- 3. Retention Cursors — per-thread retention watermarks
-- =====================================================================

CREATE TABLE agent_sdk_retention_cursors (
    thread_id       TEXT        PRIMARY KEY,
    retention_floor INTEGER     NOT NULL DEFAULT 0,
    updated_at      TEXT        NOT NULL,

    CONSTRAINT agent_sdk_retention_cursors_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    CONSTRAINT agent_sdk_retention_cursors_floor_check
        CHECK (retention_floor >= 0)
);
