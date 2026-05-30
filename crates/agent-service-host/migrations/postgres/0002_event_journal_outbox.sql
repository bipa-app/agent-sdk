-- =====================================================================
-- 0002_event_journal_outbox.sql
-- Event Journal and Transactional Outbox Storage Extensions
-- =====================================================================
--
-- Extends the durable core with three new surfaces:
--
--   1. agent_sdk_committed_events  — append-only event journal
--   2. agent_sdk_outbox            — transactional relay buffer
--   3. agent_sdk_retention_cursors — per-thread retention watermarks
--
-- These tables form the storage backbone for Phase 6 event streaming
-- and Phase 8 GA hardening.  The committed_events table is the
-- authoritative source of truth for thread-scoped event ordering; the
-- outbox is a delivery buffer that never owns ordering semantics.

-- =====================================================================
-- 1. Committed Events — append-only event journal
-- =====================================================================
--
-- Every server-committed event receives a UUID v7 event_id, a
-- monotonic per-thread sequence, and a server commit timestamp.
-- The (thread_id, sequence) pair is the durable unique key.
--
-- Batch inserts assign contiguous sequence ranges atomically, so no
-- two callers can observe the same sequence for a thread.  A database
-- backend uses SELECT ... FOR UPDATE on the sequence high-water mark
-- or relies on the UNIQUE constraint for serialisation.

CREATE TABLE agent_sdk_committed_events (
    event_id    TEXT        PRIMARY KEY,
    thread_id   TEXT        NOT NULL,
    sequence    BIGINT      NOT NULL,
    event_json  JSONB       NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,

    -- Thread ownership.
    CONSTRAINT agent_sdk_committed_events_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    -- (thread_id, sequence) is the durable unique key.
    CONSTRAINT agent_sdk_committed_events_thread_sequence_key
        UNIQUE (thread_id, sequence),

    -- Sequence is non-negative.
    CONSTRAINT agent_sdk_committed_events_sequence_check
        CHECK (sequence >= 0),

    -- Event payload is a JSON object.
    CONSTRAINT agent_sdk_committed_events_event_json_check
        CHECK (jsonb_typeof(event_json) = 'object')
);

-- Replay query: events for a thread in sequence order, optionally
-- within a range.  This is the primary read path for reconnecting
-- clients.
CREATE INDEX agent_sdk_committed_events_replay_idx
    ON agent_sdk_committed_events (thread_id, sequence);

-- Time-range queries for operational inspection and retention scans.
CREATE INDEX agent_sdk_committed_events_by_time_idx
    ON agent_sdk_committed_events (thread_id, committed_at);


-- =====================================================================
-- 2. Transactional Outbox — relay delivery buffer
-- =====================================================================
--
-- Outbox rows are inserted in the SAME SQL transaction as the
-- committed events they reference.  Relay workers claim pending rows,
-- attempt delivery, and mark them as delivered or failed.
--
-- The outbox is NOT the authority for event ordering.  Consumers
-- replay from committed_events; the outbox merely triggers delivery
-- notifications.  Terminal durable state never depends on successful
-- outbox relay.

CREATE TABLE agent_sdk_outbox (
    id              TEXT        PRIMARY KEY,
    thread_id       TEXT        NOT NULL,
    event_id        TEXT        NOT NULL,
    sequence        BIGINT      NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'pending',
    payload_json    JSONB       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    attempt_count   BIGINT      NOT NULL DEFAULT 0,
    max_attempts    BIGINT      NOT NULL DEFAULT 5,
    last_error      TEXT        NULL,
    claimed_by      TEXT        NULL,
    claimed_at      TIMESTAMPTZ NULL,
    delivered_at    TIMESTAMPTZ NULL,

    -- Thread ownership.
    CONSTRAINT agent_sdk_outbox_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    -- Event provenance.
    CONSTRAINT agent_sdk_outbox_event_fk
        FOREIGN KEY (event_id) REFERENCES agent_sdk_committed_events(event_id)
        ON DELETE CASCADE,

    -- Only known outbox statuses.
    CONSTRAINT agent_sdk_outbox_status_check
        CHECK (status IN ('pending', 'claimed', 'delivered', 'expired')),

    -- Sequence copied from committed_events; must be non-negative.
    CONSTRAINT agent_sdk_outbox_sequence_check
        CHECK (sequence >= 0),

    -- Retry budget must be at least 1.
    CONSTRAINT agent_sdk_outbox_attempt_bounds_check
        CHECK (
            attempt_count >= 0
            AND max_attempts >= 1
            AND attempt_count <= max_attempts
        ),

    -- Payload must be a JSON object.
    CONSTRAINT agent_sdk_outbox_payload_json_check
        CHECK (jsonb_typeof(payload_json) = 'object'),

    -- Claim atomicity: claimed_by/claimed_at are either both present
    -- or both absent.
    CONSTRAINT agent_sdk_outbox_claim_atomicity_check
        CHECK (
            (claimed_by IS NULL AND claimed_at IS NULL)
            OR (claimed_by IS NOT NULL AND claimed_at IS NOT NULL)
        ),

    -- Terminal state constraints:
    --   - Delivered rows must have delivered_at.
    --   - Non-delivered rows must not have delivered_at.
    CONSTRAINT agent_sdk_outbox_delivered_check
        CHECK (
            (status = 'delivered' AND delivered_at IS NOT NULL)
            OR (status <> 'delivered' AND delivered_at IS NULL)
        ),

    -- Expired rows must carry an error message.
    CONSTRAINT agent_sdk_outbox_error_check
        CHECK (
            (status = 'expired' AND last_error IS NOT NULL)
            OR (status <> 'expired' AND last_error IS NULL)
        )
);

-- Relay worker scan: pending rows ordered by next_attempt_at for
-- fair scheduling.
CREATE INDEX agent_sdk_outbox_relay_scan_idx
    ON agent_sdk_outbox (next_attempt_at, id)
    WHERE status = 'pending';

-- Thread-scoped inspection.
CREATE INDEX agent_sdk_outbox_by_thread_idx
    ON agent_sdk_outbox (thread_id, sequence);

-- Stale-claim sweep: find claimed rows that have not been delivered
-- within a timeout window.
CREATE INDEX agent_sdk_outbox_claimed_sweep_idx
    ON agent_sdk_outbox (claimed_at, id)
    WHERE status = 'claimed';


-- =====================================================================
-- 3. Retention Cursors — per-thread retention watermarks
-- =====================================================================
--
-- The retention floor is the lowest committed-event sequence number
-- that is guaranteed to still exist for a thread.  Events below the
-- floor may have been garbage-collected by the janitor.
--
-- Replay clients check the floor to detect retention gaps.  The
-- janitor advances the floor and purges old events atomically.

CREATE TABLE agent_sdk_retention_cursors (
    thread_id       TEXT        PRIMARY KEY,
    retention_floor BIGINT      NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL,

    -- Thread ownership.
    CONSTRAINT agent_sdk_retention_cursors_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,

    -- Retention floor is non-negative.
    CONSTRAINT agent_sdk_retention_cursors_floor_check
        CHECK (retention_floor >= 0)
);
