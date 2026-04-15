-- =====================================================================
-- 0005_outbox_message_kind.sql
-- ENG-7965: Phase 8.1 — Transactional Outbox Contract and Message Kinds
-- =====================================================================
--
-- Adds the logical message kind discriminator and relaxes
-- event_id / sequence to support the two Phase 8.1 kinds:
--
--   * `thread_events_available` — coalesced advisory written in the
--     same transaction as a committed-event batch.  Carries the
--     highest event_id / sequence in the batch as durable references.
--
--   * `task_wakeup` — advisory written in the same transaction as a
--     task-journal mutation that makes a task runnable.  Carries no
--     event references; the consumer resolves the task by reading
--     `agent_sdk_tasks` directly.
--
-- See `crates/agent-server/src/journal/outbox_message.rs` for the
-- typed payload shapes; the relay never encodes anything else into
-- this table.

-- ---------------------------------------------------------------------
-- 1. Add the `kind` column with a temporary default so existing rows
--    (if any) backfill cleanly to the legacy semantic — every row
--    written by ENG-7986 was logically a thread_events_available
--    message. Rewrite their payload_json into the Phase 8.1 advisory
--    shape at the same time, then drop the default so future inserts
--    must be explicit.
-- ---------------------------------------------------------------------

ALTER TABLE agent_sdk_outbox
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'thread_events_available';

UPDATE agent_sdk_outbox
SET payload_json = jsonb_build_object(
    'thread_id', thread_id,
    'last_sequence', sequence
);

ALTER TABLE agent_sdk_outbox ALTER COLUMN kind DROP DEFAULT;

ALTER TABLE agent_sdk_outbox
    ADD CONSTRAINT agent_sdk_outbox_kind_check
        CHECK (kind IN ('task_wakeup', 'thread_events_available'));

-- ---------------------------------------------------------------------
-- 2. Drop the per-event uniqueness FK and CHECK so `task_wakeup` rows
--    can carry NULL event_id / sequence.  The FK is reattached below
--    with `MATCH SIMPLE` semantics so NULLs short-circuit it.
-- ---------------------------------------------------------------------

ALTER TABLE agent_sdk_outbox
    DROP CONSTRAINT agent_sdk_outbox_event_fk;

ALTER TABLE agent_sdk_outbox
    DROP CONSTRAINT agent_sdk_outbox_sequence_check;

ALTER TABLE agent_sdk_outbox
    ALTER COLUMN event_id DROP NOT NULL,
    ALTER COLUMN sequence DROP NOT NULL;

ALTER TABLE agent_sdk_outbox
    ADD CONSTRAINT agent_sdk_outbox_event_fk
        FOREIGN KEY (event_id) REFERENCES agent_sdk_committed_events(event_id)
        ON DELETE CASCADE;

ALTER TABLE agent_sdk_outbox
    ADD CONSTRAINT agent_sdk_outbox_sequence_check
        CHECK (sequence IS NULL OR sequence >= 0);

-- ---------------------------------------------------------------------
-- 3. Enforce the Phase 8.1 transactional contract directly in SQL:
--    `thread_events_available` rows MUST carry both references;
--    `task_wakeup` rows MUST carry neither.
-- ---------------------------------------------------------------------

ALTER TABLE agent_sdk_outbox
    ADD CONSTRAINT agent_sdk_outbox_kind_payload_check
        CHECK (
            (kind = 'thread_events_available'
                AND event_id IS NOT NULL
                AND sequence IS NOT NULL)
            OR (kind = 'task_wakeup'
                AND event_id IS NULL
                AND sequence IS NULL)
        );

-- ---------------------------------------------------------------------
-- 4. Index by kind for the relay scan.  The scheduler may want to
--    drain wakeups before event-availability advisories under load;
--    a partial index on the (kind, next_attempt_at) pair keeps that
--    cheap.
-- ---------------------------------------------------------------------

CREATE INDEX agent_sdk_outbox_relay_scan_by_kind_idx
    ON agent_sdk_outbox (kind, next_attempt_at, id)
    WHERE status = 'pending';
