-- Durable "last evidence of work" timestamp for a task.
--
-- The subagent stall budget (`spec.timeout_ms`) fails a child only after it
-- has gone the whole budget with no evidence of work. Committed events alone
-- cannot answer that question:
--
--   * a child parked on ONE long tool call (a 40-minute build) commits
--     nothing at all until the tool returns — its live progress is buffered
--     in memory;
--   * a pure tool-call stream (no text, no thinking) journals nothing while
--     the provider is actively yielding frames;
--   * event retention (`event_ttl_secs`) can purge an event that fell inside
--     the stall window, making a busy child look silent.
--
-- So activity is recorded HERE, on the task row, rather than as an event:
-- retention cannot purge it, and it does not bloat the journal. It is written
-- on the heartbeat path, which already holds the task's lease, so only the
-- worker that owns the row can advance it.
--
-- Nullable on purpose: NULL means "no activity recorded yet" — rows from
-- before this migration, and a child that has not yet produced its first sign
-- of life. Readers fall back to `created_at` (spawn is the initial evidence
-- of life), which is exactly the pre-migration behaviour.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN last_activity_at TIMESTAMPTZ NULL;
