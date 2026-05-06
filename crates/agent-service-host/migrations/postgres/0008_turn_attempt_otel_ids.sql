-- Phase 9 · A7: durable OTel correlation for replay-link emission.
--
-- Captured by the worker at attempt-open time from the live OTel
-- context (`Context::current().span().span_context()`).  Subsequent
-- attempts that replay the same logical turn read these ids back out
-- and attach an `agent.replay.original_*` SpanLink to their fresh
-- chat span via `agent_sdk::observability::spans::link_to_replay_origin`.
--
-- Both columns are nullable: pre-A7 rows do not carry OTel ids, and
-- workers running without an exporter installed will continue to write
-- NULL.  Stored as raw TEXT (lower-case hex) instead of a typed
-- column so migrations stay simple — the SDK handles the
-- TraceId/SpanId hex parsing on read.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN otel_trace_id TEXT NULL;

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN otel_span_id TEXT NULL;
