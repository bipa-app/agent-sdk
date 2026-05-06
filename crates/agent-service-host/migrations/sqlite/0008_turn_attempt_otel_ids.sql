-- Phase 9 · A7: durable OTel correlation for replay-link emission
-- (SQLite dialect).  Mirror of postgres/0008_turn_attempt_otel_ids.sql.
--
-- See the Postgres variant for the full rationale.  Nullable on
-- purpose so pre-A7 rows continue to validate and workers running
-- without an OTel exporter keep writing NULL.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN otel_trace_id TEXT NULL;

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN otel_span_id TEXT NULL;
