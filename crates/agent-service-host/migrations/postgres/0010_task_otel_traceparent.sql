-- Per-task OTel parent span context, captured as the W3C `traceparent`
-- of the span this task's spans nest under.
--
-- For a root-turn task this is the inbound client span extracted from
-- the submitting gRPC call's metadata, so the daemon's `invoke_agent`
-- span continues the caller's distributed trace. For a child tool task
-- it is stamped at spawn from the root turn's `invoke_agent` span ids
-- (persisted on agent_sdk_turn_attempts) so the child's `execute_tool`
-- span nests under the turn root.
--
-- Nullable on purpose: NULL means "no trace context" — existing rows
-- from before this migration, and workers running without an OTel
-- exporter, keep it NULL, which is the correct fallback. The SDK never
-- interprets the value beyond rebuilding an OTel parent `Context`.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN otel_traceparent TEXT NULL;
