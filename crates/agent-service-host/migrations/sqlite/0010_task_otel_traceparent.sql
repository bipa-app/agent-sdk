-- Per-task OTel parent span context (W3C `traceparent`), SQLite dialect.
-- Mirror of postgres/0010_task_otel_traceparent.sql.
--
-- See the Postgres variant for the full rationale. Nullable on purpose:
-- NULL means the task has no trace parent (no inbound client trace, or a
-- worker running without an OTel exporter). Stored as TEXT — the W3C
-- `traceparent` header value the daemon rebuilds an OTel parent context
-- from.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN otel_traceparent TEXT NULL;
