-- Durable "last evidence of work" timestamp for a task, SQLite dialect.
-- Mirror of postgres/0012_task_last_activity_at.sql.
--
-- See the Postgres variant for the full rationale. Nullable on purpose: NULL
-- means "no activity recorded yet", and readers fall back to `created_at`
-- (spawn is the initial evidence of life). Stored as TEXT — an RFC 3339
-- timestamp, matching this schema's other instants.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN last_activity_at TEXT NULL;
