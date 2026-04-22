-- Per-turn caller metadata captured at task submission time (SQLite dialect).
-- Mirror of postgres/0006_task_caller_metadata.sql.
--
-- See the Postgres variant for the full rationale. Nullable on purpose:
-- NULL distinguishes "submitter attached no metadata" from an explicit
-- JSON null literal. JSON is stored as TEXT per the SQLite dialect.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN caller_metadata_json TEXT NULL;
