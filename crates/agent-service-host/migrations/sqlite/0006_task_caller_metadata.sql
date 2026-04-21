-- Per-turn caller metadata captured at task submission time (SQLite dialect).
-- Mirror of postgres/0006_task_caller_metadata.sql.
--
-- Passed through to AgentDefinition.tools_fn at turn start to enable
-- per-turn tool filtering based on application-level caller identity
-- (user kind, role, entry point, etc.). The SDK does not interpret
-- this value — it's an opaque JSON blob the application filter
-- deserializes into its own domain type.
--
-- JSON is stored as TEXT ('null' literal) to match the SQLite dialect's
-- handling of JSONB columns elsewhere in the schema.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN caller_metadata_json TEXT NOT NULL DEFAULT 'null';
