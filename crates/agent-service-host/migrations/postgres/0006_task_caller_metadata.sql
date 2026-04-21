-- Per-turn caller metadata captured at task submission time.
--
-- Passed through to AgentDefinition.tools_fn at turn start to enable
-- per-turn tool filtering based on application-level caller identity
-- (user kind, role, entry point, etc.). The SDK does not interpret
-- this value — it's an opaque JSON blob the application filter
-- deserializes into its own domain type.
--
-- Defaults to JSON null for backwards compatibility with rows inserted
-- before this column existed.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN caller_metadata_json JSONB NOT NULL DEFAULT 'null'::jsonb;
