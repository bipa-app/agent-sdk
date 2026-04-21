-- Per-turn caller metadata captured at task submission time.
--
-- Passed through to AgentDefinition.tools_fn at turn start to enable
-- per-turn tool filtering based on application-level caller identity
-- (user kind, role, entry point, etc.). The SDK does not interpret
-- this value — it's an opaque JSON blob the application filter
-- deserializes into its own domain type.
--
-- Nullable on purpose: SQL NULL means "the submitter did not attach
-- any metadata" (skip the filter, use static tools) and is
-- semantically distinct from a stored JSON `null` literal (which
-- would be an explicit, empty caller context). Existing rows from
-- before this migration are NULL, which is the correct fallback.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN caller_metadata_json JSONB NULL;
