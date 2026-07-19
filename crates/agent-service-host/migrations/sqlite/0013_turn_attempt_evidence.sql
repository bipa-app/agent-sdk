-- Durable provider-call evidence for historical cost and thinking audits
-- (SQLite dialect). Mirror of postgres/0013_turn_attempt_evidence.sql.
--
-- Nullable additions preserve old rows without backfilling guessed evidence.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN cache_creation_input_tokens INTEGER NULL
        CHECK (cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0);

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN route_provider TEXT NULL;

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN resolved_effort TEXT NULL
        CHECK (resolved_effort IS NULL OR resolved_effort IN ('low', 'medium', 'high', 'max'));
