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
    ADD COLUMN thinking_mode TEXT NULL
        CHECK (thinking_mode IS NULL
               OR thinking_mode IN ('off', 'default', 'budget', 'adaptive'));

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_budget_tokens INTEGER NULL
        CHECK ((thinking_budget_tokens IS NULL OR thinking_budget_tokens > 0)
               AND NOT (thinking_mode = 'budget' AND thinking_budget_tokens IS NULL)
               AND NOT (thinking_mode <> 'budget' AND thinking_budget_tokens IS NOT NULL));

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_effort TEXT NULL
        CHECK ((thinking_effort IS NULL
                OR thinking_effort IN ('low', 'medium', 'high', 'xhigh', 'max'))
               AND NOT (thinking_mode = 'off' AND thinking_effort IS NOT NULL));
