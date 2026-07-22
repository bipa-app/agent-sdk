-- Durable provider-call evidence for historical cost and thinking audits.
--
-- All columns are nullable so existing attempts retain their exact row shape:
-- the migration neither rewrites nor infers evidence that was not captured at
-- dispatch time.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN cache_creation_input_tokens BIGINT NULL
        CHECK (cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0);

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN route_provider TEXT NULL;

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_mode TEXT NULL
        CHECK (thinking_mode IS NULL
               OR thinking_mode IN ('off', 'default', 'budget', 'adaptive'));

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_budget_tokens BIGINT NULL
        CHECK ((thinking_budget_tokens IS NULL OR thinking_budget_tokens > 0)
               AND NOT (thinking_mode = 'budget' AND thinking_budget_tokens IS NULL)
               AND NOT (thinking_mode <> 'budget' AND thinking_budget_tokens IS NOT NULL));

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_effort TEXT NULL
        CHECK ((thinking_effort IS NULL
                OR thinking_effort IN ('low', 'medium', 'high', 'xhigh', 'max'))
               AND NOT (thinking_mode = 'off' AND thinking_effort IS NOT NULL));
