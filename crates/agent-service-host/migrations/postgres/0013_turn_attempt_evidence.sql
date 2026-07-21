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

-- 'adaptive' is not an Effort level: it records that the request asked for
-- adaptive thinking and the provider's API chose the depth, which a NULL
-- would make indistinguishable from "no thinking was requested at all".
ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN resolved_effort TEXT NULL
        CHECK (resolved_effort IS NULL
               OR resolved_effort IN ('low', 'medium', 'high', 'max', 'adaptive'));
