-- Durable provider-call evidence for historical cost and thinking audits.
--
-- All evidence columns are nullable (or default to their "absent" value) so
-- existing attempts retain their exact row shape: the migration neither
-- rewrites nor infers evidence that was not captured at dispatch time.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN cache_creation_input_tokens BIGINT NULL
        CHECK (cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0);

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN route_provider TEXT NULL;

-- Thinking is recorded as two orthogonal dimensions, mirroring the request
-- shape: adaptivity (did the request let the provider choose the depth?) and
-- effort (which level the request carried, if any). An adaptive request may
-- still carry an effort hint, so neither column can encode the other.
ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_adaptive BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN resolved_effort TEXT NULL
        CHECK (resolved_effort IS NULL
               OR resolved_effort IN ('low', 'medium', 'high', 'xhigh', 'max'));
