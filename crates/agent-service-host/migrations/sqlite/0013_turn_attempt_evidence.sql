-- Durable provider-call evidence for historical cost and thinking audits
-- (SQLite dialect). Mirror of postgres/0013_turn_attempt_evidence.sql.
--
-- Nullable additions preserve old rows without backfilling guessed evidence.

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN cache_creation_input_tokens INTEGER NULL
        CHECK (cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0);

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN route_provider TEXT NULL;

-- Thinking is recorded as two orthogonal dimensions, mirroring the request
-- shape: adaptivity (did the request let the provider choose the depth?) and
-- effort (which level the request carried, if any). An adaptive request may
-- still carry an effort hint, so neither column can encode the other.
ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN thinking_adaptive INTEGER NOT NULL DEFAULT 0
        CHECK (thinking_adaptive IN (0, 1));

ALTER TABLE agent_sdk_turn_attempts
    ADD COLUMN resolved_effort TEXT NULL
        CHECK (resolved_effort IS NULL
               OR resolved_effort IN ('low', 'medium', 'high', 'xhigh', 'max'));
