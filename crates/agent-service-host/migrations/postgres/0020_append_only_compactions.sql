ALTER TABLE agent_sdk_message_heads
    ADD COLUMN compactions_json JSONB NULL;

ALTER TABLE agent_sdk_message_heads
    ADD CONSTRAINT agent_sdk_message_heads_compactions_json_check
        CHECK (compactions_json IS NULL OR jsonb_typeof(compactions_json) = 'array');
