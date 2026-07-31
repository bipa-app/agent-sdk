ALTER TABLE agent_sdk_message_heads
    ADD COLUMN compactions_json TEXT NULL
        CHECK (compactions_json IS NULL OR json_type(compactions_json) = 'array');
