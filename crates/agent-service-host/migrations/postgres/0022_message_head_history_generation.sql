-- Count of history rewrites (replace_history, append_compaction,
-- append_repair) on the message head; append-only commits never advance it.
-- The measured compaction trigger only trusts attempts stamped with the
-- current value. Existing rows start at 0.

ALTER TABLE agent_sdk_message_heads
    ADD COLUMN history_generation BIGINT NOT NULL DEFAULT 0;

ALTER TABLE agent_sdk_message_heads
    ADD CONSTRAINT agent_sdk_message_heads_history_generation_check
        CHECK (history_generation >= 0);
