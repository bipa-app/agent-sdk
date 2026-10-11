-- Validates the check 0022 added NOT VALID. Runs in its own transaction, so
-- the scan holds only SHARE UPDATE EXCLUSIVE and message-head reads and
-- writes continue.

ALTER TABLE agent_sdk_message_heads
    VALIDATE CONSTRAINT agent_sdk_message_heads_history_generation_check;
