-- Validates the checks 0027 added NOT VALID (sorted order runs 0027 first).
-- Runs in its own transaction, so each scan holds only SHARE UPDATE
-- EXCLUSIVE and intent reads and writes continue.

ALTER TABLE agent_sdk_execution_intents
    VALIDATE CONSTRAINT agent_sdk_execution_intents_kind_check;

ALTER TABLE agent_sdk_execution_intents
    VALIDATE CONSTRAINT agent_sdk_execution_intents_resolution_check;

ALTER TABLE agent_sdk_execution_intents
    VALIDATE CONSTRAINT agent_sdk_execution_intents_attempt_check;
