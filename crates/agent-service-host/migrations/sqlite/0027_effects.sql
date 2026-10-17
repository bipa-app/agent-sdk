-- agent_sdk_execution_intents becomes the record of every effect, SQLite
-- dialect. Mirror of postgres/0027_effects.sql. SQLite checks an added
-- column's CHECK against the existing rows, which the defaults satisfy, so
-- there is no NOT VALID step and no 0029.

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN effect_kind TEXT NOT NULL DEFAULT 'tool_call'
        CONSTRAINT agent_sdk_execution_intents_kind_check
        CHECK (effect_kind IN ('tool_call', 'llm_request', 'process_start', 'git_step'));

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1
        CONSTRAINT agent_sdk_execution_intents_attempt_check
        CHECK (attempt >= 1);

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN resolution TEXT NULL
        CONSTRAINT agent_sdk_execution_intents_resolution_check
        CHECK (
            resolution IS NULL
            OR (status = 'failed' AND resolution IN ('interrupted', 'salvaged'))
        );

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN captured_output TEXT NULL;

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN journal_anchor INTEGER NULL;

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN thread_id TEXT NULL;
