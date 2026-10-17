-- agent_sdk_execution_intents becomes the record of every effect (design
-- §6.3): a tool call, a model request, a process start or a git step. An
-- effect's row commits before it runs, under its owner task's lease.
--
-- Existing rows are tool calls on their first attempt, so the two NOT NULL
-- columns take constant defaults (catalog-only on Postgres 11+, no rewrite).
-- The other columns are NULL for every row written before this migration.
--
-- The status and effect_class checks from 0003 stay as they are. An
-- interrupted, salvaged or abandoned effect is status 'failed' with error
-- set (0003's error check) and resolution naming which.
--
-- The checks land NOT VALID, so this migration scans nothing under the
-- ACCESS EXCLUSIVE lock its ALTERs take. Writes are checked at once; 0029
-- validates the existing rows. A boot that cannot get the lock within 5s
-- fails and restarts rather than queueing every intent query behind it.

SET LOCAL lock_timeout = '5s';

ALTER TABLE agent_sdk_execution_intents
    ADD COLUMN effect_kind     TEXT   NOT NULL DEFAULT 'tool_call',
    ADD COLUMN attempt         BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN resolution      TEXT   NULL,
    ADD COLUMN captured_output TEXT   NULL,
    ADD COLUMN journal_anchor  BIGINT NULL,
    ADD COLUMN thread_id       TEXT   NULL;

ALTER TABLE agent_sdk_execution_intents
    ADD CONSTRAINT agent_sdk_execution_intents_kind_check
        CHECK (effect_kind IN ('tool_call', 'llm_request', 'process_start', 'git_step')) NOT VALID,
    ADD CONSTRAINT agent_sdk_execution_intents_resolution_check
        CHECK (
            resolution IS NULL
            OR (status = 'failed' AND resolution IN ('interrupted', 'salvaged', 'abandoned'))
        ) NOT VALID,
    ADD CONSTRAINT agent_sdk_execution_intents_attempt_check
        CHECK (attempt >= 1) NOT VALID;
