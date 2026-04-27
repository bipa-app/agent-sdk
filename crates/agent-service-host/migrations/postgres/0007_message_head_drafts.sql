-- In-flight draft messages on message heads (Postgres dialect).
-- Mirror of sqlite/0007_message_head_drafts.sql.
--
-- The message projection's `history_json` is the committed conversation
-- — only updated when a turn fully completes. Long-running turns that
-- suspend at every tool boundary previously had no way to surface
-- their accumulated work if the resume LLM call later errored out:
-- `fail_root_turn` clears the parent task's `state_json` payload, so
-- the in-flight `suspended_messages` list vanished with it and the
-- next turn started from the latest committed checkpoint instead of
-- from where the failed turn actually got to.
--
-- `draft_messages_json` is the recovery slot for that work. The
-- worker writes the full `suspended_messages` snapshot on every
-- successful tool-boundary suspension, and the atomic completed-turn
-- transaction wipes it back to NULL on each commit. `recover_thread`
-- folds any non-empty draft into the next-turn view so the agent
-- picks up its partial conversation instead of starting over.
--
-- JSONB so the database can validate document shape and we can index
-- on it later if needed; nullable so a brand-new message head and a
-- freshly committed thread both represent "no in-flight draft" with
-- NULL rather than "[]".

ALTER TABLE agent_sdk_message_heads
    ADD COLUMN draft_messages_json JSONB NULL;

-- Mirror of `agent_sdk_message_heads_history_json_check`: when the
-- column is non-null it must be a JSON array, never an object or scalar.
ALTER TABLE agent_sdk_message_heads
    ADD CONSTRAINT agent_sdk_message_heads_draft_messages_json_check
        CHECK (
            draft_messages_json IS NULL
            OR jsonb_typeof(draft_messages_json) = 'array'
        );
