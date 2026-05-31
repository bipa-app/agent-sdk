-- =====================================================================
-- 0009_idempotency.sql
-- ENG-8707 (Phase 10 · E): durable at-least-once idempotency records.
-- =====================================================================
--
-- Clients that retry mutating control-plane calls (CreateThread,
-- SubmitThreadWork, ForkThread, DecideConfirmation) carry a
-- caller-supplied request_id. The dedup table used to live in process
-- memory, so a retry that arrived after a host restart was not deduped
-- and produced a duplicate root turn / double fork / double-applied
-- decision (the classic at-least-once footgun).
--
-- This table persists the dedup records so they survive restart. For
-- the submission path the row is claimed inside the SAME transaction as
-- task admission, so there is no time-of-check / time-of-use window.
--
--   request_id    — caller-supplied idempotency key (primary key)
--   kind          — operation discriminator the key was first used for
--                   ('create_thread' | 'submit_work' | 'fork_thread'
--                    | 'decide_confirmation')
--   fingerprint   — payload fingerprint; a retry with a different
--                   fingerprint under the same key is a conflict
--   result_json   — operation-specific durable references (thread id,
--                   task id, …) the transport reconstructs the original
--                   response from
--   created_at    — when the key was first claimed

CREATE TABLE agent_sdk_idempotency (
    request_id  TEXT        PRIMARY KEY,
    kind        TEXT        NOT NULL,
    fingerprint BYTEA       NOT NULL,
    result_json JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,

    CONSTRAINT agent_sdk_idempotency_kind_check
        CHECK (kind IN ('create_thread', 'submit_work', 'fork_thread', 'decide_confirmation'))
);
