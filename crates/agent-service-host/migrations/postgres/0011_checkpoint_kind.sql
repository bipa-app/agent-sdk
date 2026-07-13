-- Issue #354: explicit checkpoint provenance so the slot-shift commit
-- path can distinguish a synthetic cancel-salvage checkpoint from a
-- fully billed turn WITHOUT inferring from token usage (providers may
-- legitimately report zero usage on real completions).
ALTER TABLE agent_sdk_turn_checkpoints
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'full_turn';
