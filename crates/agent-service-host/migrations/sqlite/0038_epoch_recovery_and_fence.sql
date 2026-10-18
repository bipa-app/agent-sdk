-- Epoch fencing, SQLite dialect. Mirror of
-- postgres/0038_epoch_recovery_and_fence.sql (TIMESTAMPTZ → TEXT; no
-- lock_timeout, the daemon's write lock serializes the migration).
ALTER TABLE agent_sdk_host_epochs
    ADD COLUMN recovery TEXT NULL;
ALTER TABLE agent_sdk_host_epochs
    ADD COLUMN superseded_at TEXT NULL;
