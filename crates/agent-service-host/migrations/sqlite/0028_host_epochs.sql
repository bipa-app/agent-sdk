-- Process epochs, SQLite dialect. Mirror of postgres/0028_host_epochs.sql
-- (TIMESTAMPTZ → TEXT; no lock_timeout, the daemon's write lock
-- serializes the migration).

CREATE TABLE agent_sdk_host_epochs (
    epoch_id   TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    label      TEXT NULL
);

ALTER TABLE agent_sdk_tasks
    ADD COLUMN lease_epoch TEXT NULL;
