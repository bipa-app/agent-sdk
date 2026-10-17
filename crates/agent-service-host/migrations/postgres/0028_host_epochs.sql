-- Process epochs: which host process granted each lease.
--
-- A host writes one row here when it starts, and every lease it grants
-- from then on records that epoch in agent_sdk_tasks.lease_epoch. Worker
-- ids repeat across restarts, so the epoch tells a dead process's lease
-- from a live one. A host that is the only process on its store requeues
-- every lease of an older epoch at start; hosts that share a store keep
-- waiting for lease expiry and only stamp the column.
--
-- Both statements are catalog-only (no table rewrite, no scan). The ALTER
-- takes a brief ACCESS EXCLUSIVE lock on agent_sdk_tasks, which queues
-- behind any open transaction on the table; the lock_timeout makes a
-- blocked boot fail fast and restart instead of stalling every task
-- query behind the waiting lock. It needs sqlx's per-migration
-- transaction, so this file must not opt out of it.

SET LOCAL lock_timeout = '5s';

CREATE TABLE agent_sdk_host_epochs (
    epoch_id   TEXT        PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    label      TEXT        NULL
);

-- NULL for every lease granted before this column existed, and for every
-- lease an older build grants during a rolling deploy.
ALTER TABLE agent_sdk_tasks
    ADD COLUMN lease_epoch TEXT NULL;
