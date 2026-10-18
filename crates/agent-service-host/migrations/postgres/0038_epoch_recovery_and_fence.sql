-- Epoch fencing: how each process epoch recovers, and when a newer
-- exclusive epoch superseded it.
--
-- `recovery` is the EpochRecovery mode the process began under
-- (`lease_expiry`, `single_host`, `postgres_leader`); NULL for epochs begun
-- before this migration. A process that begins an exclusive epoch
-- (`single_host` or `postgres_leader`) sets `superseded_at` on every other
-- exclusive epoch in the same transaction. The store then refuses a
-- superseded epoch's acquires, heartbeats and lease-guarded commits, and a
-- `postgres_leader` successor reclaims at once only the leases of
-- superseded epochs. `lease_expiry` and NULL epochs are never marked.
--
-- Both statements are catalog-only on a small table (one row per process
-- start): no rewrite, no scan. The lock_timeout bounds the brief ACCESS
-- EXCLUSIVE wait, as in 0028; it needs sqlx's per-migration transaction.

SET LOCAL lock_timeout = '5s';

ALTER TABLE agent_sdk_host_epochs
    ADD COLUMN recovery TEXT NULL,
    ADD COLUMN superseded_at TIMESTAMPTZ NULL;
