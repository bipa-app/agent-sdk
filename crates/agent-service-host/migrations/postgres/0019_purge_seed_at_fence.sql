-- Purge identity becomes durable at FENCE time, not at completion: a
-- deleting row now carries the purge seed (root, scope, started_at) in the
-- same column that later holds the completed receipt, so a crash-retry
-- resumes under the first attempt's identity.

-- Rows fenced before seeds existed have no recorded identity; adopt the
-- fence write's own timestamp as started_at so the constraint below can
-- hold for every live row without inventing a scope wider than one thread.
UPDATE agent_sdk_threads
SET purge_receipt_json = jsonb_build_object(
    'root_thread_id', thread_id,
    'scope', 'thread',
    'started_at', to_char(updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
)
WHERE status = 'deleting' AND purge_receipt_json IS NULL;

ALTER TABLE agent_sdk_threads
    DROP CONSTRAINT agent_sdk_threads_purge_receipt_check;

ALTER TABLE agent_sdk_threads
    ADD CONSTRAINT agent_sdk_threads_purge_record_check
    CHECK (
        (status IN ('deleting', 'deleted') AND purge_receipt_json IS NOT NULL)
        OR (status NOT IN ('deleting', 'deleted') AND purge_receipt_json IS NULL)
    );
