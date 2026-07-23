-- Durable active -> deleting -> deleted fence for resurrection-proof purge.
ALTER TABLE agent_sdk_threads
    DROP CONSTRAINT agent_sdk_threads_status_check;

ALTER TABLE agent_sdk_threads
    ADD CONSTRAINT agent_sdk_threads_status_check
    CHECK (status IN ('active', 'completed', 'deleting', 'deleted'));

ALTER TABLE agent_sdk_threads
    ADD COLUMN purge_receipt_json JSONB NULL;

ALTER TABLE agent_sdk_threads
    ADD CONSTRAINT agent_sdk_threads_purge_receipt_check
    CHECK (
        (status = 'deleted' AND purge_receipt_json IS NOT NULL)
        OR (status <> 'deleted' AND purge_receipt_json IS NULL)
    );
