-- Earlier children of the same batch that a pending tool child waits for.
-- NULL for every row that may run as soon as it is pending, including
-- every row written before this column existed.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json JSONB NULL;
