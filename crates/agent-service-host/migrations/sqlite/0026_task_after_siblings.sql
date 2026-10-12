-- Earlier children of the same batch that a pending tool child waits for,
-- SQLite dialect. Mirror of postgres/0026_task_after_siblings.sql. Stored as
-- TEXT holding a JSON array, matching this schema's other JSON columns.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json TEXT NULL;
