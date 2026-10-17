-- SQLite dialect of postgres/0035_thread_views.sql.
--
-- The session view: a per-thread projection of the committed events, written
-- in the same transaction as the events by a store whose ViewConfig enables
-- it. A host with views off (the default) never writes these tables.
--
-- Additive only: two new, empty tables and one index on one of them. No
-- existing relation is altered or locked. No foreign key to agent_sdk_threads,
-- because creating one would lock that table; a purge deletes the rows itself.
-- A missing view row means "not materialized": the first read folds the
-- thread's retained events into it.

CREATE TABLE agent_sdk_thread_views (
    thread_id    TEXT        PRIMARY KEY,
    view_schema  INTEGER     NOT NULL,
    -- The last folded event's sequence; NULL before the first.
    sequence     INTEGER      NULL,
    next_index   INTEGER      NOT NULL,
    status_json  TEXT        NOT NULL,
    updated_at   TEXT        NOT NULL
);

CREATE TABLE agent_sdk_thread_view_items (
    thread_id   TEXT   NOT NULL,
    item_key    TEXT   NOT NULL,
    item_index  INTEGER NOT NULL,
    -- The last event that changed the item.
    sequence    INTEGER NOT NULL,
    body_json   TEXT   NOT NULL,
    PRIMARY KEY (thread_id, item_key)
);

CREATE INDEX agent_sdk_thread_view_items_by_index
    ON agent_sdk_thread_view_items (thread_id, item_index);
