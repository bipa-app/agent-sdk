-- Typed JSON documents a transition writes in the same transaction as its
-- tasks and events (design §6.4, agent-sdk `journal/document.rs`).
--
-- The owner (for example bip) keeps the schema of value_json and migrates
-- old schema_version values when it reads them; agent-sdk never reads the
-- value. A retired document keeps its row (retired_at set) so its revision
-- keeps rising across a retire and a later put.
--
-- No foreign key: a host document has no parent row, and a reference to
-- agent_sdk_threads would take SHARE ROW EXCLUSIVE on that hot table while
-- this migration runs. The thread purge retires thread and task documents
-- itself. The new table and index lock only themselves.
--
-- The key columns use the "C" collation, so keys sort byte-wise as they do
-- on SQLite and in memory, whatever the database's default collation. The
-- scope CHECK names no scope kinds: Rust refuses an unknown kind on read,
-- and SQLite could not widen such a list without a table rebuild.

CREATE TABLE agent_sdk_documents (
    scope_kind     TEXT        COLLATE "C" NOT NULL,
    scope_id       TEXT        COLLATE "C" NOT NULL,
    kind           TEXT        COLLATE "C" NOT NULL,
    doc_key        TEXT        COLLATE "C" NOT NULL,
    schema_version BIGINT      NOT NULL,
    revision       BIGINT      NOT NULL,
    value_json     JSONB       NOT NULL,
    retired_at     TIMESTAMPTZ NULL,
    updated_at     TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (scope_kind, scope_id, kind, doc_key),

    CONSTRAINT agent_sdk_documents_scope_check CHECK (
        scope_kind <> ''
        AND ((scope_kind = 'host') = (scope_id = ''))
    ),
    CONSTRAINT agent_sdk_documents_revision_check CHECK (revision >= 1),
    CONSTRAINT agent_sdk_documents_schema_version_check CHECK (schema_version >= 0)
);

-- Every live document of one kind across scopes, in key order.
CREATE INDEX agent_sdk_documents_kind_idx
    ON agent_sdk_documents (kind, scope_kind, scope_id, doc_key)
    WHERE retired_at IS NULL;
