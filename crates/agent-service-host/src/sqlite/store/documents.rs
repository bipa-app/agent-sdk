//! The `SQLite` side of [`agent_server::journal::Write::Document`] and
//! [`DocumentStore`]. `BEGIN IMMEDIATE` serializes every writer, so a
//! document's row needs no lock of its own.

use agent_server::journal::document::{
    DocKind, DocOp, DocPlan, DocScope, Document, DocumentKey, DocumentOutcome, DocumentStore,
    DocumentWrite, StoredRevision,
};
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use sqlx::{FromRow, Sqlite, SqliteConnection, Transaction};
use time::OffsetDateTime;

use super::SqliteDurableStore;

#[derive(Debug, FromRow)]
struct DocumentRecord {
    scope_kind: String,
    scope_id: String,
    kind: String,
    doc_key: String,
    schema_version: i64,
    revision: i64,
    value_json: String,
    updated_at: OffsetDateTime,
}

impl TryFrom<DocumentRecord> for Document {
    type Error = anyhow::Error;

    fn try_from(record: DocumentRecord) -> Result<Self> {
        Ok(Self {
            key: DocumentKey::new(
                DocScope::from_stored(&record.scope_kind, &record.scope_id)?,
                DocKind::new(record.kind)?,
                record.doc_key,
            ),
            schema_version: u32::try_from(record.schema_version)
                .context("document schema_version out of range")?,
            revision: u64::try_from(record.revision).context("document revision out of range")?,
            value: serde_json::from_str(&record.value_json).context("decode document value")?,
            updated_at: record.updated_at,
        })
    }
}

impl SqliteDurableStore {
    /// The live document at `key`, read on `connection` (a transaction's,
    /// to read under its lock).
    pub(super) async fn get_document_on(
        connection: &mut SqliteConnection,
        key: &DocumentKey,
    ) -> Result<Option<Document>> {
        let (scope_kind, scope_id, kind, doc_key) = key.stored();
        sqlx::query_as::<_, DocumentRecord>(
            r"
SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json, updated_at
FROM agent_sdk_documents
WHERE scope_kind = ?1 AND scope_id = ?2 AND kind = ?3 AND doc_key = ?4
  AND retired_at IS NULL
",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .bind(kind)
        .bind(doc_key)
        .fetch_optional(connection)
        .await
        .with_context(|| format!("get document {key}"))?
        .map(TryInto::try_into)
        .transpose()
    }

    /// Apply one document write inside its transition's transaction.
    pub(super) async fn apply_document_tx(
        tx: &mut Transaction<'_, Sqlite>,
        document: DocumentWrite,
        now: OffsetDateTime,
    ) -> Result<DocumentOutcome> {
        if let DocScope::Task(task) = &document.key.scope {
            ensure!(
                Self::load_task_tx(tx, task).await?.is_some(),
                "document {} names task {task}, which does not exist",
                document.key
            );
        }
        let (scope_kind, scope_id, kind, key) = document.key.stored();
        let stored = sqlx::query!(
            r#"SELECT revision, retired_at IS NULL AS "live!: bool"
               FROM agent_sdk_documents
               WHERE scope_kind = ?1 AND scope_id = ?2 AND kind = ?3 AND doc_key = ?4"#,
            scope_kind,
            scope_id,
            kind,
            key,
        )
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("read document {}", document.key))?
        .map(|row| -> Result<StoredRevision> {
            Ok(StoredRevision {
                revision: u64::try_from(row.revision).context("document revision out of range")?,
                live: row.live,
            })
        })
        .transpose()?;
        let plan = document.plan(stored).map_err(anyhow::Error::new)?;
        match (plan, &document.op) {
            (DocPlan::Put { revision }, DocOp::Put(value)) => {
                let revision = i64::try_from(revision).context("document revision out of range")?;
                let schema_version = i64::from(document.schema_version);
                let value = serde_json::to_string(value).context("encode document value")?;
                sqlx::query!(
                    r#"INSERT INTO agent_sdk_documents
                           (scope_kind, scope_id, kind, doc_key, schema_version, revision,
                            value_json, retired_at, updated_at)
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8)
                       ON CONFLICT (scope_kind, scope_id, kind, doc_key) DO UPDATE SET
                           schema_version = excluded.schema_version,
                           revision = excluded.revision,
                           value_json = excluded.value_json,
                           retired_at = NULL,
                           updated_at = excluded.updated_at"#,
                    scope_kind,
                    scope_id,
                    kind,
                    key,
                    schema_version,
                    revision,
                    value,
                    now,
                )
                .execute(&mut **tx)
                .await
                .with_context(|| format!("put document {}", document.key))?;
            }
            (DocPlan::Retire { revision }, DocOp::Retire) => {
                let revision = i64::try_from(revision).context("document revision out of range")?;
                sqlx::query!(
                    r#"UPDATE agent_sdk_documents
                       SET revision = ?5, retired_at = ?6, updated_at = ?6
                       WHERE scope_kind = ?1 AND scope_id = ?2 AND kind = ?3 AND doc_key = ?4"#,
                    scope_kind,
                    scope_id,
                    kind,
                    key,
                    revision,
                    now,
                )
                .execute(&mut **tx)
                .await
                .with_context(|| format!("retire document {}", document.key))?;
            }
            (DocPlan::Unchanged, DocOp::Retire) => {}
            (DocPlan::Put { .. } | DocPlan::Unchanged, DocOp::Retire | DocOp::Put(_))
            | (DocPlan::Retire { .. }, DocOp::Put(_)) => {
                bail!(
                    "document {} planned {plan:?} for its operation",
                    document.key
                )
            }
        }
        Ok(DocumentOutcome {
            key: document.key,
            revision: plan.live_revision(),
        })
    }

    /// Retire every live document of `thread_id` and of its tasks, in the
    /// purge's own transaction.
    pub(super) async fn retire_thread_documents_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &str,
        now: OffsetDateTime,
    ) -> Result<()> {
        sqlx::query!(
            r#"UPDATE agent_sdk_documents
               SET revision = revision + 1, retired_at = ?2, updated_at = ?2
               WHERE retired_at IS NULL
                 AND ((scope_kind = 'thread' AND scope_id = ?1)
                      OR (scope_kind = 'task'
                          AND scope_id IN (SELECT id FROM agent_sdk_tasks WHERE thread_id = ?1)))"#,
            thread_id,
            now,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("retire the documents of purged thread {thread_id}"))?;
        Ok(())
    }
}

#[async_trait]
impl DocumentStore for SqliteDurableStore {
    async fn get_document(&self, key: &DocumentKey) -> Result<Option<Document>> {
        let mut connection = self.pool.acquire().await.context("acquire connection")?;
        Self::get_document_on(&mut connection, key).await
    }

    async fn list_documents(&self, scope: &DocScope, kind: &DocKind) -> Result<Vec<Document>> {
        sqlx::query_as::<_, DocumentRecord>(
            r"
SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json, updated_at
FROM agent_sdk_documents
WHERE scope_kind = ?1 AND scope_id = ?2 AND kind = ?3 AND retired_at IS NULL
ORDER BY doc_key
",
        )
        .bind(scope.kind_str())
        .bind(scope.id_str())
        .bind(kind.as_str())
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list {kind} documents"))?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }

    async fn list_documents_by_kind(
        &self,
        kind: &DocKind,
        after: Option<&DocumentKey>,
        limit: u32,
    ) -> Result<Vec<Document>> {
        let (after_scope_kind, after_scope_id, _, after_key) =
            after.map_or(("", "", "", ""), DocumentKey::stored);
        sqlx::query_as::<_, DocumentRecord>(
            r"
SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json, updated_at
FROM agent_sdk_documents
WHERE kind = ?1 AND retired_at IS NULL
  AND (?2 = 0 OR (scope_kind, scope_id, doc_key) > (?3, ?4, ?5))
ORDER BY scope_kind, scope_id, doc_key
LIMIT ?6
",
        )
        .bind(kind.as_str())
        .bind(i64::from(after.is_some()))
        .bind(after_scope_kind)
        .bind(after_scope_id)
        .bind(after_key)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list {kind} documents"))?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }
}
