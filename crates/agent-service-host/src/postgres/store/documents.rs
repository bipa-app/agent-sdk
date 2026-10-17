//! The Postgres side of [`agent_server::journal::Write::Document`] and
//! [`DocumentStore`].
//!
//! A thread or task document is reached only under its thread or task row
//! lock, which the transition takes first. A host document has no such
//! row: the transition locks it by key (`Lock::Document`) after every task
//! row. Two first writes of the same missing key race on the primary key,
//! and the loser's unique violation is a failed expectation, not a store
//! error.

use agent_server::journal::Lock;
use agent_server::journal::document::{
    DocKind, DocOp, DocPlan, DocScope, Document, DocumentKey, DocumentOutcome, DocumentStore,
    DocumentWrite, ExpectationFailed, StoredRevision,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{Postgres, Transaction};
use time::OffsetDateTime;

use super::PostgresDurableStore;

struct DocumentRecord {
    scope_kind: String,
    scope_id: String,
    kind: String,
    doc_key: String,
    schema_version: i64,
    revision: i64,
    value_json: serde_json::Value,
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
            value: record.value_json,
            updated_at: record.updated_at,
        })
    }
}

fn is_unique_violation(error: &sqlx::Error) -> bool {
    error
        .as_database_error()
        .and_then(sqlx::error::DatabaseError::code)
        .is_some_and(|code| code == "23505")
}

impl PostgresDurableStore {
    /// Lock the host documents among a transition's `locks`, in key
    /// order, after its task rows. A key with no row locks nothing.
    pub(super) async fn lock_host_documents_tx(
        tx: &mut Transaction<'_, Postgres>,
        locks: &[Lock],
    ) -> Result<()> {
        let mut keys: Vec<_> = locks
            .iter()
            .filter_map(|lock| match lock {
                Lock::Document(key) => Some(key.stored()),
                Lock::Thread { .. } | Lock::ThreadOf { .. } | Lock::Task(_) => None,
            })
            .collect();
        keys.sort_unstable();
        keys.dedup();
        for (scope_kind, scope_id, kind, doc_key) in keys {
            sqlx::query!(
                r#"SELECT revision FROM agent_sdk_documents
                   WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3 AND doc_key = $4
                   FOR UPDATE"#,
                scope_kind,
                scope_id,
                kind,
                doc_key,
            )
            .fetch_optional(&mut **tx)
            .await
            .with_context(|| format!("lock host document {kind}/{doc_key}"))?;
        }
        Ok(())
    }

    /// Apply one document write inside its transition's transaction.
    pub(super) async fn apply_document_tx(
        tx: &mut Transaction<'_, Postgres>,
        document: DocumentWrite,
        now: OffsetDateTime,
    ) -> Result<DocumentOutcome> {
        let (scope_kind, scope_id, kind, key) = document.key.stored();
        let stored = sqlx::query!(
            r#"SELECT revision, retired_at IS NULL AS "live!"
               FROM agent_sdk_documents
               WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3 AND doc_key = $4
               FOR UPDATE"#,
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
                if stored.is_some() {
                    sqlx::query!(
                        r#"UPDATE agent_sdk_documents
                           SET schema_version = $5, revision = $6, value_json = $7,
                               retired_at = NULL, updated_at = $8
                           WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3
                             AND doc_key = $4"#,
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
                } else {
                    Self::insert_document_tx(tx, &document, schema_version, value, now).await?;
                }
            }
            (DocPlan::Retire { revision }, _) => {
                let revision = i64::try_from(revision).context("document revision out of range")?;
                sqlx::query!(
                    r#"UPDATE agent_sdk_documents
                       SET revision = $5, retired_at = $6, updated_at = $6
                       WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3 AND doc_key = $4"#,
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
            (DocPlan::Put { .. } | DocPlan::Unchanged, _) => {}
        }
        Ok(DocumentOutcome {
            key: document.key,
            revision: plan.live_revision(),
        })
    }

    /// Insert the first row of `document`'s key at revision 1. A concurrent
    /// first write that commits first makes this a failed expectation.
    async fn insert_document_tx(
        tx: &mut Transaction<'_, Postgres>,
        document: &DocumentWrite,
        schema_version: i64,
        value: &serde_json::Value,
        now: OffsetDateTime,
    ) -> Result<()> {
        let (scope_kind, scope_id, kind, key) = document.key.stored();
        let inserted = sqlx::query!(
            r#"INSERT INTO agent_sdk_documents
                   (scope_kind, scope_id, kind, doc_key, schema_version, revision,
                    value_json, retired_at, updated_at)
               VALUES ($1, $2, $3, $4, $5, 1, $6, NULL, $7)"#,
            scope_kind,
            scope_id,
            kind,
            key,
            schema_version,
            value,
            now,
        )
        .execute(&mut **tx)
        .await;
        match inserted {
            Ok(_) => Ok(()),
            Err(error) if is_unique_violation(&error) => {
                Err(anyhow::Error::new(ExpectationFailed {
                    reason: format!(
                        "document {} was first written by a concurrent transition",
                        document.key
                    ),
                }))
            }
            Err(error) => {
                Err(anyhow::Error::new(error).context(format!("put document {}", document.key)))
            }
        }
    }

    /// Retire every live document of `thread_id` and of its tasks, in the
    /// purge's own transaction, which holds the thread row.
    pub(super) async fn retire_thread_documents_tx(
        tx: &mut Transaction<'_, Postgres>,
        thread_id: &str,
        now: OffsetDateTime,
    ) -> Result<()> {
        sqlx::query!(
            r#"UPDATE agent_sdk_documents
               SET revision = revision + 1, retired_at = $2, updated_at = $2
               WHERE retired_at IS NULL
                 AND ((scope_kind = 'thread' AND scope_id = $1)
                      OR (scope_kind = 'task'
                          AND scope_id IN (SELECT id FROM agent_sdk_tasks WHERE thread_id = $1)))"#,
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
impl DocumentStore for PostgresDurableStore {
    async fn get_document(&self, key: &DocumentKey) -> Result<Option<Document>> {
        let (scope_kind, scope_id, kind, doc_key) = key.stored();
        sqlx::query_as!(
            DocumentRecord,
            r#"SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json,
                      updated_at
               FROM agent_sdk_documents
               WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3 AND doc_key = $4
                 AND retired_at IS NULL"#,
            scope_kind,
            scope_id,
            kind,
            doc_key,
        )
        .fetch_optional(&self.pool)
        .await
        .with_context(|| format!("get document {key}"))?
        .map(TryInto::try_into)
        .transpose()
    }

    async fn list_documents(&self, scope: &DocScope, kind: &DocKind) -> Result<Vec<Document>> {
        sqlx::query_as!(
            DocumentRecord,
            r#"SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json,
                      updated_at
               FROM agent_sdk_documents
               WHERE scope_kind = $1 AND scope_id = $2 AND kind = $3 AND retired_at IS NULL
               ORDER BY doc_key"#,
            scope.kind_str(),
            scope.id_str(),
            kind.as_str(),
        )
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
        sqlx::query_as!(
            DocumentRecord,
            r#"SELECT scope_kind, scope_id, kind, doc_key, schema_version, revision, value_json,
                      updated_at
               FROM agent_sdk_documents
               WHERE kind = $1 AND retired_at IS NULL
                 AND (NOT $2 OR (scope_kind, scope_id, doc_key) > ($3, $4, $5))
               ORDER BY scope_kind, scope_id, doc_key
               LIMIT $6"#,
            kind.as_str(),
            after.is_some(),
            after_scope_kind,
            after_scope_id,
            after_key,
            i64::from(limit),
        )
        .fetch_all(&self.pool)
        .await
        .with_context(|| format!("list {kind} documents"))?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }
}
