//! Typed JSON state that a [`Transition`](super::Transition) writes in the
//! same transaction as its tasks and events (design §6.4).
//!
//! A document has a scope, an owner-namespaced kind and a key. The store
//! keeps its value as opaque JSON with the owner's schema version: the
//! owner, not agent-sdk, defines the schema and migrates old versions when
//! it reads them. A document is written only through
//! [`Write::Document`](super::Write::Document), so it commits or rolls back
//! with the rest of its transition, and read through [`DocumentStore`].

use agent_sdk_foundation::ThreadId;
use async_trait::async_trait;
use time::OffsetDateTime;

use super::task::AgentTaskId;

/// The largest document value a transition may write, as serialized JSON.
pub const MAX_DOCUMENT_VALUE_BYTES: usize = 4 * 1024 * 1024;

/// The longest document key, in bytes.
pub const MAX_DOCUMENT_KEY_BYTES: usize = 512;

/// What a document belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DocScope {
    /// The host as a whole, outside any thread. A host document is locked
    /// by its own key ([`super::Lock::Document`]).
    Host,
    /// One thread. Locked through the thread row, and refused once the
    /// thread is being purged; the purge retires it.
    Thread(ThreadId),
    /// One task. Locked through the task row; the purge of the task's
    /// thread retires it.
    Task(AgentTaskId),
}

impl DocScope {
    /// The scope's kind as stored: `host`, `thread` or `task`.
    #[must_use]
    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Thread(_) => "thread",
            Self::Task(_) => "task",
        }
    }

    /// The scope's id as stored: empty for [`Self::Host`].
    #[must_use]
    pub fn id_str(&self) -> &str {
        match self {
            Self::Host => "",
            Self::Thread(thread) => &thread.0,
            Self::Task(task) => task.as_str(),
        }
    }

    /// Rebuild a scope from its stored kind and id.
    ///
    /// # Errors
    /// An unknown kind, a host scope with an id, or a thread or task scope
    /// without one.
    pub fn from_stored(kind: &str, id: &str) -> anyhow::Result<Self> {
        match (kind, id) {
            ("host", "") => Ok(Self::Host),
            ("thread", id) if !id.is_empty() => Ok(Self::Thread(ThreadId::from_string(id))),
            ("task", id) if !id.is_empty() => Ok(Self::Task(AgentTaskId::from_string(id))),
            (kind, id) => anyhow::bail!("invalid stored document scope {kind:?}/{id:?}"),
        }
    }
}

/// An owner-namespaced document kind, `<owner>.<name>`, for example
/// `bip.thread_context`: lowercase ASCII letters, digits and `_`, in two or
/// more dot-separated segments, at most 128 bytes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocKind(String);

impl DocKind {
    /// # Errors
    /// A kind that is not `<owner>.<name>` in the allowed alphabet.
    pub fn new(kind: impl Into<String>) -> anyhow::Result<Self> {
        let kind = kind.into();
        let segments_ok = kind.split('.').all(|segment| {
            !segment.is_empty()
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        });
        anyhow::ensure!(
            kind.len() <= 128 && kind.contains('.') && segments_ok,
            "document kind {kind:?} is not <owner>.<name> in [a-z0-9_]"
        );
        Ok(Self(kind))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DocKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The identity of one document. `key` is empty for a kind with one
/// document per scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DocumentKey {
    pub scope: DocScope,
    pub kind: DocKind,
    pub key: String,
}

impl DocumentKey {
    #[must_use]
    pub fn new(scope: DocScope, kind: DocKind, key: impl Into<String>) -> Self {
        Self {
            scope,
            kind,
            key: key.into(),
        }
    }
}

impl DocumentKey {
    /// The stored primary key: (scope kind, scope id, kind, key), the
    /// order every backend lists and locks documents in.
    #[must_use]
    pub fn stored(&self) -> (&'static str, &str, &str, &str) {
        (
            self.scope.kind_str(),
            self.scope.id_str(),
            self.kind.as_str(),
            &self.key,
        )
    }
}

impl std::fmt::Display for DocumentKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}/{}/{}",
            self.scope.kind_str(),
            self.scope.id_str(),
            self.kind,
            self.key
        )
    }
}

/// What the stored document must be before the write applies. A failed
/// expectation rejects the whole transition like a failed guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocExpect {
    /// Write whatever is stored.
    Any,
    /// No live document: never written, or retired.
    Absent,
    /// A live document at exactly this revision.
    Revision(u64),
}

/// The write itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocOp {
    /// Store this full value. A retired document comes back to life.
    Put(serde_json::Value),
    /// Retire the live document. Retiring a missing or retired document
    /// writes nothing.
    Retire,
}

/// One document write of a transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentWrite {
    pub key: DocumentKey,
    /// The owner's schema version of `value`. agent-sdk stores it and
    /// never reads `value`.
    pub schema_version: u32,
    pub expect: DocExpect,
    pub op: DocOp,
}

impl DocumentWrite {
    /// Put `value` at `key` under `expect`.
    #[must_use]
    pub const fn put(
        key: DocumentKey,
        schema_version: u32,
        expect: DocExpect,
        value: serde_json::Value,
    ) -> Self {
        Self {
            key,
            schema_version,
            expect,
            op: DocOp::Put(value),
        }
    }

    /// Retire the document at `key` under `expect`.
    #[must_use]
    pub const fn retire(key: DocumentKey, expect: DocExpect) -> Self {
        Self {
            key,
            schema_version: 0,
            expect,
            op: DocOp::Retire,
        }
    }

    /// Check the shape every backend relies on: a key within
    /// [`MAX_DOCUMENT_KEY_BYTES`] and without NUL, a non-empty id for a
    /// thread or task scope, and a value within
    /// [`MAX_DOCUMENT_VALUE_BYTES`].
    ///
    /// # Errors
    /// Names the first broken rule.
    pub fn validate(&self) -> anyhow::Result<()> {
        let key = &self.key;
        anyhow::ensure!(
            key.key.len() <= MAX_DOCUMENT_KEY_BYTES && !key.key.contains('\0'),
            "document key of {key} is longer than {MAX_DOCUMENT_KEY_BYTES} bytes or holds NUL"
        );
        anyhow::ensure!(
            matches!(key.scope, DocScope::Host) || !key.scope.id_str().is_empty(),
            "document {key} names an empty scope id"
        );
        if let DocOp::Put(value) = &self.op {
            let bytes = serde_json::to_vec(value)?.len();
            anyhow::ensure!(
                bytes <= MAX_DOCUMENT_VALUE_BYTES,
                "document {key} value is {bytes} bytes, over {MAX_DOCUMENT_VALUE_BYTES}"
            );
        }
        Ok(())
    }

    /// Decide what this write does to `stored`, the row as locked
    /// (`None` when the key was never written). Every backend applies this
    /// one rule.
    ///
    /// # Errors
    /// [`ExpectationFailed`] when `expect` does not hold.
    pub fn plan(&self, stored: Option<StoredRevision>) -> Result<DocPlan, ExpectationFailed> {
        let live = stored.filter(|row| row.live).map(|row| row.revision);
        let holds = match self.expect {
            DocExpect::Any => true,
            DocExpect::Absent => live.is_none(),
            DocExpect::Revision(revision) => live == Some(revision),
        };
        if !holds {
            return Err(ExpectationFailed {
                reason: format!(
                    "document {} expected {:?}, stored {}",
                    self.key,
                    self.expect,
                    live.map_or_else(
                        || "no live document".to_owned(),
                        |r| format!("revision {r}")
                    )
                ),
            });
        }
        let next = stored.map_or(1, |row| row.revision.saturating_add(1));
        Ok(match (&self.op, live) {
            (DocOp::Put(_), _) => DocPlan::Put { revision: next },
            (DocOp::Retire, Some(_)) => DocPlan::Retire { revision: next },
            (DocOp::Retire, None) => DocPlan::Unchanged,
        })
    }
}

/// The stored state a [`DocumentWrite::plan`] decides from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredRevision {
    pub revision: u64,
    /// Not retired.
    pub live: bool,
}

/// What a backend does for one document write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocPlan {
    /// Insert or replace the row at `revision`, live.
    Put { revision: u64 },
    /// Mark the row retired at `revision`.
    Retire { revision: u64 },
    /// Write nothing.
    Unchanged,
}

impl DocPlan {
    /// The revision the document has after the write, `None` when it has
    /// no live document.
    #[must_use]
    pub const fn live_revision(self) -> Option<u64> {
        match self {
            Self::Put { revision } => Some(revision),
            Self::Retire { .. } | Self::Unchanged => None,
        }
    }
}

/// A document's [`DocExpect`] did not hold. A backend returns it from the
/// write, and `commit` reports it as [`super::Rejected::Guard`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct ExpectationFailed {
    pub reason: String,
}

/// A stored, live document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Document {
    pub key: DocumentKey,
    pub schema_version: u32,
    /// Raised by one on every put and retire of the key.
    pub revision: u64,
    pub value: serde_json::Value,
    pub updated_at: OffsetDateTime,
}

/// What one [`super::Write::Document`] left behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentOutcome {
    pub key: DocumentKey,
    /// The live revision after the write; `None` once retired or never
    /// written.
    pub revision: Option<u64>,
}

/// Reads of the documents written through
/// [`super::DurableStore::commit`]. Retired documents are never returned.
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// The live document at `key`.
    ///
    /// # Errors
    /// The store failed.
    async fn get_document(&self, key: &DocumentKey) -> anyhow::Result<Option<Document>>;

    /// Every live document of `kind` in `scope`, ordered by key.
    ///
    /// # Errors
    /// The store failed.
    async fn list_documents(
        &self,
        scope: &DocScope,
        kind: &DocKind,
    ) -> anyhow::Result<Vec<Document>>;

    /// Up to `limit` live documents of `kind` in every scope, ordered by
    /// (scope kind, scope id, key), starting after `after`.
    ///
    /// # Errors
    /// The store failed.
    async fn list_documents_by_kind(
        &self,
        kind: &DocKind,
        after: Option<&DocumentKey>,
        limit: u32,
    ) -> anyhow::Result<Vec<Document>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> DocumentKey {
        DocumentKey::new(
            DocScope::Thread(ThreadId::from_string("t1")),
            DocKind::new("owner.kind").expect("kind"),
            "",
        )
    }

    fn put(expect: DocExpect) -> DocumentWrite {
        DocumentWrite::put(key(), 1, expect, serde_json::json!({"a": 1}))
    }

    const LIVE_3: StoredRevision = StoredRevision {
        revision: 3,
        live: true,
    };
    const RETIRED_3: StoredRevision = StoredRevision {
        revision: 3,
        live: false,
    };

    #[test]
    fn kinds_are_owner_namespaced() {
        for good in ["bip.todos", "a.b.c", "x1.y_2"] {
            assert!(DocKind::new(good).is_ok(), "{good}");
        }
        for bad in [
            "todos",
            "Bip.todos",
            "bip..todos",
            ".bip",
            "bip.",
            "bip.to-dos",
            "",
        ] {
            assert!(DocKind::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn expectations_read_only_the_live_revision() {
        assert_eq!(
            put(DocExpect::Absent).plan(None),
            Ok(DocPlan::Put { revision: 1 })
        );
        assert_eq!(
            put(DocExpect::Absent).plan(Some(RETIRED_3)),
            Ok(DocPlan::Put { revision: 4 }),
            "a retired document is absent, and its revision keeps rising"
        );
        assert!(put(DocExpect::Absent).plan(Some(LIVE_3)).is_err());
        assert_eq!(
            put(DocExpect::Revision(3)).plan(Some(LIVE_3)),
            Ok(DocPlan::Put { revision: 4 })
        );
        assert!(put(DocExpect::Revision(2)).plan(Some(LIVE_3)).is_err());
        assert!(put(DocExpect::Revision(3)).plan(Some(RETIRED_3)).is_err());
        assert_eq!(
            put(DocExpect::Any).plan(Some(LIVE_3)),
            Ok(DocPlan::Put { revision: 4 })
        );
    }

    #[test]
    fn retiring_nothing_writes_nothing() {
        let retire = DocumentWrite::retire(key(), DocExpect::Any);
        assert_eq!(retire.plan(None), Ok(DocPlan::Unchanged));
        assert_eq!(retire.plan(Some(RETIRED_3)), Ok(DocPlan::Unchanged));
        assert_eq!(
            retire.plan(Some(LIVE_3)),
            Ok(DocPlan::Retire { revision: 4 })
        );
    }

    #[test]
    fn validation_bounds_the_key_and_value() {
        assert!(put(DocExpect::Any).validate().is_ok());
        let mut long_key = put(DocExpect::Any);
        long_key.key.key = "k".repeat(MAX_DOCUMENT_KEY_BYTES + 1);
        assert!(long_key.validate().is_err());
        let mut empty_scope = put(DocExpect::Any);
        empty_scope.key.scope = DocScope::Thread(ThreadId::from_string(""));
        assert!(empty_scope.validate().is_err());
        let big = DocumentWrite::put(
            key(),
            1,
            DocExpect::Any,
            serde_json::Value::String("x".repeat(MAX_DOCUMENT_VALUE_BYTES)),
        );
        assert!(big.validate().is_err());
    }

    #[test]
    fn stored_scopes_round_trip() {
        for scope in [
            DocScope::Host,
            DocScope::Thread(ThreadId::from_string("t")),
            DocScope::Task(AgentTaskId::from_string("k")),
        ] {
            assert_eq!(
                DocScope::from_stored(scope.kind_str(), scope.id_str()).expect("scope"),
                scope
            );
        }
        assert!(DocScope::from_stored("host", "x").is_err());
        assert!(DocScope::from_stored("thread", "").is_err());
        assert!(DocScope::from_stored("session", "s").is_err());
    }
}
