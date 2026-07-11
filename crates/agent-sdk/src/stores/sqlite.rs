//! Embedded, single-file durable session store backed by SQLite.
//!
//! [`SqliteStore`] implements all four SDK store traits — [`MessageStore`],
//! [`StateStore`], [`EventStore`], and [`ToolExecutionStore`] — over one
//! on-disk SQLite database. Unlike the volatile `InMemory*` stores, the data it
//! records survives the process: an in-process agent can persist its
//! conversation history, state checkpoints, turn events, and tool-execution
//! ledger and pick the conversation back up after a restart.
//!
//! This is gated behind the `sqlite` cargo feature so the default build pulls
//! no SQLite dependency. The driver ([`rusqlite`] with the bundled SQLite
//! amalgamation) is synchronous, so every database call runs inside
//! [`tokio::task::spawn_blocking`] to keep the async runtime unblocked.
//!
//! # Resume semantics
//!
//! Durability is keyed by the **file path** plus the [`ThreadId`]. Reopen the
//! same path and run with the same `ThreadId` and the agent continues exactly
//! where it left off — the message history, the latest state checkpoint, every
//! stored turn's events, and the tool-execution records are all still there. A
//! fresh `ThreadId` against the same file starts an independent conversation in
//! the same database.
//!
//! # One store, four traits
//!
//! [`SqliteStore`] is cheap to [`Clone`] (it shares one
//! `Arc<Mutex<Connection>>`, not a connection pool), so the same store can back
//! every slot on the builder:
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_sdk::{builder, DefaultHooks, EventStore, SqliteStore, ThreadId};
//! # use agent_sdk::providers::AnthropicProvider;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Same file + same ThreadId across restarts == resumed conversation.
//! let store = SqliteStore::open("agent.db")?;
//! let events: Arc<dyn EventStore> = Arc::new(store.clone());
//!
//! // Custom stores require `build_with_stores()` (and explicit hooks).
//! let agent = builder::<()>()
//!     .provider(AnthropicProvider::from_env())
//!     .hooks(DefaultHooks)
//!     .message_store(store.clone())
//!     .state_store(store.clone())
//!     .event_store(events)
//!     .execution_store(store)
//!     .build_with_stores();
//! # let _ = (agent, ThreadId::new());
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_sdk_foundation::events::AgentEventEnvelope;
use agent_sdk_foundation::llm;
use agent_sdk_foundation::types::{AgentState, ThreadId, ToolExecution};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use super::{EventStore, MessageStore, StateStore, StoredTurnEvents, ToolExecutionStore};

/// How long a blocked writer waits for a competing connection to release its
/// lock before `SQLite` returns `SQLITE_BUSY`. The headline use case
/// (restart-and-resume) can briefly contend with a lingering process still
/// checkpointing the WAL, so we wait a few seconds rather than failing fast.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Schema applied on open. `CREATE TABLE IF NOT EXISTS` makes opening an
/// existing database a no-op, which is exactly the resume path.
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS agent_messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    payload   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_thread
    ON agent_messages (thread_id, id);

CREATE TABLE IF NOT EXISTS agent_states (
    thread_id TEXT PRIMARY KEY,
    payload   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_event_turns (
    thread_id TEXT NOT NULL,
    turn      INTEGER NOT NULL,
    finished  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (thread_id, turn)
);

CREATE TABLE IF NOT EXISTS agent_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    turn      INTEGER NOT NULL,
    payload   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_events_thread_turn
    ON agent_events (thread_id, turn, id);

CREATE TABLE IF NOT EXISTS agent_tool_executions (
    tool_call_id TEXT PRIMARY KEY,
    operation_id TEXT,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_tool_executions_operation
    ON agent_tool_executions (operation_id);
";

/// Single-file SQLite-backed implementation of every SDK store trait.
///
/// See the [module docs](self) for resume semantics and a wiring example.
/// Cloning shares the same single `Arc<Mutex<Connection>>` (not a pool), so a
/// clone handed to the agent builder observes everything the kept handle
/// records (and vice versa); all database calls serialize on that one mutex.
#[derive(Clone)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStore {
    /// Open (creating if absent) the `SQLite` database at `path` and ensure the
    /// store schema exists.
    ///
    /// Reopening an existing file is the resume path: the schema creation is a
    /// no-op and all previously persisted rows remain available.
    ///
    /// This constructor is synchronous — it opens the file, converts it to
    /// WAL, and applies the schema on the calling thread (waiting up to
    /// [`BUSY_TIMEOUT`] if another process holds the write lock). Call it
    /// during startup, or wrap it in `spawn_blocking` on a latency-sensitive
    /// async runtime.
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or the schema cannot
    /// be initialized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open sqlite database at {}", path.display()))?;
        // `busy_timeout` first: it makes every subsequent lock-taking
        // statement — including the WAL conversion below, the very first one
        // — wait for a competing connection instead of failing fast with
        // `SQLITE_BUSY`. Important for restart-and-resume, where a lingering
        // process may still be checkpointing the WAL when the replacement
        // opens the file.
        conn.busy_timeout(BUSY_TIMEOUT)
            .context("failed to set sqlite busy timeout")?;
        // WAL lets readers run concurrently with a writer.
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("failed to enable WAL journal mode")?;
        conn.execute_batch(SCHEMA)
            .context("failed to initialize sqlite store schema")?;
        // Stamp the on-disk format so a future schema change can migrate by
        // version instead of inferring shape from the tables. Version 1 is
        // the initial format; 0 means "pre-versioning", which is also v1.
        let version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .context("failed to read sqlite user_version")?;
        if version == 0 {
            conn.pragma_update(None, "user_version", 1)
                .context("failed to stamp sqlite user_version")?;
        }
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Run a synchronous database closure on a blocking thread.
    ///
    /// The connection is synchronous and `!Sync`, so it lives behind a
    /// [`Mutex`]; the closure runs inside [`spawn_blocking`](tokio::task::spawn_blocking)
    /// to avoid stalling the async runtime.
    async fn with_conn<T>(
        &self,
        f: impl FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    ) -> Result<T>
    where
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn
                .lock()
                .ok()
                .context("sqlite connection mutex poisoned")?;
            f(&mut guard)
        })
        .await
        .context("sqlite blocking task failed to join")?
    }

    /// Shared upsert for [`ToolExecutionStore::record_execution`] and
    /// [`ToolExecutionStore::update_execution`]: both write the full record,
    /// keyed by `tool_call_id`, refreshing the `operation_id` index column.
    async fn upsert_execution(&self, execution: ToolExecution) -> Result<()> {
        let tool_call_id = execution.tool_call_id.clone();
        let operation_id = execution.operation_id.clone();
        let payload =
            serde_json::to_string(&execution).context("failed to encode tool execution")?;
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO agent_tool_executions (tool_call_id, operation_id, payload)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(tool_call_id)
                 DO UPDATE SET operation_id = excluded.operation_id, payload = excluded.payload",
                params![tool_call_id, operation_id, payload],
            )
            .context("failed to upsert tool execution")?;
            Ok(())
        })
        .await
    }
}

#[async_trait]
impl MessageStore for SqliteStore {
    async fn append(&self, thread_id: &ThreadId, message: llm::Message) -> Result<()> {
        let thread = thread_id.0.clone();
        let payload = serde_json::to_string(&message).context("failed to encode message")?;
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO agent_messages (thread_id, payload) VALUES (?1, ?2)",
                params![thread, payload],
            )
            .context("failed to append message")?;
            Ok(())
        })
        .await
    }

    async fn get_history(&self, thread_id: &ThreadId) -> Result<Vec<llm::Message>> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT payload FROM agent_messages WHERE thread_id = ?1 ORDER BY id")
                .context("failed to prepare message history query")?;
            let rows = stmt
                .query_map(params![thread], |row| row.get::<_, String>(0))
                .context("failed to read message history")?;
            let mut messages = Vec::new();
            for row in rows {
                let payload = row.context("failed to read message row")?;
                messages.push(serde_json::from_str(&payload).context("failed to decode message")?);
            }
            Ok(messages)
        })
        .await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            conn.execute(
                "DELETE FROM agent_messages WHERE thread_id = ?1",
                params![thread],
            )
            .context("failed to clear messages")?;
            Ok(())
        })
        .await
    }

    async fn count(&self, thread_id: &ThreadId) -> Result<usize> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM agent_messages WHERE thread_id = ?1",
                    params![thread],
                    |row| row.get(0),
                )
                .context("failed to count messages")?;
            usize::try_from(count).context("message count is negative")
        })
        .await
    }

    async fn replace_history(
        &self,
        thread_id: &ThreadId,
        messages: Vec<llm::Message>,
    ) -> Result<()> {
        let thread = thread_id.0.clone();
        let payloads = messages
            .iter()
            .map(|message| serde_json::to_string(message).context("failed to encode message"))
            .collect::<Result<Vec<_>>>()?;
        self.with_conn(move |conn| {
            let tx = conn
                .transaction()
                .context("failed to begin replace_history transaction")?;
            tx.execute(
                "DELETE FROM agent_messages WHERE thread_id = ?1",
                params![thread],
            )
            .context("failed to clear existing messages")?;
            for payload in payloads {
                tx.execute(
                    "INSERT INTO agent_messages (thread_id, payload) VALUES (?1, ?2)",
                    params![thread, payload],
                )
                .context("failed to insert replacement message")?;
            }
            tx.commit()
                .context("failed to commit replace_history transaction")?;
            Ok(())
        })
        .await
    }
}

#[async_trait]
impl StateStore for SqliteStore {
    async fn save(&self, state: &AgentState) -> Result<()> {
        let thread = state.thread_id.0.clone();
        let payload = serde_json::to_string(state).context("failed to encode agent state")?;
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO agent_states (thread_id, payload) VALUES (?1, ?2)
                 ON CONFLICT(thread_id) DO UPDATE SET payload = excluded.payload",
                params![thread, payload],
            )
            .context("failed to save agent state")?;
            Ok(())
        })
        .await
    }

    async fn load(&self, thread_id: &ThreadId) -> Result<Option<AgentState>> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let payload: Option<String> = conn
                .query_row(
                    "SELECT payload FROM agent_states WHERE thread_id = ?1",
                    params![thread],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to load agent state")?;
            match payload {
                Some(payload) => Ok(Some(
                    serde_json::from_str(&payload).context("failed to decode agent state")?,
                )),
                None => Ok(None),
            }
        })
        .await
    }

    async fn delete(&self, thread_id: &ThreadId) -> Result<()> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            conn.execute(
                "DELETE FROM agent_states WHERE thread_id = ?1",
                params![thread],
            )
            .context("failed to delete agent state")?;
            Ok(())
        })
        .await
    }
}

#[async_trait]
impl EventStore for SqliteStore {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()> {
        let thread = thread_id.0.clone();
        let turn_i64 = i64::try_from(turn).context("turn index exceeds i64 range")?;
        let payload = serde_json::to_string(&envelope).context("failed to encode event")?;
        self.with_conn(move |conn| {
            // `BEGIN IMMEDIATE` takes the write lock up front so the
            // finished-turn barrier check and the insert are one atomic unit:
            // a concurrent connection cannot slip a `finish_turn` between them,
            // and the whole append is a single WAL commit instead of three.
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .context("failed to begin append transaction")?;
            let finished: Option<i64> = tx
                .query_row(
                    "SELECT finished FROM agent_event_turns WHERE thread_id = ?1 AND turn = ?2",
                    params![thread, turn_i64],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to read turn state")?;
            anyhow::ensure!(finished != Some(1), "cannot append to finished turn {turn}");
            tx.execute(
                "INSERT OR IGNORE INTO agent_event_turns (thread_id, turn, finished)
                 VALUES (?1, ?2, 0)",
                params![thread, turn_i64],
            )
            .context("failed to record turn")?;
            tx.execute(
                "INSERT INTO agent_events (thread_id, turn, payload) VALUES (?1, ?2, ?3)",
                params![thread, turn_i64, payload],
            )
            .context("failed to append event")?;
            tx.commit().context("failed to commit append transaction")?;
            Ok(())
        })
        .await
    }

    async fn finish_turn(&self, thread_id: &ThreadId, turn: usize) -> Result<()> {
        let thread = thread_id.0.clone();
        let turn_i64 = i64::try_from(turn).context("turn index exceeds i64 range")?;
        self.with_conn(move |conn| {
            // `BEGIN IMMEDIATE` makes the already-finished check and the
            // finish write atomic against a concurrent connection, so the
            // barrier cannot be double-finished or raced by an append.
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .context("failed to begin finish_turn transaction")?;
            let finished: Option<i64> = tx
                .query_row(
                    "SELECT finished FROM agent_event_turns WHERE thread_id = ?1 AND turn = ?2",
                    params![thread, turn_i64],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to read turn state")?;
            anyhow::ensure!(finished != Some(1), "turn {turn} is already finished");
            tx.execute(
                "INSERT INTO agent_event_turns (thread_id, turn, finished) VALUES (?1, ?2, 1)
                 ON CONFLICT(thread_id, turn) DO UPDATE SET finished = 1",
                params![thread, turn_i64],
            )
            .context("failed to finish turn")?;
            tx.commit()
                .context("failed to commit finish_turn transaction")?;
            Ok(())
        })
        .await
    }

    async fn get_turn(
        &self,
        thread_id: &ThreadId,
        turn: usize,
    ) -> Result<Option<StoredTurnEvents>> {
        let thread = thread_id.0.clone();
        let turn_i64 = i64::try_from(turn).context("turn index exceeds i64 range")?;
        self.with_conn(move |conn| {
            // Deferred read transaction so the turn row and its events come
            // from one WAL snapshot: without it another process could finish
            // the turn (or clear the thread) between the two queries,
            // yielding a torn `finished`/events view.
            let tx = conn
                .transaction()
                .context("failed to begin turn read transaction")?;
            let finished: Option<i64> = tx
                .query_row(
                    "SELECT finished FROM agent_event_turns WHERE thread_id = ?1 AND turn = ?2",
                    params![thread, turn_i64],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to read turn state")?;
            let Some(finished) = finished else {
                return Ok(None);
            };
            let events = {
                let mut stmt = tx
                    .prepare(
                        "SELECT payload FROM agent_events
                         WHERE thread_id = ?1 AND turn = ?2 ORDER BY id",
                    )
                    .context("failed to prepare turn events query")?;
                let rows = stmt
                    .query_map(params![thread, turn_i64], |row| row.get::<_, String>(0))
                    .context("failed to read turn events")?;
                let mut events = Vec::new();
                for row in rows {
                    let payload = row.context("failed to read event row")?;
                    events.push(serde_json::from_str(&payload).context("failed to decode event")?);
                }
                events
            };
            tx.commit().context("failed to end turn read transaction")?;
            Ok(Some(StoredTurnEvents {
                turn,
                events,
                finished: finished != 0,
            }))
        })
        .await
    }

    async fn get_turns(&self, thread_id: &ThreadId) -> Result<Vec<StoredTurnEvents>> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            // Deferred read transaction: both statements must observe one WAL
            // snapshot, or a turn appended (or a clear run) by another
            // process between them produces a torn view — events silently
            // dropped for unknown turns, or turn entries with emptied lists.
            let tx = conn
                .transaction()
                .context("failed to begin turns read transaction")?;
            let mut turns: Vec<StoredTurnEvents> = Vec::new();
            {
                let mut turn_stmt = tx
                    .prepare(
                        "SELECT turn, finished FROM agent_event_turns
                         WHERE thread_id = ?1 ORDER BY turn",
                    )
                    .context("failed to prepare turns query")?;
                let turn_rows = turn_stmt
                    .query_map(params![thread], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                    })
                    .context("failed to read turns")?;
                let mut position = std::collections::HashMap::new();
                for row in turn_rows {
                    let (turn_i64, finished) = row.context("failed to read turn row")?;
                    let turn = usize::try_from(turn_i64).context("turn index is negative")?;
                    position.insert(turn_i64, turns.len());
                    turns.push(StoredTurnEvents {
                        turn,
                        events: Vec::new(),
                        finished: finished != 0,
                    });
                }
                let mut event_stmt = tx
                    .prepare(
                        "SELECT turn, payload FROM agent_events
                         WHERE thread_id = ?1 ORDER BY turn, id",
                    )
                    .context("failed to prepare events query")?;
                let event_rows = event_stmt
                    .query_map(params![thread], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                    })
                    .context("failed to read events")?;
                for row in event_rows {
                    let (turn_i64, payload) = row.context("failed to read event row")?;
                    let envelope =
                        serde_json::from_str(&payload).context("failed to decode event")?;
                    if let Some(&index) = position.get(&turn_i64) {
                        turns[index].events.push(envelope);
                    }
                }
            }
            tx.commit()
                .context("failed to end turns read transaction")?;
            Ok(turns)
        })
        .await
    }

    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<AgentEventEnvelope>> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare("SELECT payload FROM agent_events WHERE thread_id = ?1 ORDER BY turn, id")
                .context("failed to prepare events query")?;
            let rows = stmt
                .query_map(params![thread], |row| row.get::<_, String>(0))
                .context("failed to read events")?;
            let mut events = Vec::new();
            for row in rows {
                let payload = row.context("failed to read event row")?;
                events.push(serde_json::from_str(&payload).context("failed to decode event")?);
            }
            Ok(events)
        })
        .await
    }

    async fn event_count(&self, thread_id: &ThreadId) -> Result<usize> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM agent_events WHERE thread_id = ?1",
                    params![thread],
                    |row| row.get(0),
                )
                .context("failed to count events")?;
            usize::try_from(count).context("event count is negative")
        })
        .await
    }

    async fn get_events_since(
        &self,
        thread_id: &ThreadId,
        offset: usize,
    ) -> Result<Vec<AgentEventEnvelope>> {
        let thread = thread_id.0.clone();
        let offset_i64 = i64::try_from(offset).context("offset exceeds i64 range")?;
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT payload FROM agent_events WHERE thread_id = ?1
                     ORDER BY turn, id LIMIT -1 OFFSET ?2",
                )
                .context("failed to prepare events query")?;
            let rows = stmt
                .query_map(params![thread, offset_i64], |row| row.get::<_, String>(0))
                .context("failed to read events")?;
            let mut events = Vec::new();
            for row in rows {
                let payload = row.context("failed to read event row")?;
                events.push(serde_json::from_str(&payload).context("failed to decode event")?);
            }
            Ok(events)
        })
        .await
    }

    async fn clear(&self, thread_id: &ThreadId) -> Result<()> {
        let thread = thread_id.0.clone();
        self.with_conn(move |conn| {
            let tx = conn
                .transaction()
                .context("failed to begin event clear transaction")?;
            tx.execute(
                "DELETE FROM agent_events WHERE thread_id = ?1",
                params![thread],
            )
            .context("failed to clear events")?;
            tx.execute(
                "DELETE FROM agent_event_turns WHERE thread_id = ?1",
                params![thread],
            )
            .context("failed to clear turns")?;
            tx.commit()
                .context("failed to commit event clear transaction")?;
            Ok(())
        })
        .await
    }
}

#[async_trait]
impl ToolExecutionStore for SqliteStore {
    async fn get_execution(&self, tool_call_id: &str) -> Result<Option<ToolExecution>> {
        let id = tool_call_id.to_owned();
        self.with_conn(move |conn| {
            let payload: Option<String> = conn
                .query_row(
                    "SELECT payload FROM agent_tool_executions WHERE tool_call_id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to read tool execution")?;
            match payload {
                Some(payload) => Ok(Some(
                    serde_json::from_str(&payload).context("failed to decode tool execution")?,
                )),
                None => Ok(None),
            }
        })
        .await
    }

    async fn record_execution(&self, execution: ToolExecution) -> Result<()> {
        self.upsert_execution(execution).await
    }

    async fn update_execution(&self, execution: ToolExecution) -> Result<()> {
        self.upsert_execution(execution).await
    }

    async fn get_execution_by_operation_id(
        &self,
        operation_id: &str,
    ) -> Result<Option<ToolExecution>> {
        let id = operation_id.to_owned();
        self.with_conn(move |conn| {
            // Two rows can share an `operation_id`; pick the most recently
            // *inserted* one (upserts keep their original rowid, so an
            // update to an older row does not promote it — unlike
            // `InMemoryExecutionStore`, whose index tracks the last writer).
            // Colliding operation ids are degenerate and no caller depends
            // on the tiebreak; a deterministic order is what matters here.
            let payload: Option<String> = conn
                .query_row(
                    "SELECT payload FROM agent_tool_executions WHERE operation_id = ?1
                     ORDER BY rowid DESC LIMIT 1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .context("failed to read tool execution by operation id")?;
            match payload {
                Some(payload) => Ok(Some(
                    serde_json::from_str(&payload).context("failed to decode tool execution")?,
                )),
                None => Ok(None),
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::events::{AgentEvent, AgentEventEnvelope, SequenceCounter};
    use agent_sdk_foundation::llm::Message;
    use agent_sdk_foundation::types::ToolResult;
    use anyhow::bail;
    use tempfile::tempdir;

    /// Open, write across all four stores, drop, reopen the same file, and
    /// assert every record survived — the resume contract.
    #[tokio::test]
    async fn persists_all_stores_across_reopen() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        // Session 1: write through every trait, then drop the store.
        {
            let store = SqliteStore::open(&path)?;

            MessageStore::append(&store, &thread_id, Message::user("hello")).await?;
            MessageStore::append(&store, &thread_id, Message::assistant("hi there")).await?;

            let state = AgentState::new(thread_id.clone());
            store.save(&state).await?;

            EventStore::append(
                &store,
                &thread_id,
                1,
                AgentEventEnvelope::wrap(AgentEvent::text("m1", "streamed"), &seq),
            )
            .await?;
            store.finish_turn(&thread_id, 1).await?;

            let execution = ToolExecution::new_in_flight(
                "call_1",
                thread_id.clone(),
                "my_tool",
                "My Tool",
                serde_json::json!({"k": "v"}),
                time::OffsetDateTime::now_utc(),
            );
            store.record_execution(execution).await?;
        }

        // Session 2: reopen the same file and confirm durability.
        let store = SqliteStore::open(&path)?;

        let history = store.get_history(&thread_id).await?;
        assert_eq!(history.len(), 2, "messages must survive reopen");
        assert_eq!(MessageStore::count(&store, &thread_id).await?, 2);

        let loaded = store.load(&thread_id).await?;
        assert!(loaded.is_some(), "state must survive reopen");
        assert_eq!(
            loaded.context("state present")?.thread_id,
            thread_id,
            "loaded state must match the thread"
        );

        let events = store.get_events(&thread_id).await?;
        assert_eq!(events.len(), 1, "events must survive reopen");
        assert_eq!(store.event_count(&thread_id).await?, 1);
        let turn = store
            .get_turn(&thread_id, 1)
            .await?
            .context("turn 1 present")?;
        assert!(turn.finished, "finish barrier must survive reopen");

        let execution = store
            .get_execution("call_1")
            .await?
            .context("execution present")?;
        assert_eq!(execution.tool_name, "my_tool");
        assert!(execution.is_in_flight());

        Ok(())
    }

    /// A second `ThreadId` against the same file is an independent conversation.
    #[tokio::test]
    async fn threads_are_isolated_in_one_file() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let store = SqliteStore::open(&path)?;

        let thread_a = ThreadId::new();
        let thread_b = ThreadId::new();

        MessageStore::append(&store, &thread_a, Message::user("a-only")).await?;

        assert_eq!(store.get_history(&thread_a).await?.len(), 1);
        assert!(
            store.get_history(&thread_b).await?.is_empty(),
            "a fresh thread starts empty in a shared file"
        );
        Ok(())
    }

    /// The finish barrier is durable: appending to a reopened, finished turn
    /// must still be rejected.
    #[tokio::test]
    async fn rejects_append_after_finish_across_reopen() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        {
            let store = SqliteStore::open(&path)?;
            store.finish_turn(&thread_id, 1).await?;
        }

        let store = SqliteStore::open(&path)?;
        let Err(error) = EventStore::append(
            &store,
            &thread_id,
            1,
            AgentEventEnvelope::wrap(AgentEvent::text("late", "late"), &seq),
        )
        .await
        else {
            bail!("append after finish must fail");
        };
        assert!(error.to_string().contains("cannot append to finished turn"));
        Ok(())
    }

    /// The finish barrier rejects an append to a finished turn within a single
    /// process too (not only across reopen), and an append to an unfinished
    /// turn still succeeds afterward.
    #[tokio::test]
    async fn rejects_append_after_finish_in_process() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let thread_id = ThreadId::new();
        let seq = SequenceCounter::new();

        let store = SqliteStore::open(&path)?;
        EventStore::append(
            &store,
            &thread_id,
            1,
            AgentEventEnvelope::wrap(AgentEvent::text("m1", "early"), &seq),
        )
        .await?;
        store.finish_turn(&thread_id, 1).await?;

        let Err(error) = EventStore::append(
            &store,
            &thread_id,
            1,
            AgentEventEnvelope::wrap(AgentEvent::text("late", "late"), &seq),
        )
        .await
        else {
            bail!("append after finish must fail");
        };
        assert!(error.to_string().contains("cannot append to finished turn"));

        // A different, unfinished turn still accepts appends.
        EventStore::append(
            &store,
            &thread_id,
            2,
            AgentEventEnvelope::wrap(AgentEvent::text("m2", "ok"), &seq),
        )
        .await?;
        assert_eq!(store.event_count(&thread_id).await?, 2);
        Ok(())
    }

    /// Two executions sharing one `operation_id` must resolve to the most
    /// recently written one, matching `InMemoryExecutionStore`'s latest-wins
    /// `HashMap` contract instead of returning an indeterminate row.
    #[tokio::test]
    async fn operation_id_lookup_is_latest_wins() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let store = SqliteStore::open(&path)?;
        let thread_id = ThreadId::new();

        let mut first = ToolExecution::new_in_flight(
            "call_first",
            thread_id.clone(),
            "tool",
            "Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        first.set_operation_id("op_shared");
        store.record_execution(first).await?;

        let mut second = ToolExecution::new_in_flight(
            "call_second",
            thread_id,
            "tool",
            "Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        second.set_operation_id("op_shared");
        store.record_execution(second).await?;

        assert_eq!(
            store
                .get_execution_by_operation_id("op_shared")
                .await?
                .context("op_shared resolves")?
                .tool_call_id,
            "call_second",
            "the most recently written execution must win"
        );
        Ok(())
    }

    /// Re-pointing an execution's `operation_id` stops the old id resolving.
    #[tokio::test]
    async fn operation_id_lookup_and_supersession() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("agent.db");
        let store = SqliteStore::open(&path)?;
        let thread_id = ThreadId::new();

        let mut execution = ToolExecution::new_in_flight(
            "call_op",
            thread_id,
            "async_tool",
            "Async Tool",
            serde_json::json!({}),
            time::OffsetDateTime::now_utc(),
        );
        execution.set_operation_id("op_old");
        store.record_execution(execution.clone()).await?;
        assert_eq!(
            store
                .get_execution_by_operation_id("op_old")
                .await?
                .context("op_old resolves")?
                .tool_call_id,
            "call_op"
        );

        execution.set_operation_id("op_new");
        store.update_execution(execution).await?;
        assert!(
            store
                .get_execution_by_operation_id("op_old")
                .await?
                .is_none(),
            "superseded operation id must stop resolving"
        );
        assert_eq!(
            store
                .get_execution_by_operation_id("op_new")
                .await?
                .context("op_new resolves")?
                .tool_call_id,
            "call_op"
        );

        // Completing the execution survives and is observable.
        let mut completed = store
            .get_execution("call_op")
            .await?
            .context("execution present")?;
        completed.complete(ToolResult::success("done"));
        store.update_execution(completed).await?;
        let reloaded = store
            .get_execution("call_op")
            .await?
            .context("execution present")?;
        assert!(reloaded.is_completed());
        Ok(())
    }
}
