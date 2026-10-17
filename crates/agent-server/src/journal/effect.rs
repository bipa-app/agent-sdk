//! Effects: work outside the store whose record commits before it runs
//! (design §6.3, invariants I5–I7).
//!
//! An effect is a tool call, a model request, a process start or a git
//! step. [`run_effect`] is the one way to run one: it commits the effect's
//! `Started` row with the owner task's change in one transition, runs the
//! effect, then commits the outcome, the owner's change and the event that
//! reports it in another.
//!
//! A `Started` record's owner is live: every store write that makes a task
//! terminal (its settle, a cancel of its tree, a fail-closed lease sweep)
//! also ends the task's still-`Started` records as `Abandoned`, in the same
//! transaction. A transition that ends an owner and finishes its effect
//! therefore writes the [`EffectWrite::Finish`] first. So recovery reads
//! committed state only and never infers from age: a `Started` record is an
//! effect whose live owner may run it again, an `Abandoned` one ended
//! without an outcome and is never run again. A row an earlier release left
//! `started` under a task that is already terminal reads the same way:
//! readers treat a `Started` record whose owner is terminal as `Abandoned`.
//!
//! Recovery of a `Started` effect depends on its [`ToolReplay`]: a `Safe`
//! effect runs again, an `Unsafe` one never does. A `Safe` tool call writes
//! no row at all, because running it again is its recovery whether or not
//! a row exists (a deliberate reading of I5: the hottest path pays no
//! extra transaction for a record no decision reads).
//!
//! # Rows
//!
//! The rows live in `agent_sdk_execution_intents` (migrations 0003 and
//! 0027). A tool call's key is `{child_task_id}:{tool_call_id}`, the
//! operation id every earlier release wrote and read, so an old and a new
//! process find each other's rows during a rolling deploy.
//!
//! | [`EffectState`] | `status` | `error` | `resolution` |
//! |---|---|---|---|
//! | `Started` | `started` (a legacy `pending` row reads as `Started`) | NULL | NULL |
//! | `Completed` | `completed` | NULL | NULL |
//! | `Failed` | `failed` | the error | NULL |
//! | `Interrupted` | `failed` | `interrupted` | `interrupted` |
//! | `Salvaged` | `failed` | `salvaged: aborted response` | `salvaged` |
//! | `Abandoned` | `failed` | `abandoned: its owner ended first` | `abandoned` |

use agent_sdk_foundation::{ThreadId, ToolReplay};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use super::task::AgentTaskId;
use super::transition::{Committed, DurableStore, Guard, Rejected, Transition, Write};

/// Stable error prefix for a tool call that recovery did not run again.
pub const TOOL_INTERRUPTED_PREFIX: &str = "interrupted:";

/// The key of an effect's row.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EffectKey(String);

impl EffectKey {
    /// The key of the tool call `tool_call_id` run by the tool child
    /// `child`: `{child}:{tool_call_id}`, the operation id earlier releases
    /// used, kept so mixed-version processes agree on it.
    #[must_use]
    pub fn tool(child: &AgentTaskId, tool_call_id: &str) -> Self {
        Self(format!("{}:{tool_call_id}", child.0))
    }

    /// The key of the process `name` started under `owner`'s lease:
    /// `process:{owner}:{name}`.
    #[must_use]
    pub fn process(owner: &AgentTaskId, name: &str) -> Self {
        Self(format!("process:{}:{name}", owner.0))
    }

    /// The key of the git step `step` run under `owner`'s lease:
    /// `git:{owner}:{step}`.
    #[must_use]
    pub fn git(owner: &AgentTaskId, step: &str) -> Self {
        Self(format!("git:{}:{step}", owner.0))
    }

    /// A key as a store read it back. Public because the stores live in
    /// another crate, and a host may keep a key it read to finish later.
    #[must_use]
    pub fn from_stored(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EffectKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an effect does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    ToolCall,
    LlmRequest,
    ProcessStart,
    GitStep,
}

/// The kind-specific part of an effect.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectSubject {
    Tool {
        call_id: String,
        name: String,
        input: serde_json::Value,
    },
    Llm {
        turn_attempt_id: String,
        provider: String,
        model: String,
    },
    /// A host process started for a project: `argv_digest` identifies the
    /// command line without storing it.
    Process {
        project_hash: String,
        name: String,
        argv_digest: String,
    },
    /// One step of a git operation in `repo`. `idempotency_key`, when the
    /// step has one, lets recovery check whether it already happened (a
    /// commit trailer, a ref name).
    Git {
        repo: String,
        step: String,
        idempotency_key: Option<String>,
    },
}

impl EffectSubject {
    #[must_use]
    pub const fn kind(&self) -> EffectKind {
        match self {
            Self::Tool { .. } => EffectKind::ToolCall,
            Self::Llm { .. } => EffectKind::LlmRequest,
            Self::Process { .. } => EffectKind::ProcessStart,
            Self::Git { .. } => EffectKind::GitStep,
        }
    }
}

/// Where an effect is in its life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectState {
    /// Its record committed; it may have run, in part or in full.
    Started,
    Completed,
    /// It ran and reported `error`.
    Failed {
        error: String,
    },
    /// Recovery found it `Started` and ended it without running it again.
    Interrupted {
        captured_output: Option<String>,
    },
    /// Recovery found it `Started` and kept the output it had journaled.
    Salvaged {
        captured_output: Option<String>,
    },
    /// Its owner became terminal while it was `Started`, so it ended with
    /// no outcome. It is never run again.
    Abandoned,
}

impl EffectState {
    #[must_use]
    pub const fn is_started(&self) -> bool {
        matches!(self, Self::Started)
    }

    /// The state as the `outcome` attribute of the
    /// `agent_server.effect.transitions` metric labels it.
    #[must_use]
    pub const fn outcome_label(&self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed { .. } => "failed",
            Self::Interrupted { .. } => "interrupted",
            Self::Salvaged { .. } => "salvaged",
            Self::Abandoned => "abandoned",
        }
    }
}

/// Count an effect record written with `outcome`.
#[cfg(feature = "otel")]
pub(crate) fn record_effect(kind: EffectKind, outcome: &'static str) {
    crate::observability::ServerMetrics::global().record_effect(kind, outcome);
}

/// Without `otel` there is nothing to count.
#[cfg(not(feature = "otel"))]
pub(crate) const fn record_effect(_kind: EffectKind, _outcome: &'static str) {}

/// An effect about to start: the row [`EffectWrite::Start`] inserts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectIntent {
    pub key: EffectKey,
    pub replay: ToolReplay,
    /// The task whose lease the effect runs under.
    pub owner: AgentTaskId,
    pub thread: ThreadId,
    pub subject: EffectSubject,
    pub attempt: u32,
}

impl EffectIntent {
    #[must_use]
    pub const fn kind(&self) -> EffectKind {
        self.subject.kind()
    }
}

/// An effect's row as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectRecord {
    pub key: EffectKey,
    pub replay: ToolReplay,
    pub owner: AgentTaskId,
    /// `None` for a row written before migration 0027.
    pub thread: Option<ThreadId>,
    pub subject: EffectSubject,
    pub attempt: u32,
    pub state: EffectState,
    /// The owner thread's next event sequence when the effect started:
    /// every event at or after it came later. `None` before 0027.
    pub journal_anchor: Option<u64>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

impl EffectRecord {
    #[must_use]
    pub const fn kind(&self) -> EffectKind {
        self.subject.kind()
    }

    /// The record [`EffectWrite::Start`] stores for `intent`.
    #[must_use]
    pub fn started(intent: EffectIntent, journal_anchor: Option<u64>, now: OffsetDateTime) -> Self {
        Self {
            key: intent.key,
            replay: intent.replay,
            owner: intent.owner,
            thread: Some(intent.thread),
            subject: intent.subject,
            attempt: intent.attempt,
            state: EffectState::Started,
            journal_anchor,
            created_at: now,
            updated_at: now,
        }
    }
}

/// A [`Write::Effect`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectWrite {
    /// Insert the effect's `Started` row. Needs a [`Guard::EffectFree`] on
    /// its key and a [`Guard::TaskIs`] with the owner's lease.
    Start(EffectIntent),
    /// End a `Started` effect of `owner`. Refused with
    /// [`Rejected::Guard`] unless the row exists, is `Started` and belongs
    /// to `owner`, so a second finish never overwrites an outcome.
    Finish {
        key: EffectKey,
        owner: AgentTaskId,
        state: EffectState,
    },
}

impl EffectWrite {
    #[must_use]
    pub const fn key(&self) -> &EffectKey {
        match self {
            Self::Start(intent) => &intent.key,
            Self::Finish { key, .. } => key,
        }
    }

    #[must_use]
    pub const fn owner(&self) -> &AgentTaskId {
        match self {
            Self::Start(intent) => &intent.owner,
            Self::Finish { owner, .. } => owner,
        }
    }
}

/// The column values of an effect's row, shared by the SQL backends so the
/// mapping of the table above lives in one place.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EffectRow {
    pub operation_id: String,
    pub effect_class: String,
    pub tool_call_id: String,
    pub child_task_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub status: String,
    pub error: Option<String>,
    pub effect_kind: String,
    pub attempt: i64,
    pub resolution: Option<String>,
    pub captured_output: Option<String>,
    pub journal_anchor: Option<i64>,
    pub thread_id: Option<String>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

/// `status`, `error`, `resolution` and `captured_output` for a state.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectStateColumns {
    pub status: &'static str,
    pub error: Option<String>,
    pub resolution: Option<&'static str>,
    pub captured_output: Option<String>,
}

const SALVAGED_ERROR: &str = "salvaged: aborted response";
const ABANDONED_ERROR: &str = "abandoned: its owner ended first";
const INTERRUPTED_ERROR: &str = "interrupted";

impl EffectState {
    #[doc(hidden)]
    #[must_use]
    pub fn columns(&self) -> EffectStateColumns {
        let (status, error, resolution, captured_output) = match self {
            Self::Started => ("started", None, None, None),
            Self::Completed => ("completed", None, None, None),
            Self::Failed { error } => ("failed", Some(error.clone()), None, None),
            Self::Interrupted { captured_output } => (
                "failed",
                Some(INTERRUPTED_ERROR.to_owned()),
                Some("interrupted"),
                captured_output.clone(),
            ),
            Self::Salvaged { captured_output } => (
                "failed",
                Some(SALVAGED_ERROR.to_owned()),
                Some("salvaged"),
                captured_output.clone(),
            ),
            Self::Abandoned => (
                "failed",
                Some(ABANDONED_ERROR.to_owned()),
                Some("abandoned"),
                None,
            ),
        };
        EffectStateColumns {
            status,
            error,
            resolution,
            captured_output,
        }
    }

    fn from_columns(
        status: &str,
        error: Option<String>,
        resolution: Option<&str>,
        captured_output: Option<String>,
    ) -> anyhow::Result<Self> {
        Ok(match (status, resolution) {
            ("pending" | "started", None) => Self::Started,
            ("completed", None) => Self::Completed,
            ("failed", None) => Self::Failed {
                error: error.context("a failed effect row carries an error")?,
            },
            ("failed", Some("interrupted")) => Self::Interrupted { captured_output },
            ("failed", Some("salvaged")) => Self::Salvaged { captured_output },
            ("failed", Some("abandoned")) => Self::Abandoned,
            (status, resolution) => {
                bail!("unknown effect status {status:?} with resolution {resolution:?}")
            }
        })
    }
}

#[doc(hidden)]
#[must_use]
pub const fn replay_wire(replay: ToolReplay) -> &'static str {
    match replay {
        ToolReplay::Safe => "replay_safe",
        ToolReplay::Unsafe => "side_effecting",
    }
}

/// The kind a row's `effect_kind` names, if it names one this release
/// knows.
#[doc(hidden)]
#[must_use]
pub fn kind_from_wire(kind: &str) -> Option<EffectKind> {
    match kind {
        "tool_call" => Some(EffectKind::ToolCall),
        "llm_request" => Some(EffectKind::LlmRequest),
        "process_start" => Some(EffectKind::ProcessStart),
        "git_step" => Some(EffectKind::GitStep),
        _ => None,
    }
}

fn replay_from_wire(class: &str) -> anyhow::Result<ToolReplay> {
    match class {
        "replay_safe" => Ok(ToolReplay::Safe),
        "side_effecting" | "resumable" => Ok(ToolReplay::Unsafe),
        other => bail!("unknown effect class {other:?}"),
    }
}

#[doc(hidden)]
#[must_use]
pub const fn kind_wire(kind: EffectKind) -> &'static str {
    match kind {
        EffectKind::ToolCall => "tool_call",
        EffectKind::LlmRequest => "llm_request",
        EffectKind::ProcessStart => "process_start",
        EffectKind::GitStep => "git_step",
    }
}

impl EffectRecord {
    /// The row for this record.
    ///
    /// # Errors
    /// An attempt or anchor that does not fit the column.
    #[doc(hidden)]
    pub fn to_row(&self) -> anyhow::Result<EffectRow> {
        let (tool_call_id, tool_name, input) = match &self.subject {
            EffectSubject::Tool {
                call_id,
                name,
                input,
            } => (call_id.clone(), name.clone(), input.clone()),
            EffectSubject::Llm {
                turn_attempt_id,
                provider,
                model,
            } => (
                turn_attempt_id.clone(),
                format!("llm:{provider}/{model}"),
                serde_json::json!({
                    "turn_attempt_id": turn_attempt_id,
                    "provider": provider,
                    "model": model,
                }),
            ),
            EffectSubject::Process {
                project_hash,
                name,
                argv_digest,
            } => (
                name.clone(),
                format!("process:{name}"),
                serde_json::json!({
                    "project_hash": project_hash,
                    "name": name,
                    "argv_digest": argv_digest,
                }),
            ),
            EffectSubject::Git {
                repo,
                step,
                idempotency_key,
            } => (
                step.clone(),
                format!("git:{step}"),
                serde_json::json!({
                    "repo": repo,
                    "step": step,
                    "idempotency_key": idempotency_key,
                }),
            ),
        };
        let columns = self.state.columns();
        Ok(EffectRow {
            operation_id: self.key.0.clone(),
            effect_class: replay_wire(self.replay).to_owned(),
            tool_call_id,
            child_task_id: self.owner.0.clone(),
            tool_name,
            input,
            status: columns.status.to_owned(),
            error: columns.error,
            effect_kind: kind_wire(self.kind()).to_owned(),
            attempt: i64::from(self.attempt),
            resolution: columns.resolution.map(str::to_owned),
            captured_output: columns.captured_output,
            journal_anchor: self
                .journal_anchor
                .map(i64::try_from)
                .transpose()
                .context("effect journal anchor out of range")?,
            thread_id: self.thread.as_ref().map(|thread| thread.0.clone()),
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }

    /// The record a row holds.
    ///
    /// # Errors
    /// A value no release writes.
    #[doc(hidden)]
    pub fn from_row(row: EffectRow) -> anyhow::Result<Self> {
        let optional = |name: &str| {
            row.input
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        };
        let field = |name: &str| {
            optional(name).with_context(|| {
                format!(
                    "{} effect row {} lacks {name}",
                    row.effect_kind, row.operation_id
                )
            })
        };
        let subject = match row.effect_kind.as_str() {
            "tool_call" => EffectSubject::Tool {
                call_id: row.tool_call_id.clone(),
                name: row.tool_name.clone(),
                input: row.input.clone(),
            },
            "llm_request" => EffectSubject::Llm {
                turn_attempt_id: field("turn_attempt_id")?,
                provider: field("provider")?,
                model: field("model")?,
            },
            "process_start" => EffectSubject::Process {
                project_hash: field("project_hash")?,
                name: field("name")?,
                argv_digest: field("argv_digest")?,
            },
            "git_step" => EffectSubject::Git {
                repo: field("repo")?,
                step: field("step")?,
                idempotency_key: optional("idempotency_key"),
            },
            other => bail!("unknown effect kind {other:?}"),
        };
        Ok(Self {
            state: EffectState::from_columns(
                &row.status,
                row.error,
                row.resolution.as_deref(),
                row.captured_output,
            )?,
            key: EffectKey(row.operation_id),
            replay: replay_from_wire(&row.effect_class)?,
            owner: AgentTaskId(row.child_task_id),
            thread: row.thread_id.map(ThreadId),
            subject,
            attempt: u32::try_from(row.attempt).context("effect attempt out of range")?,
            journal_anchor: row
                .journal_anchor
                .map(u64::try_from)
                .transpose()
                .context("effect journal anchor out of range")?,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

impl Transition {
    /// The effect keys whose rows [`Self::check_effects`] reads: every
    /// [`Guard::EffectFree`] and every [`EffectWrite::Finish`].
    #[must_use]
    pub fn effect_keys_read(&self) -> Vec<&EffectKey> {
        let guarded = self.guards.iter().filter_map(|guard| match guard {
            Guard::EffectFree { key } => Some(key),
            Guard::TaskIs { .. } => None,
        });
        let finished = self.effect_writes().filter_map(|write| match write {
            EffectWrite::Finish { key, .. } => Some(key),
            EffectWrite::Start(_) => None,
        });
        guarded.chain(finished).collect()
    }

    /// The transition's effect writes, in order.
    pub fn effect_writes(&self) -> impl Iterator<Item = &EffectWrite> {
        self.writes.iter().filter_map(|write| match write {
            Write::Effect(effect) => Some(effect),
            Write::SettleTask { .. }
            | Write::RejectConfirmation { .. }
            | Write::PauseOnConfirmation { .. }
            | Write::ReparkAfterSteering { .. }
            | Write::AppendCompaction { .. }
            | Write::AppendEvents { .. }
            | Write::SpawnChildren { .. } => None,
        })
    }

    /// Check the expectations the effect guards and writes carry against
    /// the rows `stored` returns, before the first write: a
    /// [`Guard::EffectFree`] key has no row, and a
    /// [`EffectWrite::Finish`] key's row is `Started` and the owner's.
    /// Every backend runs this after the task guards and before any write,
    /// so an in-memory store, which cannot roll back, never fails midway.
    ///
    /// # Errors
    /// [`Rejected::Guard`], indexed by the guard, or for a finish by the
    /// guard count plus its write index.
    pub fn check_effects<'a>(
        &self,
        stored: impl Fn(&EffectKey) -> Option<&'a EffectRecord>,
    ) -> Result<(), Rejected> {
        for (index, guard) in self.guards.iter().enumerate() {
            if let Guard::EffectFree { key } = guard
                && stored(key).is_some()
            {
                return Err(Rejected::Guard {
                    index,
                    reason: format!("effect {key} already has a record"),
                });
            }
        }
        for (index, write) in self.writes.iter().enumerate() {
            let Write::Effect(EffectWrite::Finish { key, owner, .. }) = write else {
                continue;
            };
            let reject = |reason: String| Rejected::Guard {
                index: self.guards.len() + index,
                reason,
            };
            let Some(record) = stored(key) else {
                return Err(reject(format!("effect {key} has no record")));
            };
            if &record.owner != owner {
                return Err(reject(format!(
                    "effect {key} belongs to task {}, not {owner}",
                    record.owner
                )));
            }
            if !record.state.is_started() {
                return Err(reject(format!(
                    "effect {key} already ended ({:?})",
                    record.state
                )));
            }
        }
        Ok(())
    }
}

/// The `Started` effects of `owner`, oldest first: what an owner that
/// re-acquired its task recovers before it runs again.
///
/// # Errors
/// The store could not be read.
pub async fn started_effects_for_task(
    store: &dyn DurableStore,
    owner: &AgentTaskId,
) -> anyhow::Result<Vec<EffectRecord>> {
    Ok(store
        .effects_for_task(owner)
        .await?
        .into_iter()
        .filter(|record| record.state.is_started())
        .collect())
}

/// What [`run_effect`] did.
#[derive(Debug)]
pub enum EffectRun<T> {
    /// The effect ran. `committed` is the end transition's result.
    Ran { value: T, committed: Committed },
    /// The start transition did not commit, so the effect never ran:
    /// another worker holds its record ([`Rejected::Guard`]), the owner
    /// lost its lease, or the store failed.
    Refused(Rejected),
}

/// Run one effect (design §6.3).
///
/// `start`, with a [`Guard::EffectFree`] on the key and the
/// [`EffectWrite::Start`] added, commits before `effect` runs; the caller's
/// `start` carries the owner's leased [`Guard::TaskIs`]. `end` turns the
/// effect's output into the transition that records its outcome and the
/// state to finish it with. The [`EffectWrite::Finish`] is added to that
/// transition, ahead of its writes (a write that ends the owner abandons
/// whatever is still `Started`), so the outcome, the owner's change and its
/// events commit together.
///
/// Each transition is committed once: the store retries its own lock
/// conflicts, and a second commit of `start` would only meet its own row.
///
/// # Errors
/// An error from `end`, or any refusal or error committing its transition.
/// The effect then stays `Started`, which recovery reads as having run.
pub async fn run_effect<T, F, Fut, E>(
    store: &dyn DurableStore,
    intent: EffectIntent,
    mut start: Transition,
    effect: F,
    end: E,
    now: OffsetDateTime,
) -> anyhow::Result<EffectRun<T>>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
    E: FnOnce(&T) -> anyhow::Result<(Transition, EffectState)>,
{
    let key = intent.key.clone();
    let owner = intent.owner.clone();
    let kind = intent.kind();
    start.guards.push(Guard::EffectFree { key: key.clone() });
    start.writes.push(Write::Effect(EffectWrite::Start(intent)));
    if let Err(rejected) = store.commit(start, now).await {
        record_effect(kind, "refused");
        log::warn!(
            effect_key = key.as_str(),
            effect_kind = kind_wire(kind),
            task_id = owner.as_str();
            "effect claim did not commit; the effect will not run: {rejected:#}"
        );
        return Ok(EffectRun::Refused(rejected));
    }
    record_effect(kind, "started");
    crate::fail_point!("effect.after_start");

    let value = effect().await;

    let (mut end_transition, state) = end(&value)?;
    let outcome = state.outcome_label();
    end_transition
        .writes
        .insert(0, Write::Effect(EffectWrite::Finish { key, owner, state }));
    crate::fail_point!("effect.before_finish");
    let committed = store
        .commit(end_transition, OffsetDateTime::now_utc())
        .await
        .map_err(anyhow::Error::from)?;
    record_effect(kind, outcome);
    Ok(EffectRun::Ran { value, committed })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(state: EffectState) -> EffectRecord {
        EffectRecord {
            key: EffectKey::tool(&AgentTaskId::from_string("task_1"), "call_1"),
            replay: ToolReplay::Unsafe,
            owner: AgentTaskId::from_string("task_1"),
            thread: Some(ThreadId::from_string("thread_1")),
            subject: EffectSubject::Tool {
                call_id: "call_1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            },
            attempt: 1,
            state,
            journal_anchor: Some(7),
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn a_tool_key_is_the_legacy_operation_id() {
        let key = EffectKey::tool(&AgentTaskId::from_string("task_abc"), "call_xyz");
        assert_eq!(key.as_str(), "task_abc:call_xyz");
    }

    #[test]
    fn every_state_round_trips_through_its_row() -> anyhow::Result<()> {
        for state in [
            EffectState::Started,
            EffectState::Completed,
            EffectState::Failed {
                error: "boom".into(),
            },
            EffectState::Interrupted {
                captured_output: Some("partial".into()),
            },
            EffectState::Salvaged {
                captured_output: None,
            },
            EffectState::Abandoned,
        ] {
            let stored = record(state);
            let row = stored.to_row()?;
            assert_eq!(
                row.status == "failed",
                row.error.is_some(),
                "0003's error check: {row:?}"
            );
            assert_eq!(EffectRecord::from_row(row)?, stored);
        }
        Ok(())
    }

    #[test]
    fn an_llm_subject_round_trips_through_the_legacy_columns() -> anyhow::Result<()> {
        let mut stored = record(EffectState::Started);
        stored.key = EffectKey::from_stored("llm:task_1:attempt_1");
        stored.replay = ToolReplay::Safe;
        stored.subject = EffectSubject::Llm {
            turn_attempt_id: "attempt_1".into(),
            provider: "anthropic".into(),
            model: "claude-sonnet-5-5".into(),
        };
        let row = stored.to_row()?;
        assert_eq!(row.effect_kind, "llm_request");
        assert_eq!(row.tool_name, "llm:anthropic/claude-sonnet-5-5");
        assert_eq!(EffectRecord::from_row(row)?, stored);
        Ok(())
    }

    #[test]
    fn process_and_git_subjects_round_trip_through_the_legacy_columns() -> anyhow::Result<()> {
        let owner = AgentTaskId::from_string("task_1");
        for (key, subject, tool_name) in [
            (
                EffectKey::process(&owner, "dev-server"),
                EffectSubject::Process {
                    project_hash: "p1".into(),
                    name: "dev-server".into(),
                    argv_digest: "sha256:abc".into(),
                },
                "process:dev-server",
            ),
            (
                EffectKey::git(&owner, "commit"),
                EffectSubject::Git {
                    repo: "/repo".into(),
                    step: "commit".into(),
                    idempotency_key: Some("Bip-Landing: l1".into()),
                },
                "git:commit",
            ),
            (
                EffectKey::git(&owner, "push"),
                EffectSubject::Git {
                    repo: "/repo".into(),
                    step: "push".into(),
                    idempotency_key: None,
                },
                "git:push",
            ),
        ] {
            let mut stored = record(EffectState::Started);
            stored.key = key;
            stored.subject = subject;
            let row = stored.to_row()?;
            assert_eq!(row.tool_name, tool_name);
            assert_eq!(row.effect_kind, kind_wire(stored.kind()));
            assert_eq!(EffectRecord::from_row(row)?, stored);
        }
        assert_eq!(
            EffectKey::process(&owner, "dev-server").as_str(),
            "process:task_1:dev-server"
        );
        assert_eq!(EffectKey::git(&owner, "push").as_str(), "git:task_1:push");
        Ok(())
    }

    #[test]
    fn legacy_rows_read_as_unsafe_started_tool_calls() -> anyhow::Result<()> {
        let mut row = record(EffectState::Completed).to_row()?;
        row.status = "pending".into();
        row.effect_class = "resumable".into();
        row.thread_id = None;
        row.journal_anchor = None;
        let read = EffectRecord::from_row(row)?;
        assert_eq!(read.state, EffectState::Started);
        assert_eq!(read.replay, ToolReplay::Unsafe);
        assert_eq!(read.kind(), EffectKind::ToolCall);
        Ok(())
    }

    #[test]
    fn expectations_refuse_a_claimed_key_and_a_second_finish() {
        let started = record(EffectState::Started);
        let done = record(EffectState::Completed);
        let claim = Transition {
            guards: vec![Guard::EffectFree {
                key: started.key.clone(),
            }],
            writes: Vec::new(),
        };
        assert!(claim.check_effects(|_| None).is_ok());
        assert!(matches!(
            claim.check_effects(|_| Some(&started)),
            Err(Rejected::Guard { index: 0, .. })
        ));

        let finish = Transition {
            guards: Vec::new(),
            writes: vec![Write::Effect(EffectWrite::Finish {
                key: started.key.clone(),
                owner: started.owner.clone(),
                state: EffectState::Completed,
            })],
        };
        assert!(finish.check_effects(|_| Some(&started)).is_ok());
        for stored in [None, Some(&done)] {
            assert!(matches!(
                finish.check_effects(|_| stored),
                Err(Rejected::Guard { index: 0, .. })
            ));
        }
        let mut foreign = started;
        foreign.owner = AgentTaskId::from_string("task_2");
        assert!(finish.check_effects(|_| Some(&foreign)).is_err());
    }
}
