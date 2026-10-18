//! Progress events the worker writes under a lease: the lease-guarded
//! append on every backend, the subagent frames a parked child root's tool
//! child writes on its behalf, and the root-turn stall notice.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use agent_sdk_foundation::{
    AgentEvent, PendingToolCallInfo, SubagentLiveness, SubagentPhase, ThreadId, TokenUsage,
    ToolTier,
};
use agent_server::journal::SpawnChildrenExt;
use agent_server::journal::committed_event::CommittedEvent;
use agent_server::journal::task::{
    AgentTask, ChildSpawnSpec, LeaseId, SubmittedInputItem, TaskStatus, WorkerId,
};
use agent_server::journal::thread::{PurgeScope, PurgeSeed};
use agent_server::worker::ActivityBeacon;
use agent_server::worker::registry::InMemoryAgentDefinitionRegistry;

use super::super::progress::{
    FrameError, LeaseHolder, ProgressKind, Refusal, SubagentProgressParams, append_owned,
    emit_subagent_progress, resolve_subject, subagent_progress_loop,
};
use super::super::root_stall::{RootStallLeg, stall_notice_event};
use crate::config::{RootStallNotice, ServiceConfig, StorageConfig};
use crate::host::ServiceHost;
use crate::stores::StoreRegistry;

const STALL_TOOL: &str = "bip_root_turn_stall";

/// One backend a store-level case runs on, kept alive for the case.
struct Backend {
    name: &'static str,
    storage: StorageConfig,
    _dir: Option<tempfile::TempDir>,
    #[cfg(feature = "postgres")]
    schema: Option<(sqlx::postgres::PgConnection, String)>,
}

impl Backend {
    async fn stores(&self) -> Result<StoreRegistry> {
        let stores = StoreRegistry::from_config(&self.storage, super::sample_registry())?;
        stores.initialize().await?;
        Ok(stores)
    }

    async fn drop_schema(self) -> Result<()> {
        #[cfg(feature = "postgres")]
        if let Some((mut admin, schema)) = self.schema {
            sqlx::query(sqlx::AssertSqlSafe(format!("DROP SCHEMA {schema} CASCADE")))
                .execute(&mut admin)
                .await?;
        }
        Ok(())
    }
}

/// In memory always, `SQLite` with the feature, Postgres with the feature
/// and a `TEST_DATABASE_URL`.
async fn backends() -> Result<Vec<Backend>> {
    let mut backends = vec![Backend {
        name: "memory",
        storage: StorageConfig::default(),
        _dir: None,
        #[cfg(feature = "postgres")]
        schema: None,
    }];
    #[cfg(feature = "sqlite")]
    {
        let dir = tempfile::tempdir()?;
        backends.push(Backend {
            name: "sqlite",
            storage: StorageConfig {
                backend: crate::config::StorageBackend::Sqlite {
                    path: Some(dir.path().join("progress.db").display().to_string()),
                    max_connections: None,
                },
                ..StorageConfig::default()
            },
            _dir: Some(dir),
            #[cfg(feature = "postgres")]
            schema: None,
        });
    }
    #[cfg(feature = "postgres")]
    if let Ok(database_url) =
        std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
    {
        use sqlx::Connection;

        let schema = format!("progress_pg_{}", uuid::Uuid::new_v4().simple());
        let mut admin = sqlx::postgres::PgConnection::connect(&database_url).await?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await?;
        backends.push(Backend {
            name: "postgres",
            storage: StorageConfig {
                backend: crate::config::StorageBackend::Postgres,
                postgres: crate::config::PostgresStorageConfig {
                    database_url: Some(database_url),
                    schema: Some(schema.clone()),
                    max_connections: 8,
                },
            },
            _dir: None,
            schema: Some((admin, schema)),
        });
    }
    Ok(backends)
}

/// Run `case` on every backend, dropping each Postgres schema whatever the
/// outcome.
async fn on_every_backend<F, Fut>(case: F) -> Result<()>
where
    F: Fn(StoreRegistry, &'static str) -> Fut + Sync,
    Fut: std::future::Future<Output = Result<()>> + Send,
{
    for backend in backends().await? {
        let name = backend.name;
        let outcome = case(backend.stores().await?, name).await;
        backend.drop_schema().await?;
        outcome.with_context(|| format!("on {name}"))?;
    }
    Ok(())
}

async fn running_root(
    stores: &StoreRegistry,
    thread: &ThreadId,
    now: OffsetDateTime,
) -> Result<(AgentTask, LeaseHolder)> {
    let root = AgentTask::new_root_turn_with_input(
        thread.clone(),
        vec![SubmittedInputItem::Text {
            text: "work".into(),
        }],
        now,
        3,
    );
    let id = root.id.clone();
    stores.task_store.submit_root_turn(root).await?;
    let holder = LeaseHolder {
        task: id.clone(),
        worker: WorkerId::new(),
        lease: LeaseId::new(),
    };
    let running = stores
        .task_store
        .try_acquire_task(
            &id,
            holder.worker.clone(),
            holder.lease.clone(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the root acquires")?;
    Ok((running, holder))
}

fn notice(id: &str) -> AgentEvent {
    stall_notice_event(STALL_TOOL, time::Duration::seconds(300), id)
}

async fn events_on(stores: &StoreRegistry, thread: &ThreadId) -> Result<Vec<CommittedEvent>> {
    stores.event_repo.get_events(thread).await
}

/// The append writes under the lease and reaches live followers; a lost
/// lease or a purged thread refuses it and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_owned_append_needs_the_lease_and_a_live_thread() -> Result<()> {
    on_every_backend(|stores, name| async move {
        let now = OffsetDateTime::now_utc();
        let thread = ThreadId::from_string(format!("t-owned-append-{name}"));
        let (root, holder) = running_root(&stores, &thread, now).await?;
        let mut live = stores.event_notifier.subscribe(&thread);

        append_owned(
            &stores,
            &holder,
            &thread,
            vec![notice(root.id.as_str())],
            ProgressKind::RootStallNotice,
            now,
        )
        .await
        .map_err(|refusal| anyhow::anyhow!("the holder's append was refused: {refusal:?}"))?;
        assert_eq!(events_on(&stores, &thread).await?.len(), 1);
        let published = tokio::time::timeout(std::time::Duration::from_secs(1), live.recv())
            .await
            .context("the append reached live followers")??;
        assert!(matches!(published.event, AgentEvent::ToolProgress { .. }));

        let stranger = LeaseHolder {
            lease: LeaseId::new(),
            ..holder.clone()
        };
        let refused = append_owned(
            &stores,
            &stranger,
            &thread,
            vec![notice(root.id.as_str())],
            ProgressKind::RootStallNotice,
            now,
        )
        .await;
        assert!(
            matches!(refused, Err(Refusal::LeaseLost)),
            "a lost lease must refuse the append, got {refused:?}"
        );
        assert_eq!(events_on(&stores, &thread).await?.len(), 1);

        let purged = ThreadId::from_string(format!("t-owned-append-purged-{name}"));
        stores.thread_store.get_or_create(&purged, now).await?;
        stores
            .thread_store
            .begin_purge(
                &purged,
                PurgeSeed {
                    root_thread_id: purged.clone(),
                    scope: PurgeScope::Thread,
                    started_at: now,
                },
            )
            .await?;
        let refused = append_owned(
            &stores,
            &holder,
            &purged,
            vec![notice(root.id.as_str())],
            ProgressKind::RootStallNotice,
            now,
        )
        .await;
        assert!(
            matches!(refused, Err(Refusal::Purged)),
            "a purged thread must refuse the append, got {refused:?}"
        );
        assert!(events_on(&stores, &purged).await?.is_empty());
        Ok(())
    })
    .await
}

/// Each append is counted by kind, each refusal by reason.
#[cfg(feature = "otel")]
#[tokio::test]
async fn progress_appends_are_counted_by_kind_and_refusals_by_reason() -> Result<()> {
    let meter = crate::observability::test_meter::TestMeter::install().await;
    agent_server::observability::ServerMetrics::reset_for_testing();
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let now = OffsetDateTime::now_utc();
    let thread = ThreadId::from_string("t-progress-counted");
    let (root, holder) = running_root(&stores, &thread, now).await?;
    let purged = ThreadId::from_string("t-progress-counted-purged");
    stores.thread_store.get_or_create(&purged, now).await?;
    stores
        .thread_store
        .begin_purge(
            &purged,
            PurgeSeed {
                root_thread_id: purged.clone(),
                scope: PurgeScope::Thread,
                started_at: now,
            },
        )
        .await?;
    let stranger = LeaseHolder {
        lease: LeaseId::new(),
        ..holder.clone()
    };
    for (who, target) in [(&holder, &thread), (&stranger, &thread), (&holder, &purged)] {
        let _ = append_owned(
            &stores,
            who,
            target,
            vec![notice(root.id.as_str())],
            ProgressKind::RootStallNotice,
            now,
        )
        .await;
    }
    let events = |kind| meter.counter("agent_server.progress.events", &[("kind", kind)]);
    let refused = |reason| meter.counter("agent_server.progress.refused", &[("reason", reason)]);
    assert_eq!(events("root_stall_notice"), 1);
    assert_eq!(events("subagent_progress"), 0);
    assert_eq!(refused("lease_lost"), 1);
    assert_eq!(refused("purged"), 1);
    assert_eq!(refused("store"), 0);
    Ok(())
}

/// A parent with a linked subagent child root parked on two running tool
/// children, whose open turn has used `turn_usage`.
struct ParkedChild {
    parent_thread: ThreadId,
    child_root: AgentTask,
    tools: Vec<(AgentTask, LeaseHolder)>,
}

const TURN_USAGE: TokenUsage = TokenUsage {
    input_tokens: 8,
    output_tokens: 4,
    cached_input_tokens: 0,
    cache_creation_input_tokens: 0,
};

async fn parked_child(
    stores: &StoreRegistry,
    name: &str,
    now: OffsetDateTime,
) -> Result<ParkedChild> {
    let parent_thread = ThreadId::from_string(format!("t-progress-{name}"));
    let (_parent, _invocation, child_root) =
        super::persist_subagent_fixture(stores, &parent_thread, 60_000, now).await?;
    let worker = WorkerId::new();
    let lease = LeaseId::new();
    stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            worker.clone(),
            lease.clone(),
            now + time::Duration::minutes(10),
            now,
        )
        .await?
        .context("the child root acquires")?;
    let mut payload =
        super::pending_call_suspension_with_tier(&child_root.thread_id, "probe", ToolTier::Observe);
    let first: PendingToolCallInfo = payload
        .continuation
        .payload
        .pending_tool_calls
        .first()
        .cloned()
        .context("one pending call")?;
    payload
        .continuation
        .payload
        .pending_tool_calls
        .push(PendingToolCallInfo {
            id: "call_probe_2".into(),
            ..first
        });
    payload.continuation.payload.turn_usage = TURN_USAGE;
    let (parked, children) = stores
        .task_store
        .spawn_tool_children(
            &child_root.id,
            &worker,
            &lease,
            vec![ChildSpawnSpec::new(3), ChildSpawnSpec::new(3)],
            payload,
            None,
            Vec::new(),
            now,
        )
        .await?;
    let mut tools = Vec::new();
    for child in children {
        let holder = LeaseHolder {
            task: child.id.clone(),
            worker: WorkerId::new(),
            lease: LeaseId::new(),
        };
        let running = stores
            .task_store
            .try_acquire_task(
                &child.id,
                holder.worker.clone(),
                holder.lease.clone(),
                now + time::Duration::minutes(10),
                now,
            )
            .await?
            .context("the tool child acquires")?;
        tools.push((running, holder));
    }
    Ok(ParkedChild {
        parent_thread,
        child_root: parked,
        tools,
    })
}

fn live_frames(events: &[CommittedEvent]) -> Vec<(u32, u64, Option<u32>, SubagentLiveness)> {
    events
        .iter()
        .filter_map(|committed| match &committed.event {
            AgentEvent::SubagentProgress {
                tool_count,
                total_tokens,
                current_turn,
                liveness: Some(liveness),
                ..
            } => Some((
                *tool_count,
                *total_tokens,
                *current_turn,
                (**liveness).clone(),
            )),
            _ => None,
        })
        .collect()
}

async fn report(
    stores: &StoreRegistry,
    task: &AgentTask,
    holder: &LeaseHolder,
    activity: &ActivityBeacon,
    now: OffsetDateTime,
) -> Result<()> {
    let subject = resolve_subject(stores, task)
        .await?
        .context("the task reports on a subagent")?;
    match emit_subagent_progress(stores, holder, &subject, activity, now).await {
        Ok(()) => Ok(()),
        Err(FrameError::Read(error)) => Err(error),
        Err(FrameError::Refused(refusal)) => bail!("the frame was refused: {refusal:?}"),
    }
}

/// While the child root is parked, one of its running tool children reports
/// for the invocation under its own lease: once per tick however many run,
/// waiting on tools, with the open turn's usage the child root carries.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_parked_child_root_reports_through_one_running_tool_child() -> Result<()> {
    on_every_backend(|stores, name| async move {
        let now = OffsetDateTime::now_utc();
        let parked = parked_child(&stores, name, now).await?;
        assert_eq!(parked.child_root.status, TaskStatus::WaitingOnChildren);
        let activity = ActivityBeacon::new();
        activity.bump(now);
        for (task, holder) in &parked.tools {
            report(&stores, task, holder, &activity, now).await?;
        }
        let events = events_on(&stores, &parked.parent_thread).await?;
        let frames = live_frames(&events);
        let [(tool_count, total_tokens, current_turn, liveness)] = frames.as_slice() else {
            bail!("exactly one frame for two running tool children, got {frames:?}");
        };
        assert_eq!(liveness.phase, SubagentPhase::WaitingOnTools);
        assert_eq!(*tool_count, 0, "no tool call has finished");
        assert_eq!(
            *total_tokens,
            u64::from(TURN_USAGE.input_tokens + TURN_USAGE.output_tokens),
            "the parked turn's usage is reported, so the count does not drop at the park"
        );
        assert_eq!(*current_turn, Some(1));
        assert_eq!(liveness.last_activity_at, Some(now));
        assert_eq!(liveness.llm_calls, 0);

        // A steering wake resumes the child root while its tool children
        // run: from then on the child root reports, and a tool child's frame
        // is refused at the append.
        let (task, holder) = parked
            .tools
            .iter()
            .min_by(|left, right| left.0.id.cmp(&right.0.id))
            .context("an elected tool child")?;
        let woken = stores
            .task_store
            .commit(
                agent_server::journal::Transition::enqueue_steering_resume(
                    parked.child_root.id.clone(),
                    vec![agent_sdk_foundation::llm::ContentBlock::Text {
                        text: "also check the tests".into(),
                    }],
                ),
                now,
            )
            .await
            .map_err(|rejected| anyhow::anyhow!("steering wake: {rejected}"))?;
        assert!(
            matches!(woken.steering_resumed.as_slice(), [Some(_)]),
            "the child root woke: {:?}",
            woken.steering_resumed
        );
        let subject = resolve_subject(&stores, task)
            .await?
            .context("the tool child reports on a subagent")?;
        let moved = emit_subagent_progress(&stores, holder, &subject, &activity, now).await;
        assert!(
            matches!(moved, Err(FrameError::Refused(Refusal::SubjectMoved))),
            "a tool child must not report once the child root left its park, got {moved:?}"
        );
        assert_eq!(
            live_frames(&events_on(&stores, &parked.parent_thread).await?).len(),
            1
        );

        // A lost lease ends the tool child's reports.
        let (task, holder) = parked
            .tools
            .iter()
            .min_by(|left, right| left.0.id.cmp(&right.0.id))
            .context("an elected tool child")?;
        let stale = LeaseHolder {
            lease: LeaseId::new(),
            ..holder.clone()
        };
        let subject = resolve_subject(&stores, task)
            .await?
            .context("the tool child reports on a subagent")?;
        let refused = emit_subagent_progress(&stores, &stale, &subject, &activity, now).await;
        assert!(
            matches!(refused, Err(FrameError::Refused(Refusal::LeaseLost))),
            "got {refused:?}"
        );
        assert_eq!(
            live_frames(&events_on(&stores, &parked.parent_thread).await?).len(),
            1
        );
        Ok(())
    })
    .await
}

/// The progress loop of a tool child stops once its lease is lost.
#[tokio::test]
async fn the_progress_loop_stops_when_its_lease_is_lost() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let now = OffsetDateTime::now_utc();
    let parked = parked_child(&stores, "loop", now).await?;
    let (task, holder) = parked
        .tools
        .iter()
        .min_by(|left, right| left.0.id.cmp(&right.0.id))
        .context("an elected tool child")?;
    let cancel = CancellationToken::new();
    let handle = tokio::spawn(subagent_progress_loop(SubagentProgressParams {
        stores: stores.clone(),
        task: task.clone(),
        holder: holder.clone(),
        interval: std::time::Duration::from_millis(50),
        activity: ActivityBeacon::new(),
        cancel: cancel.clone(),
    }));
    // Frames flow while the lease holds.
    let mut frames = 0;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        frames = live_frames(&events_on(&stores, &parked.parent_thread).await?).len();
        if frames >= 2 {
            break;
        }
    }
    assert!(frames >= 2, "the tool child reported, got {frames} frames");

    stores
        .task_store
        .cancel_tree(&parked.child_root.id, OffsetDateTime::now_utc())
        .await?;
    let stopped = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    cancel.cancel();
    assert!(
        stopped.is_ok(),
        "the loop must stop on its own once its lease is gone"
    );
    Ok(())
}

/// A root turn whose beacon stays silent gets one notice at the threshold
/// and one at four times it, and a new episode after any sign of work. A
/// subagent's child root gets none.
#[tokio::test]
async fn a_silent_root_turn_gets_a_notice_at_one_and_four_thresholds() -> Result<()> {
    let stores = StoreRegistry::in_memory(super::sample_registry());
    let start = OffsetDateTime::now_utc();
    let thread = ThreadId::from_string("t-root-stall");
    let (root, holder) = running_root(&stores, &thread, start).await?;
    let config = RootStallNotice {
        after_secs: 300,
        tool_name: STALL_TOOL.into(),
    };
    let mut leg = RootStallLeg::new(Some(config.clone()), &root, holder.clone(), start)
        .context("a root turn gets the leg")?;
    let at = |secs: i64| start + time::Duration::seconds(secs);
    let notices = || async {
        Ok::<_, anyhow::Error>(
            events_on(&stores, &thread)
                .await?
                .into_iter()
                .filter_map(|committed| match committed.event {
                    AgentEvent::ToolProgress { name, message, .. } if name == STALL_TOOL => {
                        Some(message)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )
    };

    leg.tick(&stores, None, at(299)).await;
    assert!(
        notices().await?.is_empty(),
        "silent less than the threshold"
    );
    leg.tick(&stores, None, at(300)).await;
    leg.tick(&stores, None, at(600)).await;
    assert_eq!(notices().await?.len(), 1, "one notice at the threshold");
    leg.tick(&stores, None, at(1_200)).await;
    leg.tick(&stores, None, at(3_000)).await;
    assert_eq!(
        notices().await?,
        vec![
            "no agent events for 300s \u{2014} cancel or keep waiting".to_owned(),
            "no agent events for 1200s \u{2014} cancel or keep waiting".to_owned(),
        ],
        "one more at four times, then none for the rest of the silence"
    );

    // Work resumes and goes silent again: a new episode.
    leg.tick(&stores, Some(at(3_100)), at(3_101)).await;
    leg.tick(&stores, Some(at(3_100)), at(3_400)).await;
    assert_eq!(notices().await?.len(), 3);

    // A turn that keeps working never gets one.
    let busy_thread = ThreadId::from_string("t-root-busy");
    let (busy, busy_holder) = running_root(&stores, &busy_thread, start).await?;
    let mut busy_leg = RootStallLeg::new(Some(config.clone()), &busy, busy_holder, start)
        .context("a root turn gets the leg")?;
    for minute in 1..=30 {
        let now = at(minute * 60);
        busy_leg.tick(&stores, Some(now), now).await;
    }
    assert!(events_on(&stores, &busy_thread).await?.is_empty());

    // A subagent's running child root is not a plain root: its silence is
    // its stall budget's business.
    let (_parent, _invocation, child_root) = super::persist_subagent_fixture(
        &stores,
        &ThreadId::from_string("t-root-stall-parent"),
        60_000,
        start,
    )
    .await?;
    let child_holder = LeaseHolder {
        task: child_root.id.clone(),
        worker: WorkerId::new(),
        lease: LeaseId::new(),
    };
    let running_child = stores
        .task_store
        .try_acquire_task(
            &child_root.id,
            child_holder.worker.clone(),
            child_holder.lease.clone(),
            start + time::Duration::minutes(10),
            start,
        )
        .await?
        .context("the child root acquires")?;
    let mut child_leg = RootStallLeg::new(Some(config), &running_child, child_holder, start)
        .context("a root turn gets the leg")?;
    child_leg.tick(&stores, None, at(10_000)).await;
    let child_notices = events_on(&stores, &running_child.thread_id)
        .await?
        .into_iter()
        .filter(|committed| {
            matches!(&committed.event, AgentEvent::ToolProgress { name, .. } if name == STALL_TOOL)
        })
        .count();
    assert_eq!(child_notices, 0, "a subagent's child root gets no notice");
    Ok(())
}

#[test]
fn a_zero_stall_notice_threshold_is_refused() {
    let config = ServiceConfig {
        worker: crate::config::WorkerConfig {
            root_stall_notice: Some(RootStallNotice {
                after_secs: 0,
                tool_name: STALL_TOOL.into(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let refused = ServiceHost::validate_config(&config);
    assert!(
        refused.is_err_and(|error| error.to_string().contains("after_secs")),
        "a zero threshold would write a notice on every tick"
    );
}

/// bip's `root_turn_stall::build_stall_event` at origin/master 8dcdd9ed,
/// copied verbatim (constants inlined) as the oracle for the wire shape.
fn bip_build_stall_event(silent: time::Duration, root_task_id: &str) -> AgentEvent {
    let silent_secs = silent.whole_seconds().max(0);
    let since_ms = u64::try_from(silent.whole_milliseconds().max(0)).unwrap_or(u64::MAX);
    AgentEvent::ToolProgress {
        id: format!("root-turn-stall-{root_task_id}"),
        name: "bip_root_turn_stall".to_owned(),
        display_name: "Turn appears stalled".to_owned(),
        stage: "stalled".to_owned(),
        message: format!("no agent events for {silent_secs}s — cancel or keep waiting"),
        data: Some(serde_json::json!({
            "since_ms": since_ms,
            "root_task_id": root_task_id,
        })),
    }
}

#[test]
fn the_stall_notice_is_byte_identical_to_bips() -> Result<()> {
    for silent in [
        time::Duration::ZERO,
        time::Duration::milliseconds(300_250),
        time::Duration::seconds(1_200),
        time::Duration::seconds(-5),
    ] {
        let ours = serde_json::to_string(&stall_notice_event(STALL_TOOL, silent, "task-1"))?;
        let bips = serde_json::to_string(&bip_build_stall_event(silent, "task-1"))?;
        assert_eq!(ours, bips);
    }
    Ok(())
}

/// End to end: a plain root turn wedged in its provider stream gets the
/// notice from its own heartbeat, keeps running, and the view folds it into
/// `stalled`.
#[tokio::test]
async fn a_wedged_root_turn_is_reported_and_keeps_running() -> Result<()> {
    let config = ServiceConfig {
        worker: crate::config::WorkerConfig {
            pool_size: 2,
            heartbeat_interval_secs: 1,
            acquisition_interval_secs: 1,
            root_stall_notice: Some(RootStallNotice {
                after_secs: 1,
                tool_name: STALL_TOOL.into(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let provider = Arc::new(super::SubagentScriptProvider::new());
    let runtime = super::subagent_timeout_runtime(provider)?;
    let host = ServiceHost::new(
        config,
        Arc::new(InMemoryAgentDefinitionRegistry::new(
            super::probe_definition(),
        )),
        runtime,
    )?;
    let stores = host.stores().clone();
    let token = host.shutdown_token();
    let thread = ThreadId::from_string("t-root-wedged");
    let root = AgentTask::new_root_turn_with_input(
        thread.clone(),
        vec![SubmittedInputItem::Text {
            text: super::HANG_CHILD_TASK.into(),
        }],
        OffsetDateTime::now_utc(),
        3,
    );
    let root_id = root.id.clone();
    stores.task_store.submit_root_turn(root).await?;
    let host_handle = tokio::spawn(async move { host.run().await });

    let mut notices = Vec::new();
    for _ in 0..300 {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        notices = events_on(&stores, &thread)
            .await?
            .into_iter()
            .filter(|committed| {
                matches!(&committed.event, AgentEvent::ToolProgress { name, .. } if name == STALL_TOOL)
            })
            .collect::<Vec<_>>();
        if !notices.is_empty() {
            break;
        }
    }
    let running = stores
        .task_store
        .get(&root_id)
        .await?
        .context("the root exists")?;
    let events = events_on(&stores, &thread).await?;
    token.cancel();
    host_handle.await??;

    let first = notices.first().context("the wedged turn got a notice")?;
    let AgentEvent::ToolProgress { id, data, .. } = &first.event else {
        bail!("a notice is a ToolProgress");
    };
    assert_eq!(id, &format!("root-turn-stall-{root_id}"));
    let data = data.as_ref().context("the notice carries data")?;
    assert_eq!(data["root_task_id"], root_id.as_str());
    assert!(data["since_ms"].as_u64().is_some_and(|ms| ms >= 1_000));
    assert_eq!(
        running.status,
        TaskStatus::Running,
        "the notice never stops the turn"
    );

    let view_config = agent_sdk_foundation::view::ViewConfig {
        stall_tool_name: Some(STALL_TOOL.into()),
        ..Default::default()
    };
    let mut status = agent_sdk_foundation::view::ThreadViewStatus::default();
    for committed in &events {
        (status, _) = agent_sdk_foundation::view::fold(
            &status,
            committed.sequence,
            committed.timestamp,
            &committed.event,
            &view_config,
        );
    }
    assert!(status.stalled, "the view reads the notice as a stall");
    Ok(())
}

/// End to end: a subagent's child root parked on a tool that reports
/// progress for several seconds keeps sending frames, waiting on tools,
/// and every count only climbs up to the completion frame.
#[tokio::test]
async fn frames_continue_while_the_child_root_waits_on_a_long_tool() -> Result<()> {
    let events = run_subagent_over_probe(
        "t-progress-long-tool",
        Arc::new(ProgressingProbe {
            secs: 4,
            started: Arc::new(AtomicUsize::new(0)),
        }),
    )
    .await?;
    let frames = live_frames(&events);
    let waiting = frames
        .iter()
        .filter(|(_, _, _, liveness)| liveness.phase == SubagentPhase::WaitingOnTools)
        .count();
    assert!(
        waiting >= 2,
        "frames kept coming while the child root was parked, got {frames:?}"
    );
    for pair in frames.windows(2) {
        let [before, after] = pair else {
            bail!("a window of two");
        };
        assert!(after.0 >= before.0, "tool_count dropped: {frames:?}");
        assert!(after.1 >= before.1, "total_tokens dropped: {frames:?}");
        assert!(after.2 >= before.2, "current_turn dropped: {frames:?}");
        assert!(after.3.llm_calls >= before.3.llm_calls, "{frames:?}");
        assert!(
            after.3.output_tokens_live >= before.3.output_tokens_live,
            "{frames:?}"
        );
    }
    let last_live = frames.last().context("a live frame")?;
    let completion = hang_progress(&events)
        .into_iter()
        .find(|(completed, _)| *completed)
        .context("a completion frame")?;
    assert!(
        completion.1 >= last_live.1,
        "the completion frame's total {} is below a live frame's {}",
        completion.1,
        last_live.1
    );
    Ok(())
}

/// A child whose only output is the progress frames written on its behalf
/// still stalls: the frames are no evidence of work, so the hung tool
/// child's frames, one a second, never push the child root's two-and-a-half
/// second budget out. They stop once the sweep cancels the tool child.
#[tokio::test]
async fn a_child_that_only_reports_progress_still_stalls() -> Result<()> {
    let started = Arc::new(AtomicUsize::new(0));
    let config = ServiceConfig {
        worker: crate::config::WorkerConfig {
            pool_size: 4,
            heartbeat_interval_secs: 1,
            acquisition_interval_secs: 1,
            sweep_interval_secs: 1,
            subagent_progress_interval_secs: 1,
            tool_stall_budget_secs: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let (stores, events, _parent_thread) = run_subagent_host(
        config,
        "t-progress-only",
        Arc::new(super::HangingProbeExecutor {
            started: Arc::clone(&started),
        }),
        // Linger after the parent completes, so a frame written after the
        // tool child lost its lease would be seen.
        std::time::Duration::from_secs(3),
    )
    .await?;
    assert!(started.load(Ordering::SeqCst) >= 1, "the probe ran");
    let hang_root =
        super::root_task_of_thread(&stores, &ThreadId::from_string("t-progress-only-hang")).await?;
    assert_eq!(
        hang_root.status,
        TaskStatus::Failed,
        "the child stalled out"
    );
    let frames = hang_progress(&events);
    let waiting = live_frames(&events)
        .iter()
        .filter(|(_, _, _, liveness)| {
            matches!(
                liveness.phase,
                SubagentPhase::WaitingOnTools | SubagentPhase::Stalled
            )
        })
        .count();
    assert!(waiting >= 1, "the hung tool child reported: {frames:?}");
    let completed_at = frames
        .iter()
        .position(|(completed, _)| *completed)
        .context("a completion frame")?;
    assert_eq!(
        completed_at,
        frames.len() - 1,
        "no frame after the child root failed and its tool child lost its lease: {frames:?}"
    );
    Ok(())
}

/// `(completed, total_tokens)` of every `call_hang` frame, in order.
fn hang_progress(events: &[CommittedEvent]) -> Vec<(bool, u64)> {
    events
        .iter()
        .filter_map(|committed| match &committed.event {
            AgentEvent::SubagentProgress {
                subagent_id,
                completed,
                total_tokens,
                ..
            } if subagent_id == "call_hang" => Some((*completed, *total_tokens)),
            _ => None,
        })
        .collect()
}

/// A tool that reports progress every 200 ms for `secs`, then succeeds.
struct ProgressingProbe {
    secs: u64,
    started: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl crate::runtime::ToolCallExecutor for ProgressingProbe {
    async fn execute_tool_call(
        &self,
        _bootstrap: &agent_server::worker::ToolTaskBootstrap,
        collector: agent_server::worker::ToolEventCollector,
        _cancel: CancellationToken,
        _artifact_store: Option<Arc<agent_sdk::ArtifactStore>>,
    ) -> Result<agent_sdk_foundation::ToolResult> {
        self.started.fetch_add(1, Ordering::SeqCst);
        let until = std::time::Instant::now() + std::time::Duration::from_secs(self.secs);
        while std::time::Instant::now() < until {
            collector.note_progress();
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        Ok(agent_sdk_foundation::ToolResult::success("probe done"))
    }
}

async fn run_subagent_over_probe(
    thread: &str,
    executor: Arc<dyn crate::runtime::ToolCallExecutor>,
) -> Result<Vec<CommittedEvent>> {
    let config = ServiceConfig {
        worker: crate::config::WorkerConfig {
            pool_size: 4,
            heartbeat_interval_secs: 1,
            acquisition_interval_secs: 1,
            sweep_interval_secs: 1,
            subagent_progress_interval_secs: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let (_stores, events, _thread) =
        run_subagent_host(config, thread, executor, std::time::Duration::ZERO).await?;
    Ok(events)
}

/// Run a parent that spawns the `hang` and `fast` subagents, where the hang
/// child calls `probe` once on `executor`; return the parent thread's
/// events once the parent completed and `linger` passed.
async fn run_subagent_host(
    config: ServiceConfig,
    thread: &str,
    executor: Arc<dyn crate::runtime::ToolCallExecutor>,
    linger: std::time::Duration,
) -> Result<(StoreRegistry, Vec<CommittedEvent>, ThreadId)> {
    let provider = Arc::new(super::SubagentScriptProvider::with_hung_tool());
    let runtime = super::subagent_timeout_runtime_with_executor(provider, executor)?;
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(
        super::probe_definition(),
    ));
    let host = ServiceHost::new(config, registry, runtime)?;
    let stores = host.stores().clone();
    let token = host.shutdown_token();
    let parent_thread = ThreadId::from_string(thread);
    let parent = AgentTask::new_root_turn_with_input(
        parent_thread.clone(),
        vec![SubmittedInputItem::Text {
            text: "coordinate the helpers".into(),
        }],
        OffsetDateTime::now_utc(),
        3,
    );
    let parent_id = parent.id.clone();
    stores.task_store.submit_root_turn(parent).await?;
    let host_handle = tokio::spawn(async move { host.run().await });
    let finished = super::wait_for_status(&stores, &parent_id, TaskStatus::Completed, 1_500).await;
    tokio::time::sleep(linger).await;
    let events = events_on(&stores, &parent_thread).await?;
    token.cancel();
    host_handle.await??;
    finished?;
    Ok((stores, events, parent_thread))
}
