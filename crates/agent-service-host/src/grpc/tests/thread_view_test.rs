//! The session view RPCs over the in-memory stores: refused when views are
//! off and for a thread being purged; a snapshot equal to the fold of the
//! events; paging; and a stream whose snapshot plus changes equals that fold
//! after each event, through a skipped notification, a lagged subscription
//! and an event committed between the subscription and the snapshot.

use super::*;
use agent_server::journal::view::{IndexedItem, ThreadView, ThreadViewStatus, ViewConfig, ViewOp};

const STALL: &str = "__view_stall";

fn enabled() -> ViewConfig {
    ViewConfig {
        enabled: true,
        stall_tool_name: Some(STALL.to_owned()),
        visible_document_kinds: Vec::new(),
    }
}

fn view_shared(view: ViewConfig) -> Result<Arc<GrpcShared>> {
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
        Vec::new(),
    )));
    view_shared_with(StoreRegistry::in_memory(registry), view)
}

fn view_shared_with(stores: StoreRegistry, view: ViewConfig) -> Result<Arc<GrpcShared>> {
    let runtime = runtime_with(
        Arc::new(ScriptedProvider::new(Vec::new())),
        Arc::new(NoopToolExecutor),
    )?;
    let mut shared = GrpcShared::new(
        stores,
        runtime,
        HealthSurface::shared(),
        CancellationToken::new(),
        time::Duration::seconds(30),
        std::time::Duration::from_secs(10),
        AdmissionConfig::default(),
    );
    shared.view = view;
    Ok(Arc::new(shared))
}

async fn thread(shared: &GrpcShared, name: &str) -> Result<ThreadId> {
    let thread_id = ThreadId::from_string(name);
    let _ = shared
        .stores
        .thread_store
        .get_or_create(&thread_id, OffsetDateTime::UNIX_EPOCH)
        .await?;
    Ok(thread_id)
}

async fn commit(shared: &GrpcShared, thread_id: &ThreadId, events: Vec<AgentEvent>) -> Result<()> {
    let committed = shared
        .stores
        .event_repo
        .commit_event_batch(thread_id, events, OffsetDateTime::UNIX_EPOCH)
        .await?;
    shared.stores.event_notifier.notify(&committed);
    Ok(())
}

fn tool_start(id: &str) -> AgentEvent {
    AgentEvent::ToolCallStart {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        input: serde_json::json!({}),
        tier: ToolTier::Observe,
    }
}

fn tool_end(id: &str) -> AgentEvent {
    AgentEvent::ToolCallEnd {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        result: ToolResult::success("ok"),
    }
}

fn text_delta(message: &str) -> AgentEvent {
    AgentEvent::TextDelta {
        message_id: message.to_owned(),
        delta: "x".to_owned(),
        emitter_task_id: None,
    }
}

fn from_ms(ms: i64) -> Result<OffsetDateTime> {
    Ok(OffsetDateTime::from_unix_timestamp_nanos(
        i128::from(ms) * 1_000_000,
    )?)
}

fn snapshot_view(snapshot: &pb::ThreadViewSnapshot) -> Result<ThreadView> {
    let status: ThreadViewStatus = serde_json::from_str(&snapshot.status_json)?;
    let items = snapshot
        .items
        .iter()
        .map(|item| {
            Ok((
                item.key.clone(),
                IndexedItem {
                    index: item.index,
                    item: serde_json::from_str(&item.body_json)?,
                    started_at: item.started_at_ms.map(from_ms).transpose()?,
                    updated_at: item.updated_at_ms.map(from_ms).transpose()?,
                },
            ))
        })
        .collect::<Result<_>>()?;
    Ok(ThreadView {
        status,
        items,
        next_index: snapshot.next_index,
    })
}

async fn folded(shared: &GrpcShared, thread_id: &ThreadId) -> Result<ThreadView> {
    let events = shared.stores.event_repo.get_events(thread_id).await?;
    Ok(ThreadView::from_events(
        events
            .iter()
            .map(|event| (event.sequence, event.timestamp, &event.event)),
        &enabled(),
    ))
}

type ViewFrames = <GrpcEventService as AgentEventService>::StreamThreadViewStream;

async fn next_frame(stream: &mut ViewFrames) -> Result<pb::stream_thread_view_response::Item> {
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .context("timed out waiting for a view frame")?
        .context("view stream ended")?
        .context("view frame error")?
        .item
        .context("empty view frame")
}

fn expect_precondition<T>(result: Result<T, Status>, what: &str) -> Result<()> {
    match result {
        Err(status) if status.code() == tonic::Code::FailedPrecondition => Ok(()),
        Err(status) => anyhow::bail!("{what}: expected FAILED_PRECONDITION, got {status:?}"),
        Ok(_) => anyhow::bail!("{what}: expected FAILED_PRECONDITION, got a response"),
    }
}

#[tokio::test]
async fn every_view_rpc_refuses_on_a_host_without_views() -> Result<()> {
    let shared = view_shared(ViewConfig::default())?;
    let thread_id = thread(&shared, "view-rpc-off").await?;
    commit(&shared, &thread_id, vec![tool_start("t1")]).await?;
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    expect_precondition(
        events
            .get_thread_view(Request::new(pb::GetThreadViewRequest {
                thread_id: thread_id.0.clone(),
                item_limit: None,
            }))
            .await,
        "GetThreadView",
    )?;
    expect_precondition(
        events
            .get_thread_view_items(Request::new(pb::GetThreadViewItemsRequest {
                thread_id: thread_id.0.clone(),
                before_index: 10,
                limit: None,
            }))
            .await,
        "GetThreadViewItems",
    )?;
    expect_precondition(
        events
            .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
                thread_id: thread_id.0.clone(),
                item_limit: None,
            }))
            .await,
        "StreamThreadView",
    )
}

#[tokio::test]
async fn the_snapshot_is_the_fold_of_the_events_and_pages_back() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-snapshot").await?;
    for i in 0..5 {
        let id = format!("t{i}");
        commit(&shared, &thread_id, vec![tool_start(&id), tool_end(&id)]).await?;
    }
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let whole = events
        .get_thread_view(Request::new(pb::GetThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    ensure!(!whole.has_older);
    ensure!(whole.sequence == Some(9));
    ensure!(snapshot_view(&whole)? == folded(&shared, &thread_id).await?);

    let newest = events
        .get_thread_view(Request::new(pb::GetThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: Some(2),
        }))
        .await?
        .into_inner();
    ensure!(newest.has_older);
    ensure!(
        newest
            .items
            .iter()
            .map(|item| item.index)
            .collect::<Vec<_>>()
            == vec![3, 4]
    );
    let older = events
        .get_thread_view_items(Request::new(pb::GetThreadViewItemsRequest {
            thread_id: thread_id.0.clone(),
            before_index: 3,
            limit: Some(2),
        }))
        .await?
        .into_inner();
    ensure!(
        older
            .items
            .iter()
            .map(|item| item.index)
            .collect::<Vec<_>>()
            == vec![1, 2]
    );
    Ok(())
}

#[tokio::test]
async fn the_stream_snapshot_plus_its_changes_equals_the_store_view() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-stream").await?;
    commit(
        &shared,
        &thread_id,
        vec![AgentEvent::start(thread_id.clone(), 1)],
    )
    .await?;
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let mut stream = events
        .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    let pb::stream_thread_view_response::Item::Snapshot(snapshot) = next_frame(&mut stream).await?
    else {
        anyhow::bail!("the stream opens with a snapshot");
    };
    let mut client = snapshot_view(&snapshot)?;

    let batches = vec![
        vec![tool_start("t1")],
        vec![text_delta("m1"), text_delta("m1")],
        vec![tool_end("t1")],
    ];
    let mut expected = 0;
    for batch in batches {
        expected += batch.len();
        commit(&shared, &thread_id, batch).await?;
    }
    for _ in 0..expected {
        let pb::stream_thread_view_response::Item::Change(change) = next_frame(&mut stream).await?
        else {
            anyhow::bail!("a committed event becomes a change");
        };
        let ops: Vec<ViewOp> = serde_json::from_str(&change.ops_json)?;
        for op in &ops {
            ensure!(
                client.apply_at(op, change.committed_at_ms.map(from_ms).transpose()?),
                "op {op:?} addresses a missing item"
            );
        }
        ensure!(client.status.sequence == Some(change.sequence));
    }
    ensure!(client == folded(&shared, &thread_id).await?);
    Ok(())
}

#[tokio::test]
async fn a_skipped_notification_is_backfilled_from_the_journal() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-gap").await?;
    commit(&shared, &thread_id, vec![tool_start("t0")]).await?;
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let mut stream = events
        .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    let pb::stream_thread_view_response::Item::Snapshot(snapshot) = next_frame(&mut stream).await?
    else {
        anyhow::bail!("the stream opens with a snapshot");
    };
    let mut client = snapshot_view(&snapshot)?;

    let _silent = shared
        .stores
        .event_repo
        .commit_event_batch(&thread_id, vec![tool_end("t0")], OffsetDateTime::UNIX_EPOCH)
        .await?;
    commit(&shared, &thread_id, vec![tool_start("t1")]).await?;
    for expected in [1, 2] {
        let pb::stream_thread_view_response::Item::Change(change) = next_frame(&mut stream).await?
        else {
            anyhow::bail!("a change");
        };
        ensure!(
            change.sequence == expected,
            "change {} in order",
            change.sequence
        );
        for op in &serde_json::from_str::<Vec<ViewOp>>(&change.ops_json)? {
            ensure!(client.apply_at(op, change.committed_at_ms.map(from_ms).transpose()?));
        }
    }
    ensure!(client == folded(&shared, &thread_id).await?);
    Ok(())
}

#[tokio::test]
async fn a_lagged_subscriber_gets_a_fresh_snapshot() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-lag").await?;
    commit(&shared, &thread_id, vec![tool_start("t0")]).await?;
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let mut stream = events
        .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: Some(5000),
        }))
        .await?
        .into_inner();
    let pb::stream_thread_view_response::Item::Snapshot(_) = next_frame(&mut stream).await? else {
        anyhow::bail!("the stream opens with a snapshot");
    };
    for i in 0..300 {
        commit(&shared, &thread_id, vec![tool_start(&format!("x{i}"))]).await?;
    }
    let want = folded(&shared, &thread_id).await?;
    let mut client: Option<ThreadView> = None;
    let mut snapshots = 0;
    while client.as_ref() != Some(&want) {
        match next_frame(&mut stream).await? {
            pb::stream_thread_view_response::Item::Snapshot(snapshot) => {
                snapshots += 1;
                client = Some(snapshot_view(&snapshot)?);
            }
            pb::stream_thread_view_response::Item::Change(change) => {
                if let Some(view) = client.as_mut() {
                    for op in &serde_json::from_str::<Vec<ViewOp>>(&change.ops_json)? {
                        ensure!(
                            view.apply_at(op, change.committed_at_ms.map(from_ms).transpose()?)
                        );
                    }
                }
            }
            pb::stream_thread_view_response::Item::Closed(_) => anyhow::bail!("closed"),
        }
    }
    ensure!(snapshots >= 1, "the lagged subscriber got a fresh snapshot");
    Ok(())
}

/// An event committed while the snapshot is being folded, past its cut-off,
/// reaches the client exactly once: the stream subscribed before it folded.
#[tokio::test]
async fn an_event_during_the_snapshot_fold_arrives_once() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-between").await?;
    commit(&shared, &thread_id, vec![tool_start("t0")]).await?;
    let (subscribed_tx, subscribed_rx) = tokio::sync::oneshot::channel();
    let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
    *super::super::thread_view::AFTER_NEXT_SEQUENCE
        .lock()
        .map_err(|_| anyhow::anyhow!("seam lock"))? = Some((subscribed_tx, resume_rx));
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let mut stream = events
        .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    let first = tokio::spawn(async move {
        let frame = next_frame(&mut stream).await;
        (frame, stream)
    });
    subscribed_rx.await?;
    commit(&shared, &thread_id, vec![tool_end("t0")]).await?;
    let _resumed = resume_tx.send(());
    let (frame, mut stream) = first.await?;
    let pb::stream_thread_view_response::Item::Snapshot(snapshot) = frame? else {
        anyhow::bail!("the stream opens with a snapshot");
    };
    let mut client = snapshot_view(&snapshot)?;
    ensure!(
        snapshot.sequence == Some(0),
        "the snapshot predates the event"
    );
    let pb::stream_thread_view_response::Item::Change(change) = next_frame(&mut stream).await?
    else {
        anyhow::bail!("the event arrives as a change");
    };
    ensure!(change.sequence == 1);
    for op in &serde_json::from_str::<Vec<ViewOp>>(&change.ops_json)? {
        ensure!(
            client.apply_at(op, change.committed_at_ms.map(from_ms).transpose()?),
            "op {op:?} addresses a missing item"
        );
    }
    ensure!(
        client == folded(&shared, &thread_id).await?,
        "the event between subscribe and snapshot arrived exactly once"
    );
    Ok(())
}

#[tokio::test]
async fn a_thread_being_purged_is_not_found() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-purge").await?;
    commit(&shared, &thread_id, vec![tool_start("t0")]).await?;
    let _ = shared
        .stores
        .thread_store
        .begin_purge(
            &thread_id,
            agent_server::journal::thread::PurgeSeed {
                root_thread_id: thread_id.clone(),
                scope: agent_server::journal::thread::PurgeScope::Thread,
                started_at: OffsetDateTime::UNIX_EPOCH,
            },
        )
        .await?;
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let codes = [
        events
            .get_thread_view(Request::new(pb::GetThreadViewRequest {
                thread_id: thread_id.0.clone(),
                item_limit: None,
            }))
            .await
            .err()
            .map(|status| status.code()),
        events
            .get_thread_view_items(Request::new(pb::GetThreadViewItemsRequest {
                thread_id: thread_id.0.clone(),
                before_index: 1,
                limit: None,
            }))
            .await
            .err()
            .map(|status| status.code()),
        events
            .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
                thread_id: thread_id.0.clone(),
                item_limit: None,
            }))
            .await
            .err()
            .map(|status| status.code()),
    ];
    ensure!(
        codes
            .iter()
            .all(|code| *code == Some(tonic::Code::NotFound)),
        "every view RPC answers NOT_FOUND for a thread being purged: {codes:?}"
    );
    Ok(())
}

/// One turn of the worker's shape, in the batches it commits them in.
#[cfg(feature = "sqlite")]
fn heavy_turn(thread_id: &ThreadId, turn: usize, output: &str) -> Vec<Vec<AgentEvent>> {
    let mut batches: Vec<Vec<AgentEvent>> = vec![vec![
        AgentEvent::user_input(
            thread_id.clone(),
            vec![agent_sdk_foundation::llm::ContentBlock::Text {
                text: "do the next step".to_owned(),
            }],
        ),
        AgentEvent::start(thread_id.clone(), turn).with_emitter_task_id("root"),
    ]];
    for (kind, message) in [
        ("thinking", format!("th{turn}")),
        ("text", format!("m{turn}")),
    ] {
        for _ in 0..4 {
            batches.push(
                (0..16)
                    .map(|_| match kind {
                        "thinking" => AgentEvent::ThinkingDelta {
                            message_id: message.clone(),
                            delta: "a few words ".to_owned(),
                            emitter_task_id: Some("root".to_owned()),
                        },
                        _ => AgentEvent::TextDelta {
                            message_id: message.clone(),
                            delta: "a few words ".to_owned(),
                            emitter_task_id: Some("root".to_owned()),
                        },
                    })
                    .collect(),
            );
        }
    }
    batches.push(vec![
        AgentEvent::Thinking {
            message_id: format!("th{turn}"),
            text: "a few words ".repeat(64),
            emitter_task_id: Some("root".to_owned()),
        },
        AgentEvent::Text {
            message_id: format!("m{turn}"),
            text: "a few words ".repeat(64),
            emitter_task_id: Some("root".to_owned()),
        },
    ]);
    for tool in 0..5 {
        let id = format!("t{turn}-{tool}");
        batches.push(vec![tool_start(&id)]);
        for step in 0..3 {
            batches.push(vec![AgentEvent::ToolProgress {
                id: id.clone(),
                name: "bash".to_owned(),
                display_name: "Bash".to_owned(),
                stage: format!("step {step}"),
                message: "running".to_owned(),
                data: None,
            }]);
        }
        batches.push(vec![AgentEvent::ToolCallEnd {
            id,
            name: "bash".to_owned(),
            display_name: "Bash".to_owned(),
            result: ToolResult::success(output.to_owned()),
        }]);
    }
    batches.push(vec![
        AgentEvent::TurnComplete {
            turn,
            usage: TokenUsage::default(),
            emitter_task_id: Some("root".to_owned()),
        },
        AgentEvent::Done {
            thread_id: thread_id.clone(),
            total_turns: turn,
            total_usage: TokenUsage::default(),
            duration: std::time::Duration::from_millis(1),
            estimated_cost_usd: None,
            emitter_task_id: Some("root".to_owned()),
        },
    ]);
    batches
}

/// The read-time fold's cost on a heavy thread, run only with
/// `AGENT_SDK_VIEW_BENCH=1` (`--release --nocapture` prints p50/p95). A
/// SQLite file store holds turns of the worker's shape: a start, the prompt,
/// text and thinking deltas in 16-delta flushes, the closing texts, five tool
/// calls with progress and ~2 KiB outputs, two subagent updates, a turn end.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn bench_read_time_fold_on_a_heavy_thread() -> Result<()> {
    if std::env::var("AGENT_SDK_VIEW_BENCH").ok().as_deref() != Some("1") {
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
        Vec::new(),
    )));
    let stores = StoreRegistry::from_config(
        &crate::config::StorageConfig {
            backend: crate::config::StorageBackend::Sqlite {
                path: Some(dir.path().join("bench.db").display().to_string()),
                max_connections: None,
            },
            ..crate::config::StorageConfig::default()
        },
        registry,
    )?;
    let shared = view_shared_with(stores, enabled())?;
    let config = enabled();
    let output = "x".repeat(2048);
    for turns in [50_usize, 250, 1000] {
        let thread_id = ThreadId::from_string(format!("bench-{turns}"));
        let _ = shared
            .stores
            .thread_store
            .get_or_create(&thread_id, OffsetDateTime::UNIX_EPOCH)
            .await?;
        for turn in 0..turns {
            let batches = heavy_turn(&thread_id, turn, &output);
            for batch in batches {
                let _ = shared
                    .stores
                    .event_repo
                    .commit_event_batch(&thread_id, batch, OffsetDateTime::UNIX_EPOCH)
                    .await?;
            }
        }
        let mut read_ms = Vec::new();
        let mut fold_ms = Vec::new();
        let mut events = 0;
        let mut items = 0;
        for _ in 0..20 {
            let started = std::time::Instant::now();
            let committed = shared.stores.event_repo.get_events(&thread_id).await?;
            read_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = std::time::Instant::now();
            let view = ThreadView::from_events(
                committed
                    .iter()
                    .map(|event| (event.sequence, event.timestamp, &event.event)),
                &config,
            );
            fold_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            events = committed.len();
            items = view.items.len();
        }
        let p = |samples: &mut Vec<f64>, percent: usize| {
            samples.sort_by(f64::total_cmp);
            samples[(samples.len() - 1) * percent / 100]
        };
        drop(super::super::thread_view::fold_thread(&shared, &thread_id).await?);
        let mut warm_ms = Vec::new();
        for turn in 0..20 {
            let id = format!("warm-{turn}");
            let mut batch =
                vec![AgentEvent::start(thread_id.clone(), turn).with_emitter_task_id("root")];
            batch.extend((0..16).map(|_| text_delta(&format!("w{turn}"))));
            batch.extend([tool_start(&id), tool_end(&id)]);
            let _ = shared
                .stores
                .event_repo
                .commit_event_batch(&thread_id, batch, OffsetDateTime::UNIX_EPOCH)
                .await?;
            let started = std::time::Instant::now();
            let view = super::super::thread_view::fold_thread(&shared, &thread_id).await?;
            let _snapshot = super::super::thread_view::snapshot_proto(&thread_id, &view, 200)?;
            drop(view);
            warm_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "fold bench: {turns} turns, {events} events, {items} items | cold read p50 {:.1} p95 {:.1} ms, cold fold p50 {:.1} p95 {:.1} ms | cached read+fold+snapshot after a 19-event turn p50 {:.2} p95 {:.2} ms",
            p(&mut read_ms, 50),
            p(&mut read_ms, 95),
            p(&mut fold_ms, 50),
            p(&mut fold_ms, 95),
            p(&mut warm_ms, 50),
            p(&mut warm_ms, 95),
        );
    }
    Ok(())
}

/// The cached view folds only the events after its sequence, and folds the
/// retained events again when the janitor pruned some it never saw.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn the_cached_view_follows_appends_and_refolds_across_a_hole() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let registry = Arc::new(InMemoryAgentDefinitionRegistry::new(mock_definition(
        Vec::new(),
    )));
    let stores = StoreRegistry::from_config(
        &crate::config::StorageConfig {
            backend: crate::config::StorageBackend::Sqlite {
                path: Some(dir.path().join("cache.db").display().to_string()),
                max_connections: None,
            },
            ..crate::config::StorageConfig::default()
        },
        registry,
    )?;
    let shared = view_shared_with(stores, enabled())?;
    let thread_id = thread(&shared, "view-rpc-cache").await?;
    commit(&shared, &thread_id, vec![tool_start("t0")]).await?;
    drop(super::super::thread_view::fold_thread(&shared, &thread_id).await?);
    commit(&shared, &thread_id, vec![tool_end("t0"), tool_start("t1")]).await?;
    let appended = super::super::thread_view::fold_thread(&shared, &thread_id).await?;
    ensure!(appended.status.sequence == Some(2));
    ensure!(
        *appended == folded(&shared, &thread_id).await?,
        "the cache folded the appends"
    );
    drop(appended);
    for i in 0..5 {
        commit(&shared, &thread_id, vec![tool_start(&format!("p{i}"))]).await?;
    }
    let _ = shared
        .stores
        .retention_store
        .advance_floor(&thread_id, 6, OffsetDateTime::UNIX_EPOCH)
        .await?;
    let refolded = super::super::thread_view::fold_thread(&shared, &thread_id)
        .await?
        .clone();
    ensure!(
        refolded == folded(&shared, &thread_id).await?,
        "a cache whose next events were pruned folds the retained events again"
    );
    Ok(())
}

/// A change carries its event's commit time, and a snapshot's items carry
/// the times of the events that put and last changed them.
#[tokio::test]
async fn changes_and_snapshot_items_carry_commit_times() -> Result<()> {
    let shared = view_shared(enabled())?;
    let thread_id = thread(&shared, "view-rpc-times").await?;
    let put_at = OffsetDateTime::from_unix_timestamp(1_800_000_000)?;
    let ended_at = put_at + time::Duration::seconds(7);
    let committed = shared
        .stores
        .event_repo
        .commit_event_batch(
            &thread_id,
            vec![AgentEvent::start(thread_id.clone(), 1), tool_start("t1")],
            put_at,
        )
        .await?;
    shared.stores.event_notifier.notify(&committed);
    let events = GrpcEventService {
        shared: Arc::clone(&shared),
    };
    let mut stream = events
        .stream_thread_view(Request::new(pb::StreamThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    let pb::stream_thread_view_response::Item::Snapshot(_) = next_frame(&mut stream).await? else {
        anyhow::bail!("the stream opens with a snapshot");
    };
    let committed = shared
        .stores
        .event_repo
        .commit_event_batch(&thread_id, vec![tool_end("t1")], ended_at)
        .await?;
    shared.stores.event_notifier.notify(&committed);
    let pb::stream_thread_view_response::Item::Change(change) = next_frame(&mut stream).await?
    else {
        anyhow::bail!("the tool's end arrives as a change");
    };
    assert_eq!(change.committed_at_ms, Some(1_800_000_007_000));

    let snapshot = events
        .get_thread_view(Request::new(pb::GetThreadViewRequest {
            thread_id: thread_id.0.clone(),
            item_limit: None,
        }))
        .await?
        .into_inner();
    let item = snapshot
        .items
        .iter()
        .find(|item| item.key == "tool:t1")
        .ok_or_else(|| anyhow::anyhow!("the tool item"))?;
    assert_eq!(
        item.started_at_ms,
        Some(1_800_000_000_000),
        "put by the tool's start"
    );
    assert_eq!(
        item.updated_at_ms,
        Some(1_800_000_007_000),
        "changed by its end"
    );
    Ok(())
}
