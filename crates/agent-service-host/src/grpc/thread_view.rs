//! The session view RPCs of `AgentEventService`: the fold of a thread's
//! committed events into its view, computed when it is read; paging; and a
//! stream of the ops each later event makes.

use std::sync::Arc;

use agent_sdk_foundation::ThreadId;
use agent_server::journal::view::{
    IndexedItem, ThreadView, ThreadViewStatus, ViewConfig, fold, signals,
};
use async_stream::try_stream;
use futures::Stream;
use tonic::Status;

use super::{GrpcShared, as_stream_status, closed_stream_response, fetch_missed_range, pb};

const DEFAULT_ITEMS: usize = 200;
const MAX_ITEMS: usize = 5000;

pub(super) fn item_limit(requested: Option<u32>) -> usize {
    requested
        .and_then(|limit| usize::try_from(limit).ok())
        .filter(|limit| *limit > 0)
        .map_or(DEFAULT_ITEMS, |limit| limit.min(MAX_ITEMS))
}

/// `FAILED_PRECONDITION` on a host that does not serve views; `NOT_FOUND`
/// for a missing thread or one being purged.
pub(super) async fn admit(shared: &GrpcShared, thread_id: &ThreadId) -> Result<(), Status> {
    if !shared.view.enabled {
        signals::stream(signals::REFUSED);
        return Err(Status::failed_precondition(
            "thread views are not served on this host",
        ));
    }
    let thread = shared.require_thread(thread_id).await?;
    if thread.status.is_purge_fenced() {
        signals::stream(signals::NOT_FOUND);
        return Err(Status::not_found(format!(
            "thread '{thread_id}' is being deleted"
        )));
    }
    Ok(())
}

fn json<T: serde::Serialize>(value: &T, what: &'static str) -> Result<String, Status> {
    serde_json::to_string(value)
        .map_err(|error| Status::internal(format!("encode {what}: {error}")))
}

fn item_proto(key: &str, indexed: &IndexedItem) -> Result<pb::ThreadViewItem, Status> {
    Ok(pb::ThreadViewItem {
        key: key.to_owned(),
        index: indexed.index,
        body_json: json(&indexed.item, "view item")?,
    })
}

/// The items below `before` (all when `None`), in timeline order, the newest
/// `limit` of them, and whether older ones exist.
fn newest_items(
    view: &ThreadView,
    before: Option<u64>,
    limit: usize,
) -> (Vec<(&str, &IndexedItem)>, bool) {
    let mut items: Vec<(&str, &IndexedItem)> = view
        .items
        .iter()
        .filter(|(_, indexed)| before.is_none_or(|before| indexed.index < before))
        .map(|(key, indexed)| (key.as_str(), indexed))
        .collect();
    items.sort_by_key(|(_, indexed)| indexed.index);
    let has_older = items.len() > limit;
    let newest = items.split_off(items.len().saturating_sub(limit));
    (newest, has_older)
}

pub(super) fn snapshot_proto(
    thread_id: &ThreadId,
    view: &ThreadView,
    limit: usize,
) -> Result<pb::ThreadViewSnapshot, Status> {
    let (items, has_older) = newest_items(view, None, limit);
    Ok(pb::ThreadViewSnapshot {
        thread_id: thread_id.0.clone(),
        status_json: json(&view.status, "view status")?,
        sequence: view.status.sequence,
        next_index: view.next_index,
        items: items
            .into_iter()
            .map(|(key, indexed)| item_proto(key, indexed))
            .collect::<Result<_, _>>()?,
        has_older,
    })
}

pub(super) fn items_proto(
    view: &ThreadView,
    before: u64,
    limit: usize,
) -> Result<Vec<pb::ThreadViewItem>, Status> {
    newest_items(view, Some(before), limit)
        .0
        .into_iter()
        .map(|(key, indexed)| item_proto(key, indexed))
        .collect()
}

/// How many threads' folded views the daemon keeps in memory.
const CACHED_VIEWS: usize = 32;

/// The folded views of the most recently read threads. A view only grows:
/// committed events are append-only, so a read folds the events after the
/// cached sequence and nothing before it.
#[derive(Default)]
pub(super) struct ViewCache {
    entries: std::sync::Mutex<std::collections::VecDeque<(ThreadId, CachedView)>>,
}

type CachedView = Arc<tokio::sync::Mutex<ThreadView>>;

impl ViewCache {
    pub(super) fn entry(&self, thread_id: &ThreadId) -> CachedView {
        let Ok(mut entries) = self.entries.lock() else {
            return CachedView::default();
        };
        let view = entries
            .iter()
            .position(|(id, _)| id == thread_id)
            .and_then(|at| entries.remove(at))
            .map(|(_, view)| view)
            .unwrap_or_default();
        entries.push_back((thread_id.clone(), Arc::clone(&view)));
        while entries.len() > CACHED_VIEWS {
            entries.pop_front();
        }
        view
    }
}

/// The thread's view, current to its latest committed event. A cached view
/// folds only the events after its sequence; when those start past it (the
/// janitor pruned some), or nothing is cached, it folds every retained event.
pub(super) async fn fold_thread(
    shared: &GrpcShared,
    thread_id: &ThreadId,
) -> Result<tokio::sync::OwnedMutexGuard<ThreadView>, Status> {
    let mut view = shared.view_cache.entry(thread_id).lock_owned().await;
    let repo = &shared.stores.event_repo;
    let next = repo
        .next_sequence(thread_id)
        .await
        .map_err(as_stream_status("reading the thread's next event sequence"))?;
    #[cfg(test)]
    {
        let hook = AFTER_NEXT_SEQUENCE
            .lock()
            .ok()
            .and_then(|mut hook| hook.take());
        if let Some((reached, resume)) = hook {
            let _sent = reached.send(());
            let _resumed = resume.await;
        }
    }
    let expected = view.status.sequence.map_or(0, |folded| folded + 1);
    if expected >= next {
        return Ok(view);
    }
    let started = std::time::Instant::now();
    let mut events = match view.status.sequence {
        Some(folded) => repo.get_events_in_range(thread_id, folded, next - 1).await,
        None => repo.get_events(thread_id).await,
    }
    .map_err(as_stream_status("reading the thread's events for its view"))?;
    events.retain(|event| event.sequence < next);
    if view.status.sequence.is_some()
        && events.first().map(|event| event.sequence) != Some(expected)
    {
        *view = ThreadView::default();
        events = repo
            .get_events(thread_id)
            .await
            .map_err(as_stream_status("reading the thread's events for its view"))?;
        events.retain(|event| event.sequence < next);
    }
    for event in &events {
        view.fold_in(event.sequence, event.timestamp, &event.event, &shared.view);
    }
    signals::folded(events.len(), started.elapsed().as_secs_f64());
    Ok(view)
}

fn snapshot_frame(
    thread_id: &ThreadId,
    view: &ThreadView,
    limit: usize,
) -> Result<pb::StreamThreadViewResponse, Status> {
    Ok(pb::StreamThreadViewResponse {
        item: Some(pb::stream_thread_view_response::Item::Snapshot(
            snapshot_proto(thread_id, view, limit)?,
        )),
    })
}

fn closed_frame(
    thread_id: &ThreadId,
    reason: pb::StreamCloseReason,
    last_sequence: Option<u64>,
) -> pb::StreamThreadViewResponse {
    let Some(pb::stream_thread_events_response::Item::Closed(closed)) =
        closed_stream_response(thread_id, reason, last_sequence).item
    else {
        return pb::StreamThreadViewResponse { item: None };
    };
    pb::StreamThreadViewResponse {
        item: Some(pb::stream_thread_view_response::Item::Closed(closed)),
    }
}

/// Fold `event` onto `status` and frame its ops; `None` for an event the
/// status already folded.
fn change_frame(
    thread_id: &ThreadId,
    status: &mut ThreadViewStatus,
    event: &agent_server::CommittedEvent,
    config: &ViewConfig,
) -> Result<Option<pb::StreamThreadViewResponse>, Status> {
    let (next, ops) = fold(
        status,
        event.sequence,
        event.timestamp,
        &event.event,
        config,
    );
    if ops.is_empty() {
        return Ok(None);
    }
    *status = next;
    Ok(Some(pb::StreamThreadViewResponse {
        item: Some(pb::stream_thread_view_response::Item::Change(
            pb::ThreadViewChange {
                thread_id: thread_id.0.clone(),
                sequence: event.sequence,
                ops_json: json(&ops, "view ops")?,
            },
        )),
    }))
}

/// Test seam inside the fold, after it read the thread's next sequence and
/// before it reads the events: it signals the first sender, then waits on
/// the second. An event committed there is past the fold's cut-off, so it
/// reaches a stream only if the stream subscribed before it folded.
#[cfg(test)]
pub(super) static AFTER_NEXT_SEQUENCE: std::sync::Mutex<
    Option<(
        tokio::sync::oneshot::Sender<()>,
        tokio::sync::oneshot::Receiver<()>,
    )>,
> = std::sync::Mutex::new(None);

/// A snapshot, then the ops of every committed event after it. The live
/// subscription opens before the snapshot's events are read, so no event
/// falls between them; events the snapshot already folded are dropped by
/// sequence. A lagged subscription or a failed backfill sends a fresh
/// snapshot.
pub(super) fn view_stream(
    shared: Arc<GrpcShared>,
    thread_id: ThreadId,
    limit: usize,
) -> impl Stream<Item = Result<pb::StreamThreadViewResponse, Status>> + Send {
    try_stream! {
        let mut live = shared.stores.event_notifier.subscribe(&thread_id);
        let view = fold_thread(shared.as_ref(), &thread_id).await?;
        let snapshot = snapshot_frame(&thread_id, &view, limit)?;
        let mut status = view.status.clone();
        drop(view);
        signals::stream(signals::OPENED);
        yield snapshot;
        loop {
            let received = tokio::select! {
                () = shared.shutdown.cancelled() => None,
                received = live.recv() => Some(received),
            };
            let resnapshot = match received {
                None | Some(Err(tokio::sync::broadcast::error::RecvError::Closed)) => {
                    signals::stream(signals::CLOSED);
                    yield closed_frame(
                        &thread_id,
                        pb::StreamCloseReason::ServerShutdown,
                        status.sequence,
                    );
                    return;
                }
                Some(Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped))) => {
                    tracing::info!(
                        thread_id = %thread_id,
                        skipped,
                        "thread view stream lagged; sending a fresh snapshot"
                    );
                    true
                }
                Some(Ok(event)) => {
                    let mut resnapshot = false;
                    if status.sequence.is_none_or(|folded| event.sequence > folded) {
                        let expected = status.sequence.map_or(0, |folded| folded + 1);
                        if event.sequence > expected {
                            match fetch_missed_range(
                                shared.as_ref(),
                                &thread_id,
                                status.sequence,
                                event.sequence,
                            )
                            .await
                            {
                                Ok(missed) => {
                                    for missed in &missed {
                                        if let Some(frame) =
                                            change_frame(&thread_id, &mut status, missed, &shared.view)?
                                        {
                                            yield frame;
                                        }
                                    }
                                }
                                Err(status) => {
                                    tracing::warn!(
                                        thread_id = %thread_id,
                                        error = %status.message(),
                                        "thread view backfill failed; sending a fresh snapshot"
                                    );
                                    resnapshot = true;
                                }
                            }
                        }
                        if resnapshot {
                        } else if let Some(frame) =
                            change_frame(&thread_id, &mut status, &event, &shared.view)?
                        {
                            yield frame;
                        }
                    }
                    resnapshot
                }
            };
            if resnapshot {
                signals::stream(signals::RESNAPSHOT);
                live = shared.stores.event_notifier.subscribe(&thread_id);
                let view = fold_thread(shared.as_ref(), &thread_id).await?;
                status = view.status.clone();
                let snapshot = snapshot_frame(&thread_id, &view, limit)?;
                drop(view);
                yield snapshot;
            }
        }
    }
}
