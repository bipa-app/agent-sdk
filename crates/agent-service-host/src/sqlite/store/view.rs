//! The session view on `SQLite`: written in the event's transaction, read
//! with a lazy materialization. See `agent_server::journal::view_store`.

use agent_sdk_foundation::ThreadId;
use agent_server::journal::CommittedEvent;
use agent_server::journal::view::{IndexedItem, ThreadView, ThreadViewStatus, VIEW_SCHEMA};
use agent_server::journal::view_store::{ThreadViewSnapshot, ThreadViewStore, ViewsDisabled};
use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{FromRow, Sqlite, Transaction};
use time::OffsetDateTime;

use super::{
    CommittedEventRecord, SqliteDurableStore, i64_from_u64, json_from_value, json_to_value,
    thread_key, u64_from_i64,
};
use crate::tx_ctx::TxCtx;
use crate::view_apply::{PartialView, materialize};

#[derive(FromRow)]
struct ViewRow {
    view_schema: i64,
    next_index: i64,
    status_json: serde_json::Value,
}

#[derive(FromRow)]
struct ItemRow {
    item_key: String,
    item_index: i64,
    body_json: serde_json::Value,
}

impl ItemRow {
    fn into_item(self) -> Result<(String, IndexedItem)> {
        Ok((
            self.item_key,
            IndexedItem {
                index: u64_from_i64(self.item_index, "view item index")?,
                item: json_from_value(self.body_json, "view item body")?,
            },
        ))
    }
}

struct LoadedView {
    schema: i64,
    status: ThreadViewStatus,
    next_index: u64,
}

impl SqliteDurableStore {
    /// Fold `committed` into the thread's view, inside the transaction that
    /// inserted them. A thread with no view row is materialized here only
    /// when this batch starts it; otherwise its first read materializes it.
    ///
    /// A batch of streaming deltas only leaves the view as it is: the next
    /// write folds them as a gap, and a read folds them before it answers.
    /// The view's sequence may then lag the thread by deltas alone.
    pub(super) async fn apply_view_tx(
        ctx: &TxCtx<'_>,
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        committed: &[CommittedEvent],
        now: OffsetDateTime,
    ) -> Result<()> {
        let Some(first) = committed.first() else {
            return Ok(());
        };
        if !ctx.view.enabled || committed.iter().all(is_delta) {
            return Ok(());
        }
        agent_server::fail_point!("view.before_apply");
        let Some(loaded) = Self::view_row_tx(tx, thread_id).await? else {
            if first.sequence == 0 {
                let view = materialize(committed, ctx.view);
                Self::write_whole_view_tx(tx, thread_id, &view, now).await?;
            }
            return Ok(());
        };
        if loaded.schema != i64::from(VIEW_SCHEMA) {
            return Self::delete_view_tx(tx, thread_id).await;
        }
        let next = loaded.status.sequence.map_or(0, |folded| folded + 1);
        let gap = if next < first.sequence {
            Self::events_between_tx(tx, thread_id, loaded.status.sequence, Some(first.sequence))
                .await?
        } else {
            Vec::new()
        };
        let mut view = PartialView::new(loaded.status, loaded.next_index);
        for event in gap.iter().chain(committed) {
            Self::fold_into_tx(ctx, tx, thread_id, &mut view, event).await?;
        }
        Self::write_partial_view_tx(tx, thread_id, &view, now).await
    }

    async fn fold_into_tx(
        ctx: &TxCtx<'_>,
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        view: &mut PartialView,
        event: &CommittedEvent,
    ) -> Result<()> {
        let ops = view.fold(event, ctx.view);
        for key in view.unloaded_keys(&ops) {
            let item = Self::view_item_tx(tx, thread_id, &key).await?;
            view.loaded(key, item);
        }
        view.apply(event.sequence, &ops);
        Ok(())
    }

    async fn view_row_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<Option<LoadedView>> {
        let row = sqlx::query_as::<_, ViewRow>(
            "SELECT view_schema, next_index, status_json FROM agent_sdk_thread_views WHERE thread_id = ?1",
        )
        .bind(thread_key(thread_id))
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("read the view of {thread_id}"))?;
        let current = i64::from(VIEW_SCHEMA);
        row.map(|row| {
            let status = if row.view_schema == current {
                json_from_value(row.status_json, "view status")?
            } else {
                ThreadViewStatus::default()
            };
            Ok(LoadedView {
                schema: row.view_schema,
                status,
                next_index: u64_from_i64(row.next_index, "view next index")?,
            })
        })
        .transpose()
    }

    async fn view_item_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        key: &str,
    ) -> Result<Option<IndexedItem>> {
        let row = sqlx::query_as::<_, ItemRow>(
            "SELECT item_key, item_index, body_json FROM agent_sdk_thread_view_items WHERE thread_id = ?1 AND item_key = ?2",
        )
        .bind(thread_key(thread_id))
        .bind(key)
        .fetch_optional(&mut **tx)
        .await
        .with_context(|| format!("read view item {key} of {thread_id}"))?;
        Ok(row
            .map(ItemRow::into_item)
            .transpose()?
            .map(|(_, item)| item))
    }

    /// Committed events after `after` (all of them when `None`) and before
    /// `before` (no bound when `None`), in sequence order.
    async fn events_between_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        after: Option<u64>,
        before: Option<u64>,
    ) -> Result<Vec<CommittedEvent>> {
        let after = after
            .map(|a| i64_from_u64(a, "view sequence"))
            .transpose()?
            .unwrap_or(-1);
        let before = before
            .map(|b| i64_from_u64(b, "first new sequence"))
            .transpose()?
            .unwrap_or(i64::MAX);
        let records = sqlx::query_as::<_, CommittedEventRecord>(
            "SELECT event_id, thread_id, sequence, event_json, committed_at FROM agent_sdk_committed_events WHERE thread_id = ?1 AND sequence > ?2 AND sequence < ?3 ORDER BY sequence",
        )
        .bind(thread_key(thread_id))
        .bind(after)
        .bind(before)
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("read the events a view of {thread_id} misses"))?;
        records.into_iter().map(TryInto::try_into).collect()
    }

    async fn write_status_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        status: &ThreadViewStatus,
        next_index: u64,
        now: OffsetDateTime,
    ) -> Result<()> {
        let thread_id_key = thread_key(thread_id);
        let schema = i64::from(VIEW_SCHEMA);
        let sequence = status
            .sequence
            .map(|s| i64_from_u64(s, "view sequence"))
            .transpose()?;
        let next_index = i64_from_u64(next_index, "view next index")?;
        let status_json = json_to_value(status, "view status")?;
        sqlx::query!(
            r"INSERT INTO agent_sdk_thread_views (thread_id, view_schema, sequence, next_index, status_json, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
ON CONFLICT (thread_id) DO UPDATE SET view_schema = excluded.view_schema, sequence = excluded.sequence,
    next_index = excluded.next_index, status_json = excluded.status_json, updated_at = excluded.updated_at",
            thread_id_key,
            schema,
            sequence,
            next_index,
            status_json,
            now,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("write the view status of {thread_id}"))?;
        Ok(())
    }

    async fn write_item_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        key: &str,
        item: &IndexedItem,
        sequence: u64,
    ) -> Result<()> {
        let thread_id_key = thread_key(thread_id);
        let index = i64_from_u64(item.index, "view item index")?;
        let sequence = i64_from_u64(sequence, "view item sequence")?;
        let body_json = json_to_value(&item.item, "view item body")?;
        sqlx::query!(
            r"INSERT INTO agent_sdk_thread_view_items (thread_id, item_key, item_index, sequence, body_json)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT (thread_id, item_key) DO UPDATE SET item_index = excluded.item_index,
    sequence = excluded.sequence, body_json = excluded.body_json",
            thread_id_key,
            key,
            index,
            sequence,
            body_json,
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("write view item {key} of {thread_id}"))?;
        Ok(())
    }

    async fn write_partial_view_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        view: &PartialView,
        now: OffsetDateTime,
    ) -> Result<()> {
        for (key, item, sequence) in view.touched() {
            Self::write_item_tx(tx, thread_id, key, item, sequence).await?;
        }
        Self::write_status_tx(tx, thread_id, view.status(), view.next_index(), now).await
    }

    async fn write_whole_view_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        view: &ThreadView,
        now: OffsetDateTime,
    ) -> Result<()> {
        let sequence = view.status.sequence.unwrap_or_default();
        for (key, item) in &view.items {
            Self::write_item_tx(tx, thread_id, key, item, sequence).await?;
        }
        Self::write_status_tx(tx, thread_id, &view.status, view.next_index, now).await
    }

    pub(super) async fn delete_view_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<()> {
        let thread_id_key = thread_key(thread_id);
        sqlx::query!(
            "DELETE FROM agent_sdk_thread_view_items WHERE thread_id = ?1",
            thread_id_key
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("delete the view items of {thread_id}"))?;
        sqlx::query!(
            "DELETE FROM agent_sdk_thread_views WHERE thread_id = ?1",
            thread_id_key
        )
        .execute(&mut **tx)
        .await
        .with_context(|| format!("delete the view of {thread_id}"))?;
        Ok(())
    }

    async fn latest_sequence_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<Option<u64>> {
        let thread_id_key = thread_key(thread_id);
        let record = sqlx::query!(
            r#"SELECT MAX(sequence) AS "latest: i64" FROM agent_sdk_committed_events WHERE thread_id = ?1"#,
            thread_id_key,
        )
        .fetch_one(&mut **tx)
        .await
        .with_context(|| format!("read the latest event of {thread_id}"))?;
        record
            .latest
            .map(|latest| u64_from_i64(latest, "latest event sequence"))
            .transpose()
    }

    /// Bring the thread's view up to its latest event: materialize it when
    /// it is missing or at another schema, fold the events it misses when an
    /// older binary appended without it. `None` for a thread with no events.
    async fn current_view_tx(
        &self,
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        now: OffsetDateTime,
    ) -> Result<Option<()>> {
        let Some(latest) = Self::latest_sequence_tx(tx, thread_id).await? else {
            return Ok(None);
        };
        let loaded = Self::view_row_tx(tx, thread_id)
            .await?
            .filter(|loaded| loaded.schema == i64::from(VIEW_SCHEMA));
        let Some(loaded) = loaded else {
            Self::delete_view_tx(tx, thread_id).await?;
            let events = Self::events_between_tx(tx, thread_id, None, None).await?;
            let view = materialize(&events, &self.view);
            Self::write_whole_view_tx(tx, thread_id, &view, now).await?;
            return Ok(Some(()));
        };
        if loaded.status.sequence == Some(latest) {
            return Ok(Some(()));
        }
        let missing = Self::events_between_tx(tx, thread_id, loaded.status.sequence, None).await?;
        let ctx = self.tx_ctx();
        let mut view = PartialView::new(loaded.status, loaded.next_index);
        for event in &missing {
            Self::fold_into_tx(&ctx, tx, thread_id, &mut view, event).await?;
        }
        Self::write_partial_view_tx(tx, thread_id, &view, now).await?;
        Ok(Some(()))
    }

    async fn view_items_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
        before_index: Option<u64>,
        limit: usize,
    ) -> Result<(Vec<(String, IndexedItem)>, bool)> {
        let before = before_index
            .map(|b| i64_from_u64(b, "view page bound"))
            .transpose()?;
        let fetch = i64::try_from(limit.saturating_add(1)).context("view page size")?;
        let rows = sqlx::query_as::<_, ItemRow>(
            "SELECT item_key, item_index, body_json FROM agent_sdk_thread_view_items WHERE thread_id = ?1 AND (?2 IS NULL OR item_index < ?2) ORDER BY item_index DESC LIMIT ?3",
        )
        .bind(thread_key(thread_id))
        .bind(before)
        .bind(fetch)
        .fetch_all(&mut **tx)
        .await
        .with_context(|| format!("read the view items of {thread_id}"))?;
        let has_older = rows.len() > limit;
        let mut items = rows
            .into_iter()
            .take(limit)
            .map(ItemRow::into_item)
            .collect::<Result<Vec<_>>>()?;
        items.reverse();
        Ok((items, has_older))
    }

    async fn purge_fenced_tx(
        tx: &mut Transaction<'_, Sqlite>,
        thread_id: &ThreadId,
    ) -> Result<bool> {
        Ok(Self::get_thread_tx(tx, thread_id)
            .await?
            .is_some_and(|thread| thread.status.is_purge_fenced()))
    }
}

const fn is_delta(event: &CommittedEvent) -> bool {
    matches!(
        event.event,
        agent_sdk_foundation::events::AgentEvent::TextDelta { .. }
            | agent_sdk_foundation::events::AgentEvent::ThinkingDelta { .. }
    )
}

#[cfg(test)]
impl SqliteDurableStore {
    /// The stored view rows as they are, without the read path's catch-up.
    pub(crate) async fn stored_view_for_test(
        &self,
        thread_id: &ThreadId,
    ) -> Result<Option<ThreadView>> {
        let mut tx = self.pool.begin().await?;
        let Some(loaded) = Self::view_row_tx(&mut tx, thread_id).await? else {
            return Ok(None);
        };
        let (items, _) = Self::view_items_tx(&mut tx, thread_id, None, 1_000_000).await?;
        Ok(Some(ThreadView {
            status: loaded.status,
            items: items.into_iter().collect(),
            next_index: loaded.next_index,
        }))
    }
}

#[async_trait]
impl ThreadViewStore for SqliteDurableStore {
    async fn thread_view(
        &self,
        thread_id: &ThreadId,
        item_limit: usize,
    ) -> Result<Option<ThreadViewSnapshot>> {
        if !self.view.enabled {
            return Err(ViewsDisabled.into());
        }
        let mut tx = self.begin().await?;
        if Self::purge_fenced_tx(&mut tx, thread_id).await? {
            return Ok(None);
        }
        if self
            .current_view_tx(&mut tx, thread_id, OffsetDateTime::now_utc())
            .await?
            .is_none()
        {
            return Ok(None);
        }
        let loaded = Self::view_row_tx(&mut tx, thread_id)
            .await?
            .context("a current view has a row")?;
        let (items, has_older) = Self::view_items_tx(&mut tx, thread_id, None, item_limit).await?;
        tx.commit().await.context("commit the view read")?;
        Ok(Some(ThreadViewSnapshot {
            status: loaded.status,
            next_index: loaded.next_index,
            items,
            has_older,
        }))
    }

    async fn thread_view_items(
        &self,
        thread_id: &ThreadId,
        before_index: u64,
        limit: usize,
    ) -> Result<Option<Vec<(String, IndexedItem)>>> {
        if !self.view.enabled {
            return Err(ViewsDisabled.into());
        }
        let mut tx = self.begin().await?;
        if Self::purge_fenced_tx(&mut tx, thread_id).await? {
            return Ok(None);
        }
        if self
            .current_view_tx(&mut tx, thread_id, OffsetDateTime::now_utc())
            .await?
            .is_none()
        {
            return Ok(None);
        }
        let (items, _) = Self::view_items_tx(&mut tx, thread_id, Some(before_index), limit).await?;
        tx.commit().await.context("commit the view page read")?;
        Ok(Some(items))
    }
}
