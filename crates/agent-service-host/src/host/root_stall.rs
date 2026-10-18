//! The root-turn stall notice: a plain root turn whose activity beacon is
//! silent for [`RootStallNotice::after`], and again for four times that,
//! gets one `ToolProgress` notice on its own thread per threshold. The
//! notice only reports; the turn keeps running and the user decides.
//!
//! The root's own heartbeat writes it, under the root's lease, so it needs
//! no sweep and stops with the lease. Silence is read from the beacon,
//! which a connectivity wait bumps, so a turn waiting out an outage is
//! never reported. Writing the notice does not bump the beacon.

use agent_sdk_foundation::AgentEvent;
use agent_server::journal::task::AgentTask;
use time::OffsetDateTime;
use tracing::{info, warn};

use super::progress::{LeaseHolder, ProgressKind, Refusal, append_owned};
use crate::config::RootStallNotice;
use crate::stores::StoreRegistry;

/// The second notice of one silence fires at this multiple of the first.
const ESCALATION_MULTIPLE: u32 = 4;

/// The notice frame, field for field what bip's root-turn stall sweep
/// wrote, so its clients and `ViewConfig::stall_tool_name` read it as
/// before.
pub(super) fn stall_notice_event(
    tool_name: &str,
    silent: time::Duration,
    root_task_id: &str,
) -> AgentEvent {
    let silent_secs = silent.whole_seconds().max(0);
    let since_ms = u64::try_from(silent.whole_milliseconds().max(0)).unwrap_or(u64::MAX);
    AgentEvent::ToolProgress {
        id: format!("root-turn-stall-{root_task_id}"),
        name: tool_name.to_owned(),
        display_name: "Turn appears stalled".to_owned(),
        stage: "stalled".to_owned(),
        message: format!("no agent events for {silent_secs}s \u{2014} cancel or keep waiting"),
        data: Some(serde_json::json!({
            "since_ms": since_ms,
            "root_task_id": root_task_id,
        })),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Eligibility {
    /// The child-root lookup has not succeeded yet.
    Unresolved,
    /// A plain root turn: the notice applies.
    PlainRoot,
    /// A subagent's child root, whose silence its stall budget handles.
    SubagentChild,
}

/// The heartbeat's side of the notice, for one acquisition of a root turn.
pub struct RootStallLeg {
    notice: RootStallNotice,
    holder: LeaseHolder,
    thread: agent_sdk_foundation::ThreadId,
    eligibility: Eligibility,
    /// Acquisition is evidence of work: silence is measured from here at the
    /// latest.
    acquired_at: OffsetDateTime,
    /// The evidence the current silence is measured from.
    episode_from: OffsetDateTime,
    /// Notices written for the current silence: 0, 1 or 2.
    written: u8,
}

impl RootStallLeg {
    /// The leg for `task`'s acquisition, or `None` when no notice is
    /// configured or `task` is not a root turn.
    pub(super) fn new(
        notice: Option<RootStallNotice>,
        task: &AgentTask,
        holder: LeaseHolder,
        acquired_at: OffsetDateTime,
    ) -> Option<Self> {
        let notice = notice?;
        (task.kind.is_root() && task.is_root()).then(|| Self {
            notice,
            holder,
            thread: task.thread_id.clone(),
            eligibility: Eligibility::Unresolved,
            acquired_at,
            episode_from: acquired_at,
            written: 0,
        })
    }

    /// One tick, after the lease was renewed. `observed` is this tick's
    /// beacon reading.
    pub(super) async fn tick(
        &mut self,
        stores: &StoreRegistry,
        observed: Option<OffsetDateTime>,
        now: OffsetDateTime,
    ) {
        if self.eligibility == Eligibility::Unresolved {
            match stores
                .task_store
                .find_subagent_invocation_for_child_root(&self.holder.task)
                .await
            {
                Ok(None) => self.eligibility = Eligibility::PlainRoot,
                Ok(Some(_)) => self.eligibility = Eligibility::SubagentChild,
                Err(error) => {
                    warn!(
                        task_id = %self.holder.task,
                        error = %error,
                        "root stall notice: child-root lookup failed; retrying next tick",
                    );
                    return;
                }
            }
        }
        match self.eligibility {
            Eligibility::PlainRoot => {}
            Eligibility::Unresolved | Eligibility::SubagentChild => return,
        }

        let evidence = observed
            .filter(|at| *at > self.acquired_at)
            .unwrap_or(self.acquired_at);
        if evidence > self.episode_from {
            self.episode_from = evidence;
            self.written = 0;
        }
        let first = time::Duration::try_from(self.notice.after()).unwrap_or(time::Duration::MAX);
        let threshold = match self.written {
            0 => first,
            1 => first.saturating_mul(i32::try_from(ESCALATION_MULTIPLE).unwrap_or(i32::MAX)),
            _ => return,
        };
        let silent = now - self.episode_from;
        if silent < threshold {
            return;
        }
        let event = stall_notice_event(&self.notice.tool_name, silent, self.holder.task.as_str());
        match append_owned(
            stores,
            &self.holder,
            &self.thread,
            vec![event],
            ProgressKind::RootStallNotice,
            now,
        )
        .await
        {
            Ok(()) => {
                self.written += 1;
                warn!(
                    task_id = %self.holder.task,
                    thread_id = %self.thread,
                    silent_secs = silent.whole_seconds(),
                    threshold_secs = threshold.whole_seconds(),
                    "root turn shows no sign of work; appended a stall notice (the turn keeps \
                     running)",
                );
            }
            // The heartbeat sees the lost lease at its next renewal; a purged
            // thread will take no notice. Either way this episode is over.
            Err(Refusal::LeaseLost | Refusal::Purged | Refusal::SubjectMoved) => {
                info!(task_id = %self.holder.task, "root stall notice refused; not retrying");
                self.written = 2;
            }
            Err(Refusal::Store(error)) => {
                warn!(
                    task_id = %self.holder.task,
                    error = %error,
                    "root stall notice hit a store error; retrying next tick",
                );
            }
        }
    }
}
