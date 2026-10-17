//! The stall budget of a tool child: how a parent's wait on a tool ends.
//!
//! A parent parked in `WaitingOnChildren` resumes only when every child is
//! terminal. A tool child hung on a dead socket keeps its lease alive, so
//! without an end of its own the parent waits forever (bip #756). The child
//! therefore owns its end: when its tool shows no evidence of work for the
//! whole budget, it settles itself `Failed` with
//! [`TerminalReason::WatchdogStall`](agent_sdk_foundation::events::TerminalReason)
//! and the parent wakes through the ordinary child-terminal edge, with a
//! [`ToolStallReport`] in place of the result it never got.
//!
//! The budget measures silence, not age: the evidence is the task's
//! [`ActivityBeacon`](super::ActivityBeacon), which the tool bumps whenever
//! it emits or calls [`ToolEventCollector::note_progress`](super::ToolEventCollector::note_progress).
//! A tool that keeps reporting runs for as long as it needs.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use agent_sdk_foundation::events::AgentEvent;

/// How long a tool may go without evidence of work before its call fails as
/// stalled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StallBudget {
    /// The host's default budget.
    #[default]
    Inherit,
    /// This budget of silence.
    After(Duration),
    /// Never stalls. Only the tool's owner should declare this: a parent
    /// waiting on such a call has no end but the tool's own return.
    Unbounded,
}

impl StallBudget {
    /// The budget in effect, with `default` standing for [`Self::Inherit`],
    /// or `None` for no budget.
    #[must_use]
    pub const fn resolve(self, default: Option<Duration>) -> Option<Duration> {
        match self {
            Self::Inherit => default,
            Self::After(budget) => Some(budget),
            Self::Unbounded => None,
        }
    }
}

/// Why a tool call was stopped as stalled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StallTrip {
    pub budget: Duration,
    /// Silence observed when the call was stopped; at least `budget`.
    pub silent_for: Duration,
    /// The newest evidence of work, or `None` if the tool never showed any.
    pub last_progress_at: Option<OffsetDateTime>,
}

/// The one-shot signal that stops a stalled tool call.
///
/// The task's heartbeat trips it; the tool run races the tool against it.
/// Cloneable; every clone observes the same trip.
#[derive(Clone, Debug, Default)]
pub struct StallSignal {
    token: CancellationToken,
    trip: Arc<OnceLock<StallTrip>>,
}

impl StallSignal {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Trip the signal. If it had already tripped, the first trip stands.
    pub fn trip(&self, trip: StallTrip) {
        let _ = self.trip.set(trip);
        self.token.cancel();
    }

    #[must_use]
    pub fn tripped_with(&self) -> Option<StallTrip> {
        self.trip.get().copied()
    }

    /// Resolves once the signal trips.
    pub async fn tripped(&self) -> StallTrip {
        loop {
            if let Some(trip) = self.tripped_with() {
                return trip;
            }
            self.token.cancelled().await;
        }
    }
}

/// Most bytes of a stalled tool's own progress text the parent gets back.
pub const STALL_PARTIAL_OUTPUT_BYTES: usize = 4096;

/// What the parent's model gets in `ToolResult::data` for a stalled call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolStallReport {
    /// Always `"stalled"`, so a reader can tell this payload apart.
    pub outcome: String,
    pub stall_budget_ms: u64,
    pub silent_for_ms: u64,
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub last_progress_at: Option<OffsetDateTime>,
    /// The tail of the progress text the tool reported before it stalled.
    pub partial_output: String,
    /// Whether `partial_output` dropped earlier text.
    pub partial_output_truncated: bool,
}

impl ToolStallReport {
    /// The report for `trip`, keeping the tail of the text in `progress`.
    #[must_use]
    pub fn new(trip: StallTrip, progress: &[AgentEvent]) -> Self {
        let (partial_output, partial_output_truncated) = progress_tail(progress);
        Self {
            outcome: "stalled".to_owned(),
            stall_budget_ms: millis(trip.budget),
            silent_for_ms: millis(trip.silent_for),
            last_progress_at: trip.last_progress_at,
            partial_output,
            partial_output_truncated,
        }
    }

    /// The text the parent's model reads in place of the tool's output.
    #[must_use]
    pub fn message(&self, tool_name: &str) -> String {
        let mut message = format!(
            "tool '{tool_name}' stalled: no progress for {}s (stall budget {}s); it was stopped \
             and its effects are unknown.",
            self.silent_for_ms / 1000,
            self.stall_budget_ms / 1000,
        );
        if self.partial_output.is_empty() {
            message.push_str(" It reported no output.");
        } else {
            message.push_str(if self.partial_output_truncated {
                " The end of its output follows.\n"
            } else {
                " Its output follows.\n"
            });
            message.push_str(&self.partial_output);
        }
        message
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// The last [`STALL_PARTIAL_OUTPUT_BYTES`] of the messages in `progress`'s
/// `ToolProgress` events, one per line, cut on a character boundary.
fn progress_tail(progress: &[AgentEvent]) -> (String, bool) {
    let mut text = String::new();
    for event in progress {
        if let AgentEvent::ToolProgress { message, .. } = event
            && !message.is_empty()
        {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(message);
        }
    }
    if text.len() <= STALL_PARTIAL_OUTPUT_BYTES {
        return (text, false);
    }
    let mut start = text.len() - STALL_PARTIAL_OUTPUT_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    (text[start..].to_owned(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(message: &str) -> AgentEvent {
        AgentEvent::ToolProgress {
            id: "call-1".to_owned(),
            name: "bash".to_owned(),
            display_name: "Bash".to_owned(),
            stage: "running".to_owned(),
            message: message.to_owned(),
            data: None,
        }
    }

    fn trip() -> StallTrip {
        StallTrip {
            budget: Duration::from_mins(15),
            silent_for: Duration::from_secs(901),
            last_progress_at: None,
        }
    }

    #[test]
    fn the_budget_resolves_against_the_host_default() {
        let default = Some(Duration::from_mins(15));
        assert_eq!(StallBudget::Inherit.resolve(default), default);
        assert_eq!(StallBudget::Inherit.resolve(None), None);
        assert_eq!(
            StallBudget::After(Duration::from_secs(5)).resolve(default),
            Some(Duration::from_secs(5)),
        );
        assert_eq!(StallBudget::Unbounded.resolve(default), None);
    }

    #[tokio::test]
    async fn the_first_trip_stands() {
        let signal = StallSignal::new();
        assert!(signal.tripped_with().is_none());
        signal.trip(trip());
        let later = StallTrip {
            silent_for: Duration::from_secs(5000),
            ..trip()
        };
        signal.clone().trip(later);
        assert_eq!(signal.tripped().await, trip());
    }

    #[test]
    fn the_report_keeps_the_tail_of_the_progress_text() {
        let long = "x".repeat(STALL_PARTIAL_OUTPUT_BYTES);
        let report = ToolStallReport::new(trip(), &[progress("first"), progress(&long)]);
        assert!(report.partial_output_truncated);
        assert_eq!(report.partial_output, long);

        let short = ToolStallReport::new(trip(), &[progress("a"), progress(""), progress("b")]);
        assert!(!short.partial_output_truncated);
        assert_eq!(short.partial_output, "a\nb");
        assert!(short.message("bash").ends_with("Its output follows.\na\nb"));
    }

    #[test]
    fn the_tail_is_cut_on_a_character_boundary() {
        let text = format!("é{}", "y".repeat(STALL_PARTIAL_OUTPUT_BYTES - 1));
        let report = ToolStallReport::new(trip(), &[progress(&text)]);
        assert!(report.partial_output_truncated);
        assert_eq!(
            report.partial_output,
            "y".repeat(STALL_PARTIAL_OUTPUT_BYTES - 1)
        );
    }

    #[test]
    fn a_silent_tool_says_so() {
        let report = ToolStallReport::new(trip(), &[]);
        assert_eq!(report.outcome, "stalled");
        assert_eq!(report.stall_budget_ms, 900_000);
        assert!(report.message("mcp_call").contains("reported no output"));
    }
}
