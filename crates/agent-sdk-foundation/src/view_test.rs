use std::time::Duration;

use proptest::prelude::*;
use time::OffsetDateTime;

use super::*;
use crate::ThreadId;
use crate::events::AgentEvent;
use crate::llm::ContentBlock;
use crate::types::{BudgetLimitKind, QuestionPayload, TokenUsage, ToolResult, ToolTier};

const STALL: &str = "__stall_notice";

fn config() -> ViewConfig {
    ViewConfig {
        enabled: true,
        stall_tool_name: Some(STALL.to_owned()),
        visible_document_kinds: Vec::new(),
    }
}

fn thread() -> ThreadId {
    ThreadId::from_string("thread-1")
}

fn start(task: &str) -> AgentEvent {
    AgentEvent::start(thread(), 1).with_emitter_task_id(task)
}

fn delta(message_id: &str, text: &str) -> AgentEvent {
    AgentEvent::TextDelta {
        message_id: message_id.to_owned(),
        delta: text.to_owned(),
        emitter_task_id: None,
    }
}

fn delta_from(message_id: &str, text: &str, task: &str) -> AgentEvent {
    AgentEvent::TextDelta {
        message_id: message_id.to_owned(),
        delta: text.to_owned(),
        emitter_task_id: Some(task.to_owned()),
    }
}

fn text(message_id: &str, body: &str) -> AgentEvent {
    AgentEvent::Text {
        message_id: message_id.to_owned(),
        text: body.to_owned(),
        emitter_task_id: None,
    }
}

fn tool_start(id: &str) -> AgentEvent {
    AgentEvent::ToolCallStart {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        input: serde_json::json!({"command": "ls"}),
        tier: ToolTier::Observe,
    }
}

fn tool_end(id: &str, success: bool) -> AgentEvent {
    AgentEvent::ToolCallEnd {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        result: if success {
            ToolResult::success("ok")
        } else {
            ToolResult::error("no")
        },
    }
}

fn confirm(id: &str) -> AgentEvent {
    AgentEvent::ToolRequiresConfirmation {
        id: id.to_owned(),
        name: "bash".to_owned(),
        display_name: "Bash".to_owned(),
        input: serde_json::json!({"command": "git push"}),
        description: "push".to_owned(),
    }
}

fn progress(id: &str, name: &str) -> AgentEvent {
    AgentEvent::ToolProgress {
        id: id.to_owned(),
        name: name.to_owned(),
        display_name: name.to_owned(),
        stage: "running".to_owned(),
        message: "still going".to_owned(),
        data: None,
    }
}

fn question(tool_call_id: &str) -> AgentEvent {
    AgentEvent::QuestionAsked {
        task_id: "root-a".to_owned(),
        questions: vec![QuestionPayload {
            tool_call_id: tool_call_id.to_owned(),
            question: "which?".to_owned(),
            header: None,
            options: Vec::new(),
            multi_select: false,
        }],
    }
}

fn done() -> AgentEvent {
    AgentEvent::Done {
        thread_id: thread(),
        total_turns: 2,
        total_usage: usage(10),
        duration: Duration::from_millis(5),
        estimated_cost_usd: None,
        emitter_task_id: None,
    }
}

fn cancelled() -> AgentEvent {
    AgentEvent::Cancelled {
        turn: 1,
        usage: usage(3),
        reason: Some(TerminalReason::UserCancel),
        emitter_task_id: None,
    }
}

fn budget() -> AgentEvent {
    AgentEvent::BudgetExceeded {
        thread_id: thread(),
        total_turns: 1,
        total_usage: usage(4),
        duration: Duration::from_millis(5),
        estimated_cost_usd: None,
        limit: BudgetLimitKind::TotalTokens,
        emitter_task_id: None,
    }
}

fn error(recoverable: bool, reason: Option<TerminalReason>) -> AgentEvent {
    AgentEvent::Error {
        message: "boom".to_owned(),
        recoverable,
        reason,
        emitter_task_id: None,
    }
}

fn retry_start(max_attempts: u32) -> AgentEvent {
    AgentEvent::AutoRetryStart {
        attempt: 1,
        max_attempts,
        delay_ms: 100,
        error_message: "overloaded".to_owned(),
    }
}

const fn usage(input_tokens: u32) -> TokenUsage {
    TokenUsage {
        input_tokens,
        output_tokens: 1,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 0,
    }
}

fn subagent_progress(id: &str, completed: bool) -> AgentEvent {
    AgentEvent::SubagentProgress {
        subagent_id: id.to_owned(),
        subagent_name: "explore".to_owned(),
        nickname: None,
        child_thread_id: None,
        child_root_task_id: None,
        subagent_task_id: None,
        max_turns: None,
        current_turn: Some(1),
        model: None,
        tool_name: "read".to_owned(),
        tool_context: "src".to_owned(),
        completed,
        success: completed,
        tool_count: 1,
        total_tokens: 10,
        input_tokens: 5,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
        liveness: None,
    }
}

fn every_variant() -> Vec<AgentEvent> {
    vec![
        AgentEvent::thread_created(thread(), Some(ThreadId::from_string("src")), Some(2)),
        start("root-a"),
        AgentEvent::user_input(
            thread(),
            vec![ContentBlock::Text {
                text: "hi".to_owned(),
            }],
        ),
        AgentEvent::Thinking {
            message_id: "th".to_owned(),
            text: "hmm".to_owned(),
            emitter_task_id: None,
        },
        AgentEvent::ThinkingDelta {
            message_id: "th2".to_owned(),
            delta: "h".to_owned(),
            emitter_task_id: None,
        },
        delta("m1", "he"),
        text("m1", "hello"),
        tool_start("t1"),
        progress("t1", "bash"),
        confirm("t1"),
        tool_end("t1", true),
        question("q1"),
        AgentEvent::TurnComplete {
            turn: 1,
            usage: usage(7),
            emitter_task_id: None,
        },
        retry_start(4),
        AgentEvent::AutoRetryEnd {
            attempt: 1,
            success: true,
            final_error: None,
        },
        AgentEvent::AccountRotation {
            provider: "anthropic".to_owned(),
            from_account: "a".to_owned(),
            to_account: "b".to_owned(),
            reason: AccountRotationReason::RateLimited,
            retry_after_seconds: None,
        },
        AgentEvent::AccountPoolExhausted {
            provider: "anthropic".to_owned(),
            account_count: 2,
            reason: AccountRotationReason::RateLimited,
            retry_after_seconds: Some(30),
        },
        AgentEvent::Refusal {
            message_id: "r1".to_owned(),
            text: Some("no".to_owned()),
        },
        AgentEvent::ContextCompacted {
            original_count: 10,
            new_count: 3,
            original_tokens: 1000,
            new_tokens: 200,
            trigger: None,
            engine: None,
            occupancy: None,
            threshold: None,
        },
        subagent_progress("t9", false),
        error(true, None),
        done(),
        budget(),
        cancelled(),
    ]
}

fn view_of(events: &[AgentEvent]) -> ThreadView {
    let at = OffsetDateTime::UNIX_EPOCH;
    ThreadView::from_events(
        events
            .iter()
            .enumerate()
            .map(|(i, event)| (i as u64, at, event)),
        &config(),
    )
}

fn item<'a>(view: &'a ThreadView, key: &str) -> Result<&'a ViewItem, String> {
    view.items
        .get(key)
        .map(|indexed| &indexed.item)
        .ok_or_else(|| format!("no item {key}"))
}

fn tool_state(view: &ThreadView, key: &str) -> Result<ToolState, String> {
    match item(view, key)? {
        ViewItem::ToolCall(tool) => Ok(tool.state),
        other => Err(format!("{key} is not a tool call: {other:?}")),
    }
}

#[test]
fn every_variant_advances_the_sequence_and_ends_with_the_status() -> Result<(), String> {
    let mut status = ThreadViewStatus::default();
    for (i, event) in every_variant().iter().enumerate() {
        let sequence = i as u64;
        let (next, ops) = fold(
            &status,
            sequence,
            OffsetDateTime::UNIX_EPOCH,
            event,
            &config(),
        );
        assert_eq!(next.sequence, Some(sequence));
        assert_eq!(next.schema, VIEW_SCHEMA);
        match ops.last() {
            Some(ViewOp::SetStatus { status: last }) => assert_eq!(**last, next),
            other => return Err(format!("last op of {event:?} is {other:?}")),
        }
        status = next;
    }
    Ok(())
}

#[test]
fn deltas_then_the_final_text_make_one_closed_item() -> Result<(), String> {
    let view = view_of(&[
        start("root-a"),
        delta("m1", "hel"),
        delta("m1", "lo"),
        text("m1", "hello!"),
    ]);
    assert_eq!(view.items.len(), 1);
    assert_eq!(
        *item(&view, "msg:m1")?,
        ViewItem::Text {
            text_kind: TextKind::Text,
            message_id: "m1".to_owned(),
            text: "hello!".to_owned(),
            open: false,
            interrupted: false,
        }
    );
    assert_eq!(view.status.open_messages.len(), 0);
    Ok(())
}

#[test]
fn an_unstreamed_final_text_is_a_closed_item() -> Result<(), String> {
    let view = view_of(&[start("root-a"), text("m1", "whole")]);
    assert!(matches!(
        item(&view, "msg:m1")?,
        ViewItem::Text {
            open: false,
            interrupted: false,
            ..
        }
    ));
    Ok(())
}

#[test]
fn a_delta_after_the_final_text_or_the_turn_end_is_dropped() -> Result<(), String> {
    let view = view_of(&[
        start("root-a"),
        delta("m1", "a"),
        text("m1", "ab"),
        delta("m1", "zzz"),
        done(),
        delta("m2", "late"),
    ]);
    assert!(matches!(item(&view, "msg:m1")?, ViewItem::Text { text, .. } if text == "ab"));
    assert!(!view.items.contains_key("msg:m2"));
    Ok(())
}

#[test]
fn a_delta_from_a_superseded_task_is_dropped() {
    let view = view_of(&[start("root-b"), delta_from("m1", "stale", "root-a")]);
    assert_eq!(view.items.len(), 0);
    let view = view_of(&[start("root-b"), delta_from("m1", "live", "root-b")]);
    assert_eq!(view.status.open_messages, vec!["msg:m1".to_owned()]);
}

#[test]
fn a_fresh_start_closes_open_items_as_interrupted() -> Result<(), String> {
    let view = view_of(&[
        start("root-a"),
        delta("m1", "partial"),
        tool_start("t1"),
        start("root-a"),
    ]);
    assert!(matches!(
        item(&view, "msg:m1")?,
        ViewItem::Text { text, open: false, interrupted: true, .. } if text == "partial"
    ));
    assert_eq!(tool_state(&view, "tool:t1")?, ToolState::Interrupted);
    assert_eq!(view.status.open_messages.len(), 0);
    assert_eq!(view.status.open_tools.len(), 0);
    assert!(matches!(view.status.turn, TurnState::Running { .. }));
    Ok(())
}

#[test]
fn a_tool_end_after_an_interrupt_still_records_the_result() -> Result<(), String> {
    let view = view_of(&[
        start("root-a"),
        tool_start("t1"),
        start("root-a"),
        tool_end("t1", true),
    ]);
    assert_eq!(tool_state(&view, "tool:t1")?, ToolState::Succeeded);
    assert_eq!(view.items.len(), 1);
    Ok(())
}

#[test]
fn a_confirmation_on_a_started_tool_is_pending_until_its_end() -> Result<(), String> {
    let view = view_of(&[start("root-a"), tool_start("t1"), confirm("t1")]);
    assert_eq!(
        tool_state(&view, "tool:t1")?,
        ToolState::AwaitingConfirmation
    );
    assert_eq!(
        view.status.pending_confirmations,
        vec![PendingConfirmation {
            key: "tool:t1".to_owned(),
            relayed: false,
        }]
    );
    let view = view_of(&[
        start("root-a"),
        tool_start("t1"),
        confirm("t1"),
        tool_end("t1", false),
    ]);
    assert_eq!(tool_state(&view, "tool:t1")?, ToolState::Failed);
    assert_eq!(view.status.pending_confirmations.len(), 0);
    assert_eq!(view.items.len(), 1);
    Ok(())
}

#[test]
fn a_relayed_confirmation_opens_lazily_and_closes_on_its_end() -> Result<(), String> {
    let open = view_of(&[confirm("relay-1"), start("root-a"), done()]);
    assert!(matches!(
        item(&open, "tool:relay-1")?,
        ViewItem::ToolCall(tool) if tool.relayed && tool.state == ToolState::AwaitingConfirmation
    ));
    assert_eq!(
        open.status.pending_confirmations,
        vec![PendingConfirmation {
            key: "tool:relay-1".to_owned(),
            relayed: true,
        }]
    );
    let closed = view_of(&[confirm("relay-1"), tool_end("relay-1", true)]);
    assert_eq!(closed.status.pending_confirmations.len(), 0);
    assert_eq!(tool_state(&closed, "tool:relay-1")?, ToolState::Succeeded);
    assert_eq!(closed.items.len(), 1);
    Ok(())
}

#[test]
fn a_question_is_pending_until_its_tool_ends_or_the_turn_resumes() -> Result<(), String> {
    let asked = view_of(&[start("root-a"), tool_start("q1"), question("q1")]);
    assert_eq!(
        asked.status.pending_questions,
        vec!["question:q1".to_owned()]
    );
    let answered = view_of(&[
        start("root-a"),
        tool_start("q1"),
        question("q1"),
        tool_end("q1", true),
    ]);
    assert_eq!(answered.status.pending_questions.len(), 0);
    assert!(matches!(
        item(&answered, "question:q1")?,
        ViewItem::Question { resolved: true, .. }
    ));
    let resumed = view_of(&[start("root-a"), question("q1"), start("root-a")]);
    assert_eq!(resumed.status.pending_questions.len(), 0);
    Ok(())
}

#[test]
fn terminal_shapes() {
    let cases: Vec<(&str, Vec<AgentEvent>, bool)> = vec![
        ("done", vec![start("root-a"), done()], true),
        ("cancelled", vec![start("root-a"), cancelled()], true),
        ("budget", vec![start("root-a"), budget()], true),
        (
            "error with a terminal reason",
            vec![
                start("root-a"),
                error(false, Some(TerminalReason::InternalError)),
            ],
            true,
        ),
        (
            "recoverable error",
            vec![start("root-a"), error(true, None)],
            false,
        ),
        (
            "non-fatal notice",
            vec![start("root-a"), error(false, None)],
            false,
        ),
        (
            "recoverable error with a terminal reason",
            vec![
                start("root-a"),
                error(true, Some(TerminalReason::InternalError)),
            ],
            true,
        ),
        (
            "refusal before done",
            vec![
                start("root-a"),
                AgentEvent::Refusal {
                    message_id: "r".to_owned(),
                    text: None,
                },
            ],
            false,
        ),
        (
            "relayed confirmation open",
            vec![start("root-a"), confirm("relay-1")],
            false,
        ),
        (
            "relayed confirmation closed",
            vec![
                start("root-a"),
                confirm("relay-1"),
                tool_end("relay-1", true),
            ],
            false,
        ),
    ];
    for (name, events, idle) in cases {
        let view = view_of(&events);
        assert_eq!(view.status.turn == TurnState::Idle, idle, "{name}");
        assert_eq!(view.status.last_outcome.is_some(), idle, "{name}");
        if idle {
            assert!(view.status.open_messages.is_empty(), "{name}");
            assert!(view.status.open_tools.is_empty(), "{name}");
            assert!(view.status.pending_questions.is_empty(), "{name}");
        }
    }
}

#[test]
fn a_cancel_mid_stream_closes_the_message_as_interrupted() -> Result<(), String> {
    let view = view_of(&[start("root-a"), delta("m1", "par"), cancelled()]);
    assert!(matches!(
        item(&view, "msg:m1")?,
        ViewItem::Text {
            open: false,
            interrupted: true,
            ..
        }
    ));
    assert!(matches!(
        view.status.last_outcome,
        Some(TurnOutcome::Cancelled {
            reason: Some(TerminalReason::UserCancel)
        })
    ));
    Ok(())
}

#[test]
fn the_stall_notice_sets_the_flag_until_the_next_event() {
    let stalled = view_of(&[start("root-a"), progress("stall-1", STALL)]);
    assert!(stalled.status.stalled);
    assert_eq!(stalled.items.len(), 0);
    let cleared = view_of(&[
        start("root-a"),
        progress("stall-1", STALL),
        delta("m1", "x"),
    ]);
    assert!(!cleared.status.stalled);
    let at = OffsetDateTime::UNIX_EPOCH;
    let no_stall_name = ThreadView::from_events(
        [
            (1, at, &start("root-a")),
            (2, at, &progress("stall-1", STALL)),
        ],
        &ViewConfig::default(),
    );
    assert!(!no_stall_name.status.stalled);
}

#[test]
fn a_connectivity_wait_is_an_open_retry_until_its_end() {
    let waiting = view_of(&[start("root-a"), retry_start(u32::MAX)]);
    assert!(
        waiting
            .status
            .retry
            .as_ref()
            .is_some_and(RetryState::is_connectivity_wait)
    );
    let ended = view_of(&[
        start("root-a"),
        retry_start(u32::MAX),
        AgentEvent::AutoRetryEnd {
            attempt: 1,
            success: true,
            final_error: None,
        },
    ]);
    assert!(ended.status.retry.is_none());
}

#[test]
fn subagent_progress_patches_its_spawning_tool_call() -> Result<(), String> {
    let view = view_of(&[
        start("root-a"),
        tool_start("t1"),
        subagent_progress("t1", false),
    ]);
    assert_eq!(view.items.len(), 1);
    assert!(matches!(
        item(&view, "tool:t1")?,
        ViewItem::ToolCall(tool)
            if tool.subagent.as_ref().is_some_and(|s| s.current_turn == Some(1))
    ));
    let orphan = view_of(&[subagent_progress("t2", true), subagent_progress("t2", true)]);
    assert_eq!(orphan.items.len(), 1);
    assert_eq!(tool_state(&orphan, "tool:t2")?, ToolState::Succeeded);
    Ok(())
}

#[test]
fn a_fork_records_its_source() {
    let view = view_of(&[AgentEvent::thread_created(
        thread(),
        Some(ThreadId::from_string("src")),
        Some(3),
    )]);
    assert_eq!(
        view.status.forked_from,
        Some(ForkOrigin {
            source_thread_id: ThreadId::from_string("src"),
            after_committed_turns: Some(3),
        })
    );
}

#[derive(Clone, Debug)]
enum Action {
    Start(u8),
    Delta(u8, Option<u8>),
    Text(u8),
    ToolStart(u8),
    ToolEnd(u8, bool),
    Confirm(u8),
    Progress(u8),
    Stall,
    Question(u8),
    Done,
    Cancel,
    Fail,
    Notice,
    RetryStart,
    RetryEnd,
    Subagent(u8, bool),
    Refusal(u8),
    UserInput,
    Compacted,
}

fn action() -> impl Strategy<Value = Action> {
    prop_oneof![
        (0..2u8).prop_map(Action::Start),
        (0..4u8, proptest::option::of(0..2u8)).prop_map(|(m, t)| Action::Delta(m, t)),
        (0..4u8).prop_map(Action::Text),
        (0..4u8).prop_map(Action::ToolStart),
        (0..6u8, any::<bool>()).prop_map(|(t, ok)| Action::ToolEnd(t, ok)),
        (0..6u8).prop_map(Action::Confirm),
        (0..4u8).prop_map(Action::Progress),
        Just(Action::Stall),
        (0..4u8).prop_map(Action::Question),
        Just(Action::Done),
        Just(Action::Cancel),
        Just(Action::Fail),
        Just(Action::Notice),
        Just(Action::RetryStart),
        Just(Action::RetryEnd),
        (0..6u8, any::<bool>()).prop_map(|(t, c)| Action::Subagent(t, c)),
        (0..2u8).prop_map(Action::Refusal),
        Just(Action::UserInput),
        Just(Action::Compacted),
    ]
}

fn to_event(action: &Action) -> AgentEvent {
    match action {
        Action::Start(task) => start(&format!("root-{task}")),
        Action::Delta(m, None) => delta(&format!("m{m}"), "x"),
        Action::Delta(m, Some(task)) => delta_from(&format!("m{m}"), "x", &format!("root-{task}")),
        Action::Text(m) => text(&format!("m{m}"), "final"),
        Action::ToolStart(t) => tool_start(&format!("t{t}")),
        Action::ToolEnd(t, ok) => tool_end(&format!("t{t}"), *ok),
        Action::Confirm(t) => confirm(&format!("t{t}")),
        Action::Progress(t) => progress(&format!("t{t}"), "bash"),
        Action::Stall => progress("stall", STALL),
        Action::Question(t) => question(&format!("t{t}")),
        Action::Done => done(),
        Action::Cancel => cancelled(),
        Action::Fail => error(false, Some(TerminalReason::InternalError)),
        Action::Notice => error(true, None),
        Action::RetryStart => retry_start(u32::MAX),
        Action::RetryEnd => AgentEvent::AutoRetryEnd {
            attempt: 1,
            success: false,
            final_error: Some("gone".to_owned()),
        },
        Action::Subagent(t, completed) => subagent_progress(&format!("t{t}"), *completed),
        Action::Refusal(m) => AgentEvent::Refusal {
            message_id: format!("m{m}"),
            text: None,
        },
        Action::UserInput => AgentEvent::user_input(
            thread(),
            vec![ContentBlock::Text {
                text: "go".to_owned(),
            }],
        ),
        Action::Compacted => AgentEvent::ContextCompacted {
            original_count: 4,
            new_count: 1,
            original_tokens: 40,
            new_tokens: 10,
            trigger: None,
            engine: None,
            occupancy: None,
            threshold: None,
        },
    }
}

fn check_invariants(view: &ThreadView) -> Result<(), TestCaseError> {
    let mut indexes: Vec<u64> = view.items.values().map(|indexed| indexed.index).collect();
    indexes.sort_unstable();
    let expected: Vec<u64> = (0..view.items.len() as u64).collect();
    prop_assert_eq!(indexes, expected, "item indexes are 0..len with no gap");
    for key in &view.status.open_messages {
        prop_assert!(
            matches!(
                view.items.get(key).map(|i| &i.item),
                Some(ViewItem::Text { open: true, .. })
            ),
            "open message {} is an open text item",
            key
        );
    }
    for key in &view.status.open_tools {
        prop_assert!(
            matches!(
                view.items.get(key).map(|i| &i.item),
                Some(ViewItem::ToolCall(_))
            ),
            "open tool {} is a tool item",
            key
        );
    }
    for pending in &view.status.pending_confirmations {
        prop_assert!(view.items.contains_key(&pending.key));
    }
    for key in &view.status.pending_questions {
        prop_assert!(view.items.contains_key(key));
    }
    for (key, indexed) in &view.items {
        if let ViewItem::Text { open: true, .. } = indexed.item {
            prop_assert!(
                view.status.open_messages.contains(key),
                "{} is open but untracked",
                key
            );
        }
    }
    prop_assert!(view.status.recently_closed.len() <= RECENTLY_CLOSED_LIMIT);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn a_snapshot_plus_the_wire_ops_after_it_equals_the_full_fold(
        actions in proptest::collection::vec(action(), 0..80),
        split in 0usize..80,
    ) {
        let events: Vec<AgentEvent> = actions.iter().map(to_event).collect();
        let split = split.min(events.len());
        let config = config();
        let at = OffsetDateTime::UNIX_EPOCH;

        let full = view_of(&events);
        check_invariants(&full)?;

        let prefix = view_of(&events[..split]);
        let snapshot_json = serde_json::to_string(&prefix).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let mut client: ThreadView =
            serde_json::from_str(&snapshot_json).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let mut store = prefix;
        for (offset, event) in events[split..].iter().enumerate() {
            let sequence = (split + offset) as u64;
            let was_running = store.status.turn != TurnState::Idle;
            let ops = store.fold(sequence, at, event, &config);
            let wire = serde_json::to_string(&ops).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ops: Vec<ViewOp> =
                serde_json::from_str(&wire).map_err(|e| TestCaseError::fail(e.to_string()))?;
            for op in &ops {
                prop_assert!(client.apply_at(op, Some(at)), "op {:?} addresses a missing item", op);
            }
            let ends_turn = match actions[split + offset] {
                Action::Done | Action::Cancel => true,
                Action::Fail => was_running,
                _ => false,
            };
            if ends_turn {
                prop_assert_eq!(&store.status.turn, &TurnState::Idle);
                prop_assert_eq!(store.status.open_messages.len(), 0);
                prop_assert_eq!(store.status.open_tools.len(), 0);
                prop_assert_eq!(store.status.pending_questions.len(), 0);
                prop_assert!(store.status.pending_confirmations.iter().all(|p| p.relayed));
            }
        }
        prop_assert_eq!(&client, &full);
        prop_assert_eq!(&store, &full);
    }

    #[test]
    fn a_client_missing_its_oldest_items_keeps_the_full_folds_order(
        actions in proptest::collection::vec(action(), 0..80),
        split in 0usize..80,
        dropped in 0usize..10,
    ) {
        let events: Vec<AgentEvent> = actions.iter().map(to_event).collect();
        let split = split.min(events.len());
        let config = config();
        let at = OffsetDateTime::UNIX_EPOCH;

        let full = view_of(&events);
        let prefix = view_of(&events[..split]);
        let mut oldest: Vec<(u64, String)> =
            prefix.items.iter().map(|(key, indexed)| (indexed.index, key.clone())).collect();
        oldest.sort_unstable();
        let mut client = prefix.clone();
        let mut removed = Vec::new();
        for (_, key) in oldest.into_iter().take(dropped) {
            client.items.remove(&key);
            removed.push(key);
        }
        let mut store = prefix;
        for (offset, event) in events[split..].iter().enumerate() {
            let sequence = (split + offset) as u64;
            for op in store.fold(sequence, at, event, &config) {
                client.apply(&op);
            }
        }

        let mut indexes: Vec<u64> = client.items.values().map(|indexed| indexed.index).collect();
        indexes.sort_unstable();
        let unique = indexes.len();
        indexes.dedup();
        prop_assert_eq!(indexes.len(), unique, "a truncated view gave two items one index");
        prop_assert!(client.next_index >= full.next_index);
        prop_assert_eq!(&client.status, &full.status);
        let mut kept: Vec<(u64, u64)> = Vec::new();
        for (key, indexed) in &client.items {
            if removed.contains(key) {
                continue;
            }
            let Some(in_full) = full.items.get(key) else {
                return Err(TestCaseError::fail(format!("{key} is not in the full fold")));
            };
            prop_assert_eq!(&indexed.item, &in_full.item, "item {} differs", key);
            kept.push((indexed.index, in_full.index));
        }
        kept.sort_unstable();
        prop_assert!(
            kept.windows(2).all(|pair| pair[0].1 < pair[1].1),
            "the truncated view orders its items differently from the full fold"
        );
    }

    #[test]
    fn folding_the_events_again_changes_nothing(
        actions in proptest::collection::vec(action(), 1..60),
        replay_from in 0usize..60,
    ) {
        let events: Vec<AgentEvent> = actions.iter().map(to_event).collect();
        let replay_from = replay_from.min(events.len() - 1);
        let full = view_of(&events);
        let mut again = full.clone();
        for (offset, event) in events[replay_from..].iter().enumerate() {
            let sequence = (replay_from + offset) as u64;
            let ops = again.fold(sequence, OffsetDateTime::UNIX_EPOCH, event, &config());
            prop_assert_eq!(ops.len(), 0);
        }
        prop_assert_eq!(&again, &full);
    }

    #[test]
    fn fold_is_deterministic(actions in proptest::collection::vec(action(), 0..60)) {
        let events: Vec<AgentEvent> = actions.iter().map(to_event).collect();
        prop_assert_eq!(view_of(&events), view_of(&events));
    }
}

#[test]
fn a_view_without_its_oldest_item_gives_a_new_item_the_next_index() {
    let mut view = view_of(&[
        start("root-a"),
        text("m1", "a"),
        text("m2", "b"),
        text("m3", "c"),
    ]);
    view.items.remove("msg:m1");
    view.fold(4, OffsetDateTime::UNIX_EPOCH, &text("m4", "d"), &config());
    let index = |key: &str| view.items.get(key).map(|indexed| indexed.index);
    assert_eq!(index("msg:m3"), Some(2));
    assert_eq!(index("msg:m4"), Some(3));
    assert_eq!(view.next_index, 4);
}

#[test]
fn an_event_at_or_below_the_folded_sequence_is_a_no_op() {
    let at = OffsetDateTime::UNIX_EPOCH;
    let mut view = view_of(&[start("root-a"), delta("m1", "ab"), done()]);
    let before = view.clone();
    assert_eq!(view.fold(1, at, &delta("m1", "ab"), &config()).len(), 0);
    assert_eq!(view.fold(0, at, &start("root-a"), &config()).len(), 0);
    assert_eq!(view.fold(2, at, &done(), &config()).len(), 0);
    assert_eq!(view, before);
    assert_eq!(view.status.sequence, Some(2));
    assert_eq!(view.status.turn, TurnState::Idle);
}

#[test]
fn a_gap_in_the_sequence_is_folded() {
    let mut view = view_of(&[start("root-a")]);
    let ops = view.fold(9, OffsetDateTime::UNIX_EPOCH, &done(), &config());
    assert!(matches!(ops.last(), Some(ViewOp::SetStatus { .. })));
    assert_eq!(view.status.sequence, Some(9));
    assert_eq!(view.status.turn, TurnState::Idle);
}

#[test]
fn a_terminal_error_after_the_turn_ended_keeps_the_outcome() {
    let view = view_of(&[
        start("root-a"),
        done(),
        error(false, Some(TerminalReason::InternalError)),
    ]);
    assert!(matches!(
        view.status.last_outcome,
        Some(TurnOutcome::Done { .. })
    ));
    assert_eq!(
        view.items
            .values()
            .filter(|indexed| matches!(indexed.item, ViewItem::TurnEnd { .. }))
            .count(),
        1
    );
    assert!(view.items.contains_key("error:2"));
}

#[test]
fn the_closed_window_keeps_the_newest_keys_and_drops_late_deltas_inside_it() {
    let mut events = vec![start("root-a"), text("m1", "final")];
    for i in 0..RECENTLY_CLOSED_LIMIT - 1 {
        events.push(tool_start(&format!("w{i}")));
        events.push(tool_end(&format!("w{i}"), true));
    }
    events.push(delta("m1", "LATE"));
    let inside = view_of(&events);
    assert!(matches!(
        inside.items.get("msg:m1").map(|indexed| &indexed.item),
        Some(ViewItem::Text { text, open: false, .. }) if text == "final"
    ));
    assert_eq!(inside.status.recently_closed.len(), RECENTLY_CLOSED_LIMIT);
    assert_eq!(
        inside.status.recently_closed.first().map(String::as_str),
        Some("msg:m1")
    );

    events.pop();
    events.push(tool_start("evicts-m1"));
    events.push(tool_end("evicts-m1", true));
    let evicted = view_of(&events);
    assert_eq!(evicted.status.recently_closed.len(), RECENTLY_CLOSED_LIMIT);
    assert!(
        !evicted
            .status
            .recently_closed
            .contains(&"msg:m1".to_owned())
    );
    assert_eq!(
        evicted.status.recently_closed.last().map(String::as_str),
        Some("tool:evicts-m1")
    );
}

#[test]
fn questions_and_refusals_stay_out_of_the_closed_window() {
    let view = view_of(&[
        start("root-a"),
        question("q1"),
        AgentEvent::Refusal {
            message_id: "r1".to_owned(),
            text: None,
        },
    ]);
    assert_eq!(view.status.recently_closed.len(), 0);
}

#[test]
fn the_first_committed_sequence_is_zero_and_folds_once() {
    let at = OffsetDateTime::UNIX_EPOCH;
    let mut view = ThreadView::default();
    view.fold(0, at, &start("root-a"), &config());
    let after_first = view.clone();
    assert_eq!(view.fold(0, at, &start("root-a"), &config()).len(), 0);
    assert_eq!(view, after_first);
    view.fold(1, at, &delta("m1", "ab"), &config());
    assert_eq!(view.fold(0, at, &start("root-a"), &config()).len(), 0);
    assert_eq!(view.fold(1, at, &delta("m1", "ab"), &config()).len(), 0);
    assert!(matches!(
        view.items.get("msg:m1").map(|indexed| &indexed.item),
        Some(ViewItem::Text { text, open: true, .. }) if text == "ab"
    ));
    assert_eq!(view.status.sequence, Some(1));
}

/// Fold `events`, the event at position `i` committed `i` seconds after the
/// epoch.
fn timed_view(events: &[AgentEvent]) -> ThreadView {
    ThreadView::from_events(
        events.iter().enumerate().map(|(i, event)| {
            let at = OffsetDateTime::UNIX_EPOCH
                + time::Duration::seconds(i64::try_from(i).unwrap_or(i64::MAX));
            (i as u64, at, event)
        }),
        &config(),
    )
}

fn turn_end(view: &ThreadView) -> Result<(TokenUsage, Option<u64>), String> {
    view.items
        .values()
        .find_map(|indexed| match &indexed.item {
            ViewItem::TurnEnd {
                usage, duration_ms, ..
            } => Some((usage.clone(), *duration_ms)),
            _ => None,
        })
        .ok_or_else(|| "a turn end".to_owned())
}

/// A turn whose model calls a tool, as the durable worker journals it: two
/// model calls (the tool round, then the round after the tool result) and
/// one `TurnComplete` with their summed usage.
fn tool_turn(turn: usize, tokens_per_call: u32) -> Vec<AgentEvent> {
    let mut both = usage(tokens_per_call);
    both.add(&usage(tokens_per_call));
    vec![
        tool_start(&format!("t{turn}")),
        tool_end(&format!("t{turn}"), true),
        delta(&format!("m{turn}"), "done"),
        text(&format!("m{turn}"), "done"),
        AgentEvent::TurnComplete {
            turn,
            usage: both,
            emitter_task_id: Some("root".to_owned()),
        },
    ]
}

#[test]
fn a_tool_turn_counts_both_model_calls_usage_once() {
    let mut events = vec![start("root")];
    events.extend(tool_turn(1, 100));
    let view = timed_view(&events);
    let mut both = usage(100);
    both.add(&usage(100));
    assert_eq!(
        view.status.usage.run_so_far, both,
        "the one TurnComplete carries both calls"
    );
    assert_eq!(view.status.usage.turns, 1);

    events.push(done());
    events.push(start("root"));
    let again = timed_view(&events);
    assert_eq!(
        again.status.usage.run_so_far,
        TokenUsage::default(),
        "a root start resets it"
    );
}

#[test]
fn a_failed_run_ends_with_the_usage_of_the_turns_it_completed_and_its_duration()
-> Result<(), String> {
    let mut events = vec![start("root")];
    events.extend(tool_turn(1, 3));
    events.extend(tool_turn(2, 4));
    events.push(error(false, Some(TerminalReason::InternalError)));
    let view = timed_view(&events);
    let (usage_at_end, duration_ms) = turn_end(&view)?;
    let mut sum = usage(3);
    sum.add(&usage(3));
    sum.add(&usage(4));
    sum.add(&usage(4));
    assert_eq!(
        usage_at_end, sum,
        "both completed turns, not the last one alone"
    );
    assert_eq!(duration_ms, Some(11_000), "Start at 0 s, the error at 11 s");

    let mut cut = vec![start("root")];
    cut.extend(tool_turn(1, 3).into_iter().take(2));
    cut.push(error(false, Some(TerminalReason::InternalError)));
    let (usage_at_end, _) = turn_end(&timed_view(&cut))?;
    assert_eq!(
        usage_at_end,
        TokenUsage::default(),
        "a turn still in flight when the run fails is not in it",
    );
    Ok(())
}

#[test]
fn a_done_turn_ends_with_its_own_duration() -> Result<(), String> {
    let mut events = vec![start("root")];
    events.extend(tool_turn(1, 3));
    events.push(done());
    let view = timed_view(&events);
    assert_eq!(
        turn_end(&view)?.1,
        Some(5),
        "Done's own duration, not the clock"
    );
    Ok(())
}

#[test]
fn items_carry_the_time_they_were_put_and_last_changed() -> Result<(), String> {
    let view = timed_view(&[
        start("root"),
        delta("m1", "a"),
        delta("m1", "b"),
        text("m1", "ab"),
    ]);
    let indexed = view.items.get("msg:m1").ok_or("the message")?;
    let at = |seconds: i64| Some(OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(seconds));
    assert_eq!(indexed.started_at, at(1), "the first delta put it");
    assert_eq!(indexed.updated_at, at(3), "the final text changed it last");

    let mut untimed = ThreadView::default();
    let (_, ops) = fold(
        &untimed.status,
        0,
        OffsetDateTime::UNIX_EPOCH,
        &start("root"),
        &config(),
    );
    for op in &ops {
        untimed.apply(op);
    }
    let (_, ops) = fold(
        &untimed.status,
        1,
        OffsetDateTime::UNIX_EPOCH,
        &delta("m1", "a"),
        &config(),
    );
    for op in &ops {
        untimed.apply(op);
    }
    let untimed = untimed.items.get("msg:m1").ok_or("the message")?;
    assert_eq!(
        (untimed.started_at, untimed.updated_at),
        (None, None),
        "apply without a time"
    );
    Ok(())
}
