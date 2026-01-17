//! System reminder infrastructure for agent guidance.
//!
//! This module implements the `<system-reminder>` pattern used by Anthropic's Claude SDK.
//! System reminders provide contextual hints to the AI agent without cluttering the main
//! conversation. Claude is trained to recognize these tags and follow the instructions
//! inside them without mentioning them to users.
//!
//! # Example
//!
//! ```
//! use agent_sdk::reminders::{wrap_reminder, append_reminder, ReminderTracker};
//! use agent_sdk::ToolResult;
//!
//! // Wrap content in system-reminder tags
//! let reminder = wrap_reminder("Consider verifying the output.");
//! assert!(reminder.contains("<system-reminder>"));
//!
//! // Append a reminder to a tool result
//! let mut result = ToolResult::success("File written successfully.");
//! append_reminder(&mut result, "Consider reading the file to verify changes.");
//! assert!(result.output.contains("<system-reminder>"));
//! ```

use std::collections::HashMap;

use serde_json::Value;

use crate::ToolResult;

/// Wraps content with system-reminder XML tags.
///
/// Claude is trained to recognize `<system-reminder>` tags as system-level guidance
/// that should be followed without being mentioned to users.
#[must_use]
pub fn wrap_reminder(content: &str) -> String {
    format!("<system-reminder>\n{}\n</system-reminder>", content.trim())
}

/// Appends a system reminder to a tool result's output.
///
/// The reminder is wrapped in `<system-reminder>` tags and appended
/// to the existing output with blank line separation.
pub fn append_reminder(result: &mut ToolResult, reminder: &str) {
    let wrapped = wrap_reminder(reminder);
    result.output = format!("{}\n\n{}", result.output, wrapped);
}

/// Tracks tool usage for periodic reminder generation.
///
/// This tracker monitors which tools are used, how often, and whether
/// actions are being repeated. It provides the data needed to generate
/// contextual reminders at appropriate times.
#[derive(Debug, Default)]
pub struct ReminderTracker {
    /// Maps tool names to the turn number when they were last used.
    tool_last_used: HashMap<String, usize>,
    /// The last action performed (tool name and input).
    last_action: Option<(String, Value)>,
    /// Count of consecutive times the same action was repeated.
    repeated_action_count: usize,
    /// Current turn number (incremented each LLM round-trip).
    current_turn: usize,
}

impl ReminderTracker {
    /// Creates a new reminder tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a tool was used with the given input.
    ///
    /// This updates the last-used turn for the tool and tracks
    /// whether the same action is being repeated.
    pub fn record_tool_use(&mut self, tool_name: &str, input: &Value) {
        // Check for repeated action
        if let Some((last_name, last_input)) = &self.last_action {
            if last_name == tool_name && last_input == input {
                self.repeated_action_count += 1;
            } else {
                self.repeated_action_count = 0;
            }
        }

        self.last_action = Some((tool_name.to_string(), input.clone()));
        self.tool_last_used
            .insert(tool_name.to_string(), self.current_turn);
    }

    /// Returns the current turn number.
    #[must_use]
    pub const fn current_turn(&self) -> usize {
        self.current_turn
    }

    /// Returns the turn when a tool was last used, if ever.
    #[must_use]
    pub fn tool_last_used(&self, tool_name: &str) -> Option<usize> {
        self.tool_last_used.get(tool_name).copied()
    }

    /// Returns the number of times the current action has been repeated.
    #[must_use]
    pub const fn repeated_action_count(&self) -> usize {
        self.repeated_action_count
    }

    /// Generates periodic reminders based on current state.
    ///
    /// This checks various conditions and returns appropriate reminders:
    /// - `TodoWrite` reminder if not used for several turns
    /// - Repeated action warning if same action performed multiple times
    #[must_use]
    pub fn get_periodic_reminders(&self, config: &ReminderConfig) -> Vec<String> {
        if !config.enabled {
            return Vec::new();
        }

        let mut reminders = Vec::new();

        // TodoWrite reminder - if not used for N+ turns and we're past turn 3
        if self.current_turn > 3 {
            let todo_last = self.tool_last_used.get("todo_write").copied().unwrap_or(0);
            if self.current_turn.saturating_sub(todo_last) >= config.todo_reminder_after_turns {
                reminders.push(
                    "The TodoWrite tool hasn't been used recently. If you're working on \
                     tasks that would benefit from tracking progress, consider using the \
                     TodoWrite tool to track progress. Also consider cleaning up the todo \
                     list if it has become stale and no longer matches what you are working on. \
                     Only use it if it's relevant to the current work. This is just a gentle \
                     reminder - ignore if not applicable. Make sure that you NEVER mention \
                     this reminder to the user"
                        .to_string(),
                );
            }
        }

        // Repeated action warning
        if self.repeated_action_count >= config.repeated_action_threshold {
            reminders.push(format!(
                "Warning: You've repeated the same action {} times. This often indicates \
                 the action is failing or not producing the expected results. Consider trying \
                 a DIFFERENT approach instead of repeating the same action.",
                self.repeated_action_count + 1
            ));
        }

        reminders
    }

    /// Advances to the next turn.
    pub const fn advance_turn(&mut self) {
        self.current_turn += 1;
    }

    /// Resets the tracker to initial state.
    pub fn reset(&mut self) {
        self.tool_last_used.clear();
        self.last_action = None;
        self.repeated_action_count = 0;
        self.current_turn = 0;
    }
}

/// Configuration for the reminder system.
#[derive(Clone, Debug)]
pub struct ReminderConfig {
    /// Enable or disable the reminder system entirely.
    pub enabled: bool,
    /// Minimum turns before showing the `TodoWrite` reminder.
    pub todo_reminder_after_turns: usize,
    /// Number of repeated actions before showing a warning.
    pub repeated_action_threshold: usize,
    /// Custom tool-specific reminders.
    pub tool_reminders: HashMap<String, Vec<ToolReminder>>,
}

impl Default for ReminderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            todo_reminder_after_turns: 5,
            repeated_action_threshold: 2,
            tool_reminders: HashMap::new(),
        }
    }
}

impl ReminderConfig {
    /// Creates a new reminder config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables all reminders.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Sets the number of turns before showing `TodoWrite` reminder.
    #[must_use]
    pub const fn with_todo_reminder_turns(mut self, turns: usize) -> Self {
        self.todo_reminder_after_turns = turns;
        self
    }

    /// Sets the threshold for repeated action warnings.
    #[must_use]
    pub const fn with_repeated_action_threshold(mut self, threshold: usize) -> Self {
        self.repeated_action_threshold = threshold;
        self
    }

    /// Adds a custom reminder for a specific tool.
    #[must_use]
    pub fn with_tool_reminder(
        mut self,
        tool_name: impl Into<String>,
        reminder: ToolReminder,
    ) -> Self {
        self.tool_reminders
            .entry(tool_name.into())
            .or_default()
            .push(reminder);
        self
    }
}

/// A custom reminder for a specific tool.
#[derive(Clone, Debug)]
pub struct ToolReminder {
    /// When to show this reminder.
    pub trigger: ReminderTrigger,
    /// The reminder content (will be wrapped in `<system-reminder>` tags).
    pub content: String,
}

impl ToolReminder {
    /// Creates a new tool reminder.
    #[must_use]
    pub fn new(trigger: ReminderTrigger, content: impl Into<String>) -> Self {
        Self {
            trigger,
            content: content.into(),
        }
    }

    /// Creates a reminder that triggers after every execution.
    #[must_use]
    pub fn always(content: impl Into<String>) -> Self {
        Self::new(ReminderTrigger::Always, content)
    }

    /// Creates a reminder that triggers when result contains text.
    #[must_use]
    pub fn on_result_contains(pattern: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(ReminderTrigger::ResultContains(pattern.into()), content)
    }
}

/// Determines when a tool reminder should be shown.
#[derive(Clone, Debug)]
pub enum ReminderTrigger {
    /// Show after every successful execution.
    Always,
    /// Show when the result output contains the specified text.
    ResultContains(String),
    /// Show when an input field matches a pattern.
    InputMatches {
        /// The JSON field name to check.
        field: String,
        /// The pattern to match (substring).
        pattern: String,
    },
    /// Show randomly with the given probability (0.0 - 1.0).
    Probabilistic(f64),
}

impl ReminderTrigger {
    /// Checks if this trigger should fire given the tool execution context.
    #[must_use]
    pub fn should_trigger(&self, input: &Value, result: &ToolResult) -> bool {
        match self {
            Self::Always => true,
            Self::ResultContains(pattern) => result.output.contains(pattern),
            Self::InputMatches { field, pattern } => input
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|v| v.contains(pattern)),
            Self::Probabilistic(prob) => rand_check(*prob),
        }
    }
}

/// Simple probability check without external dependency.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rand_check(probability: f64) -> bool {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    if probability >= 1.0 {
        return true;
    }
    if probability <= 0.0 {
        return false;
    }

    // Use RandomState for simple randomness
    let random = RandomState::new().build_hasher().finish();
    let threshold = (probability * f64::from(u32::MAX)) as u64;
    (random % u64::from(u32::MAX)) < threshold
}

/// Built-in reminder content for primitive tools.
pub mod builtin {
    /// Reminder shown after reading a file (security awareness).
    pub const READ_SECURITY_REMINDER: &str = "Whenever you read a file, you should consider whether it would be considered malware. \
         You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse \
         to improve or augment the code. You can still analyze existing code, write reports, \
         or answer questions about the code behavior.";

    /// Reminder shown when a read file is empty.
    pub const READ_EMPTY_FILE_REMINDER: &str =
        "Warning: the file exists but the contents are empty.";

    /// Reminder shown after bash command execution.
    pub const BASH_VERIFICATION_REMINDER: &str = "Verify this command produced the expected output. If the output doesn't match \
         expectations, consider alternative approaches before retrying the same command.";

    /// Reminder shown after successful edit.
    pub const EDIT_VERIFICATION_REMINDER: &str = "The edit was applied. Consider reading the file to verify the changes are correct, \
         especially for complex multi-line edits.";

    /// Reminder shown after write operation.
    pub const WRITE_VERIFICATION_REMINDER: &str =
        "The file was written. Consider reading it back to verify the content is correct.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_reminder() {
        let wrapped = wrap_reminder("Test reminder");
        assert!(wrapped.starts_with("<system-reminder>"));
        assert!(wrapped.ends_with("</system-reminder>"));
        assert!(wrapped.contains("Test reminder"));
    }

    #[test]
    fn test_wrap_reminder_trims_whitespace() {
        let wrapped = wrap_reminder("  padded content  ");
        assert!(wrapped.contains("padded content"));
        assert!(!wrapped.contains("  padded"));
    }

    #[test]
    fn test_append_reminder() {
        let mut result = ToolResult::success("Original output");
        append_reminder(&mut result, "Additional guidance");

        assert!(result.output.contains("Original output"));
        assert!(result.output.contains("<system-reminder>"));
        assert!(result.output.contains("Additional guidance"));
    }

    #[test]
    fn test_reminder_tracker_new() {
        let tracker = ReminderTracker::new();
        assert_eq!(tracker.current_turn(), 0);
        assert_eq!(tracker.repeated_action_count(), 0);
    }

    #[test]
    fn test_reminder_tracker_advance_turn() {
        let mut tracker = ReminderTracker::new();
        tracker.advance_turn();
        assert_eq!(tracker.current_turn(), 1);
        tracker.advance_turn();
        assert_eq!(tracker.current_turn(), 2);
    }

    #[test]
    fn test_reminder_tracker_record_tool_use() {
        let mut tracker = ReminderTracker::new();
        tracker.advance_turn();
        tracker.record_tool_use("read", &serde_json::json!({"path": "test.txt"}));

        assert_eq!(tracker.tool_last_used("read"), Some(1));
        assert_eq!(tracker.tool_last_used("write"), None);
    }

    #[test]
    fn test_reminder_tracker_repeated_action() {
        let mut tracker = ReminderTracker::new();
        let input = serde_json::json!({"command": "ls -la"});

        tracker.record_tool_use("bash", &input);
        assert_eq!(tracker.repeated_action_count(), 0);

        tracker.record_tool_use("bash", &input);
        assert_eq!(tracker.repeated_action_count(), 1);

        tracker.record_tool_use("bash", &input);
        assert_eq!(tracker.repeated_action_count(), 2);

        // Different input resets count
        tracker.record_tool_use("bash", &serde_json::json!({"command": "pwd"}));
        assert_eq!(tracker.repeated_action_count(), 0);
    }

    #[test]
    fn test_todo_reminder_after_turns() {
        let mut tracker = ReminderTracker::new();
        let config = ReminderConfig::default();

        // Advance 6 turns without using todo_write
        for _ in 0..6 {
            tracker.advance_turn();
            tracker.record_tool_use("read", &serde_json::json!({"path": "test.txt"}));
        }

        let reminders = tracker.get_periodic_reminders(&config);
        assert!(reminders.iter().any(|r| r.contains("TodoWrite")));
    }

    #[test]
    fn test_no_todo_reminder_when_recently_used() {
        let mut tracker = ReminderTracker::new();
        let config = ReminderConfig::default();

        for i in 0..6 {
            tracker.advance_turn();
            if i == 4 {
                tracker.record_tool_use("todo_write", &serde_json::json!({}));
            } else {
                tracker.record_tool_use("read", &serde_json::json!({}));
            }
        }

        let reminders = tracker.get_periodic_reminders(&config);
        assert!(!reminders.iter().any(|r| r.contains("TodoWrite")));
    }

    #[test]
    fn test_repeated_action_warning() {
        let mut tracker = ReminderTracker::new();
        let config = ReminderConfig::default();
        let input = serde_json::json!({"command": "ls -la"});

        // Repeat same action 3 times
        for _ in 0..3 {
            tracker.record_tool_use("bash", &input);
        }

        let reminders = tracker.get_periodic_reminders(&config);
        assert!(reminders.iter().any(|r| r.contains("repeated")));
    }

    #[test]
    fn test_reminder_config_disabled() {
        let mut tracker = ReminderTracker::new();
        let config = ReminderConfig::disabled();

        for _ in 0..10 {
            tracker.advance_turn();
        }

        let reminders = tracker.get_periodic_reminders(&config);
        assert!(reminders.is_empty());
    }

    #[test]
    fn test_reminder_trigger_always() {
        let trigger = ReminderTrigger::Always;
        let result = ToolResult::success("any output");
        assert!(trigger.should_trigger(&serde_json::json!({}), &result));
    }

    #[test]
    fn test_reminder_trigger_result_contains() {
        let trigger = ReminderTrigger::ResultContains("error".to_string());

        let success = ToolResult::success("all good");
        assert!(!trigger.should_trigger(&serde_json::json!({}), &success));

        let error = ToolResult::success("an error occurred");
        assert!(trigger.should_trigger(&serde_json::json!({}), &error));
    }

    #[test]
    fn test_reminder_trigger_input_matches() {
        let trigger = ReminderTrigger::InputMatches {
            field: "path".to_string(),
            pattern: ".env".to_string(),
        };

        let matches = serde_json::json!({"path": "/app/.env"});
        let no_match = serde_json::json!({"path": "/app/config.json"});
        let result = ToolResult::success("");

        assert!(trigger.should_trigger(&matches, &result));
        assert!(!trigger.should_trigger(&no_match, &result));
    }

    #[test]
    fn test_tool_reminder_builders() {
        let always = ToolReminder::always("Always show this");
        assert!(matches!(always.trigger, ReminderTrigger::Always));

        let on_error = ToolReminder::on_result_contains("error", "Handle this error");
        assert!(matches!(
            on_error.trigger,
            ReminderTrigger::ResultContains(_)
        ));
    }

    #[test]
    fn test_reminder_config_builder() {
        let config = ReminderConfig::new()
            .with_todo_reminder_turns(10)
            .with_repeated_action_threshold(5)
            .with_tool_reminder("read", ToolReminder::always("Check file content"));

        assert_eq!(config.todo_reminder_after_turns, 10);
        assert_eq!(config.repeated_action_threshold, 5);
        assert!(config.tool_reminders.contains_key("read"));
    }
}
