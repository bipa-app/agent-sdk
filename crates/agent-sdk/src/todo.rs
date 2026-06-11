//! TODO task tracking for agents.
//!
//! This module provides tools for agents to track tasks and show progress.
//! Task tracking helps agents organize complex work and gives users visibility
//! into what the agent is working on.
//!
//! # Example
//!
//! ```no_run
//! use agent_sdk::todo::{TodoState, TodoWriteTool, TodoReadTool};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! let state = Arc::new(RwLock::new(TodoState::new()));
//! let write_tool = TodoWriteTool::new(Arc::clone(&state));
//! let read_tool = TodoReadTool::new(state);
//! ```

use std::ffi::OsString;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

/// Status of a TODO item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Task not yet started.
    Pending,
    /// Task currently being worked on.
    InProgress,
    /// Task finished successfully.
    Completed,
}

impl TodoStatus {
    /// Returns the icon for this status.
    #[must_use]
    pub const fn icon(&self) -> &'static str {
        match self {
            Self::Pending => "○",
            Self::InProgress => "⚡",
            Self::Completed => "✓",
        }
    }
}

/// A single TODO item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// Task description in imperative form (e.g., "Fix the bug").
    pub content: String,
    /// Current status of the task.
    pub status: TodoStatus,
    /// Present continuous form shown during execution (e.g., "Fixing the bug").
    pub active_form: String,
}

impl TodoItem {
    /// Creates a new pending TODO item.
    #[must_use]
    pub fn new(content: impl Into<String>, active_form: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            status: TodoStatus::Pending,
            active_form: active_form.into(),
        }
    }

    /// Creates a new TODO item with the given status.
    #[must_use]
    pub fn with_status(
        content: impl Into<String>,
        active_form: impl Into<String>,
        status: TodoStatus,
    ) -> Self {
        Self {
            content: content.into(),
            status,
            active_form: active_form.into(),
        }
    }

    /// Returns the icon for this item's status.
    #[must_use]
    pub const fn icon(&self) -> &'static str {
        self.status.icon()
    }
}

/// Shared TODO state that can be persisted.
#[derive(Debug, Default)]
pub struct TodoState {
    /// The list of TODO items.
    pub items: Vec<TodoItem>,
    /// Optional path for persistence.
    storage_path: Option<PathBuf>,
}

impl TodoState {
    /// Creates a new empty TODO state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items: Vec::new(),
            storage_path: None,
        }
    }

    /// Creates a new TODO state with persistence.
    #[must_use]
    pub const fn with_storage(path: PathBuf) -> Self {
        Self {
            items: Vec::new(),
            storage_path: Some(path),
        }
    }

    /// Sets the storage path for persistence.
    pub fn set_storage_path(&mut self, path: PathBuf) {
        self.storage_path = Some(path);
    }

    /// Loads todos from storage if path is set.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub async fn load(&mut self) -> Result<()> {
        if let Some(ref path) = self.storage_path.as_ref().filter(|p| p.exists()) {
            let content = tokio::fs::read_to_string(path)
                .await
                .context("Failed to read todos file")?;
            self.items = serde_json::from_str(&content).context("Failed to parse todos file")?;
        }
        Ok(())
    }

    /// Saves todos to storage if path is set.
    ///
    /// The write is atomic: the JSON is written to a uniquely-named temp file
    /// in the same directory and then renamed over the target (atomic on
    /// POSIX). A crash or a cancelled future mid-write can therefore never
    /// leave a truncated/invalid file in place, which [`load`](Self::load)
    /// would otherwise fail to parse permanently.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or renamed into place.
    pub async fn save(&self) -> Result<()> {
        let Some(path) = self.storage_path.as_ref() else {
            return Ok(());
        };

        // Ensure parent directory exists.
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent)
                .await
                .context("Failed to create todos directory")?;
        }

        let content =
            serde_json::to_string_pretty(&self.items).context("Failed to serialize todos")?;

        let tmp_path = temp_sibling_path(path)?;
        tokio::fs::write(&tmp_path, content)
            .await
            .context("Failed to write temp todos file")?;

        if let Err(e) = tokio::fs::rename(&tmp_path, path).await {
            // Best-effort cleanup so a failed rename doesn't leak the temp file.
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(e).context("Failed to atomically replace todos file");
        }

        Ok(())
    }

    /// Replaces the entire TODO list.
    pub fn set_items(&mut self, items: Vec<TodoItem>) {
        self.items = items;
    }

    /// Adds a new TODO item.
    pub fn add(&mut self, item: TodoItem) {
        self.items.push(item);
    }

    /// Returns the count of items by status.
    #[must_use]
    pub fn count_by_status(&self) -> (usize, usize, usize) {
        let pending = self
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::Pending)
            .count();
        let in_progress = self
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::InProgress)
            .count();
        let completed = self
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::Completed)
            .count();
        (pending, in_progress, completed)
    }

    /// Returns the currently in-progress item, if any.
    #[must_use]
    pub fn current_task(&self) -> Option<&TodoItem> {
        self.items
            .iter()
            .find(|i| i.status == TodoStatus::InProgress)
    }

    /// Formats the TODO list for display.
    #[must_use]
    pub fn format_display(&self) -> String {
        if self.items.is_empty() {
            return "No tasks".to_string();
        }

        let (_pending, in_progress, completed) = self.count_by_status();
        let total = self.items.len();

        let mut output = format!("TODO ({completed}/{total})");

        if in_progress > 0
            && let Some(current) = self.current_task()
        {
            let _ = write!(output, " - {}", current.active_form);
        }

        output.push('\n');

        for item in &self.items {
            let _ = writeln!(output, "  {} {}", item.icon(), item.content);
        }

        output
    }

    /// Returns true if there are no items.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the number of items.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.items.len()
    }
}

/// Builds a unique sibling path (same directory as `target`) for the
/// write-then-rename in [`TodoState::save`]. Uniqueness combines the process
/// id with a monotonic counter so concurrent saves never collide on the temp
/// name.
fn temp_sibling_path(target: &Path) -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let file_name = target
        .file_name()
        .context("storage path has no file name")?;
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut tmp_name = OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(format!(".tmp.{}.{nonce}", std::process::id()));

    Ok(
        match target.parent().filter(|p| !p.as_os_str().is_empty()) {
            Some(parent) => parent.join(tmp_name),
            None => PathBuf::from(tmp_name),
        },
    )
}

/// Tool for writing/updating the TODO list.
pub struct TodoWriteTool {
    /// Shared TODO state.
    state: Arc<RwLock<TodoState>>,
}

impl TodoWriteTool {
    /// Creates a new `TodoWriteTool`.
    #[must_use]
    pub const fn new(state: Arc<RwLock<TodoState>>) -> Self {
        Self { state }
    }
}

/// Input for a single TODO item.
#[derive(Debug, Deserialize)]
struct TodoItemInput {
    content: String,
    status: TodoStatus,
    #[serde(rename = "activeForm")]
    active_form: String,
}

/// Input schema for `TodoWriteTool`.
#[derive(Debug, Deserialize)]
struct TodoWriteInput {
    todos: Vec<TodoItemInput>,
}

impl<Ctx: Send + Sync + 'static> Tool<Ctx> for TodoWriteTool {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::TodoWrite
    }

    fn display_name(&self) -> &'static str {
        "Update Tasks"
    }

    fn description(&self) -> &'static str {
        "Update the TODO list to track tasks and show progress to the user. \
         Use this tool frequently to plan complex tasks and mark progress. \
         Each item needs 'content' (imperative form like 'Fix the bug'), \
         'status' (pending/in_progress/completed), and 'activeForm' \
         (present continuous like 'Fixing the bug'). \
         Mark tasks completed immediately when done - don't batch completions."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["todos"],
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete TODO list (replaces existing)",
                    "items": {
                        "type": "object",
                        "required": ["content", "status", "activeForm"],
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Task description in imperative form (e.g., 'Fix the bug')"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of the task"
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present continuous form shown during execution (e.g., 'Fixing the bug')"
                            }
                        }
                    }
                }
            }
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe // No dangerous side effects
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: TodoWriteInput =
            serde_json::from_value(input).context("Invalid input for todo_write")?;

        let items: Vec<TodoItem> = input
            .todos
            .into_iter()
            .map(|t| TodoItem {
                content: t.content,
                status: t.status,
                active_form: t.active_form,
            })
            .collect();

        // Update the shared state, then snapshot what needs persisting and
        // drop the write guard *before* the (potentially slow) disk I/O. This
        // keeps `TodoReadTool` unblocked during the save.
        let mut state = self.state.write().await;
        state.set_items(items);
        let snapshot = TodoState {
            items: state.items.clone(),
            storage_path: state.storage_path.clone(),
        };
        let display = state.format_display();
        drop(state);

        if let Err(e) = snapshot.save().await {
            log::warn!("Failed to persist todos: {e}");
            return Ok(ToolResult::error(format!(
                "TODO list updated in memory, but persisting to storage failed: {e}\n\n{display}"
            )));
        }

        Ok(ToolResult::success(format!(
            "TODO list updated.\n\n{display}"
        )))
    }
}

/// Tool for reading the current TODO list.
pub struct TodoReadTool {
    /// Shared TODO state.
    state: Arc<RwLock<TodoState>>,
}

impl TodoReadTool {
    /// Creates a new `TodoReadTool`.
    #[must_use]
    pub const fn new(state: Arc<RwLock<TodoState>>) -> Self {
        Self { state }
    }
}

impl<Ctx: Send + Sync + 'static> Tool<Ctx> for TodoReadTool {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::TodoRead
    }

    fn display_name(&self) -> &'static str {
        "Read Tasks"
    }

    fn description(&self) -> &'static str {
        "Read the current TODO list to see task status and progress."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, _input: Value) -> Result<ToolResult> {
        let display = {
            let state = self.state.read().await;
            state.format_display()
        };

        Ok(ToolResult::success(display))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_status_icons() {
        assert_eq!(TodoStatus::Pending.icon(), "○");
        assert_eq!(TodoStatus::InProgress.icon(), "⚡");
        assert_eq!(TodoStatus::Completed.icon(), "✓");
    }

    #[test]
    fn test_todo_item_new() {
        let item = TodoItem::new("Fix the bug", "Fixing the bug");
        assert_eq!(item.content, "Fix the bug");
        assert_eq!(item.active_form, "Fixing the bug");
        assert_eq!(item.status, TodoStatus::Pending);
    }

    #[test]
    fn test_todo_state_count_by_status() {
        let mut state = TodoState::new();
        state.add(TodoItem::with_status(
            "Task 1",
            "Task 1",
            TodoStatus::Pending,
        ));
        state.add(TodoItem::with_status(
            "Task 2",
            "Task 2",
            TodoStatus::InProgress,
        ));
        state.add(TodoItem::with_status(
            "Task 3",
            "Task 3",
            TodoStatus::Completed,
        ));
        state.add(TodoItem::with_status(
            "Task 4",
            "Task 4",
            TodoStatus::Completed,
        ));

        let (pending, in_progress, completed) = state.count_by_status();
        assert_eq!(pending, 1);
        assert_eq!(in_progress, 1);
        assert_eq!(completed, 2);
    }

    #[test]
    fn test_todo_state_current_task() {
        let mut state = TodoState::new();
        state.add(TodoItem::with_status(
            "Task 1",
            "Task 1",
            TodoStatus::Pending,
        ));
        assert!(state.current_task().is_none());

        state.add(TodoItem::with_status(
            "Task 2",
            "Working on Task 2",
            TodoStatus::InProgress,
        ));
        let current = state.current_task().unwrap();
        assert_eq!(current.content, "Task 2");
        assert_eq!(current.active_form, "Working on Task 2");
    }

    #[test]
    fn test_todo_state_format_display() {
        let mut state = TodoState::new();
        assert_eq!(state.format_display(), "No tasks");

        state.add(TodoItem::with_status(
            "Fix bug",
            "Fixing bug",
            TodoStatus::InProgress,
        ));
        state.add(TodoItem::with_status(
            "Write tests",
            "Writing tests",
            TodoStatus::Pending,
        ));

        let display = state.format_display();
        assert!(display.contains("TODO (0/2)"));
        assert!(display.contains("Fixing bug"));
        assert!(display.contains("⚡ Fix bug"));
        assert!(display.contains("○ Write tests"));
    }

    #[test]
    fn test_todo_status_serde() {
        let status = TodoStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");

        let parsed: TodoStatus = serde_json::from_str("\"completed\"").unwrap();
        assert_eq!(parsed, TodoStatus::Completed);
    }

    #[tokio::test]
    async fn save_then_load_round_trips_through_storage() -> Result<()> {
        let dir = tempfile::tempdir().context("create temp dir")?;
        let path = dir.path().join("todos.json");

        let mut state = TodoState::with_storage(path.clone());
        state.add(TodoItem::with_status(
            "Fix bug",
            "Fixing bug",
            TodoStatus::InProgress,
        ));
        state.add(TodoItem::with_status(
            "Write tests",
            "Writing tests",
            TodoStatus::Pending,
        ));
        state.save().await?;

        assert!(path.exists(), "target file should exist after save");

        // The atomic write must not leave its temp sibling behind.
        let mut entries = tokio::fs::read_dir(dir.path()).await?;
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            assert!(!name.contains(".tmp."), "temp file leaked: {name}");
        }

        let mut loaded = TodoState::with_storage(path);
        loaded.load().await?;
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.items[0].content, "Fix bug");
        assert_eq!(loaded.items[1].status, TodoStatus::Pending);

        Ok(())
    }

    #[tokio::test]
    async fn save_overwrites_existing_file_atomically() -> Result<()> {
        let dir = tempfile::tempdir().context("create temp dir")?;
        let path = dir.path().join("todos.json");

        let mut first = TodoState::with_storage(path.clone());
        first.add(TodoItem::new("Old task", "Doing old task"));
        first.save().await?;

        let mut second = TodoState::with_storage(path.clone());
        second.add(TodoItem::new("New task", "Doing new task"));
        second.save().await?;

        let mut loaded = TodoState::with_storage(path);
        loaded.load().await?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.items[0].content, "New task");

        Ok(())
    }

    #[tokio::test]
    async fn save_is_noop_without_storage_path() -> Result<()> {
        let mut state = TodoState::new();
        state.add(TodoItem::new("Task", "Doing task"));
        state.save().await?;
        Ok(())
    }
}
