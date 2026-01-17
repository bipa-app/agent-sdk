use crate::reminders::{append_reminder, builtin};
use crate::{Environment, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for editing files via string replacement
pub struct EditTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> EditTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct EditInput {
    /// Path to the file to edit (also accepts `file_path` for compatibility)
    #[serde(alias = "file_path")]
    path: String,
    /// String to find and replace
    old_string: String,
    /// Replacement string
    new_string: String,
    /// Replace all occurrences (default: false)
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl<E: Environment + 'static> Tool<()> for EditTool<E> {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Edit a file by replacing a string. The old_string must match exactly and uniquely (unless replace_all is true)."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences instead of requiring unique match. Default: false"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: EditInput =
            serde_json::from_value(input).context("Invalid input for edit tool")?;

        let path = self.ctx.environment.resolve_path(&input.path);

        // Check capabilities
        if !self.ctx.capabilities.can_write(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot edit '{path}'"
            )));
        }

        // Check if file exists
        let exists = self
            .ctx
            .environment
            .exists(&path)
            .await
            .context("Failed to check file existence")?;

        if !exists {
            return Ok(ToolResult::error(format!("File not found: '{path}'")));
        }

        // Check if it's a directory
        let is_dir = self
            .ctx
            .environment
            .is_dir(&path)
            .await
            .context("Failed to check if path is directory")?;

        if is_dir {
            return Ok(ToolResult::error(format!(
                "'{path}' is a directory, cannot edit"
            )));
        }

        // Read current content
        let content = self
            .ctx
            .environment
            .read_file(&path)
            .await
            .context("Failed to read file")?;

        // Count occurrences
        let count = content.matches(&input.old_string).count();

        if count == 0 {
            return Ok(ToolResult::error(format!(
                "String not found in '{}': '{}'",
                path,
                truncate_string(&input.old_string, 100)
            )));
        }

        if count > 1 && !input.replace_all {
            return Ok(ToolResult::error(format!(
                "Found {count} occurrences of the string in '{path}'. Use replace_all: true to replace all, or provide a more specific string."
            )));
        }

        // Perform replacement
        let new_content = if input.replace_all {
            content.replace(&input.old_string, &input.new_string)
        } else {
            content.replacen(&input.old_string, &input.new_string, 1)
        };

        // Write back
        self.ctx
            .environment
            .write_file(&path, &new_content)
            .await
            .context("Failed to write file")?;

        let replacements = if input.replace_all { count } else { 1 };
        let mut result = ToolResult::success(format!(
            "Successfully replaced {replacements} occurrence(s) in '{path}'"
        ));

        // Add verification reminder
        append_reminder(&mut result, builtin::EDIT_VERIFICATION_REMINDER);

        Ok(result)
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> EditTool<InMemoryFileSystem> {
        EditTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_edit_simple_replacement() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "Hello, World!").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "World",
                    "new_string": "Rust"
                }),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("1 occurrence"));

        let content = fs.read_file("/workspace/test.txt").await?;
        assert_eq!(content, "Hello, Rust!");
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_replace_all_true() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo bar foo baz foo").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "foo",
                    "new_string": "qux",
                    "replace_all": true
                }),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("3 occurrence"));

        let content = fs.read_file("/workspace/test.txt").await?;
        assert_eq!(content, "qux bar qux baz qux");
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_multiline_replacement() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.rs", "fn main() {\n    println!(\"Hello\");\n}")
            .await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.rs",
                    "old_string": "println!(\"Hello\");",
                    "new_string": "println!(\"Hello, World!\");\n    println!(\"Goodbye!\");"
                }),
            )
            .await?;

        assert!(result.success);

        let content = fs.read_file("/workspace/test.rs").await?;
        assert!(content.contains("Hello, World!"));
        assert!(content.contains("Goodbye!"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_edit_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "content").await?;

        // Read-only capabilities
        let caps = AgentCapabilities::read_only();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "content",
                    "new_string": "new content"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_denied_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secrets/config.txt", "secret=value").await?;

        let caps = AgentCapabilities::full_access()
            .with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/secrets/config.txt",
                    "old_string": "value",
                    "new_string": "newvalue"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_edit_string_not_found() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "Hello, World!").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "Rust",
                    "new_string": "Go"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("String not found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_multiple_occurrences_without_replace_all() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo bar foo baz").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "foo",
                    "new_string": "qux"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("2 occurrences"));
        assert!(result.output.contains("replace_all"));

        // File should not have changed
        let content = fs.read_file("/workspace/test.txt").await?;
        assert_eq!(content, "foo bar foo baz");
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_file_not_found() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/nonexistent.txt",
                    "old_string": "foo",
                    "new_string": "bar"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("File not found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_directory_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.create_dir("/workspace/subdir").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/subdir",
                    "old_string": "foo",
                    "new_string": "bar"
                }),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("is a directory"));
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_empty_new_string_deletes() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "Hello, World!").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": ", World",
                    "new_string": ""
                }),
            )
            .await?;

        assert!(result.success);

        let content = fs.read_file("/workspace/test.txt").await?;
        assert_eq!(content, "Hello!");
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_special_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "特殊字符 emoji 🎉 here").await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "🎉",
                    "new_string": "🚀"
                }),
            )
            .await?;

        assert!(result.success);

        let content = fs.read_file("/workspace/test.txt").await?;
        assert!(content.contains("🚀"));
        assert!(!content.contains("🎉"));
        Ok(())
    }

    #[tokio::test]
    async fn test_edit_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), "edit");
        assert_eq!(tool.tier(), ToolTier::Confirm);
        assert!(tool.description().contains("Edit"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("old_string").is_some());
        assert!(schema["properties"].get("new_string").is_some());
        assert!(schema["properties"].get("replace_all").is_some());
    }

    #[tokio::test]
    async fn test_edit_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required fields
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_truncate_string_function() {
        assert_eq!(truncate_string("short", 10), "short");
        assert_eq!(
            truncate_string("this is a longer string", 10),
            "this is a ..."
        );
    }

    #[tokio::test]
    async fn test_edit_preserves_surrounding_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let original = "line 1\nline 2 with target\nline 3";
        fs.write_file("test.txt", original).await?;

        let tool = create_test_tool(Arc::clone(&fs), AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({
                    "path": "/workspace/test.txt",
                    "old_string": "target",
                    "new_string": "replacement"
                }),
            )
            .await?;

        assert!(result.success);

        let content = fs.read_file("/workspace/test.txt").await?;
        assert_eq!(content, "line 1\nline 2 with replacement\nline 3");
        Ok(())
    }
}
