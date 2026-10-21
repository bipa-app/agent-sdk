use crate::{
    Environment, PrimitiveToolName, Tool, ToolConcurrency, ToolContext, ToolReplay, ToolResult,
    ToolTier,
};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Tool for searching file contents using regex patterns
pub struct GrepTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> GrepTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct GrepInput {
    /// Regex pattern to search for
    pattern: String,
    /// Path to search in (file or directory)
    #[serde(default)]
    path: Option<String>,
    /// Search recursively in directories (default: true)
    #[serde(
        default = "default_recursive",
        deserialize_with = "deserialize_recursive"
    )]
    recursive: bool,
    /// Case insensitive search (default: false)
    #[serde(default, deserialize_with = "super::null_as_default")]
    case_insensitive: bool,
    /// Lines of context around each match (default: 2)
    #[serde(default = "default_context", deserialize_with = "deserialize_context")]
    context: usize,
    /// Search minified and bundled files too (default: false)
    #[serde(default, deserialize_with = "super::null_as_default")]
    include_minified: bool,
}

const fn default_context() -> usize {
    2
}

fn deserialize_context<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<usize>::deserialize(deserializer)
        .map(|context| context.unwrap_or_else(default_context))
}

const MAX_RESULTS: usize = 50;
const MAX_CONTEXT: usize = 10;
const MAX_OUTPUT_LINES: usize = 200;
/// Keeps a default search under the inline output budget (50 KiB), so it is
/// never spilled to an artifact.
const MAX_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_LINE_CHARS: usize = 200;
/// Total bytes read back for context lines; files past it show bare hits.
const CONTEXT_READ_BUDGET_BYTES: usize = 32 * 1024 * 1024;

/// Generated files whose hits drown out the source: minified bundles and
/// their source maps.
const MINIFIED_SUFFIXES: &[&str] = &[
    ".min.js",
    ".min.mjs",
    ".min.cjs",
    ".min.css",
    ".bundle.js",
    ".js.map",
    ".mjs.map",
    ".cjs.map",
    ".css.map",
];

fn is_minified_or_bundled(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    MINIFIED_SUFFIXES
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

const fn default_recursive() -> bool {
    true
}

fn deserialize_recursive<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<bool>::deserialize(deserializer)
        .map(|recursive| recursive.unwrap_or_else(default_recursive))
}

impl<E: Environment + 'static, Ctx: Send + Sync + 'static> Tool<Ctx> for GrepTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Grep
    }

    fn display_name(&self) -> &'static str {
        "Search Files"
    }

    fn description(&self) -> &'static str {
        "Search for a regex pattern in files. Returns matching lines as path:line:text with 2 lines of context as path-line-text (set context, 0 to turn it off). Minified and bundled files (*.min.js, *.js.map, …) are skipped unless include_minified is true."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ReadOnly
    }

    fn replay(&self) -> ToolReplay {
        ToolReplay::Safe
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (file or directory). Defaults to environment root."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in directories. Default: true"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search. Default: false"
                },
                "context": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_CONTEXT,
                    "description": "Lines of context before and after each match. Default: 2; 0 shows bare matches"
                },
                "include_minified": {
                    "type": "boolean",
                    "description": "Also search minified and bundled files (*.min.js, *.min.css, *.bundle.js, source maps). Default: false"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: GrepInput = GrepInput::deserialize(&input)
            .with_context(|| format!("Invalid input for grep tool: {input}"))?;

        let search_path = input.path.as_ref().map_or_else(
            || self.ctx.environment.root().to_string(),
            |p| self.ctx.environment.resolve_path(p),
        );

        // Check read capability
        if let Err(reason) = self.ctx.capabilities.check_read(&search_path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot search in '{search_path}': {reason}"
            )));
        }

        // Build pattern with case insensitivity if requested
        let pattern = if input.case_insensitive {
            format!("(?i){}", input.pattern)
        } else {
            input.pattern.clone()
        };

        // The pattern is the model's own input, so validate the regex up front
        // and report a syntax error as a correctable tool error rather than an
        // infrastructure failure. Genuine I/O errors from `grep` still surface
        // as `anyhow::Err`.
        if let Err(err) = regex::Regex::new(&pattern) {
            return Ok(ToolResult::error(format!(
                "Invalid pattern '{}': {err:#}",
                input.pattern
            )));
        }

        // NOTE: `Environment::grep` implementations MUST enforce read
        // permissions per traversed path; the root-only capability check above
        // plus the per-result filter below are defense-in-depth, not a
        // substitute.

        // Execute grep
        let matches = self
            .ctx
            .environment
            .grep(&pattern, &search_path, input.recursive)
            .await
            .context("Failed to execute grep")?;

        // A file named on its own is searched whatever its name: asking for
        // it is the request to see it.
        let skip_minified = !input.include_minified
            && !self
                .ctx
                .environment
                .is_file(&search_path)
                .await
                .unwrap_or(false);

        // Filter out matches in files the agent can't read
        let (minified, matches): (Vec<_>, Vec<_>) = matches
            .into_iter()
            .filter(|m| self.ctx.capabilities.check_read(&m.path).is_ok())
            .partition(|m| skip_minified && is_minified_or_bundled(&m.path));
        let skipped_note = (!minified.is_empty())
            .then(|| minified_note(minified.len(), minified.len() + matches.len()));

        if matches.is_empty() {
            let output = skipped_note.map_or_else(
                || format!("No matches found for pattern '{}'", input.pattern),
                |note| {
                    format!(
                        "No matches outside minified or bundled files for pattern '{}'. {note}",
                        input.pattern
                    )
                },
            );
            return Ok(ToolResult::success(output));
        }

        let count = matches.len();
        let rendered = self
            .render_matches(
                &matches[..count.min(MAX_RESULTS)],
                input.context.min(MAX_CONTEXT),
                CONTEXT_READ_BUDGET_BYTES,
            )
            .await;
        let found = if skipped_note.is_some() {
            format!("Found {count} matches outside minified or bundled files")
        } else {
            format!("Found {count} matches")
        };
        let mut output = if rendered.shown < count {
            format!("{found} (showing first {}):\n", rendered.shown)
        } else {
            format!("{found}:\n")
        };
        output.push_str(&rendered.lines.join("\n"));
        for note in rendered.note.into_iter().chain(skipped_note) {
            output.push('\n');
            output.push_str(&note);
        }

        Ok(ToolResult::success(output))
    }
}

fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

/// The note for results dropped as minified or bundled. The search behind
/// `grep` may stop at a result cap before it reaches other files, so the
/// note never claims that nothing else matched.
fn minified_note(skipped: usize, returned: usize) -> String {
    format!(
        "{skipped} of the {} the search returned {} in minified or bundled files \
         (*.min.js, source maps, ...) and {} skipped. The search stops at a result cap, so \
         it may not have reached other files: narrow `path` to the source directory, or set \
         include_minified: true to see the skipped results.",
        plural(returned, "result", "results"),
        if skipped == 1 { "was" } else { "were" },
        if skipped == 1 { "was" } else { "were" },
    )
}

struct RenderedMatches {
    lines: Vec<String>,
    bytes: usize,
    shown: usize,
    note: Option<String>,
    /// Lines up to the last shown match and its trailing context; a render
    /// cut at a cap ends there, never on context for a match it dropped.
    complete_len: usize,
}

impl RenderedMatches {
    /// Append `line` unless it would pass [`MAX_OUTPUT_LINES`] or
    /// [`MAX_OUTPUT_BYTES`].
    fn push(&mut self, line: String) -> bool {
        let bytes = self.bytes + line.len() + 1;
        if self.lines.len() >= MAX_OUTPUT_LINES || bytes > MAX_OUTPUT_BYTES {
            return false;
        }
        self.bytes = bytes;
        self.lines.push(line);
        true
    }

    fn cut(mut self) -> Self {
        self.lines.truncate(self.complete_len);
        self
    }
}

impl<E: Environment> GrepTool<E> {
    /// Render matches as `path:line:text`, with `context` lines on each side
    /// as `path-line-text` and `--` between ranges that do not touch, within
    /// [`MAX_OUTPUT_LINES`] and [`MAX_OUTPUT_BYTES`]. A file whose text cannot
    /// be read back, or that comes after `read_budget` bytes were read back,
    /// keeps its bare hits.
    async fn render_matches(
        &self,
        matches: &[crate::environment::GrepMatch],
        context: usize,
        read_budget: usize,
    ) -> RenderedMatches {
        let mut files: Vec<(&str, Vec<&crate::environment::GrepMatch>)> = Vec::new();
        for m in matches {
            match files.iter_mut().find(|(path, _)| *path == m.path) {
                Some((_, hits)) => hits.push(m),
                None => files.push((m.path.as_str(), vec![m])),
            }
        }

        let mut rendered = RenderedMatches {
            lines: Vec::new(),
            bytes: 0,
            shown: 0,
            note: None,
            complete_len: 0,
        };
        let mut read_bytes = 0_usize;
        for (path, mut hits) in files {
            hits.sort_by_key(|m| m.line_number);
            let text = if context == 0 {
                None
            } else if read_bytes > read_budget {
                rendered.note.get_or_insert_with(|| {
                    "Context was left out for later files: the context read budget ran out."
                        .to_owned()
                });
                None
            } else {
                let text = self.ctx.environment.read_file(path).await.ok();
                read_bytes = read_bytes.saturating_add(text.as_ref().map_or(0, String::len));
                text
            };
            let file_lines: Vec<&str> = text
                .as_deref()
                .map_or_else(Vec::new, |t| t.lines().collect());
            let width = if file_lines.is_empty() { 0 } else { context };

            let mut ranges: Vec<(usize, usize, Vec<&crate::environment::GrepMatch>)> = Vec::new();
            for m in hits {
                let start = m.line_number.saturating_sub(width).max(1);
                let end = m
                    .line_number
                    .saturating_add(width)
                    .min(file_lines.len().max(m.line_number));
                match ranges.last_mut() {
                    Some((_, last_end, group))
                        if width > 0 && start <= last_end.saturating_add(1) =>
                    {
                        *last_end = (*last_end).max(end);
                        group.push(m);
                    }
                    _ => ranges.push((start, end, vec![m])),
                }
            }

            let mut last_hit: Option<usize> = None;
            for (start, end, group) in ranges {
                if width > 0 && !rendered.lines.is_empty() && !rendered.push("--".to_owned()) {
                    return rendered.cut();
                }
                for line_number in start..=end {
                    let hit = group.iter().find(|m| m.line_number == line_number);
                    let line = hit.map_or_else(
                        || {
                            file_lines.get(line_number - 1).map(|text| {
                                format!(
                                    "{path}-{line_number}-{}",
                                    truncate_line(text, MAX_LINE_CHARS)
                                )
                            })
                        },
                        |hit| {
                            Some(format!(
                                "{path}:{line_number}:{}",
                                truncate_line(&hit.line_content, MAX_LINE_CHARS)
                            ))
                        },
                    );
                    let is_hit = hit.is_some();
                    let Some(line) = line else {
                        continue;
                    };
                    if !rendered.push(line) {
                        return rendered.cut();
                    }
                    if is_hit {
                        rendered.shown += 1;
                        last_hit = Some(line_number);
                    }
                    if last_hit.is_some_and(|hit| line_number <= hit.saturating_add(width)) {
                        rendered.complete_len = rendered.lines.len();
                    }
                }
            }
        }
        rendered
    }
}

/// Cut a line to `max_len` bytes. Leading whitespace is kept, so a line
/// copied from the output into an `edit` still matches the file.
fn truncate_line(s: &str, max_len: usize) -> String {
    let trimmed = s.trim_end();
    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        format!("{}...", super::truncate_str(trimmed, max_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn parse_grep_input(input: Value) -> Result<GrepInput, serde_json::Error> {
        GrepInput::deserialize(input)
    }

    #[test]
    fn grep_input_treats_null_flags_as_their_defaults() -> anyhow::Result<()> {
        let input = parse_grep_input(json!({
            "pattern": "x",
            "path": null,
            "recursive": null,
            "case_insensitive": null
        }))?;
        assert_eq!(input.path, None);
        assert!(input.recursive);
        assert!(!input.case_insensitive);
        Ok(())
    }

    #[test]
    fn grep_input_keeps_explicit_and_absent_flags() -> anyhow::Result<()> {
        let absent = parse_grep_input(json!({"pattern": "x"}))?;
        assert!(absent.recursive);
        assert!(!absent.case_insensitive);
        let explicit = parse_grep_input(json!({
            "pattern": "x",
            "recursive": false,
            "case_insensitive": true
        }))?;
        assert!(!explicit.recursive);
        assert!(explicit.case_insensitive);
        Ok(())
    }

    #[test]
    fn grep_input_still_rejects_non_boolean_flags() {
        for field in ["recursive", "case_insensitive"] {
            let mut input = json!({"pattern": "x"});
            input[field] = json!("yes");
            let error = parse_grep_input(input)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(error.contains("expected a boolean"), "{field}: {error}");
        }
    }

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> GrepTool<InMemoryFileSystem> {
        GrepTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    // ===================
    // Unit Tests
    // ===================

    #[tokio::test]
    async fn test_grep_simple_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.rs", "fn main() {\n    println!(\"Hello\");\n}")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "println"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("println"));
        assert!(result.output.contains(":2:")); // Line number
        Ok(())
    }

    #[tokio::test]
    async fn registered_grep_accepts_boolean_strings() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "WHITE_LISTED\nother").await?;
        let mut registry = crate::ToolRegistry::new();
        registry.register(create_test_tool(fs, AgentCapabilities::full_access()));
        let tool = registry
            .get("grep")
            .ok_or_else(|| anyhow::anyhow!("grep registered"))?;

        let result = tool
            .execute(
                &tool_ctx(),
                json!({"pattern": "white_listed", "case_insensitive": "true"}),
            )
            .await?;

        assert!(result.success, "{}", result.output);
        assert!(
            result.output.contains("Found 1 matches"),
            "{}",
            result.output
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_regex_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo123\nbar456\nfoo789").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "foo\\d+"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_no_matches() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "Hello, World!").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "Rust"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No matches found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // Use ASCII-only text since unicode-case feature may not be enabled
        fs.write_file("test.txt", "Hello\nHELLO\nhello").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        // Use ASCII-only case-insensitive pattern (regex supports (?i-u) for ASCII)
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "[Hh][Ee][Ll][Ll][Oo]"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 3 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_with_path() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("tests/test.rs", "fn test() {}").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"pattern": "fn", "path": "/workspace/src"}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("main.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_non_recursive() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("file.txt", "match here").await?;
        fs.write_file("subdir/nested.txt", "match nested").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match", "recursive": false}))
            .await?;

        assert!(result.success);
        // Should only find the top-level file
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("file.txt"));
        Ok(())
    }

    // ===================
    // Integration Tests
    // ===================

    #[tokio::test]
    async fn test_grep_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "content").await?;

        // No read permission
        let caps = AgentCapabilities::none();

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "content"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_filters_inaccessible_files() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("secrets/key.txt", "fn secret() {}").await?;

        // Allow src but deny secrets
        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool.execute(&tool_ctx(), json!({"pattern": "fn"})).await?;

        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        assert!(result.output.contains("main.rs"));
        assert!(!result.output.contains("key.txt"));
        Ok(())
    }

    // ===================
    // Edge Cases
    // ===================

    #[tokio::test]
    async fn test_grep_empty_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("empty.txt", "").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "anything"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("No matches found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_many_matches_truncated() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        // Create file with many matching lines
        let content: String = (1..=100)
            .map(|i| format!("match line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("many.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Found 100 matches"));
        assert!(result.output.contains("showing first 50"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_special_regex_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "foo.bar\nbaz*qux\n(parens)")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Escaped dot
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "foo\\.bar"}))
            .await?;
        assert!(result.success);
        assert!(result.output.contains("Found 1 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_multiple_files() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib.rs", "fn lib() {}").await?;
        fs.write_file("README.md", "# README").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool.execute(&tool_ctx(), json!({"pattern": "fn"})).await?;

        assert!(result.success);
        assert!(result.output.contains("Found 2 matches"));
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::Grep);
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
        assert_eq!(
            Tool::<()>::concurrency(&tool),
            crate::ToolConcurrency::ReadOnly
        );
        assert!(Tool::<()>::description(&tool).contains("Search"));

        let schema = Tool::<()>::input_schema(&tool);
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("pattern").is_some());
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("recursive").is_some());
        assert!(schema["properties"].get("case_insensitive").is_some());
    }

    #[tokio::test]
    async fn test_grep_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // Missing required pattern field
        let result = tool.execute(&tool_ctx(), json!({})).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_grep_invalid_pattern() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "content").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        // An unbalanced char class is invalid; the model should get a
        // correctable tool error, not an infrastructure failure.
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "[unclosed"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Invalid pattern"));
        Ok(())
    }

    #[tokio::test]
    async fn test_truncate_line_function() {
        assert_eq!(truncate_line("short", 10), "short");
        assert_eq!(truncate_line("  indented  ", 10), "  indented");
        assert_eq!(truncate_line("this is a longer line", 10), "this is a ...");
    }

    #[tokio::test]
    async fn test_grep_long_line_truncated() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let long_line = "match ".to_string() + &"x".repeat(300);
        fs.write_file("long.txt", &long_line).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"pattern": "match"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("..."));
        Ok(())
    }

    async fn grep_output(fs: Arc<InMemoryFileSystem>, input: Value) -> anyhow::Result<String> {
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool.execute(&tool_ctx(), input).await?;
        assert!(result.success, "{}", result.output);
        Ok(result.output)
    }

    #[tokio::test]
    async fn grep_shows_two_lines_of_context_by_default_and_merges_overlaps() -> anyhow::Result<()>
    {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let mut body: Vec<String> = (1..=14).map(|n| format!("line {n}")).collect();
        body[3] = "hit four".to_owned();
        body[5] = "hit six".to_owned();
        body[11] = "hit twelve".to_owned();
        fs.write_file("src.txt", &body.join("\n")).await?;

        let output = grep_output(fs, json!({"pattern": "hit"})).await?;

        assert_eq!(
            output,
            "Found 3 matches:\n\
             /workspace/src.txt-2-line 2\n\
             /workspace/src.txt-3-line 3\n\
             /workspace/src.txt:4:hit four\n\
             /workspace/src.txt-5-line 5\n\
             /workspace/src.txt:6:hit six\n\
             /workspace/src.txt-7-line 7\n\
             /workspace/src.txt-8-line 8\n\
             --\n\
             /workspace/src.txt-10-line 10\n\
             /workspace/src.txt-11-line 11\n\
             /workspace/src.txt:12:hit twelve\n\
             /workspace/src.txt-13-line 13\n\
             /workspace/src.txt-14-line 14"
        );
        Ok(())
    }

    #[tokio::test]
    async fn grep_context_zero_prints_bare_hits() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("a.txt", "one\nhit\nthree").await?;
        fs.write_file("b.txt", "hit").await?;

        let output = grep_output(fs, json!({"pattern": "hit", "context": 0})).await?;

        assert!(
            !output.contains("one") && !output.contains("--"),
            "{output}"
        );
        assert!(output.contains("/workspace/a.txt:2:hit"), "{output}");
        assert!(output.contains("/workspace/b.txt:1:hit"), "{output}");
        Ok(())
    }

    #[tokio::test]
    async fn grep_skips_minified_and_bundled_files_unless_asked() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/list.ts", "export const WHITE_LISTED = [];")
            .await?;
        fs.write_file("dist/index.js.map", "{\"WHITE_LISTED\":1}")
            .await?;
        fs.write_file("dist/app.min.js", "var WHITE_LISTED=1")
            .await?;

        let output = grep_output(Arc::clone(&fs), json!({"pattern": "WHITE_LISTED"})).await?;
        assert!(
            output.starts_with("Found 1 matches outside minified or bundled files:"),
            "{output}"
        );
        assert!(output.contains("/workspace/src/list.ts:1:"), "{output}");
        assert!(!output.contains("dist/"), "{output}");
        assert!(
            output.contains("2 of the 3 results the search returned were in minified or bundled")
                && output.contains("narrow `path`")
                && output.contains("include_minified: true"),
            "{output}"
        );

        let output = grep_output(
            Arc::clone(&fs),
            json!({"pattern": "WHITE_LISTED", "include_minified": true}),
        )
        .await?;
        assert!(output.starts_with("Found 3 matches:"), "{output}");
        assert!(!output.contains("skipped"), "{output}");

        let output = grep_output(fs, json!({"pattern": "nothing-matches-this"})).await?;
        assert!(output.starts_with("No matches found"), "{output}");
        Ok(())
    }

    #[tokio::test]
    async fn only_minified_results_never_read_as_no_matches() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("dist/chunk.min.js", "renderWidget()").await?;

        let output = grep_output(fs, json!({"pattern": "renderWidget"})).await?;

        assert!(!output.contains("No matches found"), "{output}");
        assert!(
            output.starts_with("No matches outside minified or bundled files")
                && output.contains("1 of the 1 result the search returned was in minified")
                && output.contains("result cap"),
            "{output}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_minified_file_named_as_the_path_is_searched() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("vendor/lib.min.js", "var needle=1").await?;

        let output = grep_output(
            fs,
            json!({"pattern": "needle", "path": "vendor/lib.min.js"}),
        )
        .await?;

        assert!(output.starts_with("Found 1 matches:"), "{output}");
        assert!(
            output.contains("/workspace/vendor/lib.min.js:1:"),
            "{output}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn context_keeps_indentation() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file(
            "a.py",
            "def f(x):\n    if x:\n        return needle\n    else:\n        pass",
        )
        .await?;

        let output = grep_output(fs, json!({"pattern": "needle"})).await?;

        assert!(output.contains("/workspace/a.py-2-    if x:"), "{output}");
        assert!(
            output.contains("/workspace/a.py:3:        return needle"),
            "{output}"
        );
        assert!(output.contains("/workspace/a.py-4-    else:"), "{output}");
        Ok(())
    }

    #[tokio::test]
    async fn context_is_clamped_to_ten_lines() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let body: Vec<String> = (1..=40)
            .map(|n| {
                if n == 20 {
                    "hit".to_owned()
                } else {
                    format!("line {n}")
                }
            })
            .collect();
        fs.write_file("c.txt", &body.join("\n")).await?;

        let output = grep_output(fs, json!({"pattern": "hit", "context": 1000})).await?;

        assert_eq!(output.lines().count(), 1 + 21, "{output}");
        assert!(output.contains("/workspace/c.txt-10-line 10"), "{output}");
        assert!(!output.contains("/workspace/c.txt-9-"), "{output}");
        Ok(())
    }

    #[tokio::test]
    async fn context_reads_stop_at_the_read_budget() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("a.txt", "before\nhit\nafter").await?;
        fs.write_file("b.txt", "before\nhit\nafter").await?;
        let matches = fs.grep("hit", "/workspace", true).await?;
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        let rendered = tool.render_matches(&matches, 2, 1).await;

        assert_eq!(rendered.shown, 2);
        let context_lines = rendered
            .lines
            .iter()
            .filter(|line| line.ends_with("-before"))
            .count();
        assert_eq!(context_lines, 1, "{:?}", rendered.lines);
        assert!(
            rendered
                .note
                .as_deref()
                .is_some_and(|note| note.contains("read budget")),
            "{:?}",
            rendered.note
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_cut_render_never_ends_on_the_next_files_leading_context() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // 33 separate five-line ranges and their separators fill 197 lines;
        // the cap then falls inside b's leading context (`--`, b 1, b 2).
        let a: Vec<String> = (1..=400)
            .map(|n| {
                if n % 10 == 5 && n < 330 {
                    format!("hit {n}")
                } else {
                    format!("fill {n}")
                }
            })
            .collect();
        fs.write_file("a.txt", &a.join("\n")).await?;
        fs.write_file(
            "b.txt",
            "b one\nb two\nhit in b\nb four\nb five\nb six\nb seven",
        )
        .await?;
        let mut matches = fs.grep("hit", "/workspace/a.txt", true).await?;
        matches.extend(fs.grep("hit", "/workspace/b.txt", true).await?);
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        let rendered = tool.render_matches(&matches, 2, usize::MAX).await;

        let last = rendered.lines.last().cloned().unwrap_or_default();
        assert!(last.starts_with("/workspace/a.txt-"), "{last}");
        assert!(
            rendered.lines.iter().all(|line| !line.contains("b.txt")),
            "{last}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn output_stays_within_the_byte_bound() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let long = "x".repeat(190);
        let body: Vec<String> = (0..50)
            .map(|n| format!("hit {n} {long}\n{long}\n{long}"))
            .collect();
        fs.write_file(
            "a/deeply/nested/directory/with/a/long/name/for/the/output/budget/file.txt",
            &body.join("\n"),
        )
        .await?;

        let output = grep_output(fs, json!({"pattern": "hit"})).await?;

        assert!(output.len() <= MAX_OUTPUT_BYTES + 200, "{}", output.len());
        assert!(output.contains("(showing first "), "{}", &output[..200]);
        Ok(())
    }

    #[tokio::test]
    async fn grep_caps_output_lines_when_context_widens_it() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let body: Vec<String> = (0..60)
            .map(|n| format!("hit {n}\nfill\nfill\nfill\nfill"))
            .collect();
        fs.write_file("many.txt", &body.join("\n")).await?;

        let output = grep_output(fs, json!({"pattern": "hit"})).await?;

        assert!(output.lines().count() <= MAX_OUTPUT_LINES + 1, "{output}");
        let line_number = |line: &str| -> usize {
            line.trim_start_matches("/workspace/many.txt")
                .trim_start_matches([':', '-'])
                .split([':', '-'])
                .next()
                .and_then(|number| number.parse().ok())
                .unwrap_or_default()
        };
        let last_hit = output
            .lines()
            .filter(|line| line.starts_with("/workspace/many.txt:"))
            .map(line_number)
            .max()
            .unwrap_or_default();
        let last_line = output.lines().last().map(line_number).unwrap_or_default();
        assert_eq!(
            last_line,
            last_hit + 2,
            "a cut render ends on its last match's context: {output}"
        );
        assert!(output.contains("/workspace/many.txt:1:hit 0"), "{output}");
        let shown_hits = output
            .lines()
            .filter(|line| line.starts_with("/workspace/many.txt:"))
            .count();
        assert!(shown_hits > 0, "{output}");
        assert!(
            output.starts_with(&format!("Found 60 matches (showing first {shown_hits}):")),
            "{output}"
        );
        Ok(())
    }

    #[test]
    fn grep_input_defaults_and_bounds_context() -> anyhow::Result<()> {
        assert_eq!(parse_grep_input(json!({"pattern": "x"}))?.context, 2);
        assert_eq!(
            parse_grep_input(json!({"pattern": "x", "context": null}))?.context,
            2
        );
        assert_eq!(
            parse_grep_input(json!({"pattern": "x", "context": 0}))?.context,
            0
        );
        Ok(())
    }
}
