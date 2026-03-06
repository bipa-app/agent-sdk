use crate::llm::ContentSource;
use crate::reminders::{append_reminder, builtin};
use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Maximum tokens allowed per file read (approximately 4 chars per token)
const MAX_TOKENS: usize = 25_000;
const CHARS_PER_TOKEN: usize = 4;

/// Tool for reading file contents
pub struct ReadTool<E: Environment> {
    ctx: PrimitiveToolContext<E>,
}

impl<E: Environment> ReadTool<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: crate::AgentCapabilities) -> Self {
        Self {
            ctx: PrimitiveToolContext::new(environment, capabilities),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadInput {
    /// Path to the file to read (also accepts `file_path` for compatibility)
    #[serde(alias = "file_path")]
    path: String,
    /// Optional line offset to start from (1-based)
    #[serde(default)]
    offset: Option<usize>,
    /// Optional number of lines to read
    #[serde(default)]
    limit: Option<usize>,
}

enum ReadContent {
    Text(String),
    NativeBinary { mime_type: &'static str },
    UnsupportedBinary,
}

impl<E: Environment + 'static> Tool<()> for ReadTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Read
    }

    fn display_name(&self) -> &'static str {
        "Read File"
    }

    fn description(&self) -> &'static str {
        "Read text files directly, and attach supported images/PDFs for native model inspection. Can optionally specify offset and limit for text files."
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Observe
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (1-based). Optional. Only applies to text files."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Optional. Only applies to text files."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: ReadInput =
            serde_json::from_value(input).context("Invalid input for read tool")?;

        let path = self.ctx.environment.resolve_path(&input.path);

        if !self.ctx.capabilities.can_read(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot read '{path}'"
            )));
        }

        let exists = self
            .ctx
            .environment
            .exists(&path)
            .await
            .context("Failed to check file existence")?;

        if !exists {
            return Ok(ToolResult::error(format!("File not found: '{path}'")));
        }

        let is_dir = self
            .ctx
            .environment
            .is_dir(&path)
            .await
            .context("Failed to check if path is directory")?;

        if is_dir {
            return Ok(ToolResult::error(format!(
                "'{path}' is a directory, not a file"
            )));
        }

        let bytes = self
            .ctx
            .environment
            .read_file_bytes(&path)
            .await
            .context("Failed to read file")?;

        let mut result = match classify_content(&path, &bytes) {
            ReadContent::Text(content) => {
                read_text_content(&path, &content, input.offset, input.limit)
            }
            ReadContent::NativeBinary { mime_type } => {
                if input.offset.is_some() || input.limit.is_some() {
                    ToolResult::error(format!(
                        "offset and limit are only supported for text files. '{path}' is a {mime_type} file."
                    ))
                } else {
                    ToolResult::success(format!(
                        "Attached '{path}' ({mime_type}, {} bytes) for native model inspection.",
                        bytes.len()
                    ))
                    .with_documents(vec![ContentSource::new(
                        mime_type,
                        base64::engine::general_purpose::STANDARD.encode(&bytes),
                    )])
                }
            }
            ReadContent::UnsupportedBinary => ToolResult::error(format!(
                "'{path}' is a binary file in an unsupported format. The read tool currently supports text files, images (PNG/JPEG/GIF/WebP), and PDF documents."
            )),
        };

        if result.success && result.output == "(empty file)" {
            append_reminder(&mut result, builtin::READ_EMPTY_FILE_REMINDER);
        }

        if result.success {
            append_reminder(&mut result, builtin::READ_SECURITY_REMINDER);
        }

        Ok(result)
    }
}

fn read_text_content(
    path: &str,
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> ToolResult {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let offset = offset.unwrap_or(1).saturating_sub(1);
    let selected_lines: Vec<&str> = lines.iter().copied().skip(offset).collect();

    let limit = if let Some(user_limit) = limit {
        user_limit
    } else {
        let selected_content_len: usize = selected_lines.iter().map(|line| line.len() + 1).sum();
        let estimated_tokens = selected_content_len / CHARS_PER_TOKEN;

        if estimated_tokens > MAX_TOKENS {
            let suggested_limit = estimate_lines_for_tokens(&selected_lines, MAX_TOKENS);
            return ToolResult::success(format!(
                "File too large to read at once (~{estimated_tokens} tokens, max {MAX_TOKENS}).\n\
                 Total lines: {total_lines}\n\n\
                 Use 'offset' and 'limit' parameters to read specific portions.\n\
                 Suggested: Start with offset=1, limit={suggested_limit} to read the first ~{MAX_TOKENS} tokens.\n\n\
                 Example: {{\"path\": \"{path}\", \"offset\": 1, \"limit\": {suggested_limit}}}"
            ));
        }

        selected_lines.len()
    };

    let selected_lines: Vec<String> = lines
        .into_iter()
        .skip(offset)
        .take(limit)
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", offset + i + 1, line))
        .collect();

    let is_empty = selected_lines.is_empty();
    let output = if is_empty {
        "(empty file)".to_string()
    } else {
        let header = if offset > 0 || limit < total_lines {
            format!(
                "Showing lines {}-{} of {} total\n",
                offset + 1,
                (offset + selected_lines.len()).min(total_lines),
                total_lines
            )
        } else {
            String::new()
        };
        format!("{header}{}", selected_lines.join("\n"))
    };

    ToolResult::success(output)
}

fn classify_content(path: &str, bytes: &[u8]) -> ReadContent {
    if let Some(mime_type) = detect_native_binary_mime(path, bytes) {
        return ReadContent::NativeBinary { mime_type };
    }

    if let Ok(content) = std::str::from_utf8(bytes) {
        return ReadContent::Text(content.to_string());
    }

    ReadContent::UnsupportedBinary
}

fn detect_native_binary_mime(path: &str, bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"%PDF-") {
        return Some("application/pdf");
    }

    if bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']) {
        return Some("image/png");
    }

    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }

    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }

    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }

    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("pdf") => Some("application/pdf"),
        Some("png") => Some("image/png"),
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        _ => None,
    }
}

/// Estimate how many lines can fit within a token budget
fn estimate_lines_for_tokens(lines: &[&str], max_tokens: usize) -> usize {
    let max_chars = max_tokens * CHARS_PER_TOKEN;
    let mut total_chars = 0;
    let mut line_count = 0;

    for line in lines {
        let line_chars = line.len() + 1;
        if total_chars + line_chars > max_chars {
            break;
        }
        total_chars += line_chars;
        line_count += 1;
    }

    line_count.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentCapabilities, InMemoryFileSystem};

    fn create_test_tool(
        fs: Arc<InMemoryFileSystem>,
        capabilities: AgentCapabilities,
    ) -> ReadTool<InMemoryFileSystem> {
        ReadTool::new(fs, capabilities)
    }

    fn tool_ctx() -> ToolContext<()> {
        ToolContext::new(())
    }

    #[tokio::test]
    async fn test_read_entire_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("line 1"));
        assert!(result.output.contains("line 2"));
        assert!(result.output.contains("line 3"));
        assert!(result.documents.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_offset() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 3}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 3-5 of 5 total"));
        assert!(result.output.contains("line 3"));
        assert!(result.output.contains("line 4"));
        assert!(result.output.contains("line 5"));
        assert!(!result.output.contains("\tline 1"));
        assert!(!result.output.contains("\tline 2"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 1-2 of 5 total"));
        assert!(result.output.contains("line 1"));
        assert!(result.output.contains("line 2"));
        assert!(!result.output.contains("\tline 3"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\nline 5")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 2, "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 2-3 of 5 total"));
        assert!(result.output.contains("line 2"));
        assert!(result.output.contains("line 3"));
        assert!(!result.output.contains("\tline 1"));
        assert!(!result.output.contains("\tline 4"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/nonexistent.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("File not found"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_directory_returns_error() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.create_dir("/workspace/subdir").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/subdir"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("is a directory"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secret.txt", "secret content").await?;

        let tool = create_test_tool(fs, AgentCapabilities::none());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/secret.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_denied_path_via_capabilities() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secrets/api_key.txt", "API_KEY=secret")
            .await?;

        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/secrets/api_key.txt"}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_allowed_path_restriction() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("src/main.rs", "fn main() {} ").await?;
        fs.write_file("config/settings.toml", "key = value").await?;

        let caps = AgentCapabilities::read_only()
            .with_denied_paths(vec![])
            .with_allowed_paths(vec!["/workspace/src/**".into()]);

        let tool = create_test_tool(Arc::clone(&fs), caps.clone());

        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/src/main.rs"}))
            .await?;
        assert!(result.success);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/config/settings.toml"}),
            )
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_empty_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("empty.txt", "").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/empty.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("(empty file)"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_with_pagination() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content: String = (1..=100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("large.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/large.txt", "offset": 50, "limit": 10}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 50-59 of 100 total"));
        assert!(result.output.contains("line 50"));
        assert!(result.output.contains("line 59"));
        assert!(!result.output.contains("\tline 49"));
        assert!(!result.output.contains("\tline 60"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_offset_beyond_file_length() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("short.txt", "line 1\nline 2").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/short.txt", "offset": 100}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("(empty file)"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_file_with_special_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content = "特殊字符\néàü\n🎉emoji\ntab\there";
        fs.write_file("special.txt", content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/special.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("特殊字符"));
        assert!(result.output.contains("éàü"));
        assert!(result.output.contains("🎉emoji"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_image_file_attaches_native_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let png = vec![
            0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 1, 2, 3, 4,
        ];
        fs.write_file_bytes("image.png", &png).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/image.png"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Attached '/workspace/image.png'"));
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].media_type, "image/png");
        Ok(())
    }

    #[tokio::test]
    async fn test_read_pdf_file_attaches_native_content() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes("doc.pdf", b"%PDF-1.7\nbody").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/doc.pdf"}))
            .await?;

        assert!(result.success);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].media_type, "application/pdf");
        Ok(())
    }

    #[tokio::test]
    async fn test_read_binary_with_offset_returns_error() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes("doc.pdf", b"%PDF-1.7\nbody").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/doc.pdf", "offset": 1}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("only supported for text files"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_unsupported_binary_returns_error() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes("archive.bin", &[0, 159, 146, 150])
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/archive.bin"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("unsupported format"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), PrimitiveToolName::Read);
        assert_eq!(tool.tier(), ToolTier::Observe);
        assert!(tool.description().contains("Read"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("path").is_some());
    }

    #[tokio::test]
    async fn test_read_invalid_input() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        let result = tool.execute(&tool_ctx(), json!({})).await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_exceeds_token_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let line = "x".repeat(100);
        let content: String = (1..=1500)
            .map(|i| format!("{i}: {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("huge.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/huge.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("File too large to read at once"));
        assert!(result.output.contains("Total lines: 1500"));
        assert!(result.output.contains("offset"));
        assert!(result.output.contains("limit"));
        Ok(())
    }

    #[tokio::test]
    async fn test_read_large_file_with_explicit_limit_bypasses_check() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let line = "x".repeat(100);
        let content: String = (1..=1500)
            .map(|i| format!("{i}: {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("huge.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/huge.txt", "offset": 1, "limit": 10}),
            )
            .await?;

        assert!(result.success);
        assert!(result.output.contains("Showing lines 1-10 of 1500 total"));
        assert!(!result.output.contains("File too large"));
        Ok(())
    }

    #[test]
    fn test_estimate_lines_for_tokens() {
        let long = "x".repeat(100);
        let lines: Vec<&str> = vec!["short line", "another short line", &long];

        let count = estimate_lines_for_tokens(&lines, 10);
        assert_eq!(count, 2);

        let count = estimate_lines_for_tokens(&lines, 1);
        assert_eq!(count, 1);
    }
}
