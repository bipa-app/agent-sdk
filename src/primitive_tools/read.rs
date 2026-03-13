use crate::llm::ContentSource;
use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Maximum characters per line before truncation.
const MAX_LINE_LENGTH: usize = 500;

/// Default maximum number of lines to return.
const DEFAULT_LIMIT: usize = 2000;

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
    #[serde(alias = "file_path")]
    path: String,
    /// 1-indexed line number to start reading from; defaults to 1.
    #[serde(
        default = "defaults::offset",
        deserialize_with = "super::deserialize_usize_from_string_or_int"
    )]
    offset: usize,
    /// Maximum number of lines to return; defaults to 2000.
    #[serde(
        default = "defaults::limit",
        deserialize_with = "super::deserialize_usize_from_string_or_int"
    )]
    limit: usize,
}

mod defaults {
    pub const fn offset() -> usize {
        1
    }
    pub const fn limit() -> usize {
        super::DEFAULT_LIMIT
    }
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
        "Read text files with 1-indexed line numbers. Also supports images (PNG/JPEG/GIF/WebP) and PDF documents."
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
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string", "pattern": "^[0-9]+$"}
                    ],
                    "description": "Line number to start from (1-based). Accepts either an integer or a numeric string. Default: 1"
                },
                "limit": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string", "pattern": "^[0-9]+$"}
                    ],
                    "description": "Maximum number of lines to return. Accepts either an integer or a numeric string. Default: 2000"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let input: ReadInput = serde_json::from_value(input.clone())
            .with_context(|| format!("Invalid input for read tool: {input}"))?;

        if input.offset == 0 {
            return Ok(ToolResult::error("offset must be a 1-indexed line number"));
        }

        if input.limit == 0 {
            return Ok(ToolResult::error("limit must be greater than zero"));
        }

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

        // Handle images and PDFs as document attachments (like codex-rs view_image).
        if let Some(media_type) = detect_media_type(&path) {
            let encoded = base64_encode(&bytes);
            return Ok(
                ToolResult::success(format!("Read {media_type} file: '{path}'"))
                    .with_documents(vec![ContentSource::new(media_type, encoded)]),
            );
        }

        // Text files: lossy UTF-8, line numbers, truncation.
        let content = String::from_utf8_lossy(&bytes);
        let collected = read_lines(&content, input.offset, input.limit);

        if collected.is_empty() {
            return Ok(ToolResult::error("offset exceeds file length"));
        }

        Ok(ToolResult::success(collected.join("\n")))
    }
}

fn read_lines(content: &str, offset: usize, limit: usize) -> Vec<String> {
    let mut collected = Vec::new();
    let mut line_number = 0usize;

    for raw_line in content.split('\n') {
        line_number += 1;

        if line_number < offset {
            continue;
        }

        if collected.len() >= limit {
            break;
        }

        // Strip trailing \r for CRLF files
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let display = truncate_line(line);
        collected.push(format!("L{line_number}: {display}"));
    }

    collected
}

fn truncate_line(line: &str) -> &str {
    if line.len() <= MAX_LINE_LENGTH {
        line
    } else {
        // Find the nearest char boundary at or before MAX_LINE_LENGTH
        let mut end = MAX_LINE_LENGTH;
        while end > 0 && !line.is_char_boundary(end) {
            end -= 1;
        }
        &line[..end]
    }
}

/// Detect supported binary media types by file extension.
fn detect_media_type(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()?
        .to_ascii_lowercase();

    match ext.to_str()? {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "pdf" => Some("application/pdf"),
        _ => None,
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
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
    async fn reads_entire_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha\nbeta\ngamma").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L1: alpha\nL2: beta\nL3: gamma");
        Ok(())
    }

    #[tokio::test]
    async fn reads_with_offset() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha\nbeta\ngamma").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 2}),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L2: beta\nL3: gamma");
        Ok(())
    }

    #[tokio::test]
    async fn reads_with_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha\nbeta\ngamma").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L1: alpha\nL2: beta");
        Ok(())
    }

    #[tokio::test]
    async fn reads_with_offset_and_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha\nbeta\ngamma\ndelta\nepsilon")
            .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 2, "limit": 2}),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L2: beta\nL3: gamma");
        Ok(())
    }

    #[tokio::test]
    async fn accepts_string_offset_and_limit() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha\nbeta\ngamma").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": "2", "limit": "1"}),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L2: beta");
        Ok(())
    }

    #[tokio::test]
    async fn errors_on_offset_zero() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "offset": 0}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("1-indexed"));
        Ok(())
    }

    #[tokio::test]
    async fn errors_on_limit_zero() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "alpha").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/test.txt", "limit": 0}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("greater than zero"));
        Ok(())
    }

    #[tokio::test]
    async fn errors_when_offset_exceeds_length() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("short.txt", "only").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/short.txt", "offset": 100}),
            )
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("offset exceeds file length"));
        Ok(())
    }

    #[tokio::test]
    async fn errors_on_nonexistent_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/nope.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("File not found"));
        Ok(())
    }

    #[tokio::test]
    async fn errors_on_directory() -> anyhow::Result<()> {
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
    async fn errors_on_permission_denied() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secret.txt", "secret").await?;

        let tool = create_test_tool(fs, AgentCapabilities::none());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/secret.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn respects_denied_paths() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("secrets/key.txt", "API_KEY=secret").await?;

        let caps =
            AgentCapabilities::read_only().with_denied_paths(vec!["/workspace/secrets/**".into()]);

        let tool = create_test_tool(fs, caps);
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/secrets/key.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("Permission denied"));
        Ok(())
    }

    #[tokio::test]
    async fn handles_crlf_line_endings() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes("crlf.txt", b"one\r\ntwo\r\n").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/crlf.txt"}))
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L1: one\nL2: two\nL3: ");
        Ok(())
    }

    #[tokio::test]
    async fn handles_non_utf8() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes(
            "bin.txt",
            &[0xff, 0xfe, b'\n', b'p', b'l', b'a', b'i', b'n', b'\n'],
        )
        .await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/bin.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.output.contains("L2: plain"));
        Ok(())
    }

    #[tokio::test]
    async fn truncates_long_lines() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let long_line = "x".repeat(MAX_LINE_LENGTH + 50);
        fs.write_file("long.txt", &long_line).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/long.txt"}))
            .await?;

        assert!(result.success);
        let expected = "x".repeat(MAX_LINE_LENGTH);
        assert_eq!(result.output, format!("L1: {expected}"));
        Ok(())
    }

    #[tokio::test]
    async fn handles_special_characters() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("special.txt", "特殊字符\néàü\n🎉emoji")
            .await?;

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
    async fn respects_limit_with_more_lines() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let content: String = (1..=100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs.write_file("many.txt", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/many.txt", "offset": 50, "limit": 3}),
            )
            .await?;

        assert!(result.success);
        assert_eq!(result.output, "L50: line 50\nL51: line 51\nL52: line 52");
        Ok(())
    }

    #[tokio::test]
    async fn tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(tool.name(), PrimitiveToolName::Read);
        assert_eq!(tool.tier(), ToolTier::Observe);

        let schema = tool.input_schema();
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("offset").is_some());
        assert!(schema["properties"].get("limit").is_some());
    }

    #[test]
    fn read_lines_basic() {
        let lines = read_lines("alpha\nbeta\ngamma", 1, 2000);
        assert_eq!(
            lines,
            vec![
                "L1: alpha".to_string(),
                "L2: beta".to_string(),
                "L3: gamma".to_string(),
            ]
        );
    }

    #[test]
    fn read_lines_with_offset_and_limit() {
        let lines = read_lines("a\nb\nc\nd\ne", 2, 2);
        assert_eq!(lines, vec!["L2: b".to_string(), "L3: c".to_string()]);
    }

    #[test]
    fn read_lines_offset_past_end_returns_empty() {
        let lines = read_lines("only", 5, 10);
        assert!(lines.is_empty());
    }

    #[test]
    fn detect_media_type_images() {
        assert_eq!(detect_media_type("photo.png"), Some("image/png"));
        assert_eq!(detect_media_type("photo.PNG"), Some("image/png"));
        assert_eq!(detect_media_type("photo.jpg"), Some("image/jpeg"));
        assert_eq!(detect_media_type("photo.jpeg"), Some("image/jpeg"));
        assert_eq!(detect_media_type("photo.gif"), Some("image/gif"));
        assert_eq!(detect_media_type("photo.webp"), Some("image/webp"));
        assert_eq!(detect_media_type("doc.pdf"), Some("application/pdf"));
        assert_eq!(detect_media_type("code.rs"), None);
        assert_eq!(detect_media_type("data.json"), None);
    }

    #[tokio::test]
    async fn reads_image_as_document() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // PNG magic bytes
        let png_bytes = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        fs.write_file_bytes("image.png", &png_bytes).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/image.png"}))
            .await?;

        assert!(result.success);
        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.documents[0].media_type, "image/png");
        Ok(())
    }

    #[tokio::test]
    async fn reads_pdf_as_document() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file_bytes("doc.pdf", b"%PDF-1.4 fake").await?;

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
    async fn text_files_have_no_documents() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        fs.write_file("test.txt", "hello").await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/test.txt"}))
            .await?;

        assert!(result.success);
        assert!(result.documents.is_empty());
        Ok(())
    }
}
