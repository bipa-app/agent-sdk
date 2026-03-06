use crate::llm::{ChatRequest, Content, ContentBlock};
use anyhow::{Result, bail};
use base64::Engine;

const ANTHROPIC_MAX_IMAGES_PER_REQUEST: usize = 100;
const ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST: usize = 5;
const ANTHROPIC_MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
const ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES: usize = 32 * 1024 * 1024;

const ANTHROPIC_SUPPORTED_IMAGE_MEDIA_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];
const ANTHROPIC_SUPPORTED_DOCUMENT_MEDIA_TYPES: &[&str] = &["application/pdf"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentPolicy {
    Unsupported,
    AnthropicInline,
}

#[derive(Debug)]
struct AttachmentRef<'a> {
    media_type: &'a str,
    data: &'a str,
    kind: AttachmentKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentKind {
    Image,
    Document,
}

pub(crate) fn validate_request_attachments(
    provider: &str,
    model: &str,
    request: &ChatRequest,
) -> Result<()> {
    let attachments = collect_attachments(request);
    if attachments.is_empty() {
        return Ok(());
    }

    match attachment_policy(provider, model) {
        AttachmentPolicy::Unsupported => bail!(
            "provider={provider} model={model} does not support image/document content blocks in this SDK yet"
        ),
        AttachmentPolicy::AnthropicInline => validate_anthropic_inline_attachments(&attachments),
    }
}

fn attachment_policy(provider: &str, model: &str) -> AttachmentPolicy {
    match provider {
        "anthropic" => AttachmentPolicy::AnthropicInline,
        "vertex" if model.starts_with("claude-") => AttachmentPolicy::AnthropicInline,
        _ => AttachmentPolicy::Unsupported,
    }
}

fn collect_attachments(request: &ChatRequest) -> Vec<AttachmentRef<'_>> {
    let mut attachments = Vec::new();

    for message in &request.messages {
        if let Content::Blocks(blocks) = &message.content {
            for block in blocks {
                match block {
                    ContentBlock::Image { source } => attachments.push(AttachmentRef {
                        media_type: &source.media_type,
                        data: &source.data,
                        kind: AttachmentKind::Image,
                    }),
                    ContentBlock::Document { source } => attachments.push(AttachmentRef {
                        media_type: &source.media_type,
                        data: &source.data,
                        kind: AttachmentKind::Document,
                    }),
                    ContentBlock::Text { .. }
                    | ContentBlock::Thinking { .. }
                    | ContentBlock::RedactedThinking { .. }
                    | ContentBlock::ToolUse { .. }
                    | ContentBlock::ToolResult { .. } => {}
                }
            }
        }
    }

    attachments
}

fn validate_anthropic_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    let mut image_count = 0;
    let mut document_count = 0;
    let mut total_inline_bytes = 0;

    for attachment in attachments {
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(attachment.data)
            .map_err(|error| anyhow::anyhow!("invalid base64 attachment data: {error}"))?;
        total_inline_bytes += attachment.data.len();

        match attachment.kind {
            AttachmentKind::Image => {
                image_count += 1;
                if !ANTHROPIC_SUPPORTED_IMAGE_MEDIA_TYPES.contains(&attachment.media_type) {
                    bail!(
                        "unsupported image media type '{}' for Anthropic/Vertex Claude attachments",
                        attachment.media_type
                    );
                }
                if decoded_bytes.len() > ANTHROPIC_MAX_IMAGE_BYTES {
                    bail!(
                        "image attachment exceeds Anthropic limit: {} bytes > {} bytes",
                        decoded_bytes.len(),
                        ANTHROPIC_MAX_IMAGE_BYTES
                    );
                }
            }
            AttachmentKind::Document => {
                document_count += 1;
                if !ANTHROPIC_SUPPORTED_DOCUMENT_MEDIA_TYPES.contains(&attachment.media_type) {
                    bail!(
                        "unsupported document media type '{}' for Anthropic/Vertex Claude attachments",
                        attachment.media_type
                    );
                }
            }
        }
    }

    if image_count > ANTHROPIC_MAX_IMAGES_PER_REQUEST {
        bail!(
            "too many image attachments for Anthropic/Vertex Claude: {} > {}",
            image_count,
            ANTHROPIC_MAX_IMAGES_PER_REQUEST
        );
    }

    if document_count > ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST {
        bail!(
            "too many document attachments for Anthropic/Vertex Claude: {} > {}",
            document_count,
            ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST
        );
    }

    if total_inline_bytes > ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES {
        bail!(
            "total inline attachment payload exceeds Anthropic/Vertex Claude limit: {} bytes > {} bytes",
            total_inline_bytes,
            ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ContentSource, Message, Role};

    fn request_with_blocks(blocks: Vec<ContentBlock>) -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: Content::Blocks(blocks),
            }],
            tools: None,
            max_tokens: 1024,
            thinking: None,
        }
    }

    #[test]
    fn test_validate_anthropic_accepts_supported_image() -> anyhow::Result<()> {
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(b"png"),
            ),
        }]);

        validate_request_attachments("anthropic", "claude-sonnet-4-6", &request)?;
        Ok(())
    }

    #[test]
    fn test_validate_anthropic_rejects_unsupported_document_type() {
        let request = request_with_blocks(vec![ContentBlock::Document {
            source: ContentSource::new(
                "application/msword",
                base64::engine::general_purpose::STANDARD.encode(b"doc"),
            ),
        }]);

        let error = validate_request_attachments("anthropic", "claude-sonnet-4-6", &request)
            .expect_err("expected unsupported media type error");
        assert!(
            error
                .to_string()
                .contains("unsupported document media type")
        );
    }

    #[test]
    fn test_validate_anthropic_rejects_large_image() {
        let oversized = vec![0_u8; ANTHROPIC_MAX_IMAGE_BYTES + 1];
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(oversized),
            ),
        }]);

        let error = validate_request_attachments("anthropic", "claude-sonnet-4-6", &request)
            .expect_err("expected large image error");
        assert!(
            error
                .to_string()
                .contains("image attachment exceeds Anthropic limit")
        );
    }

    #[test]
    fn test_validate_openai_rejects_attachments() {
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(b"png"),
            ),
        }]);

        let error = validate_request_attachments("openai", "gpt-5", &request)
            .expect_err("expected unsupported provider error");
        assert!(
            error
                .to_string()
                .contains("does not support image/document content blocks")
        );
    }

    #[test]
    fn test_validate_vertex_claude_accepts_attachments() -> anyhow::Result<()> {
        let request = request_with_blocks(vec![ContentBlock::Document {
            source: ContentSource::new(
                "application/pdf",
                base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.7"),
            ),
        }]);

        validate_request_attachments("vertex", "claude-sonnet-4-6", &request)?;
        Ok(())
    }

    #[test]
    fn test_validate_vertex_gemini_rejects_attachments() {
        let request = request_with_blocks(vec![ContentBlock::Document {
            source: ContentSource::new(
                "application/pdf",
                base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.7"),
            ),
        }]);

        let error = validate_request_attachments("vertex", "gemini-2.5-pro", &request)
            .expect_err("expected unsupported provider error");
        assert!(
            error
                .to_string()
                .contains("does not support image/document content blocks")
        );
    }
}
