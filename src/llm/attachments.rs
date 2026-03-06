use crate::llm::{ChatRequest, Content, ContentBlock};
use anyhow::{Result, bail};
use base64::Engine;

const ANTHROPIC_MAX_IMAGES_PER_REQUEST: usize = 100;
const ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST: usize = 5;
const ANTHROPIC_MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
const ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES: usize = 32 * 1024 * 1024;

const GEMINI_MAX_INLINE_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;
const OPENAI_MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const OPENAI_MAX_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;

const SUPPORTED_IMAGE_MEDIA_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];
const SUPPORTED_DOCUMENT_MEDIA_TYPES: &[&str] = &["application/pdf"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentPolicy {
    AnthropicInline,
    GeminiInline,
    OpenAIInline,
}

#[derive(Debug)]
pub(crate) struct AttachmentRef<'a> {
    pub media_type: &'a str,
    pub data: &'a str,
    pub kind: AttachmentKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttachmentKind {
    Image,
    Document,
}

pub(crate) fn request_has_attachments(request: &ChatRequest) -> bool {
    !collect_attachments(request).is_empty()
}

pub(crate) fn collect_attachments(request: &ChatRequest) -> Vec<AttachmentRef<'_>> {
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

pub(crate) fn decode_attachment_bytes(data: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| anyhow::anyhow!("invalid base64 attachment data: {error}"))
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
        Some(AttachmentPolicy::AnthropicInline) => {
            validate_anthropic_inline_attachments(&attachments)
        }
        Some(AttachmentPolicy::GeminiInline) => validate_gemini_inline_attachments(&attachments),
        Some(AttachmentPolicy::OpenAIInline) => validate_openai_inline_attachments(&attachments),
        None => bail!(
            "provider={provider} model={model} does not support image/document content blocks in this SDK yet"
        ),
    }
}

fn attachment_policy(provider: &str, model: &str) -> Option<AttachmentPolicy> {
    match provider {
        "anthropic" => Some(AttachmentPolicy::AnthropicInline),
        "openai" | "openai-responses" => Some(AttachmentPolicy::OpenAIInline),
        "gemini" => Some(AttachmentPolicy::GeminiInline),
        "vertex" if model.starts_with("claude-") => Some(AttachmentPolicy::AnthropicInline),
        "vertex" => Some(AttachmentPolicy::GeminiInline),
        _ => None,
    }
}

fn validate_media_type(kind: AttachmentKind, media_type: &str, provider_label: &str) -> Result<()> {
    match kind {
        AttachmentKind::Image if SUPPORTED_IMAGE_MEDIA_TYPES.contains(&media_type) => Ok(()),
        AttachmentKind::Document if SUPPORTED_DOCUMENT_MEDIA_TYPES.contains(&media_type) => Ok(()),
        AttachmentKind::Image => {
            bail!("unsupported image media type '{media_type}' for {provider_label} attachments")
        }
        AttachmentKind::Document => {
            bail!("unsupported document media type '{media_type}' for {provider_label} attachments")
        }
    }
}

fn validate_anthropic_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    let mut image_count = 0;
    let mut document_count = 0;
    let mut total_inline_bytes = 0;

    for attachment in attachments {
        let decoded_bytes = decode_attachment_bytes(attachment.data)?;
        total_inline_bytes += attachment.data.len();
        validate_media_type(
            attachment.kind,
            attachment.media_type,
            "Anthropic/Vertex Claude",
        )?;

        match attachment.kind {
            AttachmentKind::Image => {
                image_count += 1;
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

fn validate_gemini_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    let mut total_inline_bytes = 0;

    for attachment in attachments {
        let decoded_bytes = decode_attachment_bytes(attachment.data)?;
        validate_media_type(
            attachment.kind,
            attachment.media_type,
            "Gemini/Vertex Gemini",
        )?;
        total_inline_bytes += decoded_bytes.len();
    }

    if total_inline_bytes > GEMINI_MAX_INLINE_ATTACHMENT_BYTES {
        bail!(
            "total inline attachment payload exceeds Gemini/Vertex Gemini limit: {} bytes > {} bytes",
            total_inline_bytes,
            GEMINI_MAX_INLINE_ATTACHMENT_BYTES
        );
    }

    Ok(())
}

fn validate_openai_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    for attachment in attachments {
        let decoded_bytes = decode_attachment_bytes(attachment.data)?;
        validate_media_type(attachment.kind, attachment.media_type, "OpenAI")?;

        match attachment.kind {
            AttachmentKind::Image if decoded_bytes.len() > OPENAI_MAX_IMAGE_BYTES => {
                bail!(
                    "image attachment exceeds OpenAI inline limit: {} bytes > {} bytes",
                    decoded_bytes.len(),
                    OPENAI_MAX_IMAGE_BYTES
                );
            }
            AttachmentKind::Document if decoded_bytes.len() > OPENAI_MAX_DOCUMENT_BYTES => {
                bail!(
                    "document attachment exceeds OpenAI inline limit: {} bytes > {} bytes",
                    decoded_bytes.len(),
                    OPENAI_MAX_DOCUMENT_BYTES
                );
            }
            AttachmentKind::Image | AttachmentKind::Document => {}
        }
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
    fn test_request_has_attachments() {
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(b"png"),
            ),
        }]);
        assert!(request_has_attachments(&request));
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
    fn test_validate_openai_accepts_attachments() -> anyhow::Result<()> {
        let request = request_with_blocks(vec![ContentBlock::Document {
            source: ContentSource::new(
                "application/pdf",
                base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.7"),
            ),
        }]);

        validate_request_attachments("openai", "gpt-5", &request)?;
        Ok(())
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
    fn test_validate_vertex_gemini_accepts_attachments() -> anyhow::Result<()> {
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(b"png"),
            ),
        }]);

        validate_request_attachments("vertex", "gemini-2.5-pro", &request)?;
        Ok(())
    }
}
