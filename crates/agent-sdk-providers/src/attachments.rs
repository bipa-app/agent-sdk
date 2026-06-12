use agent_sdk_foundation::llm::{ChatRequest, Content, ContentBlock};
use anyhow::{Result, bail};

// All size caps below are measured in **decoded** bytes. This is a single,
// consistent semantic across every provider: it matches the per-image decoded
// check and Gemini's historical accounting, and avoids the two validators
// silently disagreeing on what "inline bytes" means. The decoded size is
// computed arithmetically from the base64 length (see
// [`decoded_attachment_len`]) rather than by allocating the decoded buffer.
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
    Anthropic,
    Gemini,
    OpenAI,
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

// Only the OpenAI Chat Completions provider gates its multimodal routing on
// this helper; under other feature sets it would be dead code.
#[cfg(feature = "openai")]
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
                    // Non-attachment blocks contribute nothing here, and the
                    // `_` arm covers future `#[non_exhaustive]` variants, which
                    // by definition carry no image/document attachment.
                    _ => {}
                }
            }
        }
    }

    attachments
}

/// Validate that `data` is well-formed standard (padded) base64 and return the
/// number of **decoded** bytes it represents — without allocating the decoded
/// buffer.
///
/// `validate_request_attachments` runs on every `chat`/`chat_stream` call over
/// the entire message history, so fully base64-decoding each attachment (up to
/// tens of MB) purely to read its length wasted CPU and allocations on the
/// async runtime each turn. The decoded length is instead computed
/// arithmetically from the encoded length and trailing padding.
pub(crate) fn decoded_attachment_len(data: &str) -> Result<usize> {
    let bytes = data.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return Ok(0);
    }
    if !len.is_multiple_of(4) {
        bail!("invalid base64 attachment data: length is not a multiple of 4");
    }
    let padding = bytes.iter().rev().take_while(|&&b| b == b'=').count();
    if padding > 2 {
        bail!("invalid base64 attachment data: malformed padding");
    }
    for &b in &bytes[..len - padding] {
        let valid = b.is_ascii_alphanumeric() || b == b'+' || b == b'/';
        if !valid {
            bail!("invalid base64 attachment data: unexpected character");
        }
    }
    Ok((len / 4) * 3 - padding)
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
        Some(AttachmentPolicy::Anthropic) => validate_anthropic_inline_attachments(&attachments),
        Some(AttachmentPolicy::Gemini) => validate_gemini_inline_attachments(&attachments),
        Some(AttachmentPolicy::OpenAI) => validate_openai_inline_attachments(&attachments),
        None => bail!(
            "provider={provider} model={model} does not support image/document content blocks in this SDK yet"
        ),
    }
}

fn attachment_policy(provider: &str, model: &str) -> Option<AttachmentPolicy> {
    match provider {
        "anthropic" => Some(AttachmentPolicy::Anthropic),
        "openai" | "openai-responses" => Some(AttachmentPolicy::OpenAI),
        "vertex" if model.starts_with("claude-") => Some(AttachmentPolicy::Anthropic),
        "gemini" | "vertex" => Some(AttachmentPolicy::Gemini),
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
        let decoded_len = decoded_attachment_len(attachment.data)?;
        total_inline_bytes += decoded_len;
        validate_media_type(
            attachment.kind,
            attachment.media_type,
            "Anthropic/Vertex Claude",
        )?;

        match attachment.kind {
            AttachmentKind::Image => {
                image_count += 1;
                if decoded_len > ANTHROPIC_MAX_IMAGE_BYTES {
                    bail!(
                        "image attachment exceeds Anthropic limit: {decoded_len} bytes > {ANTHROPIC_MAX_IMAGE_BYTES} bytes"
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
            "too many image attachments for Anthropic/Vertex Claude: {image_count} > {ANTHROPIC_MAX_IMAGES_PER_REQUEST}"
        );
    }

    if document_count > ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST {
        bail!(
            "too many document attachments for Anthropic/Vertex Claude: {document_count} > {ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST}"
        );
    }

    if total_inline_bytes > ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES {
        bail!(
            "total inline attachment payload exceeds Anthropic/Vertex Claude limit: {total_inline_bytes} bytes > {ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES} bytes"
        );
    }

    Ok(())
}

fn validate_gemini_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    let mut total_inline_bytes = 0;

    for attachment in attachments {
        let decoded_len = decoded_attachment_len(attachment.data)?;
        validate_media_type(
            attachment.kind,
            attachment.media_type,
            "Gemini/Vertex Gemini",
        )?;
        total_inline_bytes += decoded_len;
    }

    if total_inline_bytes > GEMINI_MAX_INLINE_ATTACHMENT_BYTES {
        bail!(
            "total inline attachment payload exceeds Gemini/Vertex Gemini limit: {total_inline_bytes} bytes > {GEMINI_MAX_INLINE_ATTACHMENT_BYTES} bytes"
        );
    }

    Ok(())
}

fn validate_openai_inline_attachments(attachments: &[AttachmentRef<'_>]) -> Result<()> {
    for attachment in attachments {
        let decoded_len = decoded_attachment_len(attachment.data)?;
        validate_media_type(attachment.kind, attachment.media_type, "OpenAI")?;

        match attachment.kind {
            AttachmentKind::Image if decoded_len > OPENAI_MAX_IMAGE_BYTES => {
                bail!(
                    "image attachment exceeds OpenAI inline limit: {decoded_len} bytes > {OPENAI_MAX_IMAGE_BYTES} bytes"
                );
            }
            AttachmentKind::Document if decoded_len > OPENAI_MAX_DOCUMENT_BYTES => {
                bail!(
                    "document attachment exceeds OpenAI inline limit: {decoded_len} bytes > {OPENAI_MAX_DOCUMENT_BYTES} bytes"
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
    use agent_sdk_foundation::llm::{ContentSource, Message, Role};
    use base64::Engine;

    fn request_with_blocks(blocks: Vec<ContentBlock>) -> ChatRequest {
        ChatRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: Content::Blocks(blocks),
            }],
            tools: None,
            max_tokens: 1024,
            max_tokens_explicit: true,
            session_id: None,
            cached_content: None,
            thinking: None,
            tool_choice: None,
            response_format: None,
            cache: None,
        }
    }

    #[cfg(feature = "openai")]
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

    #[test]
    fn test_decoded_attachment_len_matches_real_decode() -> anyhow::Result<()> {
        // The arithmetic length must equal what a real decode would yield,
        // across all three padding cases (0, 1, and 2 `=`).
        for raw in [&b""[..], b"a", b"ab", b"abc", b"abcd", b"abcde"] {
            let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
            assert_eq!(decoded_attachment_len(&encoded)?, raw.len(), "raw={raw:?}");
        }
        Ok(())
    }

    #[test]
    fn test_decoded_attachment_len_distinguishes_encoded_from_decoded() -> anyhow::Result<()> {
        // "AAAA" is 4 base64 chars -> 3 decoded bytes: the size cap counts the
        // decoded measure, not the (larger) encoded string length.
        assert_eq!(decoded_attachment_len("AAAA")?, 3);
        Ok(())
    }

    #[test]
    fn test_decoded_attachment_len_rejects_malformed() {
        // Not a multiple of 4.
        assert!(decoded_attachment_len("AAA").is_err());
        // Illegal character.
        assert!(decoded_attachment_len("AA*A").is_err());
        // Too much padding.
        assert!(decoded_attachment_len("A===").is_err());
    }

    #[test]
    fn test_gemini_total_size_counts_decoded_bytes() {
        // A payload whose decoded size exceeds the Gemini total cap is rejected
        // with the total-size error (confirming decoded-byte accounting).
        let oversized = vec![0_u8; GEMINI_MAX_INLINE_ATTACHMENT_BYTES + 1];
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(oversized),
            ),
        }]);

        let error = validate_request_attachments("gemini", "gemini-2.5-pro", &request)
            .expect_err("expected total-size error");
        assert!(
            error
                .to_string()
                .contains("total inline attachment payload exceeds Gemini"),
            "got: {error}"
        );
    }
}
