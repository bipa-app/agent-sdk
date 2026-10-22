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

// `LlmProvider::max_request_attachment_bytes` sources, kept beside the
// validator caps above so the two can never drift apart silently.
/// Anthropic's aggregate decoded request-attachment budget.
pub const ANTHROPIC_MAX_REQUEST_ATTACHMENT_BYTES: u64 =
    ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES as u64;
/// Gemini's aggregate decoded request-attachment budget.
pub const GEMINI_MAX_REQUEST_ATTACHMENT_BYTES: u64 = GEMINI_MAX_INLINE_ATTACHMENT_BYTES as u64;
/// `OpenAI` documents only per-item caps, so its aggregate borrows the smallest
/// cross-family aggregate (Gemini's) as a conservative bound.
pub const CONSERVATIVE_MAX_REQUEST_ATTACHMENT_BYTES: u64 = GEMINI_MAX_REQUEST_ATTACHMENT_BYTES;

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
// these helpers; under other feature sets they would be dead code.
#[cfg(feature = "openai")]
pub(crate) fn request_has_attachments(request: &ChatRequest) -> bool {
    !collect_attachments(request).is_empty()
}

/// Whether the request carries any document (non-image) attachment. Chat
/// Completions has a standard `image_url` content part but no portable
/// document part, so the `OpenAI` chat provider serializes images and refuses
/// documents.
#[cfg(feature = "openai")]
pub(crate) fn request_has_document_attachments(request: &ChatRequest) -> bool {
    collect_attachments(request)
        .iter()
        .any(|attachment| attachment.kind == AttachmentKind::Document)
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

/// The text an attachment is replaced with when the request cannot carry it.
pub(crate) const OMITTED_ATTACHMENT: &str = "[an image or document from an earlier message was \
     omitted to fit this request's attachment limits; read the file again to view it]";
pub(crate) const UNSUPPORTED_ATTACHMENT: &str = "[an image or document was omitted: this \
     provider route does not take attachments]";

/// What a policy lets one request carry in total.
struct RequestAttachmentLimits {
    images: usize,
    documents: usize,
    bytes: usize,
}

const fn request_limits(policy: AttachmentPolicy) -> RequestAttachmentLimits {
    match policy {
        AttachmentPolicy::Anthropic => RequestAttachmentLimits {
            images: ANTHROPIC_MAX_IMAGES_PER_REQUEST,
            documents: ANTHROPIC_MAX_DOCUMENTS_PER_REQUEST,
            bytes: ANTHROPIC_MAX_INLINE_ATTACHMENT_BYTES,
        },
        AttachmentPolicy::Gemini | AttachmentPolicy::OpenAI => RequestAttachmentLimits {
            images: usize::MAX,
            documents: usize::MAX,
            bytes: GEMINI_MAX_INLINE_ATTACHMENT_BYTES,
        },
    }
}

/// Replace the attachments `request` cannot carry with a short text note,
/// and return how many were replaced.
///
/// Images and documents stay in a conversation's history, so enough of them
/// would push every later request past the provider's per-request count or
/// total-size limit, or, on a route that takes none, fail every request. The
/// newest attachments that fit are kept; the first that does not and every
/// older one become [`OMITTED_ATTACHMENT`], so what is omitted only grows
/// from the oldest end. On a route with no attachment policy every one
/// becomes [`UNSUPPORTED_ATTACHMENT`]. A single malformed or oversized
/// attachment is left for [`validate_request_attachments`] to refuse.
pub(crate) fn fit_request_attachments(
    provider: &str,
    model: &str,
    request: &mut ChatRequest,
) -> usize {
    let policy = attachment_policy(provider, model);
    let mut positions = Vec::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        if let Content::Blocks(blocks) = &message.content {
            for (block_index, block) in blocks.iter().enumerate() {
                let (kind, data) = match block {
                    ContentBlock::Image { source } => (AttachmentKind::Image, &source.data),
                    ContentBlock::Document { source } => (AttachmentKind::Document, &source.data),
                    _ => continue,
                };
                positions.push((
                    message_index,
                    block_index,
                    kind,
                    decoded_attachment_len(data).ok(),
                ));
            }
        }
    }
    let (omitted, note) = policy.map_or((positions.len(), UNSUPPORTED_ATTACHMENT), |policy| {
        let kept = newest_that_fit(&request_limits(policy), &positions);
        (positions.len() - kept, OMITTED_ATTACHMENT)
    });
    for &(message_index, block_index, _, _) in &positions[..omitted] {
        if let Content::Blocks(blocks) = &mut request.messages[message_index].content {
            blocks[block_index] = ContentBlock::Text {
                text: note.to_owned(),
            };
        }
    }
    omitted
}

/// How many of `positions`, counted from the newest, fit `limits` before the
/// first that does not.
fn newest_that_fit(
    limits: &RequestAttachmentLimits,
    positions: &[(usize, usize, AttachmentKind, Option<usize>)],
) -> usize {
    let (mut images, mut documents, mut bytes) = (0_usize, 0_usize, 0_usize);
    let mut kept = 0;
    for (_, _, kind, decoded) in positions.iter().rev() {
        let (count, cap) = match kind {
            AttachmentKind::Image => (&mut images, limits.images),
            AttachmentKind::Document => (&mut documents, limits.documents),
        };
        let size = decoded.unwrap_or(0);
        if *count + 1 > cap || bytes.saturating_add(size) > limits.bytes {
            break;
        }
        *count += 1;
        bytes += size;
        kept += 1;
    }
    kept
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
        // The codex route is the OpenAI Responses API over ChatGPT OAuth —
        // same models, same inline image/document handling. Capabilities and
        // features already fold codex into "openai" (provider.rs); the
        // attachment policy must too, or the codex provider's own
        // validate_request_attachments call rejects images it can send.
        "openai" | "openai-responses" | "openai-codex" => Some(AttachmentPolicy::OpenAI),
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
            request_only_reminders: Vec::new(),
        }
    }

    /// `count` images of `decoded` bytes each, one per message, oldest first,
    /// each tagged by its index in its first base64 bytes.
    fn image_history(count: usize, decoded: usize) -> ChatRequest {
        let mut request = request_with_blocks(Vec::new());
        request.messages = (0..count)
            .map(|index| {
                let mut bytes = vec![0_u8; decoded];
                bytes[0] = u8::try_from(index).unwrap_or(u8::MAX);
                Message {
                    role: Role::User,
                    content: Content::Blocks(vec![
                        ContentBlock::Text {
                            text: format!("read {index}"),
                        },
                        ContentBlock::Image {
                            source: ContentSource::new(
                                "image/png",
                                base64::engine::general_purpose::STANDARD.encode(&bytes),
                            ),
                        },
                    ]),
                }
            })
            .collect();
        request
    }

    fn images_left(request: &ChatRequest) -> Vec<usize> {
        request
            .messages
            .iter()
            .enumerate()
            .filter(|(_, message)| {
                matches!(&message.content, Content::Blocks(blocks)
                    if blocks.iter().any(|block| matches!(block, ContentBlock::Image { .. })))
            })
            .map(|(index, _)| index)
            .collect()
    }

    const MAX_READ_IMAGE: usize = 3_932_160;

    #[test]
    fn images_past_anthropics_request_total_are_omitted_oldest_first() -> anyhow::Result<()> {
        let mut request = image_history(9, MAX_READ_IMAGE);
        assert!(validate_request_attachments("anthropic", "claude-opus-5-5", &request).is_err());

        let omitted = fit_request_attachments("anthropic", "claude-opus-5-5", &mut request);

        assert_eq!(omitted, 1);
        assert_eq!(images_left(&request), (1..9).collect::<Vec<_>>());
        assert!(
            matches!(&request.messages[0].content, Content::Blocks(blocks)
            if matches!(&blocks[1], ContentBlock::Text { text } if text == OMITTED_ATTACHMENT))
        );
        validate_request_attachments("anthropic", "claude-opus-5-5", &request)?;
        Ok(())
    }

    #[test]
    fn images_past_geminis_request_total_are_omitted_oldest_first() -> anyhow::Result<()> {
        let mut request = image_history(6, MAX_READ_IMAGE);
        assert!(validate_request_attachments("gemini", "gemini-3-pro", &request).is_err());

        assert_eq!(
            fit_request_attachments("gemini", "gemini-3-pro", &mut request),
            1
        );
        assert_eq!(images_left(&request), (1..6).collect::<Vec<_>>());
        validate_request_attachments("gemini", "gemini-3-pro", &request)?;
        Ok(())
    }

    #[test]
    fn anthropics_document_count_keeps_the_newest_five() {
        let mut request = request_with_blocks(
            (0..7)
                .map(|_| ContentBlock::Document {
                    source: ContentSource::new(
                        "application/pdf",
                        base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.4"),
                    ),
                })
                .collect(),
        );

        assert_eq!(
            fit_request_attachments("anthropic", "claude-opus-5-5", &mut request),
            2
        );
        assert!(validate_request_attachments("anthropic", "claude-opus-5-5", &request).is_ok());
    }

    #[test]
    fn attachments_that_fit_are_left_alone() -> anyhow::Result<()> {
        let mut request = image_history(3, 1024);
        let original = serde_json::to_value(&request.messages)?;

        assert_eq!(
            fit_request_attachments("anthropic", "claude-opus-5-5", &mut request),
            0
        );
        assert_eq!(serde_json::to_value(&request.messages)?, original);
        Ok(())
    }

    #[test]
    fn a_route_without_an_attachment_policy_gets_notes_instead_of_images() -> anyhow::Result<()> {
        let mut request = image_history(2, 1024);
        assert!(validate_request_attachments("openrouter", "some/model", &request).is_err());

        assert_eq!(
            fit_request_attachments("openrouter", "some/model", &mut request),
            2
        );
        assert_eq!(images_left(&request), Vec::<usize>::new());
        validate_request_attachments("openrouter", "some/model", &request)?;
        Ok(())
    }

    #[test]
    fn a_single_oversized_image_is_left_for_validation_to_refuse() {
        let mut request = image_history(1, 6 * 1024 * 1024);

        assert_eq!(
            fit_request_attachments("anthropic", "claude-opus-5-5", &mut request),
            0
        );
        assert!(validate_request_attachments("anthropic", "claude-opus-5-5", &request).is_err());
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
    fn test_validate_codex_accepts_supported_image() -> anyhow::Result<()> {
        // The codex route (OpenAI Responses over ChatGPT OAuth) must fold into
        // the OpenAI attachment policy like capabilities/features do — its own
        // validate_request_attachments call otherwise rejects images it can send.
        let request = request_with_blocks(vec![ContentBlock::Image {
            source: ContentSource::new(
                "image/png",
                base64::engine::general_purpose::STANDARD.encode(b"png"),
            ),
        }]);

        validate_request_attachments("openai-codex", "gpt-5.6-sol", &request)?;
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
