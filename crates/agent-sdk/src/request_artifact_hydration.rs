use crate::primitive_tools::detect_media_magic;
use crate::{ArtifactStore, llm};
use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use std::io::Read as _;
use std::sync::Arc;

const MAX_HYDRATED_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;

/// Clone a provider request and replace exact artifact-backed attachment sources with base64.
///
/// The input request is never mutated. Canonical Snapcompact frame slots degrade when their
/// backing artifact cannot be recovered; every other artifact-backed source fails closed.
///
/// # Errors
/// Returns a contextual error for a non-canonical, unavailable, oversized, or MIME-mismatched
/// artifact-backed attachment, or when the blocking hydration task cannot be joined.
#[doc(hidden)]
pub async fn hydrate_request_artifact_sources(
    request: &llm::ChatRequest,
    artifact_store: Option<Arc<ArtifactStore>>,
) -> Result<llm::ChatRequest> {
    if !request_has_artifact_sources(request) {
        return Ok(request.clone());
    }
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        hydrate_request(
            request,
            artifact_store.as_deref(),
            MAX_HYDRATED_ARTIFACT_BYTES,
        )
    })
    .await
    .context("joining provider request artifact hydration")?
}

fn request_has_artifact_sources(request: &llm::ChatRequest) -> bool {
    request.messages.iter().any(|message| {
        let llm::Content::Blocks(blocks) = &message.content else {
            return false;
        };
        blocks.iter().any(|block| {
            let (llm::ContentBlock::Image { source } | llm::ContentBlock::Document { source }) =
                block
            else {
                return false;
            };
            source.data.starts_with(crate::ARTIFACT_URI_SCHEME)
        })
    })
}

fn hydrate_request(
    mut request: llm::ChatRequest,
    artifact_store: Option<&ArtifactStore>,
    max_hydrated_bytes: u64,
) -> Result<llm::ChatRequest> {
    let mut hydrated_bytes = 0_u64;
    for (message_index, message) in request.messages.iter_mut().enumerate() {
        let canonical = llm::canonical_snapcompact_checkpoint(message);
        let canonical_frame_range = canonical.as_ref().and_then(|meta| {
            let frame_count = usize::try_from(meta.frame_count).ok()?;
            Some(3..3usize.checked_add(frame_count)?)
        });
        let llm::Content::Blocks(blocks) = &mut message.content else {
            continue;
        };

        let original_blocks = std::mem::take(blocks);
        blocks.reserve(original_blocks.len());
        for (block_index, mut block) in original_blocks.into_iter().enumerate() {
            let is_canonical_frame = canonical_frame_range
                .as_ref()
                .is_some_and(|range| range.contains(&block_index));
            let (llm::ContentBlock::Image { source } | llm::ContentBlock::Document { source }) =
                &mut block
            else {
                blocks.push(block);
                continue;
            };
            if !source.data.starts_with(crate::ARTIFACT_URI_SCHEME) {
                blocks.push(block);
                continue;
            }

            let expected_digest = is_canonical_frame
                .then(|| {
                    let manifest = canonical.as_ref()?.frame_manifest.as_ref()?;
                    let artifact_id = exact_artifact_uri_id(&source.data)?;
                    manifest
                        .iter()
                        .find(|entry| entry.artifact_id == artifact_id)
                })
                .flatten();
            let result = hydrate_source(
                source,
                artifact_store,
                expected_digest,
                &mut hydrated_bytes,
                max_hydrated_bytes,
            )
            .map_err(|failure| {
                failure.with_context(format!(
                    "hydrating provider request message {message_index} block {block_index}"
                ))
            });
            match result {
                Ok(()) => blocks.push(block),
                Err(failure) if is_canonical_frame && failure.degradable_frame => {
                    log::warn!(
                        "Omitting unavailable canonical Snapcompact frame from provider request: {:#}",
                        failure.error
                    );
                }
                Err(failure) => return Err(failure.error),
            }
        }
    }
    Ok(request)
}

struct SourceHydrationFailure {
    error: anyhow::Error,
    degradable_frame: bool,
}

impl SourceHydrationFailure {
    const fn degradable(error: anyhow::Error) -> Self {
        Self {
            error,
            degradable_frame: true,
        }
    }

    const fn fatal(error: anyhow::Error) -> Self {
        Self {
            error,
            degradable_frame: false,
        }
    }

    fn with_context(self, context: String) -> Self {
        Self {
            error: self.error.context(context),
            degradable_frame: self.degradable_frame,
        }
    }
}

fn hydrate_source(
    source: &mut llm::ContentSource,
    store: Option<&ArtifactStore>,
    expected_digest: Option<&llm::SnapcompactFrameDigest>,
    hydrated_bytes: &mut u64,
    max_hydrated_bytes: u64,
) -> std::result::Result<(), SourceHydrationFailure> {
    let artifact_id = exact_artifact_uri_id(&source.data).ok_or_else(|| {
        SourceHydrationFailure::fatal(anyhow!(
            "attachment artifact URI must be exact (artifact://<positive numeric id>)"
        ))
    })?;
    let store = store.ok_or_else(|| {
        SourceHydrationFailure::degradable(anyhow!(
            "attachment artifact URI has no current-thread ArtifactStore"
        ))
    })?;
    let mut file = store.resolve(artifact_id).map_err(|_| {
        SourceHydrationFailure::degradable(anyhow!(
            "resolving current-thread artifact {artifact_id}: unavailable"
        ))
    })?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspecting current-thread artifact {artifact_id}"))
        .map_err(SourceHydrationFailure::degradable)?;
    let expected_len = metadata.len();
    if expected_len > max_hydrated_bytes {
        return Err(SourceHydrationFailure::fatal(anyhow!(
            "artifact {artifact_id} is {expected_len} bytes; provider attachment limit is {max_hydrated_bytes} bytes"
        )));
    }
    let projected_bytes = hydrated_bytes
        .checked_add(expected_len)
        .filter(|total| *total <= max_hydrated_bytes)
        .ok_or_else(|| {
            SourceHydrationFailure::fatal(anyhow!(
                "artifact-backed provider attachments exceed aggregate decoded limit of {max_hydrated_bytes} bytes"
            ))
        })?;

    let capacity = usize::try_from(expected_len)
        .with_context(|| format!("artifact {artifact_id} length does not fit memory"))
        .map_err(SourceHydrationFailure::fatal)?;
    let mut bytes = vec![0_u8; capacity];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading current-thread artifact {artifact_id}"))
        .map_err(SourceHydrationFailure::degradable)?;
    let mut extra = [0_u8; 1];
    let extra_len = file
        .read(&mut extra)
        .with_context(|| format!("checking current-thread artifact {artifact_id} length"))
        .map_err(SourceHydrationFailure::degradable)?;
    if extra_len != 0 {
        return Err(SourceHydrationFailure::degradable(anyhow!(
            "artifact {artifact_id} changed size while being read"
        )));
    }
    if let Some(expected) = expected_digest {
        verify_frame_digest(artifact_id, &bytes, expected)?;
    }

    let detected = detect_media_magic(&bytes).ok_or_else(|| {
        SourceHydrationFailure::degradable(anyhow!(
            "artifact {artifact_id} has unsupported or corrupt media magic"
        ))
    })?;
    if detected != source.media_type {
        return Err(SourceHydrationFailure::degradable(anyhow!(
            "artifact {artifact_id} MIME mismatch: declared {}, detected {detected}",
            source.media_type
        )));
    }
    source.data = base64::engine::general_purpose::STANDARD.encode(bytes);
    *hydrated_bytes = projected_bytes;
    Ok(())
}

fn verify_frame_digest(
    artifact_id: u64,
    bytes: &[u8],
    expected: &llm::SnapcompactFrameDigest,
) -> std::result::Result<(), SourceHydrationFailure> {
    let len = bytes.len() as u64;
    if len != expected.len {
        return Err(SourceHydrationFailure::degradable(anyhow!(
            "artifact {artifact_id} is {len} bytes; canonical Snapcompact manifest pinned {} bytes",
            expected.len
        )));
    }
    let actual_sha256 = llm::sha256_hex(bytes);
    if actual_sha256 != expected.sha256 {
        return Err(SourceHydrationFailure::degradable(anyhow!(
            "artifact {artifact_id} sha256 digest mismatch: canonical Snapcompact manifest \
             pinned {}, artifact hashes to {actual_sha256}",
            expected.sha256
        )));
    }
    Ok(())
}

fn exact_artifact_uri_id(uri: &str) -> Option<u64> {
    let id = uri.strip_prefix(crate::ARTIFACT_URI_SCHEME)?;
    if id.is_empty()
        || !id.bytes().all(|byte| byte.is_ascii_digit())
        || (id.len() > 1 && id.starts_with('0'))
    {
        return None;
    }
    id.parse().ok().filter(|artifact_id| *artifact_id > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_foundation::llm::{
        Content, ContentBlock, ContentSource, Message, SNAPCOMPACT_HISTORY_IMAGE_WARNING,
        SnapcompactMetadata,
    };

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\nsnapcompact-frame";

    fn save_bytes(store: &ArtifactStore, kind: &str, bytes: &[u8]) -> Result<u64> {
        let mut bytes = bytes;
        Ok(store.save_streamed(kind, &mut bytes)?.id)
    }

    fn prime_positive_artifact_ids(store: &ArtifactStore) -> Result<()> {
        let reserved = save_bytes(store, "reserved", b"reserved artifact zero")?;
        assert_eq!(reserved, 0, "fresh fixture must reserve artifact zero");
        Ok(())
    }

    fn canonical_frame_message(source_id: u64, frame_id: u64) -> Message {
        let metadata = SnapcompactMetadata {
            source_artifact_id: source_id,
            truncated_chars: 42,
            frame_count: 1,
            frame_size: 1_932,
            source_len: None,
            source_sha256: None,
            frame_manifest: None,
        };
        Message::user_with_content(vec![
            ContentBlock::CompactionSummary {
                text: "durable exact-source summary".to_owned(),
                artifact_ids: vec![source_id, frame_id],
                snapcompact: Some(metadata),
            },
            ContentBlock::CompactionSummary {
                text: "head recovery text".to_owned(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
            ContentBlock::CompactionSummary {
                text: SNAPCOMPACT_HISTORY_IMAGE_WARNING.to_owned(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
            ContentBlock::Image {
                source: ContentSource::new("image/png", format!("artifact://{frame_id}")),
            },
            ContentBlock::CompactionSummary {
                text: "tail recovery text".to_owned(),
                artifact_ids: Vec::new(),
                snapcompact: None,
            },
        ])
    }

    fn request(message: Message) -> llm::ChatRequest {
        llm::ChatRequest::new("system", vec![message])
    }

    #[tokio::test]
    async fn canonical_frame_hydrates_ephemerally_without_mutating_uri_request() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let frame_id = save_bytes(&store, "snapcompact-frame", PNG)?;
        let durable = request(canonical_frame_message(source_id, frame_id));

        let hydrated = hydrate_request_artifact_sources(&durable, Some(store)).await?;
        let Content::Blocks(durable_blocks) = &durable.messages[0].content else {
            anyhow::bail!("durable request must use blocks");
        };
        let ContentBlock::Image { source } = &durable_blocks[3] else {
            anyhow::bail!("durable frame must remain an image");
        };
        assert_eq!(source.data, format!("artifact://{frame_id}"));

        let Content::Blocks(hydrated_blocks) = &hydrated.messages[0].content else {
            anyhow::bail!("hydrated request must use blocks");
        };
        let ContentBlock::Image { source } = &hydrated_blocks[3] else {
            anyhow::bail!("hydrated frame must remain an image");
        };
        assert_eq!(
            base64::engine::general_purpose::STANDARD.decode(&source.data)?,
            PNG
        );
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_canonical_frame_is_omitted_but_recovery_text_stays_ordered() -> Result<()>
    {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let durable = request(canonical_frame_message(source_id, 999));

        let hydrated = hydrate_request_artifact_sources(&durable, Some(store)).await?;
        let Content::Blocks(blocks) = &hydrated.messages[0].content else {
            anyhow::bail!("hydrated request must use blocks");
        };
        assert_eq!(blocks.len(), 4);
        assert!(matches!(
            &blocks[0],
            ContentBlock::CompactionSummary { text, .. } if text == "durable exact-source summary"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::CompactionSummary { text, .. } if text == "head recovery text"
        ));
        assert!(matches!(
            &blocks[3],
            ContentBlock::CompactionSummary { text, .. } if text == "tail recovery text"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn corrupt_canonical_frame_is_omitted_from_ephemeral_request() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let frame_id = save_bytes(&store, "snapcompact-frame", b"not an image")?;
        let durable = request(canonical_frame_message(source_id, frame_id));

        let hydrated = hydrate_request_artifact_sources(&durable, Some(store)).await?;
        let Content::Blocks(blocks) = &hydrated.messages[0].content else {
            anyhow::bail!("hydrated request must use blocks");
        };
        assert_eq!(blocks.len(), 4);
        assert!(blocks.iter().all(|block| !matches!(
            block,
            ContentBlock::Image { .. } | ContentBlock::Document { .. }
        )));
        Ok(())
    }

    fn pin_frame_manifest(message: &mut Message, frame_id: u64, frame_bytes: &[u8]) {
        let Content::Blocks(blocks) = &mut message.content else {
            panic!("checkpoint must use blocks");
        };
        let Some(ContentBlock::CompactionSummary {
            snapcompact: Some(metadata),
            ..
        }) = blocks.first_mut()
        else {
            panic!("checkpoint must carry Snapcompact metadata");
        };
        metadata.source_len = Some(6);
        metadata.source_sha256 = Some(llm::sha256_hex(b"source"));
        metadata.frame_manifest =
            Some(llm::snapcompact_integrity(b"source", &[(frame_id, frame_bytes)]).frame_manifest);
    }

    #[tokio::test]
    async fn digest_pinned_canonical_frame_hydrates_when_content_matches() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let frame_id = save_bytes(&store, "snapcompact-frame", PNG)?;
        let mut checkpoint = canonical_frame_message(source_id, frame_id);
        pin_frame_manifest(&mut checkpoint, frame_id, PNG);
        assert!(llm::canonical_snapcompact_checkpoint(&checkpoint).is_some());
        let durable = request(checkpoint);

        let hydrated = hydrate_request_artifact_sources(&durable, Some(store)).await?;
        let Content::Blocks(blocks) = &hydrated.messages[0].content else {
            anyhow::bail!("hydrated request must use blocks");
        };
        let ContentBlock::Image { source } = &blocks[3] else {
            anyhow::bail!("digest-matching frame must stay an image");
        };
        assert_eq!(
            base64::engine::general_purpose::STANDARD.decode(&source.data)?,
            PNG
        );
        Ok(())
    }

    #[tokio::test]
    async fn digest_pinned_canonical_frame_replaced_same_id_is_omitted() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let saved = {
            let mut png = PNG;
            store.save_streamed("snapcompact-frame", &mut png)?
        };
        let mut checkpoint = canonical_frame_message(source_id, saved.id);
        pin_frame_manifest(&mut checkpoint, saved.id, PNG);
        assert!(llm::canonical_snapcompact_checkpoint(&checkpoint).is_some());
        let replacement = b"\x89PNG\r\n\x1a\nsubstituted-frame";
        assert_eq!(replacement.len(), PNG.len());
        std::fs::write(&saved.path, replacement)?;
        let durable = request(checkpoint);

        let hydrated = hydrate_request_artifact_sources(&durable, Some(store)).await?;
        let Content::Blocks(blocks) = &hydrated.messages[0].content else {
            anyhow::bail!("hydrated request must use blocks");
        };
        assert_eq!(blocks.len(), 4);
        assert!(blocks.iter().all(|block| !matches!(
            block,
            ContentBlock::Image { .. } | ContentBlock::Document { .. }
        )));
        Ok(())
    }

    #[test]
    fn aggregate_limit_fails_canonical_multi_frame_request_instead_of_degrading() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let source_id = save_bytes(&store, "snapcompact-source", b"source")?;
        let first_frame_id = save_bytes(&store, "snapcompact-frame", PNG)?;
        let second_frame_id = save_bytes(&store, "snapcompact-frame", PNG)?;
        let mut checkpoint = canonical_frame_message(source_id, first_frame_id);
        let Content::Blocks(blocks) = &mut checkpoint.content else {
            anyhow::bail!("checkpoint must use blocks");
        };
        let ContentBlock::CompactionSummary {
            artifact_ids,
            snapcompact: Some(metadata),
            ..
        } = &mut blocks[0]
        else {
            anyhow::bail!("checkpoint must carry Snapcompact metadata");
        };
        artifact_ids.push(second_frame_id);
        metadata.frame_count = 2;
        blocks.insert(
            4,
            ContentBlock::Image {
                source: ContentSource::new("image/png", format!("artifact://{second_frame_id}")),
            },
        );
        assert!(llm::canonical_snapcompact_checkpoint(&checkpoint).is_some());
        let durable = request(checkpoint);
        let limit = u64::try_from(PNG.len() * 2 - 1)?;

        let error = hydrate_request(durable, Some(store.as_ref()), limit)
            .expect_err("aggregate overflow must fail the whole provider request");
        assert!(format!("{error:#}").contains("aggregate decoded limit"));
        Ok(())
    }

    #[tokio::test]
    async fn selector_on_noncanonical_attachment_fails_closed() -> Result<()> {
        let durable = request(Message::user_with_content(vec![ContentBlock::Image {
            source: ContentSource::new("image/png", "artifact://1:raw"),
        }]));
        let error = hydrate_request_artifact_sources(&durable, None)
            .await
            .expect_err("selector must not resolve");
        assert!(format!("{error:#}").contains("must be exact"));
        Ok(())
    }

    #[tokio::test]
    async fn cross_thread_attachment_reference_fails_closed() -> Result<()> {
        let owner_dir = tempfile::tempdir()?;
        let owner = ArtifactStore::new(owner_dir.path());
        prime_positive_artifact_ids(&owner)?;
        let artifact_id = save_bytes(&owner, "document", b"%PDF-1.7\nowned elsewhere")?;
        let current_dir = tempfile::tempdir()?;
        let current = Arc::new(ArtifactStore::new(current_dir.path()));
        let durable = request(Message::user_with_content(vec![ContentBlock::Document {
            source: ContentSource::new("application/pdf", format!("artifact://{artifact_id}")),
        }]));

        let error = hydrate_request_artifact_sources(&durable, Some(current))
            .await
            .expect_err("cross-thread artifact must not resolve");
        assert!(format!("{error:#}").contains("resolving current-thread artifact"));
        Ok(())
    }

    #[tokio::test]
    async fn noncanonical_mime_mismatch_fails_closed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let artifact_id = save_bytes(&store, "image", PNG)?;
        let durable = request(Message::user_with_content(vec![ContentBlock::Image {
            source: ContentSource::new("image/jpeg", format!("artifact://{artifact_id}")),
        }]));

        let error = hydrate_request_artifact_sources(&durable, Some(store))
            .await
            .expect_err("wrong declared MIME must fail");
        assert!(format!("{error:#}").contains("MIME mismatch"));
        Ok(())
    }

    #[tokio::test]
    async fn metadata_limit_rejects_before_artifact_allocation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path()));
        prime_positive_artifact_ids(&store)?;
        let saved = {
            let mut png = PNG;
            store.save_streamed("oversized", &mut png)?
        };
        std::fs::OpenOptions::new()
            .write(true)
            .open(&saved.path)?
            .set_len(MAX_HYDRATED_ARTIFACT_BYTES + 1)?;
        let durable = request(Message::user_with_content(vec![ContentBlock::Image {
            source: ContentSource::new("image/png", format!("artifact://{}", saved.id)),
        }]));

        let error = hydrate_request_artifact_sources(&durable, Some(store))
            .await
            .expect_err("oversized artifact must fail");
        assert!(format!("{error:#}").contains("provider attachment limit"));
        Ok(())
    }
}
