use crate::llm::ContentSource;
use crate::{Environment, PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use super::PrimitiveToolContext;

/// Maximum bytes per line before truncation.
const MAX_LINE_LENGTH: usize = 500;

/// Marker appended to a line that was truncated at `MAX_LINE_LENGTH`.
const LINE_TRUNCATION_MARKER: &str = "... [line truncated]";

/// Default maximum number of lines to return.
const DEFAULT_LIMIT: usize = 2000;

/// Total-byte floor for a read's inline output (50 KiB). A read with a
/// defaulted `limit` is bounded by this alone; an explicit `limit` widens
/// the budget to `limit * BYTES_PER_REQUESTED_LINE` so a deliberately
/// large request still fits. Nothing is lost either way — the file stays
/// on disk and the truncation notice names the continuation offset.
const TOTAL_BYTE_BUDGET_FLOOR: usize = 50 * 1024;

/// Per-line byte allowance used to scale the total budget with an
/// explicit `limit` (a formatted line is at most `MAX_LINE_LENGTH` content
/// bytes plus the `L{n}: ` prefix and truncation marker).
const BYTES_PER_REQUESTED_LINE: usize = 512;

/// Maximum size (in bytes) of a text file the tool will read into memory.
/// Larger files are rejected to avoid loading multi-GB files / dumping huge
/// payloads into the model context.
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;

/// Maximum size (in bytes) of a media file (image/PDF) that will be
/// base64-encoded and attached. Kept smaller than `MAX_FILE_BYTES` because
/// base64 inflates the payload (~1.33x) before it reaches the model context.
const MAX_MEDIA_BYTES: usize = 5 * 1024 * 1024;
/// Maximum size of a magic-sniffed artifact media attachment. Artifact
/// recovery is intentionally separate from the smaller filesystem-media cap:
/// providers accept larger recovered images/PDFs, but metadata is checked
/// before allocating their backing buffer.
const MAX_ARTIFACT_MEDIA_BYTES: usize = 32 * 1024 * 1024;

/// Hard ceiling for raw or base64 artifact-window output, further restricted
/// by the artifact store's configured inline budget.
const MAX_ARTIFACT_BYTE_WINDOW_OUTPUT_BYTES: usize = 512 * 1024;

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
    /// Maximum number of lines to return; defaults to 2000. Passing it
    /// explicitly also widens the total byte budget (see
    /// [`TOTAL_BYTE_BUDGET_FLOOR`]).
    #[serde(
        default,
        deserialize_with = "super::deserialize_optional_usize_from_string_or_int"
    )]
    limit: Option<usize>,
}

mod defaults {
    pub const fn offset() -> usize {
        1
    }
}

impl<E: Environment + 'static, Ctx: Send + Sync + 'static> Tool<Ctx> for ReadTool<E> {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::Read
    }

    fn display_name(&self) -> &'static str {
        "Read File"
    }

    fn description(&self) -> &'static str {
        "Read numbered text lines, supported media documents, and bounded artifact:// byte windows. Artifact selectors: lines=N, lines=N-M, lines=N+K (1-based lines); bytes=START+COUNT (exact UTF-8, 0-based bytes); base64=START+COUNT (exact arbitrary bytes)."
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
                    "description": "Filesystem path or artifact://ID[:SELECTOR]. Artifact selectors are :lines=N, :lines=N-M, or :lines=N+K for 1-based line windows; :bytes=START+COUNT for exact UTF-8 raw bytes; and :base64=START+COUNT for exact arbitrary bytes. Byte offsets are 0-based and counts are positive. Legacy :N, :N-M, and :N+K line selectors remain accepted. Unwindowed magic-sniffed artifact PNG/JPEG/GIF/WebP/PDF media can be attached up to 32MiB; filesystem media remains capped at 5MiB. Raw/base64 encoded output must fit both the artifact store inline budget and the 512KiB byte-window output cap."
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
                    "description": "Maximum number of lines to return. Accepts either an integer or a numeric string. Default: 2000. Output is also byte-budgeted: 50KB by default, or limit*512 bytes when limit is set explicitly; a budgeted read ends with a notice naming the continuation offset."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        let input: ReadInput = ReadInput::deserialize(&input)
            .with_context(|| format!("Invalid input for read tool: {input}"))?;

        if input.offset == 0 {
            return Ok(ToolResult::error("offset must be a 1-indexed line number"));
        }

        if input.limit == Some(0) {
            return Ok(ToolResult::error("limit must be greater than zero"));
        }

        if let Some(spec) = input
            .path
            .strip_prefix(agent_sdk_tools::artifacts::ARTIFACT_URI_SCHEME)
        {
            return read_artifact(ctx.artifact_store(), spec, &input).await;
        }

        let path = self.ctx.environment.resolve_path(&input.path);

        if let Err(reason) = self.ctx.capabilities.check_read(&path) {
            return Ok(ToolResult::error(format!(
                "Permission denied: cannot read '{path}': {reason}"
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
            // Cap media attachments before base64-encoding so an oversized
            // binary cannot be inflated into the model context.
            if bytes.len() > MAX_MEDIA_BYTES {
                return Ok(ToolResult::error(format!(
                    "Media file '{path}' is {} bytes, which exceeds the {MAX_MEDIA_BYTES}-byte attachment limit",
                    bytes.len()
                )));
            }
            let encoded = base64_encode(&bytes);
            return Ok(
                ToolResult::success(format!("Read {media_type} file: '{path}'"))
                    .with_documents(vec![ContentSource::new(media_type, encoded)]),
            );
        }

        // Cap text files before formatting them line-by-line.
        if bytes.len() > MAX_FILE_BYTES {
            return Ok(ToolResult::error(format!(
                "File '{path}' is {} bytes, which exceeds the {MAX_FILE_BYTES}-byte read limit; use offset/limit on a smaller range or a different tool",
                bytes.len()
            )));
        }

        // Text files: lossy UTF-8, line numbers, truncation. The byte
        // budget scales only with an EXPLICIT limit: a defaulted read is
        // bounded at the floor so a 2000-line log cannot dump ~1MB into
        // the context, while a caller that deliberately asks for N lines
        // gets room for them.
        let byte_budget = input.limit.map_or(TOTAL_BYTE_BUDGET_FLOOR, |limit| {
            TOTAL_BYTE_BUDGET_FLOOR.max(limit.saturating_mul(BYTES_PER_REQUESTED_LINE))
        });
        let content = String::from_utf8_lossy(&bytes);
        let collected = read_lines(
            &content,
            input.offset,
            input.limit.unwrap_or(DEFAULT_LIMIT),
            byte_budget,
        );

        if collected.is_empty() {
            return Ok(ToolResult::error("offset exceeds file length"));
        }

        Ok(ToolResult::success(collected.join("\n")))
    }
}

/// Read back a spilled artifact addressed as `artifact://<id>[:<selector>]`.
///
/// Line selectors are `:lines=N`, `:lines=N-M` (inclusive), and
/// `:lines=N+K`; the historical forms without `lines=` remain accepted.
/// Exact byte windows are `:bytes=START+COUNT` for UTF-8 output and
/// `:base64=START+COUNT` for arbitrary bytes. Byte offsets are zero-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactSelector {
    Lines {
        offset: usize,
        limit: usize,
    },
    Bytes {
        start: u64,
        count: usize,
        encoding: ArtifactByteEncoding,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactByteEncoding {
    Utf8,
    Base64,
}

enum ArtifactRead {
    Text(Vec<String>),
    Raw(String),
    Base64(String),
    Media {
        media_type: &'static str,
        bytes: Vec<u8>,
    },
}

async fn read_artifact(
    store: Option<&Arc<agent_sdk_tools::artifacts::ArtifactStore>>,
    spec: &str,
    input: &ReadInput,
) -> Result<ToolResult> {
    let Some(store) = store else {
        return Ok(ToolResult::error(
            "artifact storage is not configured for this session",
        ));
    };
    let (id_part, selector_text) = match spec.split_once(':') {
        Some((id_part, selector)) => (id_part, Some(selector)),
        None => (spec, None),
    };
    let Ok(id) = id_part.parse::<u64>() else {
        return Ok(ToolResult::error(format!(
            "artifact ID must be numeric, got: '{id_part}'"
        )));
    };
    let selector = match selector_text.map(parse_artifact_selector) {
        Some(Ok(selector)) => Some(selector),
        Some(Err(reason)) => {
            return Ok(ToolResult::error(format!(
                "invalid artifact selector ':{sel}': {reason}; use \
                 :lines=N, :lines=N-M, :lines=N+K, :bytes=START+COUNT, or \
                 :base64=START+COUNT",
                sel = selector_text.unwrap_or_default()
            )));
        }
        None => None,
    };
    let (offset, limit, explicit_limit) = match selector {
        Some(ArtifactSelector::Lines { offset, limit }) => (offset, limit, limit != usize::MAX),
        Some(ArtifactSelector::Bytes { .. }) => (1, DEFAULT_LIMIT, false),
        None => (
            input.offset,
            input.limit.unwrap_or(DEFAULT_LIMIT),
            input.limit.is_some(),
        ),
    };

    let line_windowed = selector.is_some() || offset != 1 || explicit_limit;
    let inline_budget = store.inline_budget();
    let line_byte_budget = TOTAL_BYTE_BUDGET_FLOOR.min(inline_budget);
    let window_output_cap = inline_budget.min(MAX_ARTIFACT_BYTE_WINDOW_OUTPUT_BYTES);
    let store = Arc::clone(store);
    let read = tokio::task::spawn_blocking(move || {
        let mut file = match store.resolve(id) {
            Ok(file) => file,
            Err(error) => return Ok(Err(format!("{error:#}"))),
        };
        let artifact_bytes = file
            .metadata()
            .context("inspecting artifact before read")?
            .len();

        if let Some(ArtifactSelector::Bytes {
            start,
            count,
            encoding,
        }) = selector
        {
            return read_artifact_byte_window(
                file,
                artifact_bytes,
                id,
                start,
                count,
                encoding,
                window_output_cap,
            );
        }

        if !line_windowed
            && let Some(result) =
                full_artifact_read_guard(&mut file, artifact_bytes, id, inline_budget)?
        {
            return Ok(result);
        }
        stream_artifact_lines(file, id, offset, limit, line_byte_budget, window_output_cap)
            .map(ArtifactRead::Text)
            .map(Ok)
    })
    .await
    .context("joining artifact read")??;
    match read {
        Ok(ArtifactRead::Media { media_type, bytes }) => Ok(ToolResult::success(format!(
            "Read {media_type} artifact: 'artifact://{id}'"
        ))
        .with_documents(vec![ContentSource::new(media_type, base64_encode(&bytes))])),
        Ok(ArtifactRead::Text(collected)) => {
            if collected.is_empty() {
                return Ok(ToolResult::error("offset exceeds artifact length"));
            }
            Ok(ToolResult::success(collected.join("\n")))
        }
        Ok(ArtifactRead::Raw(text) | ArtifactRead::Base64(text)) => Ok(ToolResult::success(text)),
        Err(error) => Ok(ToolResult::error(error)),
    }
}

fn artifact_media_capacity(artifact_bytes: u64) -> std::result::Result<usize, String> {
    let limit = u64::try_from(MAX_ARTIFACT_MEDIA_BYTES)
        .map_err(|_| "attachment limit cannot be represented as u64".to_string())?;
    if artifact_bytes > limit {
        return Err(format!(
            "is {artifact_bytes} bytes, which exceeds the \
             {MAX_ARTIFACT_MEDIA_BYTES}-byte attachment limit"
        ));
    }
    usize::try_from(artifact_bytes)
        .map_err(|_| format!("length {artifact_bytes} cannot be represented as usize"))
}

/// Guard a full (non-windowed) artifact read: recover media artifacts as
/// provider attachments and refuse oversized text artifacts before line
/// streaming. Magic detection deliberately precedes the inline/full-read
/// refusal: recovered media has its own provider attachment cap.
fn full_artifact_read_guard(
    file: &mut std::fs::File,
    artifact_bytes: u64,
    id: u64,
    inline_budget: usize,
) -> Result<Option<std::result::Result<ArtifactRead, String>>> {
    if let Some(media_type) = sniff_media_type(file)? {
        let capacity = match artifact_media_capacity(artifact_bytes) {
            Ok(capacity) => capacity,
            Err(error) => return Ok(Some(Err(format!("media artifact {id} {error}")))),
        };
        let mut bytes = Vec::with_capacity(capacity);
        file.take(
            u64::try_from(MAX_ARTIFACT_MEDIA_BYTES)
                .context("artifact media byte limit exceeds u64")?
                .saturating_add(1),
        )
        .read_to_end(&mut bytes)
        .context("reading media artifact")?;
        if bytes.len() > MAX_ARTIFACT_MEDIA_BYTES {
            return Ok(Some(Err(format!(
                "media artifact {id} grew beyond the \
                 {MAX_ARTIFACT_MEDIA_BYTES}-byte attachment limit"
            ))));
        }
        return Ok(Some(Ok(ArtifactRead::Media { media_type, bytes })));
    }
    if artifact_bytes > u64::try_from(inline_budget).context("inline budget exceeds u64")? {
        return Ok(Some(Err(format!(
            "artifact {id} is {artifact_bytes} bytes; read a window with \
             artifact://{id}:lines=START-END (or offset/limit), an exact UTF-8 \
             byte window with artifact://{id}:bytes=START+COUNT, or arbitrary \
             bytes with artifact://{id}:base64=START+COUNT"
        ))));
    }
    Ok(None)
}

fn read_artifact_byte_window(
    mut file: std::fs::File,
    artifact_bytes: u64,
    id: u64,
    start: u64,
    count: usize,
    encoding: ArtifactByteEncoding,
    output_cap: usize,
) -> Result<std::result::Result<ArtifactRead, String>> {
    let encoded_len = match encoding {
        ArtifactByteEncoding::Utf8 => count,
        ArtifactByteEncoding::Base64 => match base64_encoded_len(count) {
            Some(length) => length,
            None => {
                return Ok(Err(format!(
                    "base64 byte count {count} overflows its encoded length"
                )));
            }
        },
    };
    if encoded_len > output_cap {
        let max_count = match encoding {
            ArtifactByteEncoding::Utf8 => output_cap,
            ArtifactByteEncoding::Base64 => max_base64_input_bytes(output_cap),
        };
        let mode = match encoding {
            ArtifactByteEncoding::Utf8 => "raw UTF-8",
            ArtifactByteEncoding::Base64 => "base64",
        };
        return Ok(Err(format!(
            "{mode} artifact window encodes to {encoded_len} bytes, exceeding the \
             {output_cap}-byte output cap; request COUNT<={max_count}"
        )));
    }

    let count_u64 = u64::try_from(count).context("artifact byte count exceeds u64")?;
    let Some(end) = start.checked_add(count_u64) else {
        return Ok(Err(format!(
            "artifact byte window START+COUNT overflows: {start}+{count}"
        )));
    };
    if end > artifact_bytes {
        return Ok(Err(format!(
            "artifact://{id} exact byte window [{start}, {end}) exceeds artifact length \
             {artifact_bytes}; lower START or COUNT"
        )));
    }

    file.seek(SeekFrom::Start(start))
        .context("seeking to artifact byte window")?;
    let mut bytes = vec![0_u8; count];
    file.read_exact(&mut bytes)
        .context("reading exact artifact byte window")?;
    match encoding {
        ArtifactByteEncoding::Utf8 => match String::from_utf8(bytes) {
            Ok(text) => Ok(Ok(ArtifactRead::Raw(text))),
            Err(error) => {
                let invalid = error.utf8_error();
                let artifact_offset =
                    start.saturating_add(usize_to_u64_saturating(invalid.valid_up_to()));
                Ok(Err(format!(
                    "artifact://{id}:bytes={start}+{count} is not valid UTF-8 at window byte \
                     {window_offset} (artifact byte {artifact_offset}); START and START+COUNT \
                     must be UTF-8 character boundaries and all selected bytes must be valid \
                     UTF-8. Use artifact://{id}:base64={start}+{count} for arbitrary bytes",
                    window_offset = invalid.valid_up_to(),
                )))
            }
        },
        ArtifactByteEncoding::Base64 => Ok(Ok(ArtifactRead::Base64(base64_encode(&bytes)))),
    }
}

const fn base64_encoded_len(input_bytes: usize) -> Option<usize> {
    match input_bytes.checked_add(2) {
        Some(rounded) => match (rounded / 3).checked_mul(4) {
            Some(length) => Some(length),
            None => None,
        },
        None => None,
    }
}

const fn max_base64_input_bytes(output_cap: usize) -> usize {
    (output_cap / 4).saturating_mul(3)
}

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

struct ArtifactLineWindow {
    artifact_id: u64,
    offset: usize,
    limit: usize,
    byte_budget: usize,
    continuation_output_cap: usize,
    emitted_bytes: usize,
    last_emitted: usize,
    budget_reached: bool,
    collected: Vec<String>,
}

impl ArtifactLineWindow {
    fn push(
        &mut self,
        line_number: usize,
        raw_prefix: &[u8],
        raw_line_bytes: usize,
        line_start_byte: u64,
        ends_with_cr: bool,
    ) {
        if line_number < self.offset || self.collected.len() >= self.limit || self.budget_reached {
            return;
        }
        let content_bytes = raw_line_bytes.saturating_sub(usize::from(ends_with_cr));
        let captured = &raw_prefix[..raw_prefix.len().min(content_bytes)];
        let display = if content_bytes > MAX_LINE_LENGTH {
            artifact_truncated_line(
                captured,
                content_bytes,
                line_start_byte,
                self.artifact_id,
                self.continuation_output_cap,
            )
        } else {
            truncate_line(&String::from_utf8_lossy(captured))
        };
        let formatted = format!("L{line_number}: {display}");
        let payload_budget = self.byte_budget.saturating_sub(512);
        if !self.collected.is_empty()
            && self.emitted_bytes.saturating_add(formatted.len() + 1) > payload_budget
        {
            self.budget_reached = true;
            return;
        }
        self.emitted_bytes = self.emitted_bytes.saturating_add(formatted.len() + 1);
        self.collected.push(formatted);
        self.last_emitted = line_number;
    }

    const fn is_complete(&self) -> bool {
        self.budget_reached || self.collected.len() >= self.limit
    }

    fn finish(mut self, more_lines_may_follow: bool) -> Vec<String> {
        if self.budget_reached {
            self.collected.push(format!(
                "... [read byte budget of {budget} bytes reached: showing lines {offset}-{last}; \
                 more lines may follow; continue with artifact://{id}:lines={next}]",
                budget = self.byte_budget,
                offset = self.offset,
                last = self.last_emitted,
                id = self.artifact_id,
                next = self.last_emitted.saturating_add(1),
            ));
        } else if more_lines_may_follow && !self.collected.is_empty() {
            self.collected.push(format!(
                "... [showing lines {offset}-{last}; more lines may follow; continue with \
                 artifact://{id}:lines={next}]",
                offset = self.offset,
                last = self.last_emitted,
                id = self.artifact_id,
                next = self.last_emitted.saturating_add(1),
            ));
        }
        self.collected
    }
}

fn artifact_truncated_line(
    captured: &[u8],
    content_bytes: usize,
    line_start_byte: u64,
    artifact_id: u64,
    output_cap: usize,
) -> String {
    let (display, consumed, encoding) = valid_utf8_prefix_len(captured, MAX_LINE_LENGTH)
        .map_or_else(
            || {
                (
                    super::truncate_str(&String::from_utf8_lossy(captured), MAX_LINE_LENGTH)
                        .to_owned(),
                    0,
                    ArtifactByteEncoding::Base64,
                )
            },
            |consumed| {
                (
                    String::from_utf8_lossy(&captured[..consumed]).into_owned(),
                    consumed,
                    ArtifactByteEncoding::Utf8,
                )
            },
        );
    let remaining = content_bytes.saturating_sub(consumed);
    let max_count = match encoding {
        ArtifactByteEncoding::Utf8 => output_cap,
        ArtifactByteEncoding::Base64 => max_base64_input_bytes(output_cap),
    };
    let candidate_count = remaining.min(max_count);
    let (mode, count) = match encoding {
        ArtifactByteEncoding::Utf8 => {
            let continuation =
                &captured[consumed..captured.len().min(consumed.saturating_add(candidate_count))];
            match valid_utf8_prefix_len(continuation, candidate_count) {
                Some(0) | None => ("base64", remaining.min(max_base64_input_bytes(output_cap))),
                Some(count) => ("bytes", count),
            }
        }
        ArtifactByteEncoding::Base64 => ("base64", candidate_count),
    };
    let start = line_start_byte.saturating_add(usize_to_u64_saturating(consumed));
    format!(
        "{display}... [line truncated after {consumed} of {content_bytes} bytes; \
         {remaining} bytes remain; continue with exact next window \
         artifact://{artifact_id}:{mode}={start}+{count}]"
    )
}

fn valid_utf8_prefix_len(bytes: &[u8], maximum: usize) -> Option<usize> {
    let candidate = &bytes[..bytes.len().min(maximum)];
    match std::str::from_utf8(candidate) {
        Ok(_) => Some(candidate.len()),
        Err(error) if error.error_len().is_none() => Some(error.valid_up_to()),
        Err(_) => None,
    }
}

fn stream_artifact_lines(
    mut source: impl Read,
    artifact_id: u64,
    offset: usize,
    limit: usize,
    byte_budget: usize,
    continuation_output_cap: usize,
) -> Result<Vec<String>> {
    let mut window = ArtifactLineWindow {
        artifact_id,
        offset,
        limit,
        byte_budget,
        continuation_output_cap,
        emitted_bytes: 0,
        last_emitted: 0,
        budget_reached: false,
        collected: Vec::new(),
    };
    let capture_limit = MAX_LINE_LENGTH
        .saturating_add(continuation_output_cap)
        .saturating_add(4);
    let mut line_number = 1_usize;
    let mut line = Vec::with_capacity(capture_limit);
    let mut line_bytes = 0_usize;
    let mut line_start_byte = 0_u64;
    let mut absolute_byte = 0_u64;
    let mut ends_with_cr = false;
    let mut chunk = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = source
            .read(chunk.as_mut())
            .context("reading artifact stream")?;
        if read == 0 {
            window.push(
                line_number,
                &line,
                line_bytes,
                line_start_byte,
                ends_with_cr,
            );
            return Ok(window.finish(false));
        }
        for byte in &chunk[..read] {
            if *byte == b'\n' {
                window.push(
                    line_number,
                    &line,
                    line_bytes,
                    line_start_byte,
                    ends_with_cr,
                );
                if window.is_complete() {
                    return Ok(window.finish(true));
                }
                line_number = line_number.saturating_add(1);
                line.clear();
                line_bytes = 0;
                ends_with_cr = false;
                line_start_byte = absolute_byte.saturating_add(1);
            } else {
                line_bytes = line_bytes.saturating_add(1);
                ends_with_cr = *byte == b'\r';
                if line_number >= offset && !window.is_complete() && line.len() < capture_limit {
                    line.push(*byte);
                }
            }
            absolute_byte = absolute_byte.saturating_add(1);
        }
    }
}

fn parse_artifact_selector(selector: &str) -> Result<ArtifactSelector, &'static str> {
    if let Some(lines) = selector.strip_prefix("lines=") {
        let (offset, limit) = parse_line_selector(lines)?;
        return Ok(ArtifactSelector::Lines { offset, limit });
    }
    if let Some(bytes) = selector.strip_prefix("bytes=") {
        let (start, count) = parse_byte_selector(bytes)?;
        return Ok(ArtifactSelector::Bytes {
            start,
            count,
            encoding: ArtifactByteEncoding::Utf8,
        });
    }
    if let Some(bytes) = selector.strip_prefix("base64=") {
        let (start, count) = parse_byte_selector(bytes)?;
        return Ok(ArtifactSelector::Bytes {
            start,
            count,
            encoding: ArtifactByteEncoding::Base64,
        });
    }
    if selector.contains('=') {
        return Err("unknown selector kind");
    }
    let (offset, limit) = parse_line_selector(selector)?;
    Ok(ArtifactSelector::Lines { offset, limit })
}

fn parse_byte_selector(selector: &str) -> Result<(u64, usize), &'static str> {
    let Some((start, count)) = selector.split_once('+') else {
        return Err("byte selectors require START+COUNT");
    };
    let start = start
        .parse::<u64>()
        .map_err(|_| "byte START must be a non-negative integer")?;
    let count = count
        .parse::<usize>()
        .map_err(|_| "byte COUNT must be a positive integer")?;
    if count == 0 {
        return Err("byte COUNT must be greater than zero");
    }
    let count_u64 = u64::try_from(count).map_err(|_| "byte COUNT exceeds u64")?;
    start
        .checked_add(count_u64)
        .ok_or("byte START+COUNT overflows")?;
    Ok((start, count))
}

/// Parse a `read` line selector: `N`, `N-M` (inclusive), or `N+K`.
fn parse_line_selector(selector: &str) -> Result<(usize, usize), &'static str> {
    let parse_line = |raw: &str| -> Result<usize, &'static str> {
        match raw.parse::<usize>() {
            Ok(0) => Err("line numbers are 1-indexed"),
            Ok(value) => Ok(value),
            Err(_) => Err("expected a positive line number"),
        }
    };
    if let Some((start, end)) = selector.split_once('-') {
        let start = parse_line(start)?;
        let end = parse_line(end)?;
        return end
            .checked_sub(start)
            .and_then(|width| width.checked_add(1))
            .map(|limit| (start, limit))
            .ok_or("range end is before its start or too large");
    }
    if let Some((start, count)) = selector.split_once('+') {
        let start = parse_line(start)?;
        let count = parse_line(count)?;
        return Ok((start, count));
    }
    Ok((parse_line(selector)?, usize::MAX))
}

fn read_lines(content: &str, offset: usize, limit: usize, byte_budget: usize) -> Vec<String> {
    let total_lines = content.split('\n').count();
    let mut collected = Vec::new();
    let mut line_number = 0usize;
    let mut last_emitted = 0usize;
    let mut emitted_bytes = 0usize;
    let mut budget_reached = false;

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
        let formatted = format!("L{line_number}: {display}");
        // +1 for the joining newline. Always emit at least one line so a
        // budget smaller than a single line cannot yield an empty success.
        if !collected.is_empty() && emitted_bytes + formatted.len() + 1 > byte_budget {
            budget_reached = true;
            break;
        }
        emitted_bytes += formatted.len() + 1;
        collected.push(formatted);
        last_emitted = line_number;
    }

    // Unlike a silent stop, tell the model the file continues so it can read
    // further with offset/limit instead of assuming it saw everything.
    if budget_reached {
        // No bytes are destroyed — the file is intact on disk; the notice
        // names the budget and the exact continuation point.
        collected.push(format!(
            "... [read byte budget of {byte_budget} bytes reached: showing lines {offset}-{last_emitted} of {total_lines}; continue with offset={next} (an explicit limit widens the budget to limit*{BYTES_PER_REQUESTED_LINE} bytes)]",
            next = last_emitted + 1
        ));
    } else if !collected.is_empty() && last_emitted < total_lines {
        collected.push(format!(
            "... [showing lines {offset}-{last_emitted} of {total_lines}; use offset/limit to read more]"
        ));
    }

    collected
}

fn truncate_line(line: &str) -> String {
    if line.len() <= MAX_LINE_LENGTH {
        line.to_string()
    } else {
        format!(
            "{}{LINE_TRUNCATION_MARKER}",
            super::truncate_str(line, MAX_LINE_LENGTH)
        )
    }
}

fn sniff_media_type(file: &mut std::fs::File) -> Result<Option<&'static str>> {
    let mut prefix = [0_u8; 12];
    let mut read = 0;
    while read < prefix.len() {
        let count = file
            .read(&mut prefix[read..])
            .context("reading artifact media signature")?;
        if count == 0 {
            break;
        }
        read += count;
    }
    file.seek(SeekFrom::Start(0))
        .context("rewinding artifact after media detection")?;
    Ok(detect_media_magic(&prefix[..read]))
}

pub fn detect_media_magic(prefix: &[u8]) -> Option<&'static str> {
    if prefix.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if prefix.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if prefix.starts_with(b"GIF87a") || prefix.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if prefix.len() >= 12 && &prefix[..4] == b"RIFF" && &prefix[8..12] == b"WEBP" {
        Some("image/webp")
    } else if prefix.starts_with(b"%PDF-") {
        Some("application/pdf")
    } else {
        None
    }
}

/// Detect supported binary media types by file extension.
fn detect_media_type(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path).extension()?.to_ascii_lowercase();

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
        assert!(result.output.starts_with("L1: alpha\nL2: beta"));
        assert!(result.output.contains("showing lines 1-2 of 3"));
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
        assert!(result.output.starts_with("L2: beta\nL3: gamma"));
        assert!(result.output.contains("showing lines 2-3 of 5"));
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
        assert!(result.output.starts_with("L2: beta"));
        assert!(result.output.contains("showing lines 2-2 of 3"));
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
        assert!(result.output.starts_with(&format!("L1: {expected}")));
        assert!(result.output.contains(LINE_TRUNCATION_MARKER));
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
        assert!(
            result
                .output
                .starts_with("L50: line 50\nL51: line 51\nL52: line 52")
        );
        assert!(result.output.contains("showing lines 50-52 of 100"));
        Ok(())
    }

    #[tokio::test]
    async fn tool_metadata() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::Read);
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);

        let schema = Tool::<()>::input_schema(&tool);
        assert!(schema["properties"].get("path").is_some());
        assert!(schema["properties"].get("offset").is_some());
        assert!(schema["properties"].get("limit").is_some());
    }

    #[test]
    fn read_lines_basic() {
        let lines = read_lines("alpha\nbeta\ngamma", 1, 2000, TOTAL_BYTE_BUDGET_FLOOR);
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
        let lines = read_lines("a\nb\nc\nd\ne", 2, 2, TOTAL_BYTE_BUDGET_FLOOR);
        assert_eq!(
            lines,
            vec![
                "L2: b".to_string(),
                "L3: c".to_string(),
                "... [showing lines 2-3 of 5; use offset/limit to read more]".to_string(),
            ]
        );
    }

    #[test]
    fn read_lines_no_continuation_marker_when_complete() {
        let lines = read_lines("a\nb\nc", 1, 2000, TOTAL_BYTE_BUDGET_FLOOR);
        assert_eq!(
            lines,
            vec![
                "L1: a".to_string(),
                "L2: b".to_string(),
                "L3: c".to_string()
            ]
        );
    }

    #[test]
    fn read_lines_offset_past_end_returns_empty() {
        let lines = read_lines("only", 5, 10, TOTAL_BYTE_BUDGET_FLOOR);
        assert!(lines.is_empty());
    }

    #[test]
    fn read_lines_byte_budget_stops_with_notice_and_continuation() {
        // 40 lines of 100 bytes each = ~4.3KB formatted; an 1KB budget
        // stops early with the notice naming budget + continuation offset.
        let content = vec!["x".repeat(100); 40].join("\n");
        let lines = read_lines(&content, 1, 2000, 1024);

        let notice = lines.last().expect("notice line");
        assert!(
            notice.contains("read byte budget of 1024 bytes reached"),
            "notice must name the budget: {notice}"
        );
        assert!(
            notice.contains("of 40"),
            "notice must name the total line count: {notice}"
        );
        let emitted = lines.len() - 1;
        assert!(emitted < 40, "budget must stop before the line limit");
        assert!(
            notice.contains(&format!("continue with offset={}", emitted + 1)),
            "notice must name the exact continuation offset: {notice}"
        );
        let body_bytes: usize = lines[..emitted].iter().map(|l| l.len() + 1).sum();
        assert!(body_bytes <= 1024, "emitted body must fit the budget");
    }

    #[test]
    fn read_lines_always_emits_at_least_one_line() {
        // A budget smaller than one formatted line must not yield an
        // empty read (which the caller reports as offset-past-end).
        let content = "y".repeat(200);
        let lines = read_lines(&content, 1, 2000, 8);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("L1: "));
    }

    #[tokio::test]
    async fn default_read_of_oversized_file_is_byte_budgeted() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // ~80KB over 200 lines: under the old behavior the whole file
        // (well past 50KB) landed inline because 200 < 2000 lines.
        let content = vec!["z".repeat(400); 200].join("\n");
        fs.write_file("big.log", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/big.log"}))
            .await?;

        assert!(result.success);
        assert!(
            result.output.len() < 52 * 1024,
            "inline output must stay near the 50KB floor, got {} bytes",
            result.output.len()
        );
        assert!(
            result
                .output
                .contains(&format!("read byte budget of {} bytes reached", 50 * 1024)),
            "output must carry the budget notice: …{}",
            &result.output[result.output.len().saturating_sub(300)..]
        );
        assert!(result.output.contains("continue with offset="));
        Ok(())
    }

    #[tokio::test]
    async fn explicit_limit_widens_the_byte_budget() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        // Same ~80KB file: an explicit limit of 300 lines widens the
        // budget to 300*512 = 150KB, so the read completes in full.
        let content = vec!["z".repeat(400); 200].join("\n");
        fs.write_file("big.log", &content).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &tool_ctx(),
                json!({"path": "/workspace/big.log", "limit": 300}),
            )
            .await?;

        assert!(result.success);
        assert!(
            !result.output.contains("read byte budget"),
            "explicit-limit read within its widened budget must not truncate"
        );
        assert!(result.output.contains("L200: "));
        Ok(())
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

    #[tokio::test]
    async fn rejects_oversized_text_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let big = vec![b'a'; MAX_FILE_BYTES + 1];
        fs.write_file_bytes("big.txt", &big).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/big.txt"}))
            .await?;

        assert!(!result.success);
        assert!(result.output.contains("read limit"));
        Ok(())
    }

    #[tokio::test]
    async fn rejects_oversized_media_file() -> anyhow::Result<()> {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let big = vec![0u8; MAX_MEDIA_BYTES + 1];
        fs.write_file_bytes("big.png", &big).await?;

        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(&tool_ctx(), json!({"path": "/workspace/big.png"}))
            .await?;

        // Must fail before base64-encoding and never attach a document.
        assert!(!result.success);
        assert!(result.output.contains("attachment limit"));
        assert!(result.documents.is_empty());
        Ok(())
    }

    #[test]
    fn truncate_line_appends_marker() {
        let long = "x".repeat(MAX_LINE_LENGTH + 10);
        let out = truncate_line(&long);
        assert!(out.starts_with(&"x".repeat(MAX_LINE_LENGTH)));
        assert!(out.ends_with(LINE_TRUNCATION_MARKER));
    }

    #[test]
    fn truncate_line_short_unchanged() {
        assert_eq!(truncate_line("short"), "short");
    }
    // ── artifact:// read-back ────────────────────────────────────────

    fn artifact_ctx(store: Arc<agent_sdk_tools::artifacts::ArtifactStore>) -> ToolContext<()> {
        ToolContext::new(()).with_artifact_store(store)
    }

    fn artifact_fixture() -> anyhow::Result<(
        tempfile::TempDir,
        Arc<agent_sdk_tools::artifacts::ArtifactStore>,
        u64,
    )> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let content: String = (1..=100).fold(String::new(), |mut acc, n| {
            use std::fmt::Write as _;
            let _ = writeln!(acc, "line {n}");
            acc
        });
        let saved = store.save("bash", &content)?;
        Ok((dir, store, saved.id))
    }

    #[tokio::test]
    async fn unwindowed_png_artifact_returns_byte_identical_document() -> anyhow::Result<()> {
        use base64::Engine as _;

        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let png = b"\x89PNG\r\n\x1a\nsnapcompact attachment bytes".to_vec();
        let saved = store.save_streamed(
            "snapcompact-attachment",
            &mut std::io::Cursor::new(png.as_slice()),
        )?;
        let tool = create_test_tool(
            Arc::new(InMemoryFileSystem::new("/workspace")),
            AgentCapabilities::full_access(),
        );

        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{}", saved.id)}),
            )
            .await?;

        assert!(result.success, "{}", result.output);
        let [document] = result.documents.as_slice() else {
            anyhow::bail!("media artifact read must return one document");
        };
        assert_eq!(document.media_type, "image/png");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&document.data)
            .context("decoding recovered PNG")?;
        assert_eq!(decoded, png);
        Ok(())
    }

    #[test]
    fn artifact_media_size_boundaries_are_separate_from_filesystem_media() {
        let twenty_mib = 20_u64 * 1024 * 1024;
        let thirty_two_mib = 32_u64 * 1024 * 1024;

        assert_eq!(artifact_media_capacity(twenty_mib), Ok(20 * 1024 * 1024));
        assert_eq!(
            artifact_media_capacity(thirty_two_mib),
            Ok(MAX_ARTIFACT_MEDIA_BYTES)
        );
        assert!(
            artifact_media_capacity(thirty_two_mib + 1)
                .is_err_and(|error| error.contains("attachment limit"))
        );
        assert_eq!(MAX_MEDIA_BYTES, 5 * 1024 * 1024);
    }

    #[tokio::test]
    async fn rejects_over_32_mib_media_artifact_before_allocation() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let mut png = std::io::Cursor::new(&b"\x89PNG\r\n\x1a\n"[..]);
        let saved = store.save_streamed("snapcompact-attachment", &mut png)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&saved.path)?
            .set_len(usize_to_u64_saturating(
                MAX_ARTIFACT_MEDIA_BYTES.saturating_add(1),
            ))?;
        let tool = create_test_tool(
            Arc::new(InMemoryFileSystem::new("/workspace")),
            AgentCapabilities::full_access(),
        );

        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{}", saved.id)}),
            )
            .await?;

        assert!(!result.success);
        assert!(
            result.output.contains("attachment limit"),
            "{}",
            result.output
        );
        assert!(result.documents.is_empty());
        Ok(())
    }

    struct CountingReader {
        source: std::io::Cursor<Vec<u8>>,
        bytes_read: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Read for CountingReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let read = self.source.read(buffer)?;
            self.bytes_read
                .fetch_add(read, std::sync::atomic::Ordering::Relaxed);
            Ok(read)
        }
    }

    #[test]
    fn prefix_artifact_window_stops_before_consuming_eof() -> anyhow::Result<()> {
        let content = format!("first\n{}", "later\n".repeat(100_000)).into_bytes();
        let content_len = content.len();
        let bytes_read = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let reader = CountingReader {
            source: std::io::Cursor::new(content),
            bytes_read: Arc::clone(&bytes_read),
        };
        let result = stream_artifact_lines(reader, 0, 1, 1, 1024, 1024)?;
        assert_eq!(result.first().map(String::as_str), Some("L1: first"));
        assert!(result.join("\n").contains("more lines may follow"));
        assert!(
            bytes_read.load(std::sync::atomic::Ordering::Relaxed) < content_len,
            "prefix recovery must not scan to EOF"
        );
        Ok(())
    }

    #[tokio::test]
    async fn reads_artifact_with_offset_and_limit() -> anyhow::Result<()> {
        let (_dir, store, id) = artifact_fixture()?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{id}"), "offset": 5, "limit": 2}),
            )
            .await?;
        assert!(result.success, "{}", result.output);
        assert!(result.output.starts_with("L5: line 5\nL6: line 6"));
        assert!(result.output.contains("showing lines 5-6"));
        Ok(())
    }

    #[tokio::test]
    async fn reads_artifact_with_range_selector() -> anyhow::Result<()> {
        let (_dir, store, id) = artifact_fixture()?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{id}:lines=10-12")}),
            )
            .await?;
        assert!(result.success, "{}", result.output);
        assert!(
            result
                .output
                .starts_with("L10: line 10\nL11: line 11\nL12: line 12")
        );
        Ok(())
    }

    #[tokio::test]
    async fn reads_artifact_with_count_selector() -> anyhow::Result<()> {
        let (_dir, store, id) = artifact_fixture()?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{id}:lines=99+2")}),
            )
            .await?;
        assert!(result.success, "{}", result.output);
        assert!(result.output.starts_with("L99: line 99\nL100: line 100"));
        Ok(())
    }

    #[tokio::test]
    async fn long_utf8_artifact_line_reassembles_from_exact_raw_windows() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let line = "λ".repeat(60 * 1024);
        assert!(line.len() > 100 * 1024);
        let saved = store.save("snapcompact-summary", &line)?;
        let tool = create_test_tool(
            Arc::new(InMemoryFileSystem::new("/workspace")),
            AgentCapabilities::full_access(),
        );

        let result = tool
            .execute(
                &artifact_ctx(Arc::clone(&store)),
                json!({"path": format!("artifact://{}:lines=1+1", saved.id)}),
            )
            .await?;
        assert!(result.success, "{}", result.output);
        let numbered = result
            .output
            .strip_prefix("L1: ")
            .context("missing first artifact line prefix")?;
        let (displayed_prefix, _) = numbered
            .split_once("... [line truncated after")
            .context("missing exact long-line continuation marker")?;
        let mut recovered = displayed_prefix.as_bytes().to_vec();
        let window_cap = store
            .inline_budget()
            .min(MAX_ARTIFACT_BYTE_WINDOW_OUTPUT_BYTES);
        let first_count = (line.len() - recovered.len()).min(window_cap);
        assert!(
            result.output.contains(&format!(
                "artifact://{}:bytes={}+{}",
                saved.id,
                recovered.len(),
                first_count
            )),
            "{}",
            result.output
        );

        while recovered.len() < line.len() {
            let start = recovered.len();
            let count = (line.len() - start).min(window_cap);
            let chunk = tool
                .execute(
                    &artifact_ctx(Arc::clone(&store)),
                    json!({
                        "path": format!("artifact://{}:bytes={start}+{count}", saved.id)
                    }),
                )
                .await?;
            assert!(chunk.success, "{}", chunk.output);
            assert_eq!(chunk.output.len(), count);
            recovered.extend_from_slice(chunk.output.as_bytes());
        }

        assert_eq!(recovered, line.as_bytes());
        Ok(())
    }

    #[tokio::test]
    async fn binary_artifact_reassembles_from_exact_base64_windows() -> anyhow::Result<()> {
        use base64::Engine as _;

        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let original: Vec<u8> = (0_u8..=u8::MAX).cycle().take(150 * 1024 + 7).collect();
        let saved =
            store.save_streamed("binary", &mut std::io::Cursor::new(original.as_slice()))?;
        let tool = create_test_tool(
            Arc::new(InMemoryFileSystem::new("/workspace")),
            AgentCapabilities::full_access(),
        );
        let output_cap = store
            .inline_budget()
            .min(MAX_ARTIFACT_BYTE_WINDOW_OUTPUT_BYTES);
        let input_cap = max_base64_input_bytes(output_cap);
        let mut recovered = Vec::with_capacity(original.len());

        while recovered.len() < original.len() {
            let start = recovered.len();
            let count = (original.len() - start).min(input_cap);
            let result = tool
                .execute(
                    &artifact_ctx(Arc::clone(&store)),
                    json!({
                        "path": format!("artifact://{}:base64={start}+{count}", saved.id)
                    }),
                )
                .await?;
            assert!(result.success, "{}", result.output);
            assert_eq!(
                result.output.len(),
                base64_encoded_len(count).unwrap_or_default()
            );
            recovered.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(&result.output)
                    .context("decoding exact artifact byte window")?,
            );
        }

        assert_eq!(recovered, original);
        Ok(())
    }

    #[test]
    fn rejects_ambiguous_or_invalid_artifact_selectors() {
        for selector in [
            "",
            "lines=0",
            "bytes=1-2",
            "bytes=0+0",
            "base64=0",
            "base64=0+0",
            "raw=0+1",
            "bytes=18446744073709551615+1",
        ] {
            assert!(
                parse_artifact_selector(selector).is_err(),
                "selector should be rejected: {selector}"
            );
        }
    }

    #[tokio::test]
    async fn raw_byte_windows_report_utf8_range_and_output_errors() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let saved = store.save("utf8", "aéz")?;
        let tool = create_test_tool(
            Arc::new(InMemoryFileSystem::new("/workspace")),
            AgentCapabilities::full_access(),
        );

        for selector in ["bytes=1+1", "bytes=2+1"] {
            let result = tool
                .execute(
                    &artifact_ctx(Arc::clone(&store)),
                    json!({"path": format!("artifact://{}:{selector}", saved.id)}),
                )
                .await?;
            assert!(!result.success);
            assert!(
                result.output.contains("not valid UTF-8"),
                "{}",
                result.output
            );
            assert!(
                result.output.contains("character boundaries"),
                "{}",
                result.output
            );
            assert!(result.output.contains(":base64="), "{}", result.output);
        }

        let output_cap = store
            .inline_budget()
            .min(MAX_ARTIFACT_BYTE_WINDOW_OUTPUT_BYTES);
        let oversized = tool
            .execute(
                &artifact_ctx(Arc::clone(&store)),
                json!({
                    "path": format!(
                        "artifact://{}:bytes=0+{}",
                        saved.id,
                        output_cap.saturating_add(1)
                    )
                }),
            )
            .await?;
        assert!(!oversized.success);
        assert!(
            oversized.output.contains("output cap"),
            "{}",
            oversized.output
        );

        let beyond_end = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{}:base64=3+2", saved.id)}),
            )
            .await?;
        assert!(!beyond_end.success);
        assert!(
            beyond_end.output.contains("exceeds artifact length"),
            "{}",
            beyond_end.output
        );
        Ok(())
    }

    #[tokio::test]
    async fn artifact_round_trip_recovers_spilled_bytes() -> anyhow::Result<()> {
        // Spill via the budget path, then read the recovered window back —
        // the full loop the footer promises.
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let full: String = (1..=20_000).fold(String::new(), |mut acc, n| {
            use std::fmt::Write as _;
            let _ = writeln!(acc, "row {n}");
            acc
        });
        let mut spilled = ToolResult::success(full.clone());
        let saved = store
            .apply_inline_budget(&mut spilled, "bash")?
            .expect("must spill");
        assert!(
            spilled
                .output
                .ends_with(&agent_sdk_tools::artifacts::artifact_footer(saved.id))
        );

        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{}:lines=9999-10001", saved.id)}),
            )
            .await?;
        assert!(result.success, "{}", result.output);
        assert_eq!(
            result.output,
            format!(
                "L9999: row 9999\nL10000: row 10000\nL10001: row 10001\n\
                 ... [showing lines 9999-10001; more lines may follow; continue with \
                 artifact://{}:lines=10002]",
                saved.id
            )
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_read_of_oversized_artifact_suggests_selectors() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(agent_sdk_tools::artifacts::ArtifactStore::new(
            dir.path().join("artifacts"),
        ));
        let saved = store.save("bash", &"x\n".repeat(60 * 1024))?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{}", saved.id)}),
            )
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("read a window"), "{}", result.output);
        Ok(())
    }

    #[tokio::test]
    async fn artifact_recovery_clamps_adversarial_limits_to_inline_budget() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(
            agent_sdk_tools::artifacts::ArtifactStore::new(dir.path().join("artifacts"))
                .with_inline_budget(1024),
        );
        let saved = store.save("mcp", &"recovery-line\n".repeat(100_000))?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());
        for input in [
            json!({"path": format!("artifact://{}:1+{}", saved.id, usize::MAX)}),
            json!({"path": format!("artifact://{}:1-{}", saved.id, usize::MAX)}),
            json!({
                "path": format!("artifact://{}", saved.id),
                "offset": 1,
                "limit": usize::MAX
            }),
        ] {
            let result = tool
                .execute(&artifact_ctx(Arc::clone(&store)), input)
                .await?;
            assert!(result.success, "{}", result.output);
            assert!(result.output.len() <= store.inline_budget());
            assert!(result.output.contains(":lines="));
        }
        Ok(())
    }

    #[tokio::test]
    async fn artifact_errors_are_actionable() -> anyhow::Result<()> {
        let (_dir, store, id) = artifact_fixture()?;
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let tool = create_test_tool(fs, AgentCapabilities::full_access());

        // No store configured.
        let result = tool
            .execute(&tool_ctx(), json!({"path": "artifact://0"}))
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("not configured"));

        // Unknown ID names the available ones.
        let result = tool
            .execute(
                &artifact_ctx(Arc::clone(&store)),
                json!({"path": "artifact://42"}),
            )
            .await?;
        assert!(!result.success);
        assert!(
            result.output.contains("available IDs: 1"),
            "{}",
            result.output
        );

        // Non-numeric ID.
        let result = tool
            .execute(
                &artifact_ctx(Arc::clone(&store)),
                json!({"path": "artifact://latest"}),
            )
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("must be numeric"));

        // Malformed selector.
        let result = tool
            .execute(
                &artifact_ctx(store),
                json!({"path": format!("artifact://{id}:9-3")}),
            )
            .await?;
        assert!(!result.success);
        assert!(result.output.contains("range end is before its start"));
        Ok(())
    }
}
