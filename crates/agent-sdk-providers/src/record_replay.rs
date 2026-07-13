//! Deterministic record / replay provider wrapper (feature `record-replay`).
//!
//! [`RecordReplayProvider`] wraps any [`LlmProvider`] and either:
//!
//! - **records** every `chat` / `chat_stream` interaction to a JSON *cassette*
//!   file while transparently forwarding to the inner provider, or
//! - **replays** a previously-recorded cassette with no network access,
//!   serving recorded responses keyed by a fingerprint of the request.
//!
//! Identical requests are served in record order (a per-key queue), so a
//! re-prompt loop that issues the same request twice replays both turns
//! deterministically.
//!
//! This is the building block for fast, hermetic provider tests and golden
//! transcripts: record once against a live provider, then replay forever.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_sdk_foundation::llm::{
    CacheConfig, ChatOutcome, ChatRequest, ChatResponse, ContentBlock, Effort, StopReason,
    ThinkingConfig, ThinkingMode, ToolChoice, Usage,
};
use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::provider::LlmProvider;
use crate::streaming::{StreamBox, StreamDelta, StreamErrorKind};

/// Whether a [`RecordReplayProvider`] captures live traffic or serves a
/// recorded cassette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordReplayMode {
    /// Forward to the inner provider and append each interaction to the
    /// cassette.
    Record,
    /// Serve interactions from the cassette without touching the network.
    Replay,
}

/// A provider wrapper that records interactions to, or replays them from, a
/// JSON cassette file.
pub struct RecordReplayProvider {
    inner: Option<Arc<dyn LlmProvider>>,
    mode: RecordReplayMode,
    path: PathBuf,
    recorded: Mutex<Cassette>,
    replay: Mutex<HashMap<String, VecDeque<CassetteInteraction>>>,
    model: String,
}

impl RecordReplayProvider {
    /// Wrap `inner` in record mode, writing the cassette to `path` (created /
    /// overwritten as interactions are captured).
    #[must_use]
    pub fn record(inner: Arc<dyn LlmProvider>, path: impl Into<PathBuf>) -> Self {
        let model = inner.model().to_owned();
        Self {
            inner: Some(inner),
            mode: RecordReplayMode::Record,
            path: path.into(),
            recorded: Mutex::new(Cassette {
                model: model.clone(),
                entries: Vec::new(),
            }),
            replay: Mutex::new(HashMap::new()),
            model,
        }
    }

    /// Open `path` in replay mode. No inner provider is required: every
    /// interaction is served from the cassette.
    ///
    /// # Errors
    ///
    /// Returns an error when the cassette cannot be read or parsed.
    pub fn replay(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let cassette = load_cassette(&path)?;
        let model = cassette.model.clone();
        let replay = build_replay_map(&cassette);
        Ok(Self {
            inner: None,
            mode: RecordReplayMode::Replay,
            path,
            recorded: Mutex::new(Cassette::default()),
            replay: Mutex::new(replay),
            model,
        })
    }

    /// The mode this provider operates in.
    #[must_use]
    pub const fn mode(&self) -> RecordReplayMode {
        self.mode
    }

    fn record_chat(&self, key: String, outcome: &ChatOutcome) -> Result<()> {
        self.persist(CassetteEntry {
            key,
            interaction: CassetteInteraction::Chat(CassetteOutcome::from_outcome(outcome)),
        })
    }

    fn record_stream(&self, key: String, deltas: Vec<CassetteDelta>) -> Result<()> {
        self.persist(CassetteEntry {
            key,
            interaction: CassetteInteraction::Stream(deltas),
        })
    }

    /// Append an entry to the in-memory cassette and flush it to disk.
    ///
    /// The cassette is serialized while the lock is held, then the guard is
    /// dropped *before* the blocking file write so the mutex is not held across
    /// I/O.
    fn persist(&self, entry: CassetteEntry) -> Result<()> {
        let json = {
            let mut cassette = self
                .recorded
                .lock()
                .map_err(|_| anyhow!("record cassette lock poisoned"))?;
            cassette.entries.push(entry);
            serde_json::to_string_pretty(&*cassette).context("serialize cassette")?
        };
        std::fs::write(&self.path, json)
            .with_context(|| format!("write cassette to {}", self.path.display()))
    }

    fn take_replay(&self, key: &str) -> Result<CassetteInteraction> {
        let mut map = self
            .replay
            .lock()
            .map_err(|_| anyhow!("replay cassette lock poisoned"))?;
        map.get_mut(key)
            .and_then(VecDeque::pop_front)
            .with_context(|| format!("no recorded interaction for request key '{key}'"))
    }
}

#[async_trait]
impl LlmProvider for RecordReplayProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        let key = entry_key("chat", &request);
        match self.mode {
            RecordReplayMode::Record => {
                let inner = self
                    .inner
                    .as_ref()
                    .context("record mode requires an inner provider")?;
                let outcome = inner.chat(request).await?;
                self.record_chat(key, &outcome)?;
                Ok(outcome)
            }
            RecordReplayMode::Replay => match self.take_replay(&key)? {
                CassetteInteraction::Chat(outcome) => Ok(outcome.into_outcome()),
                CassetteInteraction::Stream(_) => {
                    bail!("recorded interaction for '{key}' is a stream, not a chat")
                }
            },
        }
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        let key = entry_key("stream", &request);
        match self.mode {
            RecordReplayMode::Record => {
                let inner = self.inner.clone();
                Box::pin(async_stream::stream! {
                    let Some(inner) = inner else {
                        yield Err(anyhow!("record mode requires an inner provider"));
                        return;
                    };
                    let mut stream = inner.chat_stream(request);
                    let mut captured: Vec<CassetteDelta> = Vec::new();
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(delta) => {
                                let terminal_error = matches!(delta, StreamDelta::Error { .. });
                                captured.push(CassetteDelta::from_delta(&delta));
                                if terminal_error {
                                    // A caller stops reading at a terminal error and drops the
                                    // stream, so this generator is never resumed past the yield
                                    // below and the end-of-stream write would never run. Persist
                                    // the cassette while control is still here, or every recorded
                                    // failure interaction is lost.
                                    if let Err(error) =
                                        self.record_stream(key, std::mem::take(&mut captured))
                                    {
                                        log::warn!(
                                            "record/replay: failed to persist stream cassette: {error}"
                                        );
                                    }
                                    yield Ok(delta);
                                    return;
                                }
                                yield Ok(delta);
                            }
                            Err(error) => {
                                // A transport failure produced no terminal delta, so there is no
                                // faithful interaction to record: replaying the captured prefix
                                // would end the stream cleanly and hide the failure.
                                yield Err(error);
                                return;
                            }
                        }
                    }
                    if let Err(error) = self.record_stream(key, captured) {
                        log::warn!("record/replay: failed to persist stream cassette: {error}");
                    }
                })
            }
            RecordReplayMode::Replay => Box::pin(async_stream::stream! {
                match self.take_replay(&key) {
                    Ok(CassetteInteraction::Stream(deltas)) => {
                        for delta in deltas {
                            yield Ok(delta.into_delta());
                        }
                    }
                    Ok(CassetteInteraction::Chat(_)) => {
                        yield Err(anyhow!(
                            "recorded interaction for '{key}' is a chat, not a stream"
                        ));
                    }
                    Err(error) => yield Err(error),
                }
            }),
        }
    }

    /// Delegate live model discovery to the inner provider when one is present
    /// (record mode) so wrapping never silently loses `list_models`. In replay
    /// mode there is no inner provider and no recorded model list, so this
    /// reports the operation as unsupported.
    async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
        match &self.inner {
            Some(inner) => inner.list_models().await,
            None => bail!("list_models is not supported in replay mode (no inner provider)"),
        }
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "record-replay"
    }
}

// ── Cassette file format ──────────────────────────────────────────────

#[derive(Default, Serialize, Deserialize)]
struct Cassette {
    #[serde(default)]
    model: String,
    #[serde(default)]
    entries: Vec<CassetteEntry>,
}

#[derive(Serialize, Deserialize)]
struct CassetteEntry {
    key: String,
    interaction: CassetteInteraction,
}

#[derive(Clone, Serialize, Deserialize)]
enum CassetteInteraction {
    Chat(CassetteOutcome),
    Stream(Vec<CassetteDelta>),
}

#[derive(Clone, Serialize, Deserialize)]
enum CassetteOutcome {
    Success(CassetteResponse),
    /// Recorded rate limit, with the parsed `Retry-After` delay in millis.
    RateLimited(Option<u64>),
    InvalidRequest(String),
    ServerError(String),
}

impl CassetteOutcome {
    fn from_outcome(outcome: &ChatOutcome) -> Self {
        match outcome {
            ChatOutcome::Success(response) => {
                Self::Success(CassetteResponse::from_response(response))
            }
            ChatOutcome::RateLimited(retry_after) => {
                Self::RateLimited(retry_after.map(millis_from_duration))
            }
            ChatOutcome::InvalidRequest(msg) => Self::InvalidRequest(msg.clone()),
            ChatOutcome::ServerError(msg) => Self::ServerError(msg.clone()),
            // `ChatOutcome` is `#[non_exhaustive]`; record an unknown outcome
            // as a server error so replay still surfaces a failure.
            _ => Self::ServerError("unrecognized provider outcome".to_owned()),
        }
    }

    fn into_outcome(self) -> ChatOutcome {
        match self {
            Self::Success(response) => ChatOutcome::Success(response.into_response()),
            Self::RateLimited(ms) => ChatOutcome::RateLimited(ms.map(Duration::from_millis)),
            Self::InvalidRequest(msg) => ChatOutcome::InvalidRequest(msg),
            Self::ServerError(msg) => ChatOutcome::ServerError(msg),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CassetteResponse {
    id: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<StopReason>,
    usage: Usage,
}

impl CassetteResponse {
    fn from_response(response: &ChatResponse) -> Self {
        Self {
            id: response.id.clone(),
            content: response.content.clone(),
            model: response.model.clone(),
            stop_reason: response.stop_reason,
            usage: response.usage.clone(),
        }
    }

    fn into_response(self) -> ChatResponse {
        ChatResponse {
            id: self.id,
            content: self.content,
            model: self.model,
            stop_reason: self.stop_reason,
            usage: self.usage,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum CassetteDelta {
    TextDelta {
        delta: String,
        block_index: usize,
    },
    ThinkingDelta {
        delta: String,
        block_index: usize,
    },
    ToolUseStart {
        id: String,
        name: String,
        block_index: usize,
        thought_signature: Option<String>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        block_index: usize,
    },
    SignatureDelta {
        delta: String,
        block_index: usize,
    },
    RedactedThinking {
        data: String,
        block_index: usize,
    },
    OpaqueReasoning {
        provider: String,
        data: serde_json::Value,
        block_index: usize,
    },
    Usage(Usage),
    Done {
        stop_reason: Option<StopReason>,
    },
    Error {
        message: String,
        kind: CassetteErrorKind,
        /// Retry delay a rate-limited error carried, in millis.
        ///
        /// A separate optional field rather than a payload on
        /// [`CassetteErrorKind::RateLimited`] so cassettes recorded before
        /// retry hints were captured still deserialize.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
}

impl CassetteDelta {
    fn from_delta(delta: &StreamDelta) -> Self {
        match delta {
            StreamDelta::TextDelta { delta, block_index } => Self::TextDelta {
                delta: delta.clone(),
                block_index: *block_index,
            },
            StreamDelta::ThinkingDelta { delta, block_index } => Self::ThinkingDelta {
                delta: delta.clone(),
                block_index: *block_index,
            },
            StreamDelta::ToolUseStart {
                id,
                name,
                block_index,
                thought_signature,
            } => Self::ToolUseStart {
                id: id.clone(),
                name: name.clone(),
                block_index: *block_index,
                thought_signature: thought_signature.clone(),
            },
            StreamDelta::ToolInputDelta {
                id,
                delta,
                block_index,
            } => Self::ToolInputDelta {
                id: id.clone(),
                delta: delta.clone(),
                block_index: *block_index,
            },
            StreamDelta::SignatureDelta { delta, block_index } => Self::SignatureDelta {
                delta: delta.clone(),
                block_index: *block_index,
            },
            StreamDelta::RedactedThinking { data, block_index } => Self::RedactedThinking {
                data: data.clone(),
                block_index: *block_index,
            },
            StreamDelta::OpaqueReasoning {
                provider,
                data,
                block_index,
            } => Self::OpaqueReasoning {
                provider: provider.clone(),
                data: data.clone(),
                block_index: *block_index,
            },
            StreamDelta::Usage(usage) => Self::Usage(usage.clone()),
            StreamDelta::Done { stop_reason } => Self::Done {
                stop_reason: *stop_reason,
            },
            StreamDelta::Error { message, kind } => Self::Error {
                message: message.clone(),
                kind: CassetteErrorKind::from_kind(*kind),
                retry_after_ms: kind.retry_after().map(millis_from_duration),
            },
        }
    }

    fn into_delta(self) -> StreamDelta {
        match self {
            Self::TextDelta { delta, block_index } => StreamDelta::TextDelta { delta, block_index },
            Self::ThinkingDelta { delta, block_index } => {
                StreamDelta::ThinkingDelta { delta, block_index }
            }
            Self::ToolUseStart {
                id,
                name,
                block_index,
                thought_signature,
            } => StreamDelta::ToolUseStart {
                id,
                name,
                block_index,
                thought_signature,
            },
            Self::ToolInputDelta {
                id,
                delta,
                block_index,
            } => StreamDelta::ToolInputDelta {
                id,
                delta,
                block_index,
            },
            Self::SignatureDelta { delta, block_index } => {
                StreamDelta::SignatureDelta { delta, block_index }
            }
            Self::RedactedThinking { data, block_index } => {
                StreamDelta::RedactedThinking { data, block_index }
            }
            Self::OpaqueReasoning {
                provider,
                data,
                block_index,
            } => StreamDelta::OpaqueReasoning {
                provider,
                data,
                block_index,
            },
            Self::Usage(usage) => StreamDelta::Usage(usage),
            Self::Done { stop_reason } => StreamDelta::Done { stop_reason },
            Self::Error {
                message,
                kind,
                retry_after_ms,
            } => StreamDelta::Error {
                message,
                kind: kind.into_kind(retry_after_ms.map(Duration::from_millis)),
            },
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum CassetteErrorKind {
    RateLimited,
    ServerError,
    InvalidRequest,
    Unknown,
}

impl CassetteErrorKind {
    const fn from_kind(kind: StreamErrorKind) -> Self {
        match kind {
            StreamErrorKind::RateLimited(_) => Self::RateLimited,
            StreamErrorKind::ServerError => Self::ServerError,
            StreamErrorKind::InvalidRequest => Self::InvalidRequest,
            // `StreamErrorKind` is `#[non_exhaustive]`.
            _ => Self::Unknown,
        }
    }

    const fn into_kind(self, retry_after: Option<Duration>) -> StreamErrorKind {
        match self {
            Self::RateLimited => StreamErrorKind::RateLimited(retry_after),
            Self::ServerError => StreamErrorKind::ServerError,
            Self::InvalidRequest => StreamErrorKind::InvalidRequest,
            Self::Unknown => StreamErrorKind::Unknown,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

fn millis_from_duration(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// A stable fingerprint of every caller-visible field that can change a
/// provider response.
///
/// The canonical JSON stays in-process and is immediately hashed; only the
/// digest is stored in a cassette key or included in an error, so request
/// fields such as session identifiers never appear in logs.
fn fingerprint(request: &ChatRequest) -> String {
    let canonical = serde_json::json!({
        "system": request.system,
        "messages": request.messages,
        "tools": request.tools,
        "max_tokens": request.max_tokens,
        "max_tokens_explicit": request.max_tokens_explicit,
        "session_id": request.session_id,
        "cached_content": request.cached_content,
        "thinking": canonical_thinking(request.thinking.as_ref()),
        "tool_choice": canonical_tool_choice(request.tool_choice.as_ref()),
        "response_format": request.response_format,
        "cache": canonical_cache(request.cache.as_ref()),
    });
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    // FNV-1a (64-bit): self-contained and stable across runs, unlike the
    // randomized `RandomState` hasher.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn canonical_thinking(thinking: Option<&ThinkingConfig>) -> serde_json::Value {
    let Some(thinking) = thinking else {
        return serde_json::Value::Null;
    };

    let mode = match &thinking.mode {
        ThinkingMode::Enabled { budget_tokens } => serde_json::json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }),
        ThinkingMode::Adaptive => serde_json::json!({"type": "adaptive"}),
    };
    let effort = thinking.effort.map(|effort| match effort {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::Max => "max",
    });

    serde_json::json!({
        "mode": mode,
        "effort": effort,
    })
}

fn canonical_tool_choice(choice: Option<&ToolChoice>) -> serde_json::Value {
    match choice {
        None => serde_json::Value::Null,
        Some(ToolChoice::Auto) => serde_json::Value::String("auto".to_owned()),
        Some(ToolChoice::Tool(name)) => serde_json::json!({
            "type": "tool",
            "name": name,
        }),
    }
}

fn canonical_cache(cache: Option<&CacheConfig>) -> serde_json::Value {
    let Some(cache) = cache else {
        return serde_json::Value::Null;
    };

    serde_json::json!({
        "enabled": cache.enabled,
        "ttl": cache
            .ttl
            .map(agent_sdk_foundation::llm::CacheTtl::as_wire_str),
        "max_breakpoints": cache.max_breakpoints,
    })
}

fn entry_key(method: &str, request: &ChatRequest) -> String {
    format!("{method}:{}", fingerprint(request))
}

fn load_cassette(path: &Path) -> Result<Cassette> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("read cassette {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("parse cassette {}", path.display()))
}

fn build_replay_map(cassette: &Cassette) -> HashMap<String, VecDeque<CassetteInteraction>> {
    let mut map: HashMap<String, VecDeque<CassetteInteraction>> = HashMap::new();
    for entry in &cassette.entries {
        map.entry(entry.key.clone())
            .or_default()
            .push_back(entry.interaction.clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    use agent_sdk_foundation::llm::{CacheTtl, Message};

    /// A scripted inner provider used only in record mode.
    struct InnerProvider {
        model: &'static str,
        chat_outcome: ChatOutcome,
        deltas: Vec<StreamDelta>,
    }

    #[async_trait]
    impl LlmProvider for InnerProvider {
        async fn chat(&self, _request: ChatRequest) -> Result<ChatOutcome> {
            Ok(self.chat_outcome.clone())
        }

        async fn list_models(&self) -> Result<Vec<crate::provider::ModelInfo>> {
            Ok(vec![crate::provider::ModelInfo {
                id: "inner-discovered-model".to_owned(),
                display_name: None,
                context_window: None,
                max_output_tokens: None,
            }])
        }

        fn chat_stream(&self, _request: ChatRequest) -> StreamBox<'_> {
            let deltas = self.deltas.clone();
            Box::pin(async_stream::stream! {
                for delta in deltas {
                    yield Ok(delta);
                }
            })
        }

        fn model(&self) -> &str {
            self.model
        }

        fn provider(&self) -> &'static str {
            "inner"
        }
    }

    fn success_outcome(text: &str) -> ChatOutcome {
        ChatOutcome::Success(ChatResponse {
            id: "resp-1".to_owned(),
            content: vec![ContentBlock::Text {
                text: text.to_owned(),
            }],
            model: "inner-model".to_owned(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                input_tokens: 7,
                output_tokens: 3,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        })
    }

    fn temp_cassette_path() -> PathBuf {
        std::env::temp_dir().join(format!("agent-sdk-cassette-{}.json", uuid::Uuid::new_v4()))
    }

    fn request() -> ChatRequest {
        ChatRequest::new("sys", vec![Message::user("hello")])
    }

    #[test]
    fn fingerprint_distinguishes_response_shaping_request_controls() {
        let baseline = request();
        let mut explicitly_default_max_tokens = baseline.clone();
        explicitly_default_max_tokens.max_tokens_explicit = true;
        let mut cached_content = baseline.clone();
        cached_content.cached_content = Some("cached-content-1".to_owned());

        let variants = [
            baseline.clone(),
            baseline
                .clone()
                .with_thinking(ThinkingConfig::adaptive_with_effort(Effort::High)),
            baseline
                .clone()
                .with_tool_choice(ToolChoice::Tool("lookup".to_owned())),
            baseline.clone().with_cache(
                CacheConfig::enabled()
                    .with_ttl(CacheTtl::OneHour)
                    .with_max_breakpoints(2),
            ),
            baseline.with_session_id("session-1"),
            cached_content,
            explicitly_default_max_tokens,
        ];

        let mut fingerprints = std::collections::HashSet::new();
        for variant in &variants {
            assert!(
                fingerprints.insert(fingerprint(variant)),
                "response-shaping request controls must not share cassette keys"
            );
        }
    }

    #[tokio::test]
    async fn chat_round_trips_through_cassette() -> Result<()> {
        let path = temp_cassette_path();
        let inner = Arc::new(InnerProvider {
            model: "inner-model",
            chat_outcome: success_outcome("recorded answer"),
            deltas: Vec::new(),
        });

        // Record.
        let recorder = RecordReplayProvider::record(inner, &path);
        let live_outcome = recorder.chat(request()).await?;
        assert!(
            matches!(&live_outcome, ChatOutcome::Success(r) if r.first_text() == Some("recorded answer"))
        );

        // Replay — no inner provider, served straight from disk.
        let player = RecordReplayProvider::replay(&path)?;
        assert_eq!(player.mode(), RecordReplayMode::Replay);
        assert_eq!(player.model(), "inner-model");
        let replayed = player.chat(request()).await?;
        match replayed {
            ChatOutcome::Success(response) => {
                assert_eq!(response.first_text(), Some("recorded answer"));
                assert_eq!(response.usage.input_tokens, 7);
                assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
            }
            other => panic!("expected Success, got {other:?}"),
        }

        // A request with no recorded entry fails deterministically.
        let missing = player
            .chat(ChatRequest::new("other", vec![Message::user("nope")]))
            .await;
        assert!(missing.is_err());

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[tokio::test]
    async fn list_models_delegates_in_record_and_errors_in_replay() -> Result<()> {
        let path = temp_cassette_path();
        let inner = Arc::new(InnerProvider {
            model: "inner-model",
            chat_outcome: success_outcome("x"),
            deltas: Vec::new(),
        });

        // Record mode delegates discovery to the inner provider instead of
        // returning the trait-default "unsupported" error.
        let recorder = RecordReplayProvider::record(inner, &path);
        let models = recorder.list_models().await?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "inner-discovered-model");

        // Persist a cassette so replay has a file to load (list_models is not
        // captured to the cassette).
        recorder.chat(request()).await?;

        // Replay mode has no inner provider and no recorded model list, so it
        // reports the operation as unsupported rather than panicking.
        let player = RecordReplayProvider::replay(&path)?;
        assert!(player.list_models().await.is_err());

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[tokio::test]
    async fn stream_round_trips_through_cassette() -> Result<()> {
        let path = temp_cassette_path();
        let inner = Arc::new(InnerProvider {
            model: "inner-model",
            chat_outcome: success_outcome("unused"),
            deltas: vec![
                StreamDelta::TextDelta {
                    delta: "hel".to_owned(),
                    block_index: 0,
                },
                StreamDelta::TextDelta {
                    delta: "lo".to_owned(),
                    block_index: 0,
                },
                StreamDelta::Done {
                    stop_reason: Some(StopReason::EndTurn),
                },
            ],
        });

        // Record the stream (pass-through).
        let recorder = RecordReplayProvider::record(inner, &path);
        let mut text = String::new();
        let mut stream = recorder.chat_stream(request());
        while let Some(item) = stream.next().await {
            if let StreamDelta::TextDelta { delta, .. } = item? {
                text.push_str(&delta);
            }
        }
        drop(stream);
        assert_eq!(text, "hello");

        // Replay the stream from disk.
        let player = RecordReplayProvider::replay(&path)?;
        let mut replayed = String::new();
        let mut stop_seen = false;
        let mut stream = player.chat_stream(request());
        while let Some(item) = stream.next().await {
            match item? {
                StreamDelta::TextDelta { delta, .. } => replayed.push_str(&delta),
                StreamDelta::Done { .. } => stop_seen = true,
                _ => {}
            }
        }
        assert_eq!(replayed, "hello");
        assert!(stop_seen, "Done delta should replay");

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn cassette_delta_round_trips_opaque_reasoning_exactly() {
        let delta = StreamDelta::OpaqueReasoning {
            provider: "test-provider".to_owned(),
            data: serde_json::json!({
                "id": "reasoning_1",
                "summary": [],
                "encrypted_content": "ciphertext"
            }),
            block_index: 4,
        };

        let replayed = CassetteDelta::from_delta(&delta).into_delta();
        assert!(matches!(
            replayed,
            StreamDelta::OpaqueReasoning {
                provider,
                data,
                block_index: 4,
            } if provider == "test-provider"
                && data == serde_json::json!({
                    "id": "reasoning_1",
                    "summary": [],
                    "encrypted_content": "ciphertext"
                })
        ));
    }

    #[tokio::test]
    async fn error_stream_is_recorded_even_though_the_caller_drops_it() -> Result<()> {
        // The real lifecycle: a consumer stops at the terminal error and drops
        // the stream, so the generator never reaches end-of-stream. The cassette
        // (with the rate-limit hint) must still be on disk afterwards.
        let path = temp_cassette_path();
        let inner = Arc::new(InnerProvider {
            model: "inner-model",
            chat_outcome: success_outcome("unused"),
            deltas: vec![
                StreamDelta::TextDelta {
                    delta: "partial".to_owned(),
                    block_index: 0,
                },
                StreamDelta::Error {
                    message: "Rate limited".to_owned(),
                    kind: StreamErrorKind::RateLimited(Some(Duration::from_secs(30))),
                },
            ],
        });

        let recorder = RecordReplayProvider::record(inner, &path);
        {
            let mut stream = recorder.chat_stream(request());
            while let Some(item) = stream.next().await {
                if matches!(item?, StreamDelta::Error { .. }) {
                    break;
                }
            }
            // Dropped here, mid-stream, exactly as the agent loop drops it.
        }

        let player = RecordReplayProvider::replay(&path)?;
        let mut replayed = Vec::new();
        let mut stream = player.chat_stream(request());
        while let Some(item) = stream.next().await {
            replayed.push(item?);
        }
        drop(stream);

        assert!(
            replayed.iter().any(|delta| matches!(
                delta,
                StreamDelta::Error {
                    kind: StreamErrorKind::RateLimited(Some(delay)),
                    ..
                } if *delay == Duration::from_secs(30)
            )),
            "the recorded error delta and its retry hint must survive the drop, got {replayed:?}"
        );

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn cassette_delta_round_trips_the_rate_limit_retry_hint() -> Result<()> {
        let delta = StreamDelta::Error {
            message: "Rate limited".to_owned(),
            kind: StreamErrorKind::RateLimited(Some(Duration::from_secs(30))),
        };

        let recorded = CassetteDelta::from_delta(&delta);
        let json = serde_json::to_string(&recorded)?;
        let replayed = serde_json::from_str::<CassetteDelta>(&json)?.into_delta();

        assert!(matches!(
            replayed,
            StreamDelta::Error {
                kind: StreamErrorKind::RateLimited(Some(delay)),
                ..
            } if delay == Duration::from_secs(30)
        ));
        Ok(())
    }

    #[test]
    fn cassette_error_without_a_recorded_hint_replays_as_hintless() -> Result<()> {
        // Cassettes recorded before retry hints existed carry no
        // `retry_after_ms`; they must still deserialize.
        let json = r#"{"Error":{"message":"Rate limited","kind":"RateLimited"}}"#;
        let replayed = serde_json::from_str::<CassetteDelta>(json)?.into_delta();

        assert!(matches!(
            replayed,
            StreamDelta::Error {
                kind: StreamErrorKind::RateLimited(None),
                ..
            }
        ));
        Ok(())
    }
}
