//! Claude through Amazon Bedrock's runtime API.
//!
//! The request is the Anthropic Messages body, sent to `InvokeModel`
//! (`POST /model/{modelId}/invoke`) or `InvokeModelWithResponseStream`
//! (`POST /model/{modelId}/invoke-with-response-stream`). Bedrock takes the
//! API version and the beta flags in the body (`anthropic_version:
//! "bedrock-2023-05-31"`, `anthropic_beta: [...]`) and the model in the path,
//! so the body carries no `model` and no `stream`
//! (<https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html>).
//! A streamed response is AWS event-stream, not SSE: each `chunk` event
//! carries one Anthropic stream event, base64-encoded, and failures arrive as
//! typed exception events
//! (<https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html>).
//!
//! The provider reuses the Anthropic request types, cache breakpoints,
//! attachments, thinking and stream parser. It reports `provider() ==
//! "anthropic"`, the API shape, so capabilities, attachment limits and
//! pricing resolve on the Anthropic rows, and `route() == "bedrock"`, the
//! endpoint that serves the call. [`model`](LlmProvider::model) is the
//! Anthropic model id; the Bedrock model id (an inference profile id or ARN)
//! only goes on the wire. Pricing on the Anthropic rows matches Bedrock's
//! `global.` profiles; AWS prices the geographic profiles (`us.`, `eu.`, ...)
//! of recent models at a premium those rows do not carry.

pub mod credentials;
mod event_stream;
mod sigv4;

use std::time::Duration;

use agent_sdk_foundation::llm::{
    ChatOutcome, ChatRequest, ChatResponse, StopReason, ThinkingConfig, ThinkingMode, ToolChoice,
    Usage,
};
use anyhow::Result;
use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::Value;

pub use aws_credential_types::Credentials;
pub use aws_credential_types::provider::error::CredentialsError;
pub use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
pub use credentials::{BedrockAuth, WebIdentityCredentials, WebIdentityToken};

use crate::attachments::{fit_request_attachments, validate_request_attachments};
use crate::impls::anthropic::{
    AnthropicProvider, CacheRegions, MODEL_FABLE_5, MODEL_FABLE_51, MODEL_HAIKU_45, MODEL_HAIKU_55,
    MODEL_OPUS_5, MODEL_OPUS_46, MODEL_OPUS_47, MODEL_OPUS_48, MODEL_OPUS_55, MODEL_SONNET_5,
    MODEL_SONNET_45, MODEL_SONNET_46, MODEL_SONNET_55, data, json_output_format, schema,
};
use crate::provider::{LlmProvider, StructuredOutputSupport, thinking_for_forced_tool};
use crate::streaming::{
    StreamBox, StreamDelta, StreamErrorKind, reqwest_body_error_delta, reqwest_error_delta,
};

/// The `anthropic_version` Bedrock requires in the body.
const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
/// `SigV4` signing name of the Bedrock runtime endpoint.
const SIGNING_SERVICE: &str = "bedrock";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Bedrock asks for this beta with `output_config.effort` on Claude Opus 4.5;
/// adaptive-thinking models take effort without it.
const EFFORT_BETA: &str = "effort-2025-11-24";
/// The beta `thinking.block_binding` needs.
const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";
const EVENT_STREAM_CONTENT_TYPE: &str = "application/vnd.amazon.eventstream";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Deadline for the response headers of a streaming request. Bedrock compiles
/// a structured-output schema it has not seen in the last 24 hours before it
/// answers, which AWS says can take about 90 seconds.
const STREAM_HEADERS_TIMEOUT: Duration = Duration::from_mins(2);
/// Longest silence between event-stream bytes before the stream is treated as
/// a stalled connection. Kept under the 350-second idle timeout AWS documents
/// for NAT gateways and load balancers.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_mins(5);
/// Total deadline of a non-streaming request, as on the other providers.
const CHAT_REQUEST_TIMEOUT: Duration = Duration::from_mins(5);
const DEFAULT_SAFE_MAX_OUTPUT_TOKENS: u32 = 32_000;

/// Largest request body Bedrock accepts for Claude: 20 MB
/// (<https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy>).
const MAX_BODY_BYTES: usize = 20_000_000;
/// Aggregate decoded attachment budget: base64 grows it by a third, which
/// leaves about 3 MB of the body for text.
pub const BEDROCK_MAX_REQUEST_ATTACHMENT_BYTES: u64 = 12 * 1024 * 1024;
/// Bedrock compiles a structured-output schema with at most 12 optional
/// properties (<https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html>).
const MAX_JSON_OUTPUT_OPTIONAL_PROPERTIES: usize = 12;

/// A cross-Region inference profile: the prefix of a system-defined profile
/// id. Every Claude model current on Bedrock is served through one; the bare
/// model id answers 400 for on-demand throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockInferenceProfile {
    /// Routes to any commercial Region.
    Global,
    Us,
    Eu,
    Apac,
    Jp,
    Au,
    Ca,
    In,
}

impl BedrockInferenceProfile {
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Us => "us",
            Self::Eu => "eu",
            Self::Apac => "apac",
            Self::Jp => "jp",
            Self::Au => "au",
            Self::Ca => "ca",
            Self::In => "in",
        }
    }
}

/// What Bedrock documents for one Anthropic model.
struct BedrockModel {
    anthropic_id: &'static str,
    base_id: &'static str,
    profiles: &'static [BedrockInferenceProfile],
    /// The model card lists `tool_choice` types `auto` and `none` only.
    refuses_forced_tool: bool,
    /// Preserved thinking: each thinking block is bound to the conversation
    /// before it, so [`PrefixMismatchBehavior`] applies.
    binds_thinking: bool,
}

/// The Claude models of this SDK's catalog that Bedrock's runtime serves, from
/// the AWS model cards
/// (<https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html>,
/// read 2026-11-17). Left out: Claude Mythos 5 (not on the runtime endpoint),
/// Claude Sonnet 4 (end of life on Bedrock 2026-10-14), Claude Opus 4 and the
/// 3.5 line (no Bedrock model card). Each listed model supports structured
/// outputs and prompt caching on Bedrock.
const BEDROCK_MODELS: &[BedrockModel] = {
    use BedrockInferenceProfile::{Au, Ca, Eu, Global, In, Jp, Us};
    &[
        BedrockModel {
            anthropic_id: MODEL_FABLE_51,
            base_id: "anthropic.claude-fable-5-1",
            profiles: &[Us, Global],
            refuses_forced_tool: false,
            binds_thinking: true,
        },
        BedrockModel {
            anthropic_id: MODEL_FABLE_5,
            base_id: "anthropic.claude-fable-5",
            profiles: &[Us, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: "claude-mythos-5-1",
            base_id: "anthropic.claude-mythos-5-1",
            profiles: &[Us, Global],
            refuses_forced_tool: false,
            binds_thinking: true,
        },
        BedrockModel {
            anthropic_id: MODEL_OPUS_55,
            base_id: "anthropic.claude-opus-5-5",
            profiles: &[Us, Eu, Au, Jp, Global],
            refuses_forced_tool: true,
            binds_thinking: true,
        },
        BedrockModel {
            anthropic_id: MODEL_OPUS_5,
            base_id: "anthropic.claude-opus-5",
            profiles: &[Us, Eu, Au, In, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_OPUS_48,
            base_id: "anthropic.claude-opus-4-8",
            profiles: &[Us, Eu, Jp, Au, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_OPUS_47,
            base_id: "anthropic.claude-opus-4-7",
            profiles: &[Us, Eu, Jp, Au, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_OPUS_46,
            base_id: "anthropic.claude-opus-4-6-v1",
            profiles: &[Us, Eu, Au, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: "claude-opus-4-5-20251101",
            base_id: "anthropic.claude-opus-4-5-20251101-v1:0",
            profiles: &[Us, Eu, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_SONNET_55,
            base_id: "anthropic.claude-sonnet-5-5",
            profiles: &[Us, Ca, Eu, Au, Global],
            refuses_forced_tool: true,
            binds_thinking: true,
        },
        BedrockModel {
            anthropic_id: MODEL_SONNET_5,
            base_id: "anthropic.claude-sonnet-5",
            profiles: &[Us, Eu, Au, In, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_SONNET_46,
            base_id: "anthropic.claude-sonnet-4-6",
            profiles: &[Us, Eu, Au, Jp, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_SONNET_45,
            base_id: "anthropic.claude-sonnet-4-5-20250929-v1:0",
            profiles: &[Us, Eu, Au, Jp, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
        BedrockModel {
            anthropic_id: MODEL_HAIKU_55,
            base_id: "anthropic.claude-haiku-5-5",
            profiles: &[Us, Ca, Eu, Au, Jp, Global],
            refuses_forced_tool: false,
            binds_thinking: true,
        },
        BedrockModel {
            anthropic_id: MODEL_HAIKU_45,
            base_id: "anthropic.claude-haiku-4-5-20251001-v1:0",
            profiles: &[Us, Eu, Au, Jp, In, Global],
            refuses_forced_tool: false,
            binds_thinking: false,
        },
    ]
};

/// Percent-encode a model id as one URL path segment: everything outside RFC
/// 3986's unreserved set, `:` and `/` included, as the AWS SDKs encode it.
fn encode_path_segment(segment: &str) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
            encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
        }
    }
    encoded
}

/// The model card of an Anthropic model id. An undated alias such as
/// `claude-sonnet-4-5` names its dated snapshot, as it does for the
/// capability table.
fn bedrock_model(anthropic_model: &str) -> Option<&'static BedrockModel> {
    BEDROCK_MODELS
        .iter()
        .find(|model| model.anthropic_id == anthropic_model)
        .or_else(|| {
            BEDROCK_MODELS.iter().find(|model| {
                crate::model_capabilities::snapshot_resolves_to(anthropic_model, model.anthropic_id)
            })
        })
}

/// Bedrock's base model id for an Anthropic model id, when AWS documents one.
#[must_use]
pub fn bedrock_base_model_id(anthropic_model: &str) -> Option<&'static str> {
    bedrock_model(anthropic_model).map(|model| model.base_id)
}

/// The id of the system-defined inference profile that serves
/// `anthropic_model` through `profile`, when AWS documents that pair.
#[must_use]
pub fn bedrock_inference_profile_id(
    anthropic_model: &str,
    profile: BedrockInferenceProfile,
) -> Option<String> {
    bedrock_model(anthropic_model)
        .filter(|model| model.profiles.contains(&profile))
        .map(|model| format!("{}.{}", profile.prefix(), model.base_id))
}

/// Claude on Amazon Bedrock's runtime API.
#[derive(Clone)]
pub struct BedrockProvider {
    client: reqwest::Client,
    auth: BedrockAuth,
    region: String,
    endpoint: String,
    model: String,
    model_id: String,
    thinking: Option<ThinkingConfig>,
    prefix_mismatch: PrefixMismatchBehavior,
    stream_headers_timeout: Duration,
    stream_idle_timeout: Duration,
}

/// What Bedrock does with a replayed `thinking` block whose conversation
/// changed since the block was made.
///
/// The change can be an edited message, or a different `tools` list, such as
/// a harness that adds tools mid-conversation. Models with preserved thinking
/// (Sonnet 5.5, Opus 5.5, Haiku 5.5, Fable 5.1 and later) bind each block to
/// the conversation before it, and Bedrock refuses a mismatched block on
/// accounts where that check is enforced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixMismatchBehavior {
    /// The request fails with a 400 (`InvalidRequest`), and so does every
    /// later request that replays the block. Bedrock's default; the request
    /// carries no `block_binding`.
    #[default]
    Error,
    /// Bedrock drops the block, unbilled, and answers. The model loses that
    /// reasoning and may think again, which costs output tokens. Sent as
    /// `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` with the
    /// `thinking-binding-controls-2026-08-01` beta, on requests that carry
    /// thinking to a model whose card has preserved thinking; other models'
    /// requests are unchanged.
    DropBlock,
}

impl BedrockProvider {
    /// A provider for the Anthropic model `model`, sent to Bedrock as
    /// `model_id`: an inference profile id (`us.anthropic.claude-sonnet-5-5`),
    /// a base model id, or an inference profile ARN.
    #[must_use]
    pub fn new(
        auth: BedrockAuth,
        region: impl Into<String>,
        model: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        let region = region.into();
        let client = crate::tls_roots::http_client_builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .tcp_keepalive(TCP_KEEPALIVE)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .build()
            .unwrap_or_else(|error| {
                log::error!(
                    "failed to build Bedrock HTTP client with timeouts, \
                     falling back to reqwest defaults: {error}"
                );
                reqwest::Client::default()
            });
        Self {
            client,
            auth,
            endpoint: format!("https://bedrock-runtime.{region}.amazonaws.com"),
            region,
            model: model.into(),
            model_id: model_id.into(),
            thinking: None,
            prefix_mismatch: PrefixMismatchBehavior::Error,
            stream_headers_timeout: STREAM_HEADERS_TIMEOUT,
            stream_idle_timeout: STREAM_IDLE_TIMEOUT,
        }
    }

    /// A provider for `model` through the documented inference profile of
    /// `profile`.
    ///
    /// # Errors
    ///
    /// Returns an error when AWS documents no such profile for the model.
    pub fn with_inference_profile(
        auth: BedrockAuth,
        region: impl Into<String>,
        model: impl Into<String>,
        profile: BedrockInferenceProfile,
    ) -> Result<Self> {
        let model = model.into();
        let model_id = bedrock_inference_profile_id(&model, profile).ok_or_else(|| {
            anyhow::anyhow!(
                "Bedrock documents no {} inference profile for {model}",
                profile.prefix()
            )
        })?;
        Ok(Self::new(auth, region, model, model_id))
    }

    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// What Bedrock does with a replayed thinking block whose conversation
    /// changed. Set [`PrefixMismatchBehavior::DropBlock`] for a harness whose
    /// tool list grows mid-conversation, or a conversation that already failed
    /// on such a block never answers again.
    #[must_use]
    pub const fn with_prefix_mismatch(mut self, behavior: PrefixMismatchBehavior) -> Self {
        self.prefix_mismatch = behavior;
        self
    }

    /// The behavior [`Self::with_prefix_mismatch`] set.
    #[must_use]
    pub const fn prefix_mismatch(&self) -> PrefixMismatchBehavior {
        self.prefix_mismatch
    }

    /// Send requests to another endpoint (a VPC endpoint, or a test server).
    /// `SigV4` still signs for the provider's region.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let endpoint = endpoint.into();
        endpoint
            .trim_end_matches('/')
            .clone_into(&mut self.endpoint);
        self
    }

    /// Override the streaming deadlines: response headers, and the longest
    /// silence between event-stream bytes.
    #[must_use]
    pub const fn with_stream_stall_timeouts(
        mut self,
        headers_timeout: Duration,
        idle_timeout: Duration,
    ) -> Self {
        self.stream_headers_timeout = headers_timeout;
        self.stream_idle_timeout = idle_timeout;
        self
    }

    /// The model id sent to Bedrock.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    fn invoke_url(&self, streaming: bool) -> String {
        let action = if streaming {
            "invoke-with-response-stream"
        } else {
            "invoke"
        };
        format!(
            "{}/model/{}/{action}",
            self.endpoint,
            encode_path_segment(&self.model_id)
        )
    }

    fn bedrock_model(&self) -> Option<&'static BedrockModel> {
        bedrock_model(&self.model)
    }

    fn is_adaptive_thinking_model(&self) -> bool {
        crate::model_capabilities::get_model_capabilities("anthropic", &self.model)
            .is_some_and(|caps| caps.rejects_budget_thinking)
    }

    /// JSON outputs on Bedrock: the model card lists structured outputs and
    /// the schema stays within Bedrock's optional-property limit.
    fn takes_json_output(&self, format: &agent_sdk_foundation::llm::ResponseFormat) -> bool {
        self.bedrock_model().is_some()
            && schema::json_output_schema(&format.schema).is_ok_and(|wire| {
                optional_property_count(&wire) <= MAX_JSON_OUTPUT_OPTIONAL_PROPERTIES
            })
    }

    fn effective_max_tokens(&self, request: &ChatRequest) -> u32 {
        if request.max_tokens_explicit {
            request.max_tokens
        } else {
            self.default_max_tokens()
        }
    }

    /// Resolve thinking and attachments, then build the request body. `Err`
    /// is a refusal message: the request never reaches the wire.
    fn prepare(&self, request: &mut ChatRequest) -> Result<bytes::Bytes, String> {
        self.resolve_speed_tier()
            .map_err(|error| error.to_string())?;
        let thinking = self
            .resolve_thinking_config(request.thinking.as_ref())
            .map(|thinking| thinking_for_forced_tool(thinking, request.tool_choice.as_ref()))
            .map_err(|error| error.to_string())?;
        fit_request_attachments(self.provider(), self.model(), request);
        validate_request_attachments(self.provider(), self.model(), request)
            .map_err(|error| error.to_string())?;
        self.body(request, thinking.as_ref())
    }

    fn body(
        &self,
        request: &ChatRequest,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<bytes::Bytes, String> {
        if let Some(ToolChoice::Tool(name)) = &request.tool_choice
            && self
                .bedrock_model()
                .is_some_and(|model| model.refuses_forced_tool)
        {
            return Err(format!(
                "{} on Bedrock takes tool_choice auto or none only, so this request cannot \
                 force the tool `{name}` (AWS model card); a structured output whose schema \
                 Bedrock's JSON outputs cannot take has no fallback on this model",
                self.model
            ));
        }
        let format = match &request.response_format {
            Some(format) if self.bedrock_model().is_some() && !self.takes_json_output(format) => {
                let forces_a_tool = matches!(request.tool_choice, Some(ToolChoice::Tool(_)));
                if forces_a_tool {
                    None
                } else {
                    return Err(format!(
                        "Bedrock's JSON outputs cannot take the `{}` schema: it must close \
                         every object and list at most {MAX_JSON_OUTPUT_OPTIONAL_PROPERTIES} \
                         optional properties",
                        format.name
                    ));
                }
            }
            _ => json_output_format(request, self.bedrock_model().is_some())
                .map_err(|error| error.to_string())?,
        };

        let CacheRegions {
            tools: tools_cache,
            system: system_cache,
            messages: messages_cache,
        } = AnthropicProvider::cache_regions(request);
        let messages = AnthropicProvider::build_cached_api_messages(request, messages_cache);
        let tools = data::build_api_tools_with_cache(request, tools_cache);
        let effort = thinking.and_then(|thinking| thinking.effort);
        let api_request = data::ApiMessagesRequest {
            model: None,
            metadata: None,
            max_tokens: self.effective_max_tokens(request),
            system: data::build_api_system_prompt(&request.system, system_cache),
            messages: &messages,
            tools: tools.as_deref(),
            tool_choice: request
                .tool_choice
                .as_ref()
                .map(data::ApiToolChoice::from_tool_choice),
            stream: false,
            thinking: thinking.and_then(data::ApiThinkingConfig::from_thinking_config),
            output_config: data::ApiOutputConfig::new(effort, format),
            speed: None,
            anthropic_version: Some(ANTHROPIC_VERSION),
        };

        let mut body = serde_json::to_value(&api_request)
            .map_err(|error| format!("failed to serialize the Bedrock request: {error}"))?;
        let mut betas = self.betas(effort.is_some());
        if let Value::Object(fields) = &mut body {
            // The operation, not the body, says whether Bedrock streams.
            fields.remove("stream");
            // A model that binds nothing refuses no replay, and may not take the beta.
            let drop_block = match self.prefix_mismatch {
                PrefixMismatchBehavior::Error => false,
                PrefixMismatchBehavior::DropBlock => self
                    .bedrock_model()
                    .is_some_and(|model| model.binds_thinking),
            };
            if drop_block && let Some(Value::Object(thinking)) = fields.get_mut("thinking") {
                thinking.insert(
                    "block_binding".to_owned(),
                    serde_json::json!({ "prefix_mismatch_behavior": "drop_block" }),
                );
                betas.push(THINKING_BINDING_BETA);
            }
            if !betas.is_empty() {
                fields.insert(
                    "anthropic_beta".to_owned(),
                    Value::Array(betas.into_iter().map(Value::from).collect()),
                );
            }
        }
        let bytes = serde_json::to_vec(&body)
            .map_err(|error| format!("failed to serialize the Bedrock request: {error}"))?;
        if bytes.len() > MAX_BODY_BYTES {
            return Err(format!(
                "the Bedrock request body is {} bytes; Bedrock accepts at most {MAX_BODY_BYTES}",
                bytes.len()
            ));
        }
        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "Bedrock request model={} model_id={} body_bytes={}",
                self.model,
                self.model_id,
                bytes.len()
            );
        }
        Ok(bytes.into())
    }

    /// Betas go in the body. Budget-thinking models interleave thinking with
    /// tool calls only under the interleaved-thinking beta; adaptive models do
    /// it on their own and take effort without a beta.
    fn betas(&self, effort_on_wire: bool) -> Vec<&'static str> {
        if self.is_adaptive_thinking_model() {
            return Vec::new();
        }
        let mut betas = vec![INTERLEAVED_THINKING_BETA];
        if effort_on_wire {
            betas.push(EFFORT_BETA);
        }
        betas
    }

    /// Sign (or bear) and send one request. Returns the web identity
    /// credentials it signed with, which an expired-token rejection drops.
    async fn send(
        &self,
        body: &bytes::Bytes,
        streaming: bool,
    ) -> Result<(reqwest::Response, Option<Credentials>), SendError> {
        let url = self.invoke_url(streaming);
        let accept = if streaming {
            EVENT_STREAM_CONTENT_TYPE
        } else {
            "application/json"
        };
        let mut builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", accept);
        if !streaming {
            builder = builder.timeout(CHAT_REQUEST_TIMEOUT);
        }
        let mut signed_with = None;
        builder = match &self.auth {
            BedrockAuth::ApiKey(key) => builder.bearer_auth(key),
            BedrockAuth::Credentials(provider) => {
                let credentials = provider
                    .provide_credentials()
                    .await
                    .map_err(|error| SendError::from_credentials(&error))?;
                Self::sign(builder, &url, body, &self.region, credentials)?
            }
            BedrockAuth::WebIdentity(source) => {
                let credentials = source
                    .credentials()
                    .await
                    .map_err(|error| SendError::from_credentials(&error))?;
                signed_with = Some(credentials.clone());
                Self::sign(builder, &url, body, &self.region, credentials)?
            }
        };
        let response = builder
            .body(body.clone())
            .send()
            .await
            .map_err(SendError::Transport)?;
        Ok((response, signed_with))
    }

    fn sign(
        builder: reqwest::RequestBuilder,
        url: &str,
        body: &bytes::Bytes,
        region: &str,
        credentials: Credentials,
    ) -> Result<reqwest::RequestBuilder, SendError> {
        let parsed = url::Url::parse(url)
            .map_err(|error| SendError::Refused(format!("Bedrock endpoint {url}: {error}")))?;
        let host = match (parsed.host_str(), parsed.port()) {
            (Some(host), Some(port)) => format!("{host}:{port}"),
            (Some(host), None) => host.to_owned(),
            (None, _) => {
                return Err(SendError::Refused(format!(
                    "Bedrock endpoint {url} has no host"
                )));
            }
        };
        let headers = sigv4::signed_headers(
            &sigv4::SigningInput {
                method: "POST",
                url,
                host: &host,
                payload: body,
                region,
                service: SIGNING_SERVICE,
                time: std::time::SystemTime::now(),
            },
            credentials,
        )
        .map_err(|error| SendError::Refused(format!("{error:#}")))?;
        Ok(headers.into_iter().fold(builder, |builder, (name, value)| {
            builder.header(name, value)
        }))
    }

    /// Whether a rejection means the signing credentials expired, after which
    /// those credentials are dropped so the retry fetches new ones. Only web
    /// identity credentials are retried here; a `Credentials` provider that
    /// caches and refreshes (as `aws-config`'s do) recovers on its next call.
    async fn expired_credentials(
        &self,
        rejection: &Rejection,
        signed_with: Option<&Credentials>,
    ) -> bool {
        match (&self.auth, rejection, signed_with) {
            (
                BedrockAuth::WebIdentity(source),
                Rejection::ExpiredCredentials(_),
                Some(rejected),
            ) => {
                source.invalidate(rejected).await;
                true
            }
            (
                BedrockAuth::ApiKey(_) | BedrockAuth::Credentials(_) | BedrockAuth::WebIdentity(_),
                _,
                _,
            ) => false,
        }
    }

    /// Send a streaming request, retrying once on expired credentials. `Err`
    /// is the delta that ends the stream.
    async fn open_stream(&self, body: &bytes::Bytes) -> Result<reqwest::Response, StreamDelta> {
        let mut retried_credentials = false;
        loop {
            let response = match tokio::time::timeout(
                self.stream_headers_timeout,
                self.send(body, true),
            )
            .await
            {
                Ok(Ok(sent)) => sent,
                Ok(Err(SendError::Transport(error))) => {
                    return Err(reqwest_error_delta("request failed", &error));
                }
                Ok(Err(SendError::Refused(message))) => {
                    return Err(StreamDelta::Error {
                        message,
                        kind: StreamErrorKind::InvalidRequest,
                    });
                }
                Ok(Err(SendError::Credentials(message))) => {
                    return Err(StreamDelta::Error {
                        message,
                        kind: StreamErrorKind::ServerError,
                    });
                }
                Err(_elapsed) => {
                    let secs = self.stream_headers_timeout.as_secs();
                    log::error!(
                        "Bedrock streaming request timed out awaiting response headers after {secs}s"
                    );
                    return Err(StreamDelta::Error {
                        message: format!(
                            "request timed out awaiting response headers after {secs}s"
                        ),
                        kind: StreamErrorKind::ConnectionLost,
                    });
                }
            };
            let (response, signed_with) = response;
            if response.status().is_success() {
                return Ok(response);
            }
            let rejection = rejection(response).await;
            if !retried_credentials
                && self
                    .expired_credentials(&rejection, signed_with.as_ref())
                    .await
            {
                retried_credentials = true;
                continue;
            }
            return Err(rejection.into_delta());
        }
    }

    fn map_response(&self, api_response: data::ApiResponse) -> ChatResponse {
        ChatResponse {
            id: api_response.id,
            content: data::map_content_blocks(api_response.content),
            model: self.model.clone(),
            stop_reason: api_response.stop_reason.as_ref().map(data::map_stop_reason),
            usage: Usage {
                served_speed: None,
                input_tokens: api_response.usage.total_input_tokens(),
                output_tokens: api_response.usage.output,
                cached_input_tokens: api_response.usage.cached_input_tokens(),
                cache_creation_input_tokens: api_response.usage.cache_creation_input_tokens(),
                request_input_tokens: Some(api_response.usage.total_input_tokens()),
            },
        }
    }
}

enum SendError {
    /// The credential source or the signer refused; retrying will not help.
    Refused(String),
    /// Fetching credentials failed in a way a retry may fix.
    Credentials(String),
    Transport(reqwest::Error),
}

impl SendError {
    fn from_credentials(error: &CredentialsError) -> Self {
        let message = format!(
            "Bedrock credentials: {}",
            aws_smithy_types::error::display::DisplayErrorContext(error)
        );
        match error {
            CredentialsError::InvalidConfiguration(_)
            | CredentialsError::CredentialsNotLoaded(_) => Self::Refused(message),
            _ => Self::Credentials(message),
        }
    }
}

/// A non-success answer, classified for the retry layers.
#[derive(Debug, PartialEq, Eq)]
enum Rejection {
    RateLimited(Option<Duration>),
    ServerError(String),
    InvalidRequest(String),
    ExpiredCredentials(String),
}

impl Rejection {
    fn into_outcome(self) -> ChatOutcome {
        match self {
            Self::RateLimited(retry_after) => ChatOutcome::RateLimited(retry_after),
            Self::ServerError(message) => ChatOutcome::ServerError(message),
            Self::InvalidRequest(message) | Self::ExpiredCredentials(message) => {
                ChatOutcome::InvalidRequest(message)
            }
        }
    }

    fn into_delta(self) -> StreamDelta {
        let (message, kind) = match self {
            Self::RateLimited(retry_after) => (
                "Bedrock throttled the request".to_owned(),
                StreamErrorKind::RateLimited(retry_after),
            ),
            Self::ServerError(message) => (message, StreamErrorKind::ServerError),
            Self::InvalidRequest(message) | Self::ExpiredCredentials(message) => {
                (message, StreamErrorKind::InvalidRequest)
            }
        };
        StreamDelta::Error { message, kind }
    }
}

/// The AWS error type of a response (`x-amzn-ErrorType`, without the
/// namespace AWS appends after a colon).
fn error_type(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let value = headers.get("x-amzn-errortype")?.to_str().ok()?;
    let name = value.split(':').next()?.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

/// Classify a non-success status. 408 (`ModelTimeoutException`) and 424
/// (`ModelErrorException`) are 4xx statuses AWS says to retry; a 400
/// `ServiceQuotaExceededException` is a quota, not a bad request. Other 4xx
/// keep the body verbatim, so a prompt-too-long message reaches the caller's
/// compaction trigger unchanged.
fn classify(
    status: StatusCode,
    error_type: Option<&str>,
    retry_after: Option<Duration>,
    body: &str,
) -> Rejection {
    let message = error_type.map_or_else(
        || format!("Bedrock {status}: {body}"),
        |error_type| format!("Bedrock {status} {error_type}: {body}"),
    );
    match (status.as_u16(), error_type) {
        (429 | 529, _) | (_, Some("ServiceQuotaExceededException")) => {
            Rejection::RateLimited(retry_after)
        }
        (403, Some("ExpiredTokenException")) => Rejection::ExpiredCredentials(message),
        (408 | 424, _) => Rejection::ServerError(message),
        (code, _) if code >= 500 => Rejection::ServerError(message),
        _ => Rejection::InvalidRequest(message),
    }
}

async fn rejection(response: reqwest::Response) -> Rejection {
    let status = response.status();
    let error_type = error_type(response.headers());
    let retry_after = crate::http::retry_after_from_headers(response.headers());
    let body = response.text().await.unwrap_or_default();
    let rejection = classify(status, error_type.as_deref(), retry_after, &body);
    match &rejection {
        Rejection::ServerError(message) => log::error!("{message}"),
        Rejection::InvalidRequest(message) | Rejection::ExpiredCredentials(message) => {
            log::warn!("{message}");
        }
        Rejection::RateLimited(_) => log::warn!("Bedrock {status} throttled: {body}"),
    }
    rejection
}

/// Classify an event-stream exception. `modelStreamErrorException` carries
/// the model's own status, which decides between throttling, a bad request
/// and a transient failure. An exception this SDK does not know is retried.
fn exception_kind(exception_type: &str, payload: &Value) -> StreamErrorKind {
    match exception_type {
        "throttlingException" | "serviceQuotaExceededException" | "modelNotReadyException" => {
            StreamErrorKind::RateLimited(None)
        }
        "validationException" | "accessDeniedException" | "resourceNotFoundException" => {
            StreamErrorKind::InvalidRequest
        }
        "modelStreamErrorException" => {
            match payload.get("originalStatusCode").and_then(Value::as_u64) {
                Some(429 | 529) => StreamErrorKind::RateLimited(None),
                Some(code) if (400..500).contains(&code) && code != 408 => {
                    StreamErrorKind::InvalidRequest
                }
                _ => StreamErrorKind::ServerError,
            }
        }
        _ => StreamErrorKind::ServerError,
    }
}

fn exception_delta(exception_type: &str, payload: &[u8]) -> StreamDelta {
    let payload: Value = serde_json::from_slice(payload).unwrap_or(Value::Null);
    let message = payload
        .get("message")
        .or_else(|| payload.get("Message"))
        .and_then(Value::as_str)
        .unwrap_or("no message");
    let kind = exception_kind(exception_type, &payload);
    log::warn!("Bedrock stream exception {exception_type}: {message}");
    StreamDelta::Error {
        message: format!("Bedrock stream exception ({exception_type}): {message}"),
        kind,
    }
}

/// Log what Bedrock removed from a request before the model read it: one
/// `thinking_dropped` entry for each replayed thinking block that
/// [`PrefixMismatchBehavior::DropBlock`] let it drop, with its path.
fn log_input_transformations(model: &str, transformations: Option<&Value>) {
    let Some(Value::Array(entries)) = transformations else {
        return;
    };
    if entries.is_empty() {
        return;
    }
    let dropped = entries
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("thinking_dropped"))
        .count();
    log::info!(
        "Bedrock {model} dropped {dropped} replayed thinking block(s) whose conversation \
         changed: {}",
        Value::Array(entries.clone())
    );
}

/// The Anthropic stream event a `chunk` carries, as JSON text.
fn chunk_event(payload: &[u8]) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    struct PayloadPart {
        bytes: String,
    }
    let part: PayloadPart = serde_json::from_slice(payload)
        .map_err(|error| format!("Bedrock chunk is not a PayloadPart: {error}"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(part.bytes.as_bytes())
        .map_err(|error| format!("Bedrock chunk bytes are not base64: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("Bedrock chunk is not UTF-8: {error}"))
}

/// Count the properties a schema leaves optional, across every object it
/// describes.
fn optional_property_count(schema: &Value) -> usize {
    const SUBSCHEMA: &[&str] = &[
        "items",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "additionalProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    ];
    const SUBSCHEMA_LIST: &[&str] = &["prefixItems", "anyOf", "allOf", "oneOf"];
    const SUBSCHEMA_MAP: &[&str] = &[
        "properties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
    ];
    let Value::Object(node) = schema else {
        return 0;
    };
    let mut count = 0;
    if let Some(Value::Object(properties)) = node.get("properties") {
        let required: Vec<&str> = node
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        count += properties
            .keys()
            .filter(|name| !required.contains(&name.as_str()))
            .count();
    }
    for keyword in SUBSCHEMA {
        if let Some(child) = node.get(*keyword) {
            count += optional_property_count(child);
        }
    }
    for keyword in SUBSCHEMA_LIST {
        if let Some(Value::Array(children)) = node.get(*keyword) {
            count += children.iter().map(optional_property_count).sum::<usize>();
        }
    }
    for keyword in SUBSCHEMA_MAP {
        if let Some(Value::Object(children)) = node.get(*keyword) {
            count += children
                .values()
                .map(optional_property_count)
                .sum::<usize>();
        }
    }
    count
}

/// What a streamed response has shown so far, fed one event-stream message
/// at a time.
struct EventStreamState {
    usage: data::SseUsageState,
    tool_ids: std::collections::HashMap<usize, String>,
    pending_stop_reason: Option<StopReason>,
    received_message_stop: bool,
    stream_errored: bool,
    /// Set by a message that ends the stream: an exception, an error, or a
    /// chunk that cannot be read.
    finished: bool,
    served_route: String,
    served_model: String,
}

impl EventStreamState {
    fn new(served_route: String, served_model: String) -> Self {
        Self {
            usage: data::SseUsageState::default(),
            tool_ids: std::collections::HashMap::new(),
            pending_stop_reason: None,
            received_message_stop: false,
            stream_errored: false,
            finished: false,
            served_route,
            served_model,
        }
    }

    fn fail(&mut self, message: String, kind: StreamErrorKind) -> Vec<StreamDelta> {
        self.finished = true;
        vec![StreamDelta::Error { message, kind }]
    }

    fn corrupt(&mut self, error: &event_stream::DecodeError) -> Vec<StreamDelta> {
        log::error!("Bedrock event stream is corrupt: {error}");
        self.fail(error.to_string(), StreamErrorKind::ServerError)
    }

    fn on_message(
        &mut self,
        message: &aws_smithy_types::event_stream::Message,
    ) -> Vec<StreamDelta> {
        match event_stream::header(message, ":message-type") {
            Some("event") if event_stream::header(message, ":event-type") == Some("chunk") => {
                match chunk_event(message.payload()) {
                    Ok(event) => self.on_chunk(&event),
                    Err(error) => {
                        log::error!("{error}");
                        self.fail(error, StreamErrorKind::ServerError)
                    }
                }
            }
            Some("exception") => {
                let exception_type =
                    event_stream::header(message, ":exception-type").unwrap_or("unknownException");
                self.finished = true;
                vec![exception_delta(exception_type, message.payload())]
            }
            Some("error") => {
                let code = event_stream::header(message, ":error-code").unwrap_or("unknown");
                let text = event_stream::header(message, ":error-message").unwrap_or("no message");
                log::error!("Bedrock stream error {code}: {text}");
                self.fail(
                    format!("Bedrock stream error ({code}): {text}"),
                    StreamErrorKind::ServerError,
                )
            }
            other => {
                log::debug!(
                    "Bedrock event-stream message {other:?} ({:?}) ignored",
                    event_stream::header(message, ":event-type")
                );
                Vec::new()
            }
        }
    }

    fn on_chunk(&mut self, event: &str) -> Vec<StreamDelta> {
        let Some(event_type) = data::stream_event_type(event) else {
            log::warn!("Bedrock chunk without an event type");
            return Vec::new();
        };
        if event_type == "message_start"
            && let Ok(value) = serde_json::from_str::<Value>(event)
        {
            log_input_transformations(
                &self.served_model,
                value
                    .get("input_transformations")
                    .or_else(|| value.pointer("/message/input_transformations")),
            );
        }
        let mut deltas = Vec::new();
        if let Some(delta) = data::parse_stream_event(
            &event_type,
            event,
            &mut self.usage,
            &mut self.tool_ids,
            &mut self.pending_stop_reason,
        ) {
            self.stream_errored |= matches!(delta, StreamDelta::Error { .. });
            deltas.push(delta);
        }
        if event_type == "message_stop" {
            self.received_message_stop = true;
            self.finished = true;
            deltas.push(StreamDelta::Done {
                stop_reason: self.pending_stop_reason.take(),
                served_route: Some(self.served_route.clone()),
                served_model: Some(self.served_model.clone()),
            });
        }
        deltas
    }

    /// The error a stream that ended before `message_stop` reports, unless an
    /// error event already said why.
    fn end(&self, partial_frame_bytes: usize) -> Option<StreamDelta> {
        if self.received_message_stop || self.stream_errored {
            return None;
        }
        log::warn!(
            "Bedrock event stream ended without message_stop ({partial_frame_bytes} bytes of a partial frame)"
        );
        Some(StreamDelta::Error {
            message: "Stream ended unexpectedly without completion".to_owned(),
            kind: StreamErrorKind::ServerError,
        })
    }
}

#[async_trait]
impl LlmProvider for BedrockProvider {
    async fn chat(&self, mut request: ChatRequest) -> Result<ChatOutcome> {
        let body = match self.prepare(&mut request) {
            Ok(body) => body,
            Err(refusal) => return Ok(ChatOutcome::InvalidRequest(refusal)),
        };
        let mut retried_credentials = false;
        let response = loop {
            let (response, signed_with) = match self.send(&body, false).await {
                Ok(sent) => sent,
                Err(SendError::Refused(message)) => {
                    return Ok(ChatOutcome::InvalidRequest(message));
                }
                Err(SendError::Credentials(message)) => return Err(anyhow::anyhow!(message)),
                Err(SendError::Transport(error)) => {
                    return Err(anyhow::anyhow!("request failed: {error}"));
                }
            };
            if response.status().is_success() {
                break response;
            }
            let rejection = rejection(response).await;
            if !retried_credentials
                && self
                    .expired_credentials(&rejection, signed_with.as_ref())
                    .await
            {
                retried_credentials = true;
                continue;
            }
            return Ok(rejection.into_outcome());
        };
        let bytes = response
            .bytes()
            .await
            .map_err(|error| anyhow::anyhow!("failed to read response body: {error}"))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| anyhow::anyhow!("failed to parse response: {error}"))?;
        log_input_transformations(&self.model, value.get("input_transformations"));
        let api_response: data::ApiResponse = serde_json::from_value(value)
            .map_err(|error| anyhow::anyhow!("failed to parse response: {error}"))?;
        Ok(ChatOutcome::Success(self.map_response(api_response)))
    }

    fn chat_stream(&self, mut request: ChatRequest) -> StreamBox<'_> {
        let served_route = self.route().to_owned();
        let served_model = self.model().to_owned();
        Box::pin(async_stream::stream! {
            let body = match self.prepare(&mut request) {
                Ok(body) => body,
                Err(refusal) => {
                    yield Ok(StreamDelta::Error {
                        message: refusal,
                        kind: StreamErrorKind::InvalidRequest,
                    });
                    return;
                }
            };

            let response = match self.open_stream(&body).await {
                Ok(response) => response,
                Err(delta) => {
                    yield Ok(delta);
                    return;
                }
            };

            let mut bytes = response.bytes_stream();
            let mut decoder = event_stream::Decoder::default();
            let mut state = EventStreamState::new(served_route, served_model);
            loop {
                let next = match tokio::time::timeout(self.stream_idle_timeout, bytes.next()).await {
                    Ok(next) => next,
                    Err(_elapsed) => {
                        log::error!(
                            "Bedrock event stream timed out: no bytes for {}s",
                            self.stream_idle_timeout.as_secs()
                        );
                        yield Ok(StreamDelta::Error {
                            message: format!(
                                "Bedrock event stream timed out: no bytes for {}s",
                                self.stream_idle_timeout.as_secs()
                            ),
                            kind: StreamErrorKind::ConnectionLost,
                        });
                        return;
                    }
                };
                let Some(chunk) = next else { break };
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Ok(reqwest_body_error_delta("stream error", &error));
                        return;
                    }
                };
                decoder.push(&chunk);
                loop {
                    let deltas = match decoder.next_message() {
                        Ok(Some(message)) => state.on_message(&message),
                        Ok(None) => break,
                        Err(error) => state.corrupt(&error),
                    };
                    for delta in deltas {
                        yield Ok(delta);
                    }
                    if state.finished {
                        return;
                    }
                }
            }
            if let Some(delta) = state.end(decoder.pending()) {
                yield Ok(delta);
            }
        })
    }

    fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
        let Some(thinking) = thinking else {
            return Ok(());
        };
        if self
            .capabilities()
            .is_some_and(|caps| !caps.supports_thinking)
        {
            return Err(anyhow::anyhow!(
                "thinking is not supported for route={} model={}",
                self.route(),
                self.model()
            ));
        }
        if matches!(thinking.mode, ThinkingMode::Adaptive)
            && !self
                .capabilities()
                .is_some_and(|caps| caps.supports_adaptive_thinking)
        {
            return Err(anyhow::anyhow!(
                "adaptive thinking is not supported for route={} model={}",
                self.route(),
                self.model()
            ));
        }
        if self.is_adaptive_thinking_model()
            && matches!(thinking.mode, ThinkingMode::Enabled { .. })
        {
            return Err(anyhow::anyhow!(
                "budget_tokens thinking is rejected for route={} model={}; use \
                 ThinkingConfig::adaptive() or ThinkingConfig::default_with_effort(_) instead",
                self.route(),
                self.model()
            ));
        }
        Ok(())
    }

    async fn probe_connectivity(&self) -> bool {
        crate::provider::probe_http_reachability(&self.client, &self.endpoint).await
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &'static str {
        "anthropic"
    }

    fn route(&self) -> &'static str {
        "bedrock"
    }

    fn supports_historical_image_blocks(&self) -> bool {
        true
    }

    fn max_request_attachment_bytes(&self) -> Option<u64> {
        Some(BEDROCK_MAX_REQUEST_ATTACHMENT_BYTES)
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        self.thinking.as_ref()
    }

    fn structured_output_support(&self) -> StructuredOutputSupport {
        if self.bedrock_model().is_some() {
            StructuredOutputSupport::Native
        } else {
            StructuredOutputSupport::ToolForcing
        }
    }

    fn structured_output_support_for(
        &self,
        format: &agent_sdk_foundation::llm::ResponseFormat,
    ) -> StructuredOutputSupport {
        if self.takes_json_output(format) {
            StructuredOutputSupport::Native
        } else {
            StructuredOutputSupport::ToolForcing
        }
    }

    fn default_max_tokens(&self) -> u32 {
        let model_max = self
            .capabilities()
            .and_then(crate::model_capabilities::ModelCapabilities::default_output_tokens)
            .or_else(|| {
                crate::model_capabilities::default_max_output_tokens(self.provider(), self.model())
            })
            .unwrap_or(4096);
        model_max.clamp(4096, DEFAULT_SAFE_MAX_OUTPUT_TOKENS)
    }
}

#[cfg(test)]
mod tests;
