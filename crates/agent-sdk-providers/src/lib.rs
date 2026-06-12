//! LLM provider trait, streaming primitives, and first-party provider implementations.
//!
//! This crate defines the [`LlmProvider`] trait that all LLM backends implement,
//! streaming types for incremental response processing, attachment validation,
//! model capability metadata, and seven first-party provider implementations.
//!
//! # Provider Implementations
//!
//! | Provider | Module | Protocol |
//! |----------|--------|----------|
//! | Anthropic (Messages API) | [`impls::anthropic`] | SSE streaming |
//! | OpenAI (Chat Completions) | [`impls::openai`] | SSE streaming |
//! | OpenAI (Responses API) | [`impls::openai_responses`] | SSE streaming |
//! | OpenAI Codex / ChatGPT | [`impls::openai_codex_responses`] | WebSocket |
//! | Google Gemini | [`impls::gemini`] | SSE streaming |
//! | Google Vertex AI | [`impls::vertex`] | SSE streaming |
//! | Cloudflare AI Gateway | [`impls::cloudflare_ai_gateway`] | Proxy wrapper |

#![forbid(unsafe_code)]

// The attachment-validation helpers are only ever called from a provider
// impl (each gated behind its provider feature), so the whole module is dead
// code when no provider is compiled in. Gate it on "any provider feature" to
// keep `--no-default-features` builds warning- (and `-D warnings`-) clean.
#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "openai-codex",
    feature = "gemini",
    feature = "vertex",
    feature = "cloudflare",
))]
pub mod attachments;
pub mod fallback;
// `Retry-After` header parsing for 429 responses. Only ever called from a
// provider impl (each gated behind its provider feature), so it is dead code
// when no provider is compiled in — gate it on "any provider feature" to keep
// `--no-default-features` builds `-D warnings`-clean.
#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "openai-codex",
    feature = "gemini",
    feature = "vertex",
    feature = "cloudflare",
))]
pub(crate) mod http;
pub mod impls;
pub mod model_capabilities;
#[cfg(feature = "model-discovery")]
pub mod model_catalog;
pub mod provider;
#[cfg(feature = "record-replay")]
pub mod record_replay;
pub mod refresh;
pub mod router;
pub mod search;
pub mod streaming;
pub mod structured;

// Convenience re-exports — provider trait and streaming
pub use fallback::FallbackProvider;
pub use provider::{LlmProvider, ModelInfo, StructuredOutputSupport, collect_stream};
#[cfg(feature = "record-replay")]
pub use record_replay::{RecordReplayMode, RecordReplayProvider};
pub use refresh::{RefreshingProvider, is_unauthorized_error};
pub use router::{ModelRouter, ModelTier, TaskComplexity};
pub use streaming::{StreamAccumulator, StreamBox, StreamDelta};
pub use structured::{
    StructuredConfig, StructuredOutput, StructuredOutputError, StructuredStream,
    StructuredStreamUpdate, run_structured, run_structured_stream,
};

// Re-export all core LLM types so consumers can `use agent_sdk_providers::*`
pub use agent_sdk_foundation::llm::*;

// Provider re-exports — each is gated behind its provider feature.
#[cfg(feature = "cloudflare")]
pub use impls::CloudflareAIGatewayProvider;
#[cfg(feature = "gemini")]
pub use impls::GeminiProvider;
#[cfg(feature = "openai-codex")]
pub use impls::OpenAICodexResponsesProvider;
#[cfg(feature = "vertex")]
pub use impls::VertexProvider;
#[cfg(feature = "anthropic")]
pub use impls::{AnthropicProvider, is_oauth_token};
#[cfg(feature = "openai")]
pub use impls::{OpenAIProvider, OpenAIResponsesProvider};

// Model capabilities
pub use model_capabilities::{
    ModelCapabilities, PricePoint, Pricing, SourceStatus, get_model_capabilities,
    supported_model_capabilities,
};

// Dynamic model-discovery: third-party capability/pricing feed + layered registry.
#[cfg(feature = "model-discovery")]
pub use model_catalog::{
    CatalogEntry, ModelCatalogSource, ModelRegistry, ModelsDevSource, OpenRouterSource,
    ResolvedModel, ResolvedSource,
};

// Search provider
pub use search::{BraveSearchProvider, SearchProvider, SearchResponse, SearchResult};
