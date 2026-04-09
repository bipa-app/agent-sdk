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

pub mod attachments;
pub mod impls;
pub mod model_capabilities;
pub mod provider;
pub mod router;
pub mod search;
pub mod streaming;

// Convenience re-exports — provider trait and streaming
pub use provider::{LlmProvider, collect_stream};
pub use router::{ModelRouter, ModelTier, TaskComplexity};
pub use streaming::{StreamAccumulator, StreamBox, StreamDelta};

// Re-export all core LLM types so consumers can `use agent_sdk_providers::*`
pub use agent_sdk_core::llm::*;

// Provider re-exports
pub use impls::{
    AnthropicProvider, CloudflareAIGatewayProvider, GeminiProvider, OpenAICodexResponsesProvider,
    OpenAIProvider, OpenAIResponsesProvider, VertexProvider, is_oauth_token,
};

// Model capabilities
pub use model_capabilities::{
    ModelCapabilities, PricePoint, Pricing, SourceStatus, get_model_capabilities,
    supported_model_capabilities,
};

// Search provider
pub use search::{BraveSearchProvider, SearchProvider, SearchResponse, SearchResult};
