//! First-party LLM provider implementations.
//!
//! Each provider is gated behind a cargo feature so consumers only compile
//! (and only pull the transitive dependencies of) the providers they use.
//! The `anthropic` feature is enabled by default. `vertex` reuses the
//! Anthropic + Gemini codecs, and `cloudflare` wraps Anthropic + `OpenAI` +
//! Gemini, so those features imply their dependencies in `Cargo.toml`.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "cloudflare")]
pub mod cloudflare_ai_gateway;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai-codex")]
pub mod openai_codex_responses;
#[cfg(feature = "openai")]
pub mod openai_responses;
#[cfg(feature = "vertex")]
pub mod vertex;

// Shared live model-listing HTTP helper. Only the Anthropic, OpenAI, and
// Gemini provider impls call it (vertex/cloudflare pull those in), so gate it
// behind exactly those features to avoid a dead-code error in single-provider
// or no-default-features builds.
#[cfg(any(feature = "anthropic", feature = "openai", feature = "gemini"))]
pub(crate) mod model_listing;

#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicProvider, is_oauth_token};
#[cfg(feature = "cloudflare")]
pub use cloudflare_ai_gateway::CloudflareAIGatewayProvider;
#[cfg(feature = "gemini")]
pub use gemini::GeminiProvider;
#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;
#[cfg(feature = "openai-codex")]
pub use openai_codex_responses::OpenAICodexResponsesProvider;
#[cfg(feature = "openai")]
pub use openai_responses::OpenAIResponsesProvider;
#[cfg(feature = "vertex")]
pub use vertex::VertexProvider;
