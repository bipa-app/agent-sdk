//! LLM Provider implementations.
//!
//! Re-exported from [`agent_sdk_providers::impls`]. Each provider is gated
//! behind its matching cargo feature (`anthropic` is enabled by default).

#[cfg(feature = "anthropic")]
pub mod anthropic {
    //! Anthropic Messages API provider.
    pub use agent_sdk_providers::impls::anthropic::*;
}
#[cfg(feature = "cloudflare")]
pub mod cloudflare_ai_gateway {
    //! Cloudflare AI Gateway proxy provider.
    pub use agent_sdk_providers::impls::cloudflare_ai_gateway::*;
}
#[cfg(feature = "gemini")]
pub mod gemini {
    //! Google Gemini API provider.
    pub use agent_sdk_providers::impls::gemini::*;
}
#[cfg(feature = "openai")]
pub mod openai {
    //! `OpenAI` Chat Completions API provider.
    pub use agent_sdk_providers::impls::openai::*;
}
#[cfg(feature = "openai-codex")]
pub mod openai_codex_responses {
    //! `OpenAI` Codex / `ChatGPT` WebSocket provider.
    pub use agent_sdk_providers::impls::openai_codex_responses::*;
}
#[cfg(feature = "openai")]
pub mod openai_responses {
    //! `OpenAI` Responses API provider.
    pub use agent_sdk_providers::impls::openai_responses::*;
}
#[cfg(feature = "vertex")]
pub mod vertex {
    //! Google Vertex AI provider.
    pub use agent_sdk_providers::impls::vertex::*;
}

#[cfg(feature = "cloudflare")]
pub use agent_sdk_providers::impls::CloudflareAIGatewayProvider;
#[cfg(feature = "gemini")]
pub use agent_sdk_providers::impls::GeminiProvider;
#[cfg(feature = "openai-codex")]
pub use agent_sdk_providers::impls::OpenAICodexResponsesProvider;
#[cfg(feature = "vertex")]
pub use agent_sdk_providers::impls::VertexProvider;
#[cfg(feature = "anthropic")]
pub use agent_sdk_providers::impls::{AnthropicProvider, is_oauth_token};
#[cfg(feature = "openai")]
pub use agent_sdk_providers::impls::{OpenAIProvider, OpenAIResponsesProvider};
