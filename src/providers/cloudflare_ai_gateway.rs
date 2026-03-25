//! Cloudflare AI Gateway provider implementation.
//!
//! Routes requests through [Cloudflare's AI Gateway](https://developers.cloudflare.com/ai-gateway/)
//! using **provider-native proxy** endpoints, preserving all provider-specific
//! features (prompt caching, extended thinking, adaptive thinking, etc.).
//!
//! Unlike the Unified API (`/compat`), the provider-native proxy keeps each
//! provider's request/response format intact — the gateway only swaps the base
//! URL and handles authentication via BYOK.
//!
//! # Why provider-native over unified?
//!
//! The unified endpoint translates everything to `OpenAI` Chat Completions
//! format, which **loses** critical features:
//! - Anthropic prompt caching (`cache_control: ephemeral`) — significant cost savings
//! - Anthropic adaptive / budgeted thinking
//! - Gemini `cachedContent` handles
//! - Anthropic thought signatures for tool verification
//!
//! The provider-native proxy preserves all of these.
//!
//! # BYOK authentication
//!
//! Provider API keys are stored in the Cloudflare dashboard. At runtime only a
//! Cloudflare API token is needed — no provider secrets in code.
//!
//! # Example
//!
//! ```no_run
//! use agent_sdk::providers::CloudflareAIGatewayProvider;
//!
//! // BYOK — CF token is the only secret at runtime
//! let provider = CloudflareAIGatewayProvider::anthropic_sonnet(
//!     "your-cf-api-token",
//!     "your-cf-account-id",
//!     "your-gateway-id",
//! );
//!
//! // Pass-through — provider key at runtime, optional gateway auth
//! let provider = CloudflareAIGatewayProvider::anthropic(
//!     "your-anthropic-key".to_string(),
//!     "your-cf-account-id",
//!     "your-gateway-id",
//!     "claude-sonnet-4-6".to_string(),
//! ).with_gateway_token("your-cf-api-token");
//! ```

use crate::llm::{ChatOutcome, ChatRequest, LlmProvider, StreamBox, ThinkingConfig};
use crate::model_capabilities::ModelCapabilities;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::gemini::GeminiProvider;
use crate::providers::openai::OpenAIProvider;
use anyhow::Result;
use async_trait::async_trait;

const GATEWAY_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1";
const CF_AIG_AUTH_HEADER: &str = "cf-aig-authorization";

/// Upstream provider that the gateway routes to.
#[derive(Clone)]
enum Inner {
    Anthropic(AnthropicProvider),
    OpenAI(OpenAIProvider),
    Gemini(GeminiProvider),
}

/// Cloudflare AI Gateway LLM provider.
///
/// Wraps an upstream provider (Anthropic, `OpenAI`, or Gemini) and routes
/// requests through the provider-native proxy endpoint, preserving all
/// provider-specific features including prompt caching, extended thinking,
/// and streaming.
///
/// The gateway provides analytics, caching, rate limiting, logging, and
/// automatic fallback on top.
#[derive(Clone)]
pub struct CloudflareAIGatewayProvider {
    inner: Inner,
}

fn gateway_base(account_id: &str, gateway_id: &str, provider_segment: &str) -> String {
    format!("{GATEWAY_BASE_URL}/{account_id}/{gateway_id}/{provider_segment}")
}

fn byok_headers(cf_token: &str) -> Vec<(String, String)> {
    vec![(CF_AIG_AUTH_HEADER.to_owned(), format!("Bearer {cf_token}"))]
}

impl CloudflareAIGatewayProvider {
    // ========================================================================
    // Anthropic (provider-native: /anthropic/v1/messages)
    // ========================================================================

    /// Route to any Anthropic model via the provider-native proxy.
    ///
    /// In **BYOK mode** pass an empty `api_key` and call
    /// [`with_gateway_token`](Self::with_gateway_token).
    /// In **pass-through mode** pass the Anthropic API key directly.
    #[must_use]
    pub fn anthropic(api_key: String, account_id: &str, gateway_id: &str, model: String) -> Self {
        let base_url = gateway_base(account_id, gateway_id, "anthropic");
        let inner = AnthropicProvider::new(api_key, model).with_base_url(base_url);
        Self {
            inner: Inner::Anthropic(inner),
        }
    }

    /// Route to Claude Sonnet 4.6 — BYOK mode (CF token only).
    #[must_use]
    pub fn anthropic_sonnet(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::anthropic(
            String::new(),
            account_id,
            gateway_id,
            "claude-sonnet-4-6".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    /// Route to Claude Opus 4.6 — BYOK mode (CF token only).
    #[must_use]
    pub fn anthropic_opus(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::anthropic(
            String::new(),
            account_id,
            gateway_id,
            "claude-opus-4-6".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    // ========================================================================
    // OpenAI (provider-native: /openai/chat/completions)
    // ========================================================================

    /// Route to any `OpenAI` model via the provider-native proxy.
    #[must_use]
    pub fn openai(api_key: String, account_id: &str, gateway_id: &str, model: String) -> Self {
        let base_url = gateway_base(account_id, gateway_id, "openai");
        let inner = OpenAIProvider::with_base_url(api_key, model, base_url);
        Self {
            inner: Inner::OpenAI(inner),
        }
    }

    /// Route to GPT-5.4 — BYOK mode.
    #[must_use]
    pub fn openai_gpt54(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::openai(String::new(), account_id, gateway_id, "gpt-5.4".to_owned())
            .with_gateway_token(cf_token)
    }

    /// Route to GPT-5.4 Mini — BYOK mode.
    #[must_use]
    pub fn openai_gpt54_mini(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::openai(
            String::new(),
            account_id,
            gateway_id,
            "gpt-5.4-mini".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    /// Route to GPT-5.4 Nano — BYOK mode.
    #[must_use]
    pub fn openai_gpt54_nano(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::openai(
            String::new(),
            account_id,
            gateway_id,
            "gpt-5.4-nano".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    // ========================================================================
    // Gemini (provider-native: /google-ai-studio/v1beta/models/...)
    // ========================================================================

    /// Route to any Gemini model via the provider-native proxy.
    ///
    /// Automatically switches to header-based auth (`x-goog-api-key`) as
    /// required by the gateway.
    #[must_use]
    pub fn gemini(api_key: String, account_id: &str, gateway_id: &str, model: String) -> Self {
        let base_url = gateway_base(account_id, gateway_id, "google-ai-studio/v1beta");
        let inner = GeminiProvider::new(api_key, model)
            .with_base_url(base_url)
            .with_header_auth();
        Self {
            inner: Inner::Gemini(inner),
        }
    }

    /// Route to Gemini 3.1 Pro — BYOK mode.
    #[must_use]
    pub fn gemini_pro(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::gemini(
            String::new(),
            account_id,
            gateway_id,
            "gemini-3.1-pro-preview".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    /// Route to Gemini 3 Flash — BYOK mode.
    #[must_use]
    pub fn gemini_flash(cf_token: &str, account_id: &str, gateway_id: &str) -> Self {
        Self::gemini(
            String::new(),
            account_id,
            gateway_id,
            "gemini-3-flash-preview".to_owned(),
        )
        .with_gateway_token(cf_token)
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set the Cloudflare AI Gateway authentication token.
    ///
    /// Sent via the `cf-aig-authorization` header. In BYOK mode this is the
    /// only auth needed. In pass-through mode it authenticates with the
    /// gateway while the provider API key authenticates with the upstream.
    #[must_use]
    pub fn with_gateway_token(mut self, token: &str) -> Self {
        let headers = byok_headers(token);
        match &mut self.inner {
            Inner::Anthropic(p) => {
                *p = std::mem::replace(p, AnthropicProvider::new(String::new(), String::new()))
                    .with_extra_headers(headers);
            }
            Inner::OpenAI(p) => {
                *p = std::mem::replace(p, OpenAIProvider::new(String::new(), String::new()))
                    .with_extra_headers(headers);
            }
            Inner::Gemini(p) => {
                *p = std::mem::replace(p, GeminiProvider::new(String::new(), String::new()))
                    .with_extra_headers(headers);
            }
        }
        self
    }

    /// Set the provider-owned thinking configuration.
    #[must_use]
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        match &mut self.inner {
            Inner::Anthropic(p) => {
                *p = std::mem::replace(p, AnthropicProvider::new(String::new(), String::new()))
                    .with_thinking(thinking);
            }
            Inner::OpenAI(p) => {
                *p = std::mem::replace(p, OpenAIProvider::new(String::new(), String::new()))
                    .with_thinking(thinking);
            }
            Inner::Gemini(p) => {
                *p = std::mem::replace(p, GeminiProvider::new(String::new(), String::new()))
                    .with_thinking(thinking);
            }
        }
        self
    }
}

#[async_trait]
impl LlmProvider for CloudflareAIGatewayProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome> {
        match &self.inner {
            Inner::Anthropic(p) => p.chat(request).await,
            Inner::OpenAI(p) => p.chat(request).await,
            Inner::Gemini(p) => p.chat(request).await,
        }
    }

    fn chat_stream(&self, request: ChatRequest) -> StreamBox<'_> {
        match &self.inner {
            Inner::Anthropic(p) => p.chat_stream(request),
            Inner::OpenAI(p) => p.chat_stream(request),
            Inner::Gemini(p) => p.chat_stream(request),
        }
    }

    fn model(&self) -> &str {
        match &self.inner {
            Inner::Anthropic(p) => p.model(),
            Inner::OpenAI(p) => p.model(),
            Inner::Gemini(p) => p.model(),
        }
    }

    fn provider(&self) -> &'static str {
        "cloudflare-ai-gateway"
    }

    fn configured_thinking(&self) -> Option<&ThinkingConfig> {
        match &self.inner {
            Inner::Anthropic(p) => p.configured_thinking(),
            Inner::OpenAI(p) => p.configured_thinking(),
            Inner::Gemini(p) => p.configured_thinking(),
        }
    }

    fn capabilities(&self) -> Option<&'static ModelCapabilities> {
        match &self.inner {
            Inner::Anthropic(p) => p.capabilities(),
            Inner::OpenAI(p) => p.capabilities(),
            Inner::Gemini(p) => p.capabilities(),
        }
    }

    fn validate_thinking_config(&self, thinking: Option<&ThinkingConfig>) -> Result<()> {
        match &self.inner {
            Inner::Anthropic(p) => p.validate_thinking_config(thinking),
            Inner::OpenAI(p) => p.validate_thinking_config(thinking),
            Inner::Gemini(p) => p.validate_thinking_config(thinking),
        }
    }

    fn default_max_tokens(&self) -> u32 {
        match &self.inner {
            Inner::Anthropic(p) => p.default_max_tokens(),
            Inner::OpenAI(p) => p.default_max_tokens(),
            Inner::Gemini(p) => p.default_max_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_sonnet_byok() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "claude-sonnet-4-6");
        assert_eq!(p.provider(), "cloudflare-ai-gateway");
    }

    #[test]
    fn anthropic_opus_byok() {
        let p = CloudflareAIGatewayProvider::anthropic_opus("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "claude-opus-4-6");
    }

    #[test]
    fn openai_gpt54_byok() {
        let p = CloudflareAIGatewayProvider::openai_gpt54("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "gpt-5.4");
    }

    #[test]
    fn openai_gpt54_mini_byok() {
        let p = CloudflareAIGatewayProvider::openai_gpt54_mini("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "gpt-5.4-mini");
    }

    #[test]
    fn openai_gpt54_nano_byok() {
        let p = CloudflareAIGatewayProvider::openai_gpt54_nano("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "gpt-5.4-nano");
    }

    #[test]
    fn gemini_pro_byok() {
        let p = CloudflareAIGatewayProvider::gemini_pro("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "gemini-3.1-pro-preview");
    }

    #[test]
    fn gemini_flash_byok() {
        let p = CloudflareAIGatewayProvider::gemini_flash("cf-tok", "acct", "gw");
        assert_eq!(p.model(), "gemini-3-flash-preview");
    }

    #[test]
    fn capabilities_resolve_anthropic() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t", "a", "g");
        let caps = p.capabilities().unwrap();
        assert_eq!(caps.provider, "anthropic");
        assert_eq!(caps.model_id, "claude-sonnet-4-6");
        assert!(caps.supports_adaptive_thinking);
    }

    #[test]
    fn capabilities_resolve_openai() {
        let p = CloudflareAIGatewayProvider::openai_gpt54("t", "a", "g");
        let caps = p.capabilities().unwrap();
        assert_eq!(caps.provider, "openai");
        assert_eq!(caps.model_id, "gpt-5.4");
    }

    #[test]
    fn capabilities_resolve_gemini() {
        let p = CloudflareAIGatewayProvider::gemini_pro("t", "a", "g");
        let caps = p.capabilities().unwrap();
        assert_eq!(caps.provider, "gemini");
    }

    #[test]
    fn pass_through_with_gateway_token() {
        let p = CloudflareAIGatewayProvider::anthropic(
            "sk-ant-key".to_string(),
            "acct",
            "gw",
            "claude-sonnet-4-6".to_string(),
        )
        .with_gateway_token("cf-tok");
        assert_eq!(p.model(), "claude-sonnet-4-6");
    }

    #[test]
    fn with_thinking_is_applied() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t", "a", "g")
            .with_thinking(ThinkingConfig::adaptive());
        assert!(p.configured_thinking().is_some());
    }

    #[test]
    fn provider_is_cloneable() {
        let p = CloudflareAIGatewayProvider::anthropic_sonnet("t", "a", "g");
        let cloned = p.clone();
        assert_eq!(p.model(), cloned.model());
    }

    #[test]
    fn gateway_url_format() {
        assert_eq!(
            gateway_base("my-acct", "my-gw", "anthropic"),
            "https://gateway.ai.cloudflare.com/v1/my-acct/my-gw/anthropic"
        );
    }
}
