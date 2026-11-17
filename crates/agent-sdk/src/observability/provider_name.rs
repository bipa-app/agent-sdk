//! Provider name normalization to `GenAI` semconv canonical values.

use agent_sdk_providers::LlmProvider;

/// The `gen_ai.provider.name` of the endpoint that serves `provider`'s calls.
///
/// A provider that speaks one vendor's API through another platform reports
/// the vendor as [`LlmProvider::provider`] and the platform as
/// [`LlmProvider::route`]; telemetry names the platform.
#[must_use]
pub fn of<P: LlmProvider + ?Sized>(provider: &P) -> &'static str {
    match provider.route() {
        "bedrock" => "aws.bedrock",
        _ => normalize(provider.provider()),
    }
}

/// Normalize an SDK provider identifier to a canonical `GenAI` semconv
/// `gen_ai.provider.name` value.
#[must_use]
pub fn normalize(sdk_provider_id: &'static str) -> &'static str {
    match sdk_provider_id {
        "anthropic" => "anthropic",
        "openai" | "openai-responses" | "openai-codex" => "openai",
        "gemini" => "gcp.gemini",
        "vertex" => "gcp.vertex_ai",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Routed(&'static str, &'static str);

    #[async_trait::async_trait]
    impl LlmProvider for Routed {
        async fn chat(
            &self,
            _request: agent_sdk_providers::ChatRequest,
        ) -> anyhow::Result<agent_sdk_providers::ChatOutcome> {
            anyhow::bail!("not called")
        }

        fn model(&self) -> &'static str {
            "claude-sonnet-5-5"
        }

        fn provider(&self) -> &'static str {
            self.0
        }

        fn route(&self) -> &'static str {
            self.1
        }
    }

    #[test]
    fn a_bedrock_route_is_named_aws_bedrock() {
        assert_eq!(of(&Routed("anthropic", "bedrock")), "aws.bedrock");
        assert_eq!(of(&Routed("anthropic", "anthropic")), "anthropic");
        assert_eq!(of(&Routed("vertex", "vertex")), "gcp.vertex_ai");
        assert_eq!(of(&Routed("openai", "openrouter")), "openai");
    }

    #[test]
    fn anthropic_maps_to_anthropic() {
        assert_eq!(normalize("anthropic"), "anthropic");
    }

    #[test]
    fn openai_maps_to_openai() {
        assert_eq!(normalize("openai"), "openai");
    }

    #[test]
    fn openai_responses_maps_to_openai() {
        assert_eq!(normalize("openai-responses"), "openai");
    }

    #[test]
    fn openai_codex_maps_to_openai() {
        assert_eq!(normalize("openai-codex"), "openai");
    }

    #[test]
    fn gemini_maps_to_gcp_gemini() {
        assert_eq!(normalize("gemini"), "gcp.gemini");
    }

    #[test]
    fn vertex_maps_to_gcp_vertex_ai() {
        assert_eq!(normalize("vertex"), "gcp.vertex_ai");
    }

    #[test]
    fn unknown_provider_passes_through() {
        assert_eq!(normalize("custom"), "custom");
    }
}
