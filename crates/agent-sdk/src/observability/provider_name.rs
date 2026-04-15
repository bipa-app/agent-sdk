//! Provider name normalization to `GenAI` semconv canonical values.

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
