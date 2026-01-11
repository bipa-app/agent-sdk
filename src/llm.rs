pub mod router;
pub mod types;

pub use router::{ModelRouter, ModelTier, TaskComplexity};
pub use types::*;

use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatOutcome>;
    fn model(&self) -> &str;
    fn provider(&self) -> &'static str;
}
