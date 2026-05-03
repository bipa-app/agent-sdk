//! LLM provider trait, streaming types, and message data types.
//!
//! This module re-exports from [`agent_sdk_providers`] and [`agent_sdk_core`]
//! so that existing `crate::llm::*` paths continue to resolve.

pub mod attachments {
    //! Attachment validation (facade for external consumers).
    #[allow(unused_imports)]
    pub use agent_sdk_providers::attachments::*;
}

pub mod router {
    //! Model routing.
    pub use agent_sdk_providers::router::*;
}

pub mod streaming {
    //! Streaming types.
    pub use agent_sdk_providers::streaming::*;
}

pub mod types {
    //! LLM data types.
    pub use agent_sdk_core::llm::*;
}

// Re-export everything at the llm:: level for backward compatibility
pub use agent_sdk_core::llm::*;
pub use agent_sdk_providers::provider::{LlmProvider, collect_stream};
pub use agent_sdk_providers::router::{ModelRouter, ModelTier, TaskComplexity};
pub use agent_sdk_providers::streaming::{
    StreamAccumulator, StreamBox, StreamDelta, StreamErrorKind,
};
