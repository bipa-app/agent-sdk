//! Agent SDK - A Rust SDK for building LLM-powered agents.
//!
//! This crate provides the building blocks for creating AI agents with:
//! - Tool execution and lifecycle hooks
//! - Streaming event-based architecture
//! - Provider-agnostic LLM interface
//! - Built-in primitive tools for file operations
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::{AgentLoop, AgentConfig};
//!
//! let agent = AgentLoop::builder()
//!     .provider(your_provider)
//!     .config(AgentConfig::default())
//!     .build();
//!
//! let events = agent.run("Hello!").await;
//! ```

#![forbid(unsafe_code)]
