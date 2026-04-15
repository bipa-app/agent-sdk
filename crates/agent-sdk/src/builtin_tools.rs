//! Bundled registration for the full set of built-in SDK tools.
//!
//! The SDK ships many ready-to-use tools (primitives, todo tracking,
//! web search and fetch, user-interaction).  Hosts that want to run
//! these through the v2 durable tool runtime need them registered in a
//! shared [`ToolRegistry<Ctx>`] so `agent-service-host`'s
//! `RegistryToolExecutor` can dispatch them by name.
//!
//! This module centralises the registration plumbing so hosts don't
//! have to know the constructor shape of every tool family.  Each
//! family is exposed as its own focused helper and a top-level
//! [`register_builtin_tools`] wires the common bundle.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//! use agent_sdk::builtin_tools::{register_builtin_tools, BuiltinToolsConfig};
//! use agent_sdk::todo::TodoState;
//! use agent_sdk::{AgentCapabilities, InMemoryFileSystem, ToolRegistry};
//!
//! let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
//! let mut registry = ToolRegistry::<()>::new();
//! register_builtin_tools(&mut registry, BuiltinToolsConfig {
//!     environment: fs,
//!     capabilities: AgentCapabilities::full_access(),
//!     todo_state: Some(Arc::new(RwLock::new(TodoState::new()))),
//!     link_fetch: true,
//! });
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

use agent_sdk_tools::tools::ToolRegistry;

use crate::primitive_tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool};
use crate::todo::{TodoReadTool, TodoState, TodoWriteTool};
use crate::web::{LinkFetchTool, SearchProvider, WebSearchTool};
use crate::{AgentCapabilities, Environment};

/// Configuration for [`register_builtin_tools`].
///
/// Every field except `environment` and `capabilities` is optional
/// because the corresponding tool family has side effects or
/// dependencies that must be provisioned by the host.
///
/// The tool families with more complex constructors — in particular
/// [`AskUserQuestionTool`](crate::user_interaction::AskUserQuestionTool)
/// (requires a pair of mpsc channels) and
/// [`WebSearchTool`] (requires a typed
/// [`SearchProvider`]) — are **not** wired through this config to
/// avoid over-coupling it to transport or provider choices.  Register
/// them with [`register_web_search`] or directly against the registry.
pub struct BuiltinToolsConfig<E: Environment + 'static> {
    /// Filesystem environment used by the primitive tools.
    pub environment: Arc<E>,
    /// Capability policy enforced by the primitive tools.
    pub capabilities: AgentCapabilities,
    /// Shared todo state.  When `Some`, registers `TodoRead` / `TodoWrite`.
    pub todo_state: Option<Arc<RwLock<TodoState>>>,
    /// When `true`, registers [`LinkFetchTool`] with default settings.
    pub link_fetch: bool,
}

/// Register the full bundle of built-in tools described by `config`.
///
/// This is a convenience wrapper that delegates to the per-family
/// helpers ([`register_primitives`], [`register_todo_tools`],
/// [`register_link_fetch`]).  Hosts that need finer control can call
/// the helpers directly.
pub fn register_builtin_tools<Ctx, E>(
    registry: &mut ToolRegistry<Ctx>,
    config: BuiltinToolsConfig<E>,
) where
    Ctx: Send + Sync + 'static,
    E: Environment + 'static,
{
    register_primitives(registry, config.environment, config.capabilities);

    if let Some(state) = config.todo_state {
        register_todo_tools(registry, state);
    }

    if config.link_fetch {
        register_link_fetch(registry);
    }
}

/// Register the six filesystem / shell primitives
/// (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`).
pub fn register_primitives<Ctx, E>(
    registry: &mut ToolRegistry<Ctx>,
    environment: Arc<E>,
    capabilities: AgentCapabilities,
) where
    Ctx: Send + Sync + 'static,
    E: Environment + 'static,
{
    registry.register(ReadTool::new(
        Arc::clone(&environment),
        capabilities.clone(),
    ));
    registry.register(WriteTool::new(
        Arc::clone(&environment),
        capabilities.clone(),
    ));
    registry.register(EditTool::new(
        Arc::clone(&environment),
        capabilities.clone(),
    ));
    registry.register(BashTool::new(
        Arc::clone(&environment),
        capabilities.clone(),
    ));
    registry.register(GlobTool::new(
        Arc::clone(&environment),
        capabilities.clone(),
    ));
    registry.register(GrepTool::new(environment, capabilities));
}

/// Register [`TodoReadTool`] and [`TodoWriteTool`] sharing `state`.
pub fn register_todo_tools<Ctx>(registry: &mut ToolRegistry<Ctx>, state: Arc<RwLock<TodoState>>)
where
    Ctx: Send + Sync + 'static,
{
    registry.register(TodoReadTool::new(Arc::clone(&state)));
    registry.register(TodoWriteTool::new(state));
}

/// Register [`LinkFetchTool`] with default HTTP client and
/// SSRF-protecting [`UrlValidator`](crate::web::security::UrlValidator).
pub fn register_link_fetch<Ctx>(registry: &mut ToolRegistry<Ctx>)
where
    Ctx: Send + Sync + 'static,
{
    registry.register(LinkFetchTool::new());
}

/// Register [`WebSearchTool`] backed by `provider`.
///
/// Kept separate from [`BuiltinToolsConfig`] so hosts can pick a
/// concrete [`SearchProvider`] (Brave, custom, etc.) without that
/// choice leaking into every call site of [`register_builtin_tools`].
pub fn register_web_search<Ctx, P>(registry: &mut ToolRegistry<Ctx>, provider: P)
where
    Ctx: Send + Sync + 'static,
    P: SearchProvider + 'static,
{
    registry.register(WebSearchTool::new(provider));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryFileSystem;

    #[test]
    fn register_builtin_tools_wires_primitives_and_todo_and_fetch() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let todo = Arc::new(RwLock::new(TodoState::new()));
        let mut registry = ToolRegistry::<()>::new();

        register_builtin_tools(
            &mut registry,
            BuiltinToolsConfig {
                environment: fs,
                capabilities: AgentCapabilities::full_access(),
                todo_state: Some(todo),
                link_fetch: true,
            },
        );

        for expected in [
            "read",
            "write",
            "edit",
            "bash",
            "glob",
            "grep",
            "todo_read",
            "todo_write",
            "link_fetch",
        ] {
            assert!(
                registry.get(expected).is_some(),
                "expected '{expected}' registered",
            );
        }
    }

    #[test]
    fn register_primitives_registers_exactly_six_tools() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let mut registry = ToolRegistry::<()>::new();
        register_primitives(&mut registry, fs, AgentCapabilities::read_only());

        for expected in ["read", "write", "edit", "bash", "glob", "grep"] {
            assert!(
                registry.get(expected).is_some(),
                "expected '{expected}' registered",
            );
        }
    }

    #[test]
    fn builtin_tools_config_skips_optional_families_when_unset() {
        let fs = Arc::new(InMemoryFileSystem::new("/workspace"));
        let mut registry = ToolRegistry::<()>::new();
        register_builtin_tools(
            &mut registry,
            BuiltinToolsConfig {
                environment: fs,
                capabilities: AgentCapabilities::full_access(),
                todo_state: None,
                link_fetch: false,
            },
        );

        assert!(registry.get("read").is_some());
        assert!(registry.get("todo_read").is_none());
        assert!(registry.get("link_fetch").is_none());
    }
}
