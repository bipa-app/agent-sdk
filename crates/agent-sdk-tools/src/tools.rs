//! Tool definition and registry.
//!
//! Tools allow the LLM to perform actions in the real world. This module provides:
//!
//! - [`Tool`] trait - Define custom tools the LLM can call
//! - [`ToolName`] trait - Marker trait for strongly-typed tool names
//! - [`PrimitiveToolName`] - Tool names for SDK's built-in tools
//! - [`DynamicToolName`] - Tool names created at runtime (MCP bridges)
//! - [`ToolRegistry`] - Collection of available tools
//! - [`ToolContext`] - Context passed to tool execution
//! - [`ListenExecuteTool`] - Tools that listen for updates, then execute later
//!
//! # Implementing a Tool
//!
//! ```
//! use agent_sdk_tools::tools::{Tool, ToolContext, DynamicToolName};
//! use agent_sdk_foundation::types::{ToolResult, ToolTier};
//! use serde_json::{json, Value};
//! use std::future::Future;
//!
//! struct MyTool;
//!
//! // No #[async_trait] needed - Rust 1.75+ supports native async traits
//! impl Tool<()> for MyTool {
//!     type Name = DynamicToolName;
//!
//!     fn name(&self) -> DynamicToolName { DynamicToolName::new("my_tool") }
//!     // `display_name` defaults to "" — override it for nicer UI.
//!     fn description(&self) -> &'static str { "Does something useful" }
//!     fn input_schema(&self) -> Value { json!({ "type": "object" }) }
//!     fn tier(&self) -> ToolTier { ToolTier::Observe }
//!
//!     fn execute(
//!         &self,
//!         _ctx: &ToolContext<()>,
//!         _input: Value,
//!     ) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
//!         async move { Ok(ToolResult::success("Done!")) }
//!     }
//! }
//! ```

use crate::authority::{EventAuthority, LocalEventAuthority};
use crate::seed::{HostDependencies, ToolContextSeed};
use crate::stores::EventStore;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::llm;
use agent_sdk_foundation::types::{ToolOutcome, ToolResult, ToolTier};
use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Tool Name Types
// ============================================================================

/// Marker trait for tool names.
///
/// Tool names must be serializable (for storage/logging) and deserializable
/// (for parsing from LLM responses). The string representation is derived
/// from serde serialization.
///
/// # Example
///
/// ```ignore
/// #[derive(Serialize, Deserialize)]
/// #[serde(rename_all = "snake_case")]
/// pub enum MyToolName {
///     Read,
///     Write,
/// }
///
/// impl ToolName for MyToolName {}
/// ```
pub trait ToolName: Send + Sync + Serialize + DeserializeOwned + 'static {}

/// Helper to get string representation of a tool name via serde.
///
/// Returns `"<unknown_tool>"` if serialization fails (should never happen
/// with properly implemented `ToolName` types that use `#[derive(Serialize)]`).
#[must_use]
pub fn tool_name_to_string<N: ToolName>(name: &N) -> String {
    serde_json::to_string(name)
        .unwrap_or_else(|_| "\"<unknown_tool>\"".to_string())
        .trim_matches('"')
        .to_string()
}

/// Parse a tool name from string via serde.
///
/// The input is encoded as a JSON string with `serde_json::to_string` (not
/// interpolated with `format!`) so names containing quotes or backslashes —
/// possible for [`DynamicToolName`]s bridged from remote MCP servers — are
/// escaped correctly and round-trip with [`tool_name_to_string`].
///
/// # Errors
/// Returns error if the string doesn't match a valid tool name.
pub fn tool_name_from_str<N: ToolName>(s: &str) -> Result<N, serde_json::Error> {
    let json = serde_json::to_string(s)?;
    serde_json::from_str(&json)
}

/// Tool names for SDK's built-in primitive tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveToolName {
    Read,
    Write,
    Edit,
    MultiEdit,
    Bash,
    Glob,
    Grep,
    NotebookRead,
    NotebookEdit,
    TodoRead,
    TodoWrite,
    AskUser,
    LinkFetch,
    WebSearch,
}

impl ToolName for PrimitiveToolName {}

/// Dynamic tool name for runtime-created tools (MCP bridges, subagents).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DynamicToolName(String);

impl DynamicToolName {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl ToolName for DynamicToolName {}

// ============================================================================
// Progress Stage Types (for AsyncTool)
// ============================================================================

/// Marker trait for tool progress stages (type-safe, like [`ToolName`]).
///
/// Progress stages are used by async tools to indicate the current phase
/// of a long-running operation. They must be serializable for event streaming.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, Debug, Serialize, Deserialize)]
/// #[serde(rename_all = "snake_case")]
/// pub enum PixTransferStage {
///     Initiated,
///     Processing,
///     SentToBank,
/// }
///
/// impl ProgressStage for PixTransferStage {}
/// ```
pub trait ProgressStage: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {}

/// Helper to get string representation of a progress stage via serde.
///
/// Returns `"<unknown_stage>"` if serialization fails (should never happen with
/// properly implemented `ProgressStage` types). This mirrors
/// [`tool_name_to_string`]'s non-panicking fallback so a failing `Serialize`
/// impl cannot panic the turn loop on the async-tool progress hot path.
#[must_use]
pub fn stage_to_string<S: ProgressStage>(stage: &S) -> String {
    serde_json::to_string(stage)
        .unwrap_or_else(|_| "\"<unknown_stage>\"".to_string())
        .trim_matches('"')
        .to_string()
}

/// Status update from an async tool operation.
#[derive(Clone, Debug, Serialize)]
pub enum ToolStatus<S: ProgressStage> {
    /// Operation is making progress
    Progress {
        stage: S,
        message: String,
        data: Option<serde_json::Value>,
    },

    /// Operation completed successfully
    Completed(ToolResult),

    /// Operation failed
    Failed(ToolResult),
}

/// Type-erased status for the agent loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ErasedToolStatus {
    /// Operation is making progress
    Progress {
        stage: String,
        message: String,
        data: Option<serde_json::Value>,
    },
    /// Operation completed successfully
    Completed(ToolResult),
    /// Operation failed
    Failed(ToolResult),
}

/// Update emitted from a `listen()` stream.
///
/// This models workflows where a runtime prepares an operation over time, and
/// execution happens later using an operation identifier and revision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ListenToolUpdate {
    /// Preparation is still running and should keep listening.
    Listening {
        /// Opaque operation identifier used for later execute/cancel calls.
        operation_id: String,
        /// Monotonic revision number for optimistic concurrency.
        revision: u64,
        /// Human-readable status message.
        message: String,
        /// Optional current snapshot for UI rendering.
        snapshot: Option<serde_json::Value>,
        /// Optional expiration timestamp (RFC3339).
        #[serde(with = "time::serde::rfc3339::option")]
        expires_at: Option<OffsetDateTime>,
    },

    /// Preparation is complete and execution can be confirmed.
    Ready {
        /// Opaque operation identifier used for later execute/cancel calls.
        operation_id: String,
        /// Monotonic revision number for optimistic concurrency.
        revision: u64,
        /// Human-readable status message.
        message: String,
        /// Snapshot shown in confirmation UI.
        snapshot: serde_json::Value,
        /// Optional expiration timestamp (RFC3339).
        #[serde(with = "time::serde::rfc3339::option")]
        expires_at: Option<OffsetDateTime>,
    },

    /// Operation is no longer valid.
    Invalidated {
        /// Opaque operation identifier.
        operation_id: String,
        /// Human-readable reason.
        message: String,
        /// Whether caller may recover by starting a new listen operation.
        recoverable: bool,
    },
}

/// Reason for stopping a listen session.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ListenStopReason {
    /// User explicitly rejected confirmation.
    UserRejected,
    /// Agent policy/hook blocked execution before confirmation.
    Blocked,
    /// Consumer disconnected while listen stream was active.
    StreamDisconnected,
    /// Listen stream ended unexpectedly before terminal state.
    StreamEnded,
}

impl<S: ProgressStage> From<ToolStatus<S>> for ErasedToolStatus {
    fn from(status: ToolStatus<S>) -> Self {
        match status {
            ToolStatus::Progress {
                stage,
                message,
                data,
            } => Self::Progress {
                stage: stage_to_string(&stage),
                message,
                data,
            },
            ToolStatus::Completed(r) => Self::Completed(r),
            ToolStatus::Failed(r) => Self::Failed(r),
        }
    }
}

/// Context passed to tool execution
#[derive(Clone)]
pub struct ToolContext<Ctx> {
    /// Application-specific context (e.g., `user_id`, db connection)
    pub app: Ctx,
    /// Tool-specific metadata
    pub metadata: HashMap<String, Value>,
    /// Optional event store for tools to emit turn-scoped events.
    event_store: Option<Arc<dyn EventStore>>,
    /// Thread associated with the bound event store.
    event_thread_id: Option<agent_sdk_foundation::types::ThreadId>,
    /// Turn associated with the bound event store.
    event_turn: Option<usize>,
    /// Optional event authority for wrapping events in envelopes
    event_authority: Option<Arc<dyn EventAuthority>>,
    /// Optional cancellation token for propagating cancellation to subtasks
    cancel_token: Option<CancellationToken>,
    /// Optional semaphore for limiting concurrent subagent threads.
    subagent_semaphore: Option<Arc<tokio::sync::Semaphore>>,
    /// Optional per-tool execution timeout enforced at the SDK boundary.
    ///
    /// When set, the agent loop races each tool's `execute()` future
    /// against this duration. A tool that does not finish within the
    /// budget is stopped at the boundary and reported with a synthetic
    /// timeout [`ToolResult`] so the `tool_use` / `tool_result` pair stays
    /// balanced. Tools that hold OS resources (subprocesses, sockets) must
    /// observe the [cooperative-cancel contract](Tool#cooperative-cancellation)
    /// so the timeout actually reclaims them.
    tool_timeout: Option<std::time::Duration>,
}

impl<Ctx> ToolContext<Ctx> {
    #[must_use]
    pub fn new(app: Ctx) -> Self {
        Self {
            app,
            metadata: HashMap::new(),
            event_store: None,
            event_thread_id: None,
            event_turn: None,
            event_authority: None,
            cancel_token: None,
            subagent_semaphore: None,
            tool_timeout: None,
        }
    }

    /// Reconstruct a `ToolContext` from a durable seed and host-provided
    /// runtime dependencies.
    ///
    /// This is the authoritative reconstruction path.  Workers should use
    /// this (or a host's [`crate::seed::ExecutionContextFactory`]) instead
    /// of chaining builder methods, so that the context shape is
    /// deterministic and auditable.
    ///
    /// The event authority is constructed internally from
    /// [`ToolContextSeed::sequence_offset`] to guarantee monotonic
    /// sequencing — callers cannot accidentally supply a misaligned
    /// authority.
    #[must_use]
    pub fn from_seed(seed: &ToolContextSeed, app: Ctx, deps: HostDependencies) -> Self {
        let authority: Arc<dyn EventAuthority> =
            Arc::new(LocalEventAuthority::with_offset(seed.sequence_offset));
        Self {
            app,
            metadata: seed.metadata.clone(),
            event_store: Some(deps.event_store),
            event_thread_id: Some(seed.thread_id.clone()),
            event_turn: Some(seed.turn),
            event_authority: Some(authority),
            cancel_token: Some(deps.cancel_token),
            subagent_semaphore: deps.subagent_semaphore,
            tool_timeout: None,
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Bind the tool context to the event store for a specific thread/turn.
    #[must_use]
    pub fn with_event_store(
        mut self,
        store: Arc<dyn EventStore>,
        thread_id: agent_sdk_foundation::types::ThreadId,
        turn: usize,
        authority: Arc<dyn EventAuthority>,
    ) -> Self {
        self.event_store = Some(store);
        self.event_thread_id = Some(thread_id);
        self.event_turn = Some(turn);
        self.event_authority = Some(authority);
        self
    }

    /// Emit an event through the configured event store (if set).
    ///
    /// The event is wrapped in an [`agent_sdk_foundation::AgentEventEnvelope`] with a unique ID,
    /// sequence number, and timestamp before publishing.
    ///
    /// # Errors
    /// Returns an error if the configured event store cannot persist the event.
    pub async fn emit_event(&self, event: AgentEvent) -> Result<()>
    where
        Ctx: Sync,
    {
        let Some((store, authority, thread_id, turn)) = self
            .event_store
            .as_ref()
            .zip(self.event_authority.as_ref())
            .zip(self.event_thread_id.as_ref())
            .zip(self.event_turn)
            .map(|(((store, authority), thread_id), turn)| (store, authority, thread_id, turn))
        else {
            // Surface the misconfiguration instead of silently dropping the
            // event: a tool written for the durable host but run under a
            // hand-built `ToolContext::new()` would otherwise lose every
            // emitted event with no trace, undermining the audit trail.
            let kind = serde_json::to_value(&event)
                .ok()
                .and_then(|v| {
                    v.get("type")
                        .and_then(|t| t.as_str().map(ToOwned::to_owned))
                })
                .unwrap_or_else(|| "unknown".to_string());
            log::warn!(
                "ToolContext::emit_event called on an unbound context; dropping {kind} event \
                 (no event store/authority/thread/turn bound)"
            );
            return Ok(());
        };
        let envelope = authority.wrap(event);
        store.append(thread_id, turn, envelope).await
    }

    /// Get a clone of the event authority (if set).
    ///
    /// This is useful for tools that spawn subprocesses (like subagents)
    /// and need to wrap events with the same sequencing authority as the
    /// parent's turn log.
    #[must_use]
    pub fn event_authority(&self) -> Option<Arc<dyn EventAuthority>> {
        self.event_authority.clone()
    }

    /// Set the cancellation token for propagating cancellation to subtasks.
    #[must_use]
    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    /// Get the cancellation token (if set).
    ///
    /// Used by tools that spawn long-running subtasks (like subagents)
    /// to propagate cancellation from the parent.
    #[must_use]
    pub fn cancel_token(&self) -> Option<CancellationToken> {
        self.cancel_token.clone()
    }

    /// Set the per-tool execution timeout enforced at the SDK boundary.
    ///
    /// The agent loop populates this from `AgentConfig::tool_timeout_ms`;
    /// callers can also set it directly when constructing a context.
    #[must_use]
    pub const fn with_tool_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.tool_timeout = Some(timeout);
        self
    }

    /// Get the per-tool execution timeout (if set).
    ///
    /// Read by the agent loop's SDK-boundary execution race; tools do not
    /// normally need to consult this themselves.
    #[must_use]
    pub const fn tool_timeout(&self) -> Option<std::time::Duration> {
        self.tool_timeout
    }

    /// Set a shared semaphore for limiting concurrent subagent threads.
    #[must_use]
    pub fn with_subagent_semaphore(mut self, semaphore: Arc<tokio::sync::Semaphore>) -> Self {
        self.subagent_semaphore = Some(semaphore);
        self
    }

    /// Get the subagent thread-limiting semaphore (if set).
    #[must_use]
    pub fn subagent_semaphore(&self) -> Option<Arc<tokio::sync::Semaphore>> {
        self.subagent_semaphore.clone()
    }
}

// ============================================================================
// Tool Trait
// ============================================================================

/// Definition of a tool that can be called by the agent.
///
/// Tools have a strongly-typed `Name` associated type that determines
/// how the tool name is serialized for LLM communication.
///
/// # Native Async Support
///
/// This trait uses Rust's native async functions in traits (stabilized in Rust 1.75).
/// You do NOT need the `async_trait` crate to implement this trait.
///
/// # Cooperative cancellation
///
/// The agent loop races every tool's `execute()` future against the run's
/// [`ToolContext::cancel_token`] and, when configured, against
/// [`ToolContext::tool_timeout`]. If either fires the SDK drops the
/// in-flight `execute()` future and synthesises a balanced `tool_result`
/// (`"Cancelled by user"` or a timeout message). Dropping a future runs
/// its destructors but cannot, on its own, reclaim OS resources a tool
/// has handed to the kernel.
///
/// **Subprocess contract:** a tool that spawns a child process MUST make
/// the process die when its `execute()` future is dropped. The two
/// supported ways to satisfy this are:
///
/// * Build the command with `tokio::process::Command::kill_on_drop(true)`,
///   so the child is killed when the `Child` handle is dropped together
///   with the cancelled future (this is what the SDK's MCP stdio transport
///   does), or
/// * Observe [`ToolContext::cancel_token`] directly and `kill()` the child
///   when it fires.
///
/// A tool that holds a subprocess open without either of these will leak
/// the process when cancelled or timed out — the synthesised `tool_result`
/// keeps the conversation balanced, but the orphaned OS process is the
/// tool author's bug, not the SDK's.
pub trait Tool<Ctx>: Send + Sync {
    /// The type of name for this tool.
    type Name: ToolName;

    /// Returns the tool's strongly-typed name.
    fn name(&self) -> Self::Name;

    /// Human-readable display name for UI (e.g., "Read File" vs "read").
    ///
    /// Defaults to the empty string. Override for better UX.
    fn display_name(&self) -> &'static str {
        ""
    }

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool.
    ///
    /// Defaults to [`ToolTier::Confirm`] (fail-closed): a tool author who
    /// forgets to declare a tier gets confirmation gating, not silent
    /// auto-execution. Read-only tools should explicitly opt in to
    /// [`ToolTier::Observe`].
    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    /// Execute the tool with the given input.
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolResult>> + Send;
}

// ============================================================================
// TypedTool Trait (typed input + runtime validation / self-correction)
// ============================================================================

/// A tool whose model-emitted arguments are validated against a typed,
/// deserializable [`Input`](TypedTool::Input) **before** [`execute`](TypedTool::execute)
/// runs.
///
/// Today a raw [`serde_json::Value`] is handed straight to [`Tool::execute`],
/// so a malformed tool call reaches tool code unvalidated. `TypedTool` closes
/// that gap: you declare a `Serialize` / `Deserialize` argument struct as
/// [`Input`](TypedTool::Input), and the runtime deserializes the model's args
/// into it at the dispatch boundary. On a deserialization/validation failure
/// the runtime synthesises a structured error [`ToolResult`] (carrying the
/// serde error message) so the model can self-correct on its next turn —
/// `execute` is **never** called with invalid arguments.
///
/// # Relationship to [`Tool`]
///
/// `TypedTool` is the typed, opt-in *sugar* layer; [`Tool`] remains the
/// untyped baseline. A [`TypedTool`] becomes a full [`Tool`] through
/// [`TypedToolAdapter`] (mirroring how [`SimpleTool`] becomes a [`Tool`] via
/// [`SimpleToolAdapter`]). Register one with
/// [`ToolRegistry::register_typed`], which wraps it in the adapter for you;
/// the adapter performs the deserialize-then-dispatch (or
/// deserialize-then-synthesise-error) described above.
///
/// # Back-compat / migration
///
/// Existing [`Tool`] impls (and [`SimpleTool`] / [`DynamicToolName`] tools)
/// keep compiling and running unchanged — they stay on the `Value`-in
/// baseline, which is the identity passthrough (a `Value` always
/// "deserializes" into a `Value`). Migrate a tool to typed args by moving its
/// `impl Tool<Ctx>` to `impl TypedTool<Ctx>`, setting `type Input = MyArgs`,
/// and changing `execute`'s signature from `input: Value` to `input: MyArgs`.
/// The hand-written [`input_schema`](TypedTool::input_schema) JSON stays
/// user-declared; this trait does **not** auto-derive a schema from `Input`.
///
/// # Example
///
/// ```
/// use agent_sdk_tools::tools::{TypedTool, ToolContext};
/// use agent_sdk_foundation::types::ToolResult;
/// use serde::{Deserialize, Serialize};
/// use serde_json::{json, Value};
/// use std::future::Future;
///
/// #[derive(Debug, Serialize, Deserialize)]
/// struct WeatherArgs {
///     city: String,
/// }
///
/// struct WeatherTool;
///
/// impl TypedTool<()> for WeatherTool {
///     type Input = WeatherArgs;
///
///     fn name(&self) -> &'static str { "get_weather" }
///     fn description(&self) -> &'static str { "Get current weather for a city" }
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": { "city": { "type": "string" } },
///             "required": ["city"]
///         })
///     }
///
///     fn execute(
///         &self,
///         _ctx: &ToolContext<()>,
///         input: WeatherArgs,
///     ) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
///         async move { Ok(ToolResult::success(format!("Weather in {}: Sunny", input.city))) }
///     }
/// }
/// ```
///
/// Like [`SimpleTool`], a `TypedTool` has a single fixed `&'static str`
/// [`name`](TypedTool::name) (mapping to [`DynamicToolName`] via
/// [`TypedToolAdapter`]). Reach for a hand-written [`Tool`] with a
/// strongly-typed [`ToolName`] when the name must be computed at runtime or
/// constrained to an enum.
pub trait TypedTool<Ctx>: Send + Sync {
    /// The typed input the model's arguments are deserialized into before
    /// [`execute`](TypedTool::execute) runs.
    ///
    /// Must be [`DeserializeOwned`] (to parse model args), [`Serialize`] (so
    /// the typed value round-trips for logging/storage), and `Send + 'static`
    /// (to cross the async dispatch boundary).
    type Input: DeserializeOwned + Serialize + Send + 'static;

    /// The tool's name as sent to (and parsed from) the model.
    fn name(&self) -> &'static str;

    /// Human-readable display name for UI. Defaults to an empty string.
    fn display_name(&self) -> &'static str {
        ""
    }

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// User-declared JSON schema for the tool's input parameters.
    ///
    /// This stays hand-written JSON — it is **not** auto-derived from
    /// [`Input`](TypedTool::Input). Keeping the schema explicit lets the
    /// declared provider-facing contract diverge from the Rust type when that
    /// is useful (descriptions, examples, provider-specific keywords).
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool. Defaults to [`ToolTier::Confirm`]
    /// (fail-closed); read-only tools should opt in to [`ToolTier::Observe`].
    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    /// Execute the tool with the already-validated, typed input.
    ///
    /// The runtime guarantees `input` deserialized cleanly from the model's
    /// arguments; a malformed call is turned into a structured error
    /// [`ToolResult`] before this method is reached, so implementations never
    /// see invalid arguments.
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Self::Input,
    ) -> impl Future<Output = Result<ToolResult>> + Send;
}

/// Synthesise the structured validation-error [`ToolResult`] returned to the
/// model when its arguments fail to deserialize into a [`TypedTool::Input`].
///
/// Factored out (and `pub`) so the exact self-correction wording is
/// consistent with [`TypedToolAdapter`] and is directly unit-testable. The
/// error is an *error* [`ToolResult`] (not a thrown `anyhow::Error`): it flows
/// through the normal balanced `tool_use` / `tool_result` path so history
/// stays balanced and the model gets a concrete, machine-actionable hint on
/// its next turn.
#[must_use]
pub fn invalid_tool_input_result(tool_name: &str, error: &serde_json::Error) -> ToolResult {
    ToolResult::error(format!(
        "Invalid arguments for tool `{tool_name}`: {error}. \
         The arguments did not match the tool's input schema — \
         re-read the schema and call the tool again with corrected arguments."
    ))
}

/// Deserialize raw model args into a typed `Input`, or synthesise the
/// structured validation-error result.
///
/// Returns `Ok(typed)` for the happy path and `Err(result)` carrying the
/// balanced error [`ToolResult`] for the self-correction path.
/// [`TypedToolAdapter`] uses this to ensure [`TypedTool::execute`] is never
/// reached with invalid arguments.
///
/// # Errors
/// Returns the synthesised error [`ToolResult`] when `raw` does not
/// deserialize into `Input`.
pub fn validate_tool_input<Input>(tool_name: &str, raw: Value) -> Result<Input, ToolResult>
where
    Input: DeserializeOwned,
{
    serde_json::from_value(raw).map_err(|error| invalid_tool_input_result(tool_name, &error))
}

/// Adapter that turns any [`TypedTool`] into a full [`Tool`].
///
/// It gives the wrapped tool `Name = DynamicToolName`, deserializes the
/// model's `Value` arguments into [`TypedTool::Input`] before dispatching, and
/// synthesises a structured validation-error [`ToolResult`] when that fails.
///
/// You rarely name this type directly — register a [`TypedTool`] with
/// [`ToolRegistry::register_typed`], which wraps it for you. The adapter
/// pattern (rather than a blanket `impl Tool for T: TypedTool`) is required
/// for coherence: a blanket impl would conflict with the existing
/// [`SimpleToolAdapter`] impl, because the compiler cannot rule out a
/// downstream `TypedTool` impl for `SimpleToolAdapter`.
///
/// This adapter is also where the typed `Input` is threaded through the
/// erased-tool machinery without leaking the generic into trait objects: the
/// registry's [`ErasedTool`] wrapper still only ever sees `Value`, while the
/// concrete `Input` type (and the deserialize) live here, inside the adapter's
/// concrete `T`.
pub struct TypedToolAdapter<T> {
    inner: T,
}

impl<T> TypedToolAdapter<T> {
    /// Wrap a [`TypedTool`] so it can be used anywhere a [`Tool`] is expected.
    pub const fn new(tool: T) -> Self {
        Self { inner: tool }
    }

    /// Unwrap the inner [`TypedTool`].
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<Ctx, T> Tool<Ctx> for TypedToolAdapter<T>
where
    T: TypedTool<Ctx>,
    Ctx: Send + Sync,
{
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(TypedTool::name(&self.inner))
    }

    fn display_name(&self) -> &'static str {
        TypedTool::display_name(&self.inner)
    }

    fn description(&self) -> &'static str {
        TypedTool::description(&self.inner)
    }

    fn input_schema(&self) -> Value {
        TypedTool::input_schema(&self.inner)
    }

    fn tier(&self) -> ToolTier {
        TypedTool::tier(&self.inner)
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        match validate_tool_input::<<T as TypedTool<Ctx>>::Input>(
            TypedTool::name(&self.inner),
            input,
        ) {
            Ok(typed) => TypedTool::execute(&self.inner, ctx, typed).await,
            // A validation failure is returned as an error `ToolResult`,
            // never `?`-bailed: it must reach the model as a balanced
            // `tool_result` for self-correction. `execute` is not called.
            Err(result) => Ok(result),
        }
    }
}

// ============================================================================
// ToolLogic Trait (execute-only companion for the derive macros)
// ============================================================================

/// The `execute`-only half of a tool, used as the target of the
/// `#[derive(Tool)]` / `#[derive(TypedTool)]` ergonomics macros.
///
/// The derives generate everything *except* the behaviour — `name`,
/// `description`, `input_schema`, `tier` come from `#[tool(...)]` attributes —
/// and delegate execution to this trait. You implement `ToolLogic` to supply
/// the one thing a macro cannot: the `execute` body.
///
/// It is deliberately a **trait** (not an inherent method): a trait-method
/// `async fn` that performs no `await` is fine, whereas an inherent one trips
/// `clippy::unused_async`. Writing the body here keeps trivial, fully
/// synchronous tools lint-clean without an `#[allow]`.
///
/// You rarely name this trait in prose — the derive docs show it in context —
/// but the shape is:
///
/// ```
/// use agent_sdk_tools::tools::{ToolLogic, ToolContext};
/// use agent_sdk_foundation::types::ToolResult;
/// use serde_json::Value;
///
/// struct MyTool;
///
/// impl ToolLogic<()> for MyTool {
///     type Input = Value; // typed tools set this to their `Input` struct
///
///     async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> anyhow::Result<ToolResult> {
///         Ok(ToolResult::success(format!("got {input}")))
///     }
/// }
/// ```
pub trait ToolLogic<Ctx>: Send + Sync {
    /// The input the tool's `execute` receives. For `#[derive(Tool)]` this is
    /// [`serde_json::Value`]; for `#[derive(TypedTool)]` it is the typed
    /// `Input` (validated before `execute` runs).
    type Input;

    /// The tool's behaviour. Receives the (already-validated, for typed tools)
    /// input.
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Self::Input,
    ) -> impl Future<Output = Result<ToolResult>> + Send;
}

// ============================================================================
// SimpleTool Trait
// ============================================================================

/// An ergonomic [`Tool`] whose name is a plain string.
///
/// Most custom tools don't need a strongly-typed [`ToolName`] enum — they have
/// a single, fixed name. `SimpleTool` lets you write a tool by returning a
/// `&str` from [`name`](SimpleTool::name) instead of defining a `ToolName`
/// type and an associated [`Tool::Name`].
///
/// Any `SimpleTool` is automatically a [`Tool`] (via a blanket impl) with
/// `Name = DynamicToolName`, so it can be registered and used exactly like a
/// hand-written `Tool`.
///
/// # Example
///
/// ```
/// use agent_sdk_tools::tools::{SimpleTool, ToolContext};
/// use agent_sdk_foundation::types::ToolResult;
/// use serde_json::{json, Value};
/// use std::future::Future;
///
/// struct WeatherTool;
///
/// impl SimpleTool<()> for WeatherTool {
///     fn name(&self) -> &'static str { "get_weather" }
///     fn description(&self) -> &'static str { "Get current weather for a city" }
///     fn input_schema(&self) -> Value {
///         json!({ "type": "object", "properties": { "city": { "type": "string" } } })
///     }
///
///     fn execute(
///         &self,
///         _ctx: &ToolContext<()>,
///         input: Value,
///     ) -> impl Future<Output = anyhow::Result<ToolResult>> + Send {
///         async move {
///             let city = input["city"].as_str().unwrap_or("Unknown");
///             Ok(ToolResult::success(format!("Weather in {city}: Sunny")))
///         }
///     }
/// }
/// ```
pub trait SimpleTool<Ctx>: Send + Sync {
    /// The tool's name as sent to (and parsed from) the LLM.
    ///
    /// Returns `&'static str` because a simple tool has one fixed name; reach
    /// for the full [`Tool`] trait with a [`DynamicToolName`] when the name is
    /// computed at runtime.
    fn name(&self) -> &'static str;

    /// Human-readable display name for UI.
    ///
    /// Defaults to an empty string; override for a friendlier label.
    fn display_name(&self) -> &'static str {
        ""
    }

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool. Defaults to [`ToolTier::Confirm`]
    /// (fail-closed); read-only tools should opt in to [`ToolTier::Observe`].
    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    /// Execute the tool with the given input.
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolResult>> + Send;
}

/// Adapter that turns any [`SimpleTool`] into a full [`Tool`] with
/// `Name = DynamicToolName`.
///
/// You rarely name this type directly — register a [`SimpleTool`] with
/// [`ToolRegistry::register_simple`], which wraps it for you. Use this adapter
/// explicitly only when you need a `Tool` value (e.g. to pass to code that is
/// generic over [`Tool`]).
pub struct SimpleToolAdapter<T> {
    inner: T,
}

impl<T> SimpleToolAdapter<T> {
    /// Wrap a [`SimpleTool`] so it can be used anywhere a [`Tool`] is expected.
    pub const fn new(tool: T) -> Self {
        Self { inner: tool }
    }

    /// Unwrap the inner [`SimpleTool`].
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<Ctx, T> Tool<Ctx> for SimpleToolAdapter<T>
where
    T: SimpleTool<Ctx>,
{
    type Name = DynamicToolName;

    fn name(&self) -> DynamicToolName {
        DynamicToolName::new(SimpleTool::name(&self.inner))
    }

    fn display_name(&self) -> &'static str {
        SimpleTool::display_name(&self.inner)
    }

    fn description(&self) -> &'static str {
        SimpleTool::description(&self.inner)
    }

    fn input_schema(&self) -> Value {
        SimpleTool::input_schema(&self.inner)
    }

    fn tier(&self) -> ToolTier {
        SimpleTool::tier(&self.inner)
    }

    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolResult>> + Send {
        SimpleTool::execute(&self.inner, ctx, input)
    }
}

// ============================================================================
// AsyncTool Trait
// ============================================================================

/// A tool that performs long-running async operations.
///
/// `AsyncTool`s have two phases:
/// 1. `execute()` - Start the operation (lightweight, returns quickly)
/// 2. `check_status()` - Stream progress until completion
///
/// The actual work should happen externally (background task, external service)
/// and persist results to a durable store. The tool is just an orchestrator.
///
/// # Example
///
/// ```ignore
/// impl AsyncTool<MyCtx> for ExecutePixTransferTool {
///     type Name = PixToolName;
///     type Stage = PixTransferStage;
///
///     async fn execute(&self, ctx: &ToolContext<MyCtx>, input: Value) -> Result<ToolOutcome> {
///         let params = parse_input(&input)?;
///         let operation_id = ctx.app.pix_service.start_transfer(params).await?;
///         Ok(ToolOutcome::in_progress(
///             operation_id,
///             format!("PIX transfer of {} initiated", params.amount),
///         ))
///     }
///
///     fn check_status(&self, ctx: &ToolContext<MyCtx>, operation_id: &str)
///         -> impl Stream<Item = ToolStatus<PixTransferStage>> + Send
///     {
///         async_stream::stream! {
///             loop {
///                 let status = ctx.app.pix_service.get_status(operation_id).await;
///                 match status {
///                     PixStatus::Success { id } => {
///                         yield ToolStatus::Completed(ToolResult::success(id));
///                         break;
///                     }
///                     _ => yield ToolStatus::Progress { ... };
///                 }
///                 tokio::time::sleep(Duration::from_millis(500)).await;
///             }
///         }
///     }
/// }
/// ```
pub trait AsyncTool<Ctx>: Send + Sync {
    /// The type of name for this tool.
    type Name: ToolName;
    /// The type of progress stages for this tool.
    type Stage: ProgressStage;

    /// Returns the tool's strongly-typed name.
    fn name(&self) -> Self::Name;

    /// Human-readable display name for UI. Defaults to the empty string.
    fn display_name(&self) -> &'static str {
        ""
    }

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool. Defaults to [`ToolTier::Confirm`]
    /// (fail-closed); read-only tools should opt in to [`ToolTier::Observe`].
    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    /// Execute the tool. Returns immediately with one of:
    /// - Success/Failed: Operation completed synchronously
    /// - `InProgress`: Operation started, use `check_status()` to stream updates
    ///
    /// # Errors
    /// Returns an error if tool execution fails.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Future<Output = Result<ToolOutcome>> + Send;

    /// Stream status updates for an in-progress operation.
    /// Must yield until Completed or Failed.
    fn check_status(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
    ) -> impl Stream<Item = ToolStatus<Self::Stage>> + Send;
}

// ============================================================================
// ListenExecuteTool Trait
// ============================================================================

/// A tool whose runtime has two phases:
/// 1. `listen()` - starts preparation and streams updates
/// 2. `execute()` - performs final execution after confirmation
///
/// This abstraction is useful when runtime state can expire or evolve before
/// execution (quotes, challenge windows, leases, approvals).
///
/// Ordering note: the agent loop consumes `listen()` updates before
/// `AgentHooks::pre_tool_use()` runs. Hooks can therefore block `execute()`, but
/// any side effects done during `listen()` have already happened.
pub trait ListenExecuteTool<Ctx>: Send + Sync {
    /// The type of name for this tool.
    type Name: ToolName;

    /// Returns the tool's strongly-typed name.
    fn name(&self) -> Self::Name;

    /// Human-readable display name for UI. Defaults to the empty string.
    fn display_name(&self) -> &'static str {
        ""
    }

    /// Human-readable description of what the tool does.
    fn description(&self) -> &'static str;

    /// JSON schema for the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Permission tier for this tool.
    fn tier(&self) -> ToolTier {
        ToolTier::Confirm
    }

    /// Start and stream runtime preparation updates.
    fn listen(
        &self,
        ctx: &ToolContext<Ctx>,
        input: Value,
    ) -> impl Stream<Item = ListenToolUpdate> + Send;

    /// Execute using operation ID and optimistic concurrency revision.
    ///
    /// # Errors
    /// Returns an error if execution fails or revision is stale.
    fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
        expected_revision: u64,
    ) -> impl Future<Output = Result<ToolResult>> + Send;

    /// Stop a listen operation (best effort).
    ///
    /// # Errors
    /// Returns an error if cancellation fails.
    fn cancel(
        &self,
        _ctx: &ToolContext<Ctx>,
        _operation_id: &str,
        _reason: ListenStopReason,
    ) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }
}

// ============================================================================
// Type-Erased Tool (for Registry)
// ============================================================================

/// Type-erased tool trait for registry storage.
///
/// This allows tools with different `Name` associated types to be stored
/// in the same registry by erasing the type information.
///
/// # Example
///
/// ```ignore
/// for tool in registry.all() {
///     println!("Tool: {} - {}", tool.name_str(), tool.description());
/// }
/// ```
#[async_trait]
pub trait ErasedTool<Ctx>: Send + Sync {
    /// Get the tool name as a string.
    fn name_str(&self) -> &str;
    /// Get a human-friendly display name for the tool.
    fn display_name(&self) -> &'static str;
    /// Get the tool description.
    fn description(&self) -> &'static str;
    /// Get the JSON schema for tool inputs.
    fn input_schema(&self) -> Value;
    /// Get the tool's permission tier.
    fn tier(&self) -> ToolTier;
    /// Execute the tool with the given input.
    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult>;
}

/// Wrapper that erases the Name associated type from a Tool.
struct ToolWrapper<T, Ctx>
where
    T: Tool<Ctx>,
{
    inner: T,
    name_cache: String,
    _marker: PhantomData<Ctx>,
}

impl<T, Ctx> ToolWrapper<T, Ctx>
where
    T: Tool<Ctx>,
{
    fn new(tool: T) -> Self {
        let name_cache = tool_name_to_string(&tool.name());
        Self {
            inner: tool,
            name_cache,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T, Ctx> ErasedTool<Ctx> for ToolWrapper<T, Ctx>
where
    T: Tool<Ctx> + 'static,
    Ctx: Send + Sync + 'static,
{
    fn name_str(&self) -> &str {
        &self.name_cache
    }

    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn tier(&self) -> ToolTier {
        self.inner.tier()
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        self.inner.execute(ctx, input).await
    }
}

// ============================================================================
// Type-Erased AsyncTool (for Registry)
// ============================================================================

/// Type-erased async tool trait for registry storage.
///
/// This allows async tools with different `Name` and `Stage` associated types
/// to be stored in the same registry by erasing the type information.
#[async_trait]
pub trait ErasedAsyncTool<Ctx>: Send + Sync {
    /// Get the tool name as a string.
    fn name_str(&self) -> &str;
    /// Get a human-friendly display name for the tool.
    fn display_name(&self) -> &'static str;
    /// Get the tool description.
    fn description(&self) -> &'static str;
    /// Get the JSON schema for tool inputs.
    fn input_schema(&self) -> Value;
    /// Get the tool's permission tier.
    fn tier(&self) -> ToolTier;
    /// Execute the tool with the given input.
    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolOutcome>;
    /// Stream status updates for an in-progress operation (type-erased).
    fn check_status_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        operation_id: &'a str,
    ) -> Pin<Box<dyn Stream<Item = ErasedToolStatus> + Send + 'a>>;
}

/// Wrapper that erases the Name and Stage associated types from an [`AsyncTool`].
struct AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx>,
{
    inner: T,
    name_cache: String,
    _marker: PhantomData<Ctx>,
}

impl<T, Ctx> AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx>,
{
    fn new(tool: T) -> Self {
        let name_cache = tool_name_to_string(&tool.name());
        Self {
            inner: tool,
            name_cache,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T, Ctx> ErasedAsyncTool<Ctx> for AsyncToolWrapper<T, Ctx>
where
    T: AsyncTool<Ctx> + 'static,
    Ctx: Send + Sync + 'static,
{
    fn name_str(&self) -> &str {
        &self.name_cache
    }

    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn tier(&self) -> ToolTier {
        self.inner.tier()
    }

    async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolOutcome> {
        self.inner.execute(ctx, input).await
    }

    fn check_status_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        operation_id: &'a str,
    ) -> Pin<Box<dyn Stream<Item = ErasedToolStatus> + Send + 'a>> {
        use futures::StreamExt;
        let stream = self.inner.check_status(ctx, operation_id);
        Box::pin(stream.map(ErasedToolStatus::from))
    }
}

// ============================================================================
// Type-Erased ListenExecuteTool (for Registry)
// ============================================================================

/// Type-erased listen/execute tool trait for registry storage.
#[async_trait]
pub trait ErasedListenTool<Ctx>: Send + Sync {
    /// Get the tool name as a string.
    fn name_str(&self) -> &str;
    /// Get a human-friendly display name for the tool.
    fn display_name(&self) -> &'static str;
    /// Get the tool description.
    fn description(&self) -> &'static str;
    /// Get the JSON schema for tool inputs.
    fn input_schema(&self) -> Value;
    /// Get the tool's permission tier.
    fn tier(&self) -> ToolTier;
    /// Start listen stream.
    fn listen_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        input: Value,
    ) -> Pin<Box<dyn Stream<Item = ListenToolUpdate> + Send + 'a>>;
    /// Execute using a prepared operation.
    async fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
        expected_revision: u64,
    ) -> Result<ToolResult>;
    /// Cancel operation.
    async fn cancel(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
        reason: ListenStopReason,
    ) -> Result<()>;
}

/// Wrapper that erases the Name associated type from a [`ListenExecuteTool`].
struct ListenToolWrapper<T, Ctx>
where
    T: ListenExecuteTool<Ctx>,
{
    inner: T,
    name_cache: String,
    _marker: PhantomData<Ctx>,
}

impl<T, Ctx> ListenToolWrapper<T, Ctx>
where
    T: ListenExecuteTool<Ctx>,
{
    fn new(tool: T) -> Self {
        let name_cache = tool_name_to_string(&tool.name());
        Self {
            inner: tool,
            name_cache,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T, Ctx> ErasedListenTool<Ctx> for ListenToolWrapper<T, Ctx>
where
    T: ListenExecuteTool<Ctx> + 'static,
    Ctx: Send + Sync + 'static,
{
    fn name_str(&self) -> &str {
        &self.name_cache
    }

    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    fn tier(&self) -> ToolTier {
        self.inner.tier()
    }

    fn listen_stream<'a>(
        &'a self,
        ctx: &'a ToolContext<Ctx>,
        input: Value,
    ) -> Pin<Box<dyn Stream<Item = ListenToolUpdate> + Send + 'a>> {
        let stream = self.inner.listen(ctx, input);
        Box::pin(stream)
    }

    async fn execute(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
        expected_revision: u64,
    ) -> Result<ToolResult> {
        self.inner
            .execute(ctx, operation_id, expected_revision)
            .await
    }

    async fn cancel(
        &self,
        ctx: &ToolContext<Ctx>,
        operation_id: &str,
        reason: ListenStopReason,
    ) -> Result<()> {
        self.inner.cancel(ctx, operation_id, reason).await
    }
}

/// Registry of available tools.
///
/// Tools are stored with their names erased to allow different `Name` types
/// in the same registry. The registry uses string-based lookup for LLM
/// compatibility.
///
/// Supports both synchronous [`Tool`]s and asynchronous [`AsyncTool`]s.
pub struct ToolRegistry<Ctx> {
    tools: HashMap<String, Arc<dyn ErasedTool<Ctx>>>,
    async_tools: HashMap<String, Arc<dyn ErasedAsyncTool<Ctx>>>,
    listen_tools: HashMap<String, Arc<dyn ErasedListenTool<Ctx>>>,
}

impl<Ctx> Clone for ToolRegistry<Ctx> {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
            async_tools: self.async_tools.clone(),
            listen_tools: self.listen_tools.clone(),
        }
    }
}

impl<Ctx: Send + Sync + 'static> Default for ToolRegistry<Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Ctx: Send + Sync + 'static> ToolRegistry<Ctx> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            async_tools: HashMap::new(),
            listen_tools: HashMap::new(),
        }
    }

    /// Evict any existing registration for `name` across **all three** maps so
    /// a name lives in exactly one map, then warn about the replacement.
    ///
    /// Without this, re-registering a name silently replaced the tool, and the
    /// same name registered as both (say) a sync and a listen tool coexisted —
    /// [`len`](ToolRegistry::len) double-counted it and
    /// [`to_llm_tools`](ToolRegistry::to_llm_tools) emitted two definitions with
    /// identical names (which providers reject). A remote MCP server could also
    /// silently shadow a vetted built-in (`read`, `bash`). We keep the
    /// non-breaking last-registration-wins behavior but make it loud; callers
    /// that need fail-closed semantics should use the `try_register*` variants.
    fn evict_existing(&mut self, name: &str, new_kind: &str) {
        // Evict from all three maps (each is side-effecting and must run); a
        // name lives in at most one map, so the listen > async > sync ordering
        // only disambiguates the pathological double-registration case.
        let previous_kind = [
            (self.listen_tools.remove(name).is_some(), "listen"),
            (self.async_tools.remove(name).is_some(), "async"),
            (self.tools.remove(name).is_some(), "sync"),
        ]
        .into_iter()
        .find_map(|(removed, kind)| removed.then_some(kind));
        if let Some(previous_kind) = previous_kind {
            log::warn!(
                "tool registry: name {name:?} already registered as a {previous_kind} tool; \
                 replacing it with a {new_kind} tool (last registration wins)"
            );
        }
    }

    /// Error if `name` is already registered in any of the three maps.
    fn ensure_unique(&self, name: &str) -> Result<()> {
        anyhow::ensure!(
            !self.tools.contains_key(name)
                && !self.async_tools.contains_key(name)
                && !self.listen_tools.contains_key(name),
            "tool {name:?} is already registered",
        );
        Ok(())
    }

    /// Register a synchronous tool in the registry.
    ///
    /// The tool's name is converted to a string via serde serialization
    /// and used as the lookup key. If the name is already registered (in any
    /// map), the previous tool is evicted and a warning is logged; use
    /// [`try_register`](ToolRegistry::try_register) for fail-closed semantics.
    pub fn register<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool<Ctx> + 'static,
    {
        let wrapper = ToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.evict_existing(&name, "sync");
        self.tools.insert(name, Arc::new(wrapper));
        self
    }

    /// Register a synchronous tool, returning an error on name collision.
    ///
    /// Unlike [`register`](ToolRegistry::register), this never silently
    /// replaces an existing tool — it checks all three maps and fails if the
    /// name is taken. Useful for registering untrusted (e.g. MCP-supplied)
    /// tools without letting them squat over vetted built-ins.
    ///
    /// # Errors
    /// Returns an error if a tool with the same name is already registered.
    pub fn try_register<T>(&mut self, tool: T) -> Result<&mut Self>
    where
        T: Tool<Ctx> + 'static,
    {
        let wrapper = ToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.ensure_unique(&name)?;
        self.tools.insert(name, Arc::new(wrapper));
        Ok(self)
    }

    /// Register a [`SimpleTool`] — a tool whose name is a plain `&str` and
    /// which needs no [`ToolName`] type.
    ///
    /// The tool is wrapped in a [`SimpleToolAdapter`] (giving it
    /// `Name = DynamicToolName`) and registered like any other [`Tool`].
    /// This is the lowest-ceremony way to add a first custom tool.
    pub fn register_simple<T>(&mut self, tool: T) -> &mut Self
    where
        T: SimpleTool<Ctx> + 'static,
    {
        self.register(SimpleToolAdapter::new(tool))
    }

    /// Register a [`TypedTool`] — a tool whose model-emitted arguments are
    /// deserialized into a typed [`TypedTool::Input`] and validated **before**
    /// `execute` runs.
    ///
    /// The tool is wrapped in a [`TypedToolAdapter`] (giving it
    /// `Name = DynamicToolName`) and registered like any other [`Tool`]. A
    /// malformed tool call is turned into a structured validation-error
    /// [`ToolResult`] at the dispatch boundary so the model can self-correct;
    /// `execute` is never reached with invalid arguments.
    pub fn register_typed<T>(&mut self, tool: T) -> &mut Self
    where
        T: TypedTool<Ctx> + 'static,
    {
        self.register(TypedToolAdapter::new(tool))
    }

    /// Register an async tool in the registry.
    ///
    /// Async tools have two phases: execute (lightweight, starts operation)
    /// and `check_status` (streams progress until completion).
    pub fn register_async<T>(&mut self, tool: T) -> &mut Self
    where
        T: AsyncTool<Ctx> + 'static,
    {
        let wrapper = AsyncToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.evict_existing(&name, "async");
        self.async_tools.insert(name, Arc::new(wrapper));
        self
    }

    /// Register an async tool, returning an error on name collision.
    ///
    /// The fail-closed counterpart to [`register_async`](ToolRegistry::register_async).
    ///
    /// # Errors
    /// Returns an error if a tool with the same name is already registered.
    pub fn try_register_async<T>(&mut self, tool: T) -> Result<&mut Self>
    where
        T: AsyncTool<Ctx> + 'static,
    {
        let wrapper = AsyncToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.ensure_unique(&name)?;
        self.async_tools.insert(name, Arc::new(wrapper));
        Ok(self)
    }

    /// Register a listen/execute tool in the registry.
    ///
    /// Listen/execute tools start by streaming updates via `listen()`, then run
    /// final execution with `execute()` once confirmed.
    pub fn register_listen<T>(&mut self, tool: T) -> &mut Self
    where
        T: ListenExecuteTool<Ctx> + 'static,
    {
        let wrapper = ListenToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.evict_existing(&name, "listen");
        self.listen_tools.insert(name, Arc::new(wrapper));
        self
    }

    /// Register a listen/execute tool, returning an error on name collision.
    ///
    /// The fail-closed counterpart to [`register_listen`](ToolRegistry::register_listen).
    ///
    /// # Errors
    /// Returns an error if a tool with the same name is already registered.
    pub fn try_register_listen<T>(&mut self, tool: T) -> Result<&mut Self>
    where
        T: ListenExecuteTool<Ctx> + 'static,
    {
        let wrapper = ListenToolWrapper::new(tool);
        let name = wrapper.name_str().to_string();
        self.ensure_unique(&name)?;
        self.listen_tools.insert(name, Arc::new(wrapper));
        Ok(self)
    }

    /// Get a synchronous tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ErasedTool<Ctx>>> {
        self.tools.get(name)
    }

    /// Get an async tool by name.
    #[must_use]
    pub fn get_async(&self, name: &str) -> Option<&Arc<dyn ErasedAsyncTool<Ctx>>> {
        self.async_tools.get(name)
    }

    /// Get a listen/execute tool by name.
    #[must_use]
    pub fn get_listen(&self, name: &str) -> Option<&Arc<dyn ErasedListenTool<Ctx>>> {
        self.listen_tools.get(name)
    }

    /// Check if a tool name refers to an async tool.
    #[must_use]
    pub fn is_async(&self, name: &str) -> bool {
        self.async_tools.contains_key(name)
    }

    /// Check if a tool name refers to a listen/execute tool.
    #[must_use]
    pub fn is_listen(&self, name: &str) -> bool {
        self.listen_tools.contains_key(name)
    }

    /// Get all registered synchronous tools.
    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn ErasedTool<Ctx>>> {
        self.tools.values()
    }

    /// Get all registered async tools.
    pub fn all_async(&self) -> impl Iterator<Item = &Arc<dyn ErasedAsyncTool<Ctx>>> {
        self.async_tools.values()
    }

    /// Get all registered listen/execute tools.
    pub fn all_listen(&self) -> impl Iterator<Item = &Arc<dyn ErasedListenTool<Ctx>>> {
        self.listen_tools.values()
    }

    /// Get the number of registered tools (sync + async).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len() + self.async_tools.len() + self.listen_tools.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.async_tools.is_empty() && self.listen_tools.is_empty()
    }

    /// Filter tools by a predicate.
    ///
    /// Removes tools for which the predicate returns false.
    /// The predicate receives the tool name.
    /// Applies to both sync and async tools.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.filter(|name| name != "bash");
    /// ```
    pub fn filter<F>(&mut self, predicate: F)
    where
        F: Fn(&str) -> bool,
    {
        self.tools.retain(|name, _| predicate(name));
        self.async_tools.retain(|name, _| predicate(name));
        self.listen_tools.retain(|name, _| predicate(name));
    }

    /// Convert all tools (sync + async + listen) to LLM tool
    /// definitions. The output is sorted by tool name so the order
    /// is deterministic across builds and across calls.
    ///
    /// Determinism matters for **prompt caching**. Anthropic's
    /// `cache_control: ephemeral` keys on the byte content of the
    /// system + tool list. Anything that perturbs the order of the
    /// tool list invalidates the cache. The three backing maps are
    /// `HashMap`s, whose `values()` order is randomized (DoS-safe
    /// `RandomState` by default), so two consecutive turns with the
    /// same registered tool set were producing different orderings
    /// and silently zeroing the cache hit rate.
    ///
    /// Sorting by name is the cheapest fix that holds across
    /// insertion order, internal map type changes, and concurrent
    /// builds. The tool count is small (tens, not thousands) so the
    /// sort cost is negligible compared to a single LLM call.
    #[must_use]
    pub fn to_llm_tools(&self) -> Vec<llm::Tool> {
        /// Build the LLM tool descriptor from the accessors every erased tool
        /// trait shares. Extracted so the five-field `llm::Tool` literal —
        /// whose byte content is prompt-cache load-bearing — exists in exactly
        /// one place across the sync / async / listen iterators.
        fn descriptor(
            name: &str,
            display_name: &str,
            description: &str,
            input_schema: Value,
            tier: ToolTier,
        ) -> llm::Tool {
            llm::Tool {
                name: name.to_string(),
                description: description.to_string(),
                input_schema,
                display_name: display_name.to_string(),
                tier,
            }
        }

        let mut tools: Vec<_> = self
            .tools
            .values()
            .map(|tool| {
                descriptor(
                    tool.name_str(),
                    tool.display_name(),
                    tool.description(),
                    tool.input_schema(),
                    tool.tier(),
                )
            })
            .collect();

        tools.extend(self.async_tools.values().map(|tool| {
            descriptor(
                tool.name_str(),
                tool.display_name(),
                tool.description(),
                tool.input_schema(),
                tool.tier(),
            )
        }));

        tools.extend(self.listen_tools.values().map(|tool| {
            descriptor(
                tool.name_str(),
                tool.display_name(),
                tool.description(),
                tool.input_schema(),
                tool.tier(),
            )
        }));

        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    // Test tool name enum for tests
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum TestToolName {
        MockTool,
        AnotherTool,
    }

    impl ToolName for TestToolName {}

    struct MockTool;

    impl Tool<()> for MockTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::MockTool
        }

        fn display_name(&self) -> &'static str {
            "Mock Tool"
        }

        fn description(&self) -> &'static str {
            "A mock tool for testing"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                }
            })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
            let message = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("no message");
            Ok(ToolResult::success(format!("Received: {message}")))
        }
    }

    #[test]
    fn test_tool_name_serialization() {
        let name = TestToolName::MockTool;
        assert_eq!(tool_name_to_string(&name), "mock_tool");

        let parsed: TestToolName = tool_name_from_str("mock_tool").unwrap();
        assert_eq!(parsed, TestToolName::MockTool);
    }

    #[test]
    fn test_dynamic_tool_name() {
        let name = DynamicToolName::new("my_mcp_tool");
        assert_eq!(tool_name_to_string(&name), "my_mcp_tool");
        assert_eq!(name.as_str(), "my_mcp_tool");
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        assert_eq!(registry.len(), 1);
        assert!(registry.get("mock_tool").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_to_llm_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        let llm_tools = registry.to_llm_tools();
        assert_eq!(llm_tools.len(), 1);
        assert_eq!(llm_tools[0].name, "mock_tool");
    }

    #[test]
    fn to_llm_tools_returns_alphabetical_order() {
        let mut registry = ToolRegistry::new();
        // Register in non-alphabetical order so the assertion would
        // fail if we ever returned insertion order again.
        registry.register(MockTool); // "mock_tool"
        registry.register(AnotherTool); // "another_tool"

        let names: Vec<String> = registry
            .to_llm_tools()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["another_tool", "mock_tool"]);
    }

    #[test]
    fn to_llm_tools_is_deterministic_across_calls() {
        // Regression: prompt caching depends on byte-stable tool list
        // ordering. The `HashMap` behind the registry randomizes its
        // `values()` order, so without an explicit sort two consecutive
        // builds with the same registered set could ship different
        // tool orderings to the LLM and silently invalidate the cache.
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        let first: Vec<String> = registry
            .to_llm_tools()
            .into_iter()
            .map(|t| t.name)
            .collect();

        for _ in 0..32 {
            let next: Vec<String> = registry
                .to_llm_tools()
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert_eq!(next, first, "tool ordering must be stable across calls");
        }
    }

    struct AnotherTool;

    impl Tool<()> for AnotherTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::AnotherTool
        }

        fn display_name(&self) -> &'static str {
            "Another Tool"
        }

        fn description(&self) -> &'static str {
            "Another tool for testing"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
            Ok(ToolResult::success("Done"))
        }
    }

    #[test]
    fn test_filter_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        assert_eq!(registry.len(), 2);

        // Filter out mock_tool
        registry.filter(|name| name != "mock_tool");

        assert_eq!(registry.len(), 1);
        assert!(registry.get("mock_tool").is_none());
        assert!(registry.get("another_tool").is_some());
    }

    #[test]
    fn test_filter_tools_keep_all() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        registry.filter(|_| true);

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_filter_tools_remove_all() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(AnotherTool);

        registry.filter(|_| false);

        assert!(registry.is_empty());
    }

    #[test]
    fn test_display_name() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);

        let tool = registry.get("mock_tool").unwrap();
        assert_eq!(tool.display_name(), "Mock Tool");
    }

    struct ListenMockTool;

    impl ListenExecuteTool<()> for ListenMockTool {
        type Name = TestToolName;

        fn name(&self) -> TestToolName {
            TestToolName::MockTool
        }

        fn display_name(&self) -> &'static str {
            "Listen Mock Tool"
        }

        fn description(&self) -> &'static str {
            "A listen/execute mock tool for testing"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        fn listen(
            &self,
            _ctx: &ToolContext<()>,
            _input: Value,
        ) -> impl futures::Stream<Item = ListenToolUpdate> + Send {
            futures::stream::iter(vec![ListenToolUpdate::Ready {
                operation_id: "op_1".to_string(),
                revision: 1,
                message: "ready".to_string(),
                snapshot: serde_json::json!({"ok": true}),
                expires_at: None,
            }])
        }

        async fn execute(
            &self,
            _ctx: &ToolContext<()>,
            _operation_id: &str,
            _expected_revision: u64,
        ) -> Result<ToolResult> {
            Ok(ToolResult::success("Executed"))
        }
    }

    #[test]
    fn test_listen_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register_listen(ListenMockTool);

        assert_eq!(registry.len(), 1);
        assert!(registry.get_listen("mock_tool").is_some());
        assert!(registry.is_listen("mock_tool"));
    }

    // ── TypedTool: typed input + validation / self-correction ───────────

    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Serialize, Deserialize)]
    struct GreetArgs {
        name: String,
        // Required so a missing/typo'd field is a hard validation error.
        greeting: String,
    }

    /// A typed tool that records whether `execute` was reached, so tests can
    /// assert the validation boundary never calls `execute` with bad args.
    struct GreetTool {
        executed: Arc<AtomicBool>,
    }

    impl TypedTool<()> for GreetTool {
        type Input = GreetArgs;

        fn name(&self) -> &'static str {
            "greet"
        }

        fn description(&self) -> &'static str {
            "Greet someone by name"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "greeting": { "type": "string" }
                },
                "required": ["name", "greeting"]
            })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: GreetArgs) -> Result<ToolResult> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(ToolResult::success(format!(
                "{}, {}!",
                input.greeting, input.name
            )))
        }
    }

    #[tokio::test]
    async fn typed_tool_happy_path_receives_typed_input() -> Result<()> {
        let executed = Arc::new(AtomicBool::new(false));
        let adapter = TypedToolAdapter::new(GreetTool {
            executed: executed.clone(),
        });
        let ctx = ToolContext::new(());

        let result = Tool::execute(
            &adapter,
            &ctx,
            serde_json::json!({ "name": "Ada", "greeting": "Hello" }),
        )
        .await?;

        assert!(executed.load(Ordering::SeqCst), "execute must be called");
        assert!(result.success);
        assert_eq!(result.output, "Hello, Ada!");
        Ok(())
    }

    #[tokio::test]
    async fn typed_tool_invalid_args_self_correct_without_executing() -> Result<()> {
        let executed = Arc::new(AtomicBool::new(false));
        let adapter = TypedToolAdapter::new(GreetTool {
            executed: executed.clone(),
        });
        let ctx = ToolContext::new(());

        // `greeting` is missing — must not deserialize into `GreetArgs`.
        let result = Tool::execute(&adapter, &ctx, serde_json::json!({ "name": "Ada" })).await?;

        assert!(
            !executed.load(Ordering::SeqCst),
            "execute must NOT be called with invalid arguments"
        );
        assert!(!result.success, "validation failure is an error result");
        assert!(
            result.output.contains("Invalid arguments for tool `greet`"),
            "error must identify the tool: {}",
            result.output
        );
        assert!(
            result.output.contains("greeting"),
            "error must surface the serde message naming the bad field: {}",
            result.output
        );
        Ok(())
    }

    #[tokio::test]
    async fn typed_tool_wrong_type_self_corrects() -> Result<()> {
        let executed = Arc::new(AtomicBool::new(false));
        let adapter = TypedToolAdapter::new(GreetTool {
            executed: executed.clone(),
        });
        let ctx = ToolContext::new(());

        // `name` is a number, not a string.
        let result = Tool::execute(
            &adapter,
            &ctx,
            serde_json::json!({ "name": 42, "greeting": "Hi" }),
        )
        .await?;

        assert!(!executed.load(Ordering::SeqCst));
        assert!(!result.success);
        Ok(())
    }

    /// Back-compat: a `TypedTool` whose `Input = Value` is the identity
    /// passthrough — any JSON deserializes, mirroring today's untyped tools.
    struct ValueTypedTool;

    impl TypedTool<()> for ValueTypedTool {
        type Input = Value;

        fn name(&self) -> &'static str {
            "value_typed"
        }

        fn description(&self) -> &'static str {
            "Accepts any JSON, like an untyped tool"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
            Ok(ToolResult::success(input.to_string()))
        }
    }

    #[tokio::test]
    async fn typed_tool_value_input_is_identity_passthrough() -> Result<()> {
        let adapter = TypedToolAdapter::new(ValueTypedTool);
        let ctx = ToolContext::new(());

        // Arbitrary shape — Value always "deserializes".
        let result = Tool::execute(
            &adapter,
            &ctx,
            serde_json::json!({ "anything": [1, 2, 3], "nested": { "ok": true } }),
        )
        .await?;

        assert!(result.success);
        Ok(())
    }

    #[test]
    fn register_typed_exposes_tool_via_registry() -> Result<()> {
        let mut registry = ToolRegistry::new();
        registry.register_typed(GreetTool {
            executed: Arc::new(AtomicBool::new(false)),
        });

        assert_eq!(registry.len(), 1);
        let tool = registry.get("greet").context("typed tool registered")?;
        // The user-declared schema flows through unchanged.
        assert_eq!(tool.input_schema()["required"][0], "name");
        Ok(())
    }

    #[test]
    fn invalid_tool_input_result_is_balanced_error() -> Result<()> {
        let Err(err) = serde_json::from_str::<GreetArgs>("{}") else {
            anyhow::bail!("empty object must fail to deserialize GreetArgs");
        };
        let result = invalid_tool_input_result("greet", &err);

        assert!(!result.success);
        assert!(result.output.contains("greet"));
        assert!(result.output.contains("call the tool again"));
        Ok(())
    }

    // ── Fail-closed tier + display_name defaults (findings 8 & 19) ───────

    /// A tool that overrides neither `tier()` nor `display_name()`, exercising
    /// the trait defaults.
    struct DefaultsTool;

    impl Tool<()> for DefaultsTool {
        type Name = DynamicToolName;

        fn name(&self) -> DynamicToolName {
            DynamicToolName::new("defaults")
        }

        fn description(&self) -> &'static str {
            "uses trait defaults"
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        async fn execute(&self, _ctx: &ToolContext<()>, _input: Value) -> Result<ToolResult> {
            Ok(ToolResult::success("ok"))
        }
    }

    #[test]
    fn tool_trait_defaults_are_fail_closed() {
        let tool = DefaultsTool;
        // display_name defaults to "" (finding 19: the doc-claimed default now
        // actually exists).
        assert_eq!(Tool::display_name(&tool), "");
        // tier defaults to Confirm so a side-effecting tool whose author forgot
        // to declare a tier is gated, not auto-run (finding 8).
        assert_eq!(Tool::tier(&tool), ToolTier::Confirm);
    }

    // ── Registry name-collision handling (findings 9 & 10) ───────────────

    #[test]
    fn re_registering_same_name_replaces_without_duplicates() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool);
        registry.register(MockTool); // same name "mock_tool"

        assert_eq!(registry.len(), 1, "re-register must replace, not add");
        let names: Vec<String> = registry
            .to_llm_tools()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["mock_tool"]);
    }

    #[test]
    fn cross_kind_name_collision_keeps_single_entry() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool); // sync "mock_tool"
        registry.register_listen(ListenMockTool); // listen "mock_tool"

        // The listen registration evicts the sync one — a name lives in exactly
        // one map, so `len()` and `to_llm_tools()` never double-count it.
        assert_eq!(registry.len(), 1);
        assert!(registry.is_listen("mock_tool"));
        assert!(
            registry.get("mock_tool").is_none(),
            "the shadowed sync tool must be evicted"
        );
        let names: Vec<String> = registry
            .to_llm_tools()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["mock_tool"], "no duplicate LLM definitions");
    }

    #[test]
    fn try_register_rejects_name_collision() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool); // "mock_tool"

        assert!(
            registry.try_register(MockTool).is_err(),
            "duplicate sync name must be rejected"
        );
        assert!(
            registry.try_register_listen(ListenMockTool).is_err(),
            "cross-map duplicate (squatting) must be rejected"
        );
        assert_eq!(
            registry.len(),
            1,
            "rejected registrations must not be stored"
        );
    }

    // ── Non-panicking serde helpers (findings 16 & 17) ───────────────────

    #[derive(Clone)]
    struct FailingStage;

    impl Serialize for FailingStage {
        fn serialize<S>(&self, _serializer: S) -> core::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom("intentionally unserializable"))
        }
    }

    impl<'de> Deserialize<'de> for FailingStage {
        fn deserialize<D>(_deserializer: D) -> core::result::Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            Ok(Self)
        }
    }

    impl ProgressStage for FailingStage {}

    #[test]
    fn stage_to_string_falls_back_instead_of_panicking() {
        // A ProgressStage whose Serialize impl fails must not panic the turn
        // loop on the async-tool progress hot path.
        assert_eq!(stage_to_string(&FailingStage), "<unknown_stage>");
    }

    #[test]
    fn tool_name_from_str_round_trips_special_characters() -> Result<()> {
        // Names with quotes/backslashes (possible from remote MCP servers) must
        // be JSON-escaped, not interpolated raw, so parsing succeeds.
        let name: DynamicToolName =
            tool_name_from_str("weird\"name\\with-escapes").context("must parse escaped name")?;
        assert_eq!(name.as_str(), "weird\"name\\with-escapes");
        Ok(())
    }

    // ── emit_event surfaces unbound misuse instead of silently dropping ──

    #[tokio::test]
    async fn emit_event_persists_when_bound_and_is_noop_when_unbound() -> Result<()> {
        use crate::stores::InMemoryEventStore;
        use agent_sdk_foundation::types::ThreadId;

        let store = Arc::new(InMemoryEventStore::new());
        let thread_id = ThreadId::new();
        let authority: Arc<dyn EventAuthority> = Arc::new(LocalEventAuthority::new());

        let bound =
            ToolContext::new(()).with_event_store(store.clone(), thread_id.clone(), 1, authority);
        bound.emit_event(AgentEvent::text("m1", "hi")).await?;
        assert_eq!(
            store.event_count(&thread_id).await?,
            1,
            "a bound context persists the event"
        );

        // An unbound context is a no-op (the fix also logs a warning) — it must
        // not silently append elsewhere or error.
        let unbound = ToolContext::new(());
        unbound.emit_event(AgentEvent::text("m2", "lost")).await?;
        assert_eq!(
            store.event_count(&thread_id).await?,
            1,
            "an unbound context changes nothing"
        );
        Ok(())
    }
}
