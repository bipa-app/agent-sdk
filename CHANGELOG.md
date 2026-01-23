# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-01-23

### Breaking Changes

- **Removed `ToolTier::RequiresPin`** - The SDK now only has `Observe` and `Confirm` tiers. Applications that need PIN verification should implement this at the application layer using hooks.

- **Removed `ToolDecision::RequiresPin`** - Hooks no longer return this variant. Use `RequiresConfirmation` or `Block` instead.

- **Removed `PendingAction`** - Applications now manage the pending action lifecycle externally.

- **Removed `ToolRequiresPin` event** - This event is no longer emitted by the agent loop.

- **Tool trait now requires `type Name` associated type** - All `Tool` implementations must specify a tool name type:
  ```rust
  impl Tool<()> for MyTool {
      type Name = DynamicToolName;  // or your custom ToolName type
      fn name(&self) -> DynamicToolName { DynamicToolName::new("my_tool") }
      // ...
  }
  ```

- **`agent.run()` now returns a tuple** - The return type is `(mpsc::Receiver<AgentEvent>, impl Future<Output = AgentRunState>)` instead of just a receiver.

- **Input is now wrapped in `AgentInput`** - Use `AgentInput::Text(...)` instead of passing a plain string.

### Added

- **Typed tool names** - Tool names are now strongly typed via the `ToolName` trait and associated type:
  - `PrimitiveToolName` enum for SDK's built-in tools
  - `DynamicToolName` for runtime/MCP tools
  - `tool_name_to_string()` and `tool_name_from_str()` helpers

- **`display_name()` method on Tool trait** - Tools can now provide a human-readable display name for UIs.

- **Yield/Resume pattern** - Agent can yield execution when a tool requires confirmation via `AgentRunState::AwaitingConfirmation`, then resume with `AgentInput::Resume`.

- **Single-turn execution** - New `run_turn()` method for external orchestration (e.g., message queues). Returns `TurnOutcome` indicating whether more turns are needed.

- **`AgentRunState` enum** - New return type indicating agent completion status: `Done`, `Error`, or `AwaitingConfirmation`.

- **`AgentContinuation`** - Opaque state for resuming agent execution after yielding.

- **Streaming support** - LLM providers now support streaming responses via `stream_chat_completion`.

### Fixed

- **Token usage tracking in resume cases** - Previously, token usage was zeroed when resuming after tool confirmation. Now properly tracks usage from the LLM call that generated the tool calls.

### Changed

- **Tool trait no longer requires `#[async_trait]`** - Rust 1.75+ native async traits are used instead.

- **Refactored agent loop** - Internal refactoring into smaller helper functions for better maintainability.

## [0.1.0] - 2025-01-15

### Added

- **Agent Loop**: Core orchestration for LLM conversations with tool calling
  - Event-driven streaming architecture
  - Configurable turn limits and retry behavior
  - Thread-based conversation management

- **LLM Abstraction**: Provider-agnostic interface for chat completions
  - `LlmProvider` trait for implementing custom providers
  - Built-in Anthropic provider (Claude models)
  - OpenAI provider implementation
  - Google Gemini provider implementation

- **Tool System**: Define and register custom tools
  - `Tool` trait for implementing tools
  - `ToolRegistry` for managing available tools
  - JSON schema validation for tool inputs
  - Tool tiers for security classification (Observe, Confirm, RequiresPin)

- **Lifecycle Hooks**: Pre/post tool execution hooks
  - `AgentHooks` trait for custom hook implementations
  - `AllowAllHooks` for development/testing
  - `LoggingHooks` for observability
  - `ToolDecision` for controlling tool execution

- **Environment Abstraction**: File and command execution
  - `Environment` trait for file system operations
  - `InMemoryFileSystem` for testing and sandboxing
  - `LocalFileSystem` for production use
  - Command execution with output capture

- **Primitive Tools**: Built-in tools for file operations
  - `ReadTool` - Read file contents
  - `WriteTool` - Create/overwrite files
  - `EditTool` - Make targeted edits to files
  - `GlobTool` - Find files by pattern
  - `GrepTool` - Search file contents
  - `BashTool` - Execute shell commands

- **Persistence**: Trait-based storage
  - `MessageStore` for conversation history
  - `StateStore` for agent state
  - `InMemoryStore` default implementation

- **Capabilities**: Security model for agent operations
  - `AgentCapabilities` for fine-grained permission control
  - Read-only, write, and exec toggles

- **Subagent System**: Nested agent execution
  - Real-time progress events during execution

- **MCP Support**: Model Context Protocol integration

- **Web Tools**: Fetch and search capabilities

### Security

- `#[forbid(unsafe_code)]` enforced across the codebase
- Capability-based security model
- Tool tier system for operation classification
