# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
