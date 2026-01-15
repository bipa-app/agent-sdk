# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Agent SDK, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

### What to Expect

- We will acknowledge receipt within 48 hours
- We will provide an initial assessment within 7 days
- We will work with you to understand and resolve the issue
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Considerations

### Agent Capabilities

The SDK includes a capability-based security model (`AgentCapabilities`) to control what agents can do:

- `read_only()` - Only allows file reading
- `with_write(bool)` - Controls file write access
- `with_exec(bool)` - Controls command execution

Always use the minimum required capabilities for your use case.

### Tool Execution

- Tools are categorized by tier (`ToolTier::Observe`, `ToolTier::Confirm`, `ToolTier::RequiresPin`)
- Implement `AgentHooks` to add confirmation flows for sensitive operations
- Review tool inputs before execution in production environments

### API Keys

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Consider using a secrets manager in production

### File System Access

- The `InMemoryFileSystem` is useful for sandboxed testing
- When using `LocalFileSystem`, be mindful of the paths agents can access
- Consider chrooting or containerizing agents that need file system access

## Best Practices

1. **Validate inputs** - Always validate tool inputs before processing
2. **Limit scope** - Use read-only capabilities when write access isn't needed
3. **Monitor usage** - Implement logging hooks to track agent actions
4. **Rate limit** - Implement rate limiting for tool calls in production
5. **Review outputs** - Be cautious about exposing raw LLM outputs to end users
