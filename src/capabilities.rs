use serde::{Deserialize, Serialize};

/// Capabilities that control what the agent can do.
///
/// This provides a security model for primitive tools (Read, Write, Grep, Glob, Bash).
/// Paths are matched using glob patterns, commands using regex patterns.
///
/// By default, everything is allowed — the SDK is unopinionated and leaves
/// security policy to the client. Use the builder methods to configure restrictions.
///
/// # Example
///
/// ```rust
/// use agent_sdk::AgentCapabilities;
///
/// // Read-only agent that can only access src/ directory
/// let caps = AgentCapabilities::read_only()
///     .with_allowed_paths(vec!["src/**/*".into()]);
///
/// // Full access agent with some restrictions
/// let caps = AgentCapabilities::full_access()
///     .with_denied_paths(vec!["**/.env*".into(), "**/secrets/**".into()]);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Can read files
    pub read: bool,
    /// Can write/edit files
    pub write: bool,
    /// Can execute shell commands
    pub exec: bool,
    /// Allowed path patterns (glob). Empty means all paths allowed.
    pub allowed_paths: Vec<String>,
    /// Denied path patterns (glob). Takes precedence over `allowed_paths`.
    pub denied_paths: Vec<String>,
    /// Allowed commands (regex patterns). Empty means all commands allowed when `exec=true`.
    pub allowed_commands: Vec<String>,
    /// Denied commands (regex patterns). Takes precedence over `allowed_commands`.
    pub denied_commands: Vec<String>,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self::full_access()
    }
}

impl AgentCapabilities {
    /// Create capabilities with no access (must explicitly enable)
    #[must_use]
    pub const fn none() -> Self {
        Self {
            read: false,
            write: false,
            exec: false,
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_commands: vec![],
            denied_commands: vec![],
        }
    }

    /// Create read-only capabilities
    #[must_use]
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            exec: false,
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_commands: vec![],
            denied_commands: vec![],
        }
    }

    /// Create full access capabilities
    #[must_use]
    pub const fn full_access() -> Self {
        Self {
            read: true,
            write: true,
            exec: true,
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_commands: vec![],
            denied_commands: vec![],
        }
    }

    /// Builder: enable read access
    #[must_use]
    pub const fn with_read(mut self, enabled: bool) -> Self {
        self.read = enabled;
        self
    }

    /// Builder: enable write access
    #[must_use]
    pub const fn with_write(mut self, enabled: bool) -> Self {
        self.write = enabled;
        self
    }

    /// Builder: enable exec access
    #[must_use]
    pub const fn with_exec(mut self, enabled: bool) -> Self {
        self.exec = enabled;
        self
    }

    /// Builder: set allowed paths
    #[must_use]
    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths;
        self
    }

    /// Builder: set denied paths
    #[must_use]
    pub fn with_denied_paths(mut self, paths: Vec<String>) -> Self {
        self.denied_paths = paths;
        self
    }

    /// Builder: set allowed commands
    #[must_use]
    pub fn with_allowed_commands(mut self, commands: Vec<String>) -> Self {
        self.allowed_commands = commands;
        self
    }

    /// Builder: set denied commands
    #[must_use]
    pub fn with_denied_commands(mut self, commands: Vec<String>) -> Self {
        self.denied_commands = commands;
        self
    }

    /// Check read permission, returning the denial reason on failure.
    ///
    /// # Errors
    ///
    /// Returns the denial reason when read is disabled, the path matches a
    /// denied pattern, or the path is not in the allowed list.
    pub fn check_read(&self, path: &str) -> Result<(), String> {
        if !self.read {
            return Err("read access is disabled".into());
        }
        self.check_path(path)
    }

    /// Check write permission, returning the denial reason on failure.
    ///
    /// # Errors
    ///
    /// Returns the denial reason when write is disabled, the path matches a
    /// denied pattern, or the path is not in the allowed list.
    pub fn check_write(&self, path: &str) -> Result<(), String> {
        if !self.write {
            return Err("write access is disabled".into());
        }
        self.check_path(path)
    }

    /// Check exec permission, returning the denial reason on failure.
    ///
    /// # Errors
    ///
    /// Returns the denial reason when exec is disabled, the command matches a
    /// denied pattern, or the command is not in the allowed list.
    pub fn check_exec(&self, command: &str) -> Result<(), String> {
        if !self.exec {
            return Err("command execution is disabled".into());
        }
        self.check_command(command)
    }

    /// Returns `true` if reading `path` is allowed.
    #[must_use]
    pub fn can_read(&self, path: &str) -> bool {
        self.check_read(path).is_ok()
    }

    /// Returns `true` if writing `path` is allowed.
    #[must_use]
    pub fn can_write(&self, path: &str) -> bool {
        self.check_write(path).is_ok()
    }

    /// Returns `true` if executing `command` is allowed.
    #[must_use]
    pub fn can_exec(&self, command: &str) -> bool {
        self.check_exec(command).is_ok()
    }

    /// Check whether a path passes the allow/deny rules, returning
    /// the specific denial reason on failure.
    ///
    /// # Errors
    ///
    /// Returns the denial reason when the path matches a denied pattern
    /// or is not in the allowed list.
    pub fn check_path(&self, path: &str) -> Result<(), String> {
        // Denied patterns take precedence
        for pattern in &self.denied_paths {
            if glob_match(pattern, path) {
                return Err(format!("path matches denied pattern '{pattern}'"));
            }
        }

        // If allowed_paths is empty, all non-denied paths are allowed
        if self.allowed_paths.is_empty() {
            return Ok(());
        }

        // Check if path matches any allowed pattern
        for pattern in &self.allowed_paths {
            if glob_match(pattern, path) {
                return Ok(());
            }
        }

        Err(format!(
            "path not in allowed list (allowed: [{}])",
            self.allowed_paths.join(", ")
        ))
    }

    /// Check whether a command passes the allow/deny rules, returning
    /// the specific denial reason on failure.
    ///
    /// # Errors
    ///
    /// Returns the denial reason when the command matches a denied pattern
    /// or is not in the allowed list.
    pub fn check_command(&self, command: &str) -> Result<(), String> {
        // Denied patterns take precedence
        for pattern in &self.denied_commands {
            if regex_match(pattern, command) {
                return Err(format!("command matches denied pattern '{pattern}'"));
            }
        }

        // If allowed_commands is empty, all non-denied commands are allowed
        if self.allowed_commands.is_empty() {
            return Ok(());
        }

        // Check if command matches any allowed pattern
        for pattern in &self.allowed_commands {
            if regex_match(pattern, command) {
                return Ok(());
            }
        }

        Err(format!(
            "command not in allowed list (allowed: [{}])",
            self.allowed_commands.join(", ")
        ))
    }
}

/// Simple glob matching (supports * and ** wildcards)
fn glob_match(pattern: &str, path: &str) -> bool {
    // Handle special case: pattern is just **
    if pattern == "**" {
        return true; // Matches everything
    }

    // Escape regex special characters except * and ?
    let mut escaped = String::new();
    for c in pattern.chars() {
        match c {
            '.' | '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }

    // Handle glob patterns:
    // - **/ at start or middle: zero or more path components (including leading /)
    // - /** at end: matches everything after
    // - * : matches any characters except /
    let pattern = escaped
        .replace("**/", "\x00") // **/ -> placeholder
        .replace("/**", "\x01") // /** -> placeholder
        .replace('*', "[^/]*") // * -> match non-slash characters
        .replace('\x00', "(.*/)?") // **/ as optional prefix (handles absolute paths)
        .replace('\x01', "(/.*)?"); // /** as optional suffix

    let regex = format!("^{pattern}$");
    regex_match(&regex, path)
}

/// Simple regex matching (returns false on invalid patterns)
fn regex_match(pattern: &str, text: &str) -> bool {
    regex::Regex::new(pattern)
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_has_no_deny_lists() {
        let caps = AgentCapabilities::default();

        // Default is permissive — no paths or commands are denied
        assert!(caps.check_path("src/main.rs").is_ok());
        assert!(caps.check_path(".env").is_ok());
        assert!(caps.check_path("/workspace/secrets/key.txt").is_ok());
        assert!(caps.check_command("any command").is_ok());
    }

    #[test]
    fn test_full_access_allows_everything() {
        let caps = AgentCapabilities::full_access();

        assert!(caps.check_read("/any/path").is_ok());
        assert!(caps.check_write("/any/path").is_ok());
        assert!(caps.check_exec("any command").is_ok());
    }

    #[test]
    fn test_read_only_cannot_write() {
        let caps = AgentCapabilities::read_only();

        assert!(caps.check_read("src/main.rs").is_ok());
        assert!(caps.check_write("src/main.rs").is_err());
        assert!(caps.check_exec("ls").is_err());
    }

    #[test]
    fn test_client_configured_denied_paths() {
        let caps = AgentCapabilities::full_access().with_denied_paths(vec![
            "**/.env".into(),
            "**/.env.*".into(),
            "**/secrets/**".into(),
            "**/*.pem".into(),
        ]);

        // Denied paths (relative)
        assert!(caps.check_path(".env").is_err());
        assert!(caps.check_path("config/.env.local").is_err());
        assert!(caps.check_path("app/secrets/key.txt").is_err());
        assert!(caps.check_path("certs/server.pem").is_err());

        // Denied paths (absolute — after resolve_path)
        assert!(caps.check_path("/workspace/.env").is_err());
        assert!(caps.check_path("/workspace/.env.production").is_err());
        assert!(caps.check_path("/workspace/secrets/key.txt").is_err());
        assert!(caps.check_path("/workspace/certs/server.pem").is_err());

        // Normal files still allowed
        assert!(caps.check_path("src/main.rs").is_ok());
        assert!(caps.check_path("/workspace/src/main.rs").is_ok());
        assert!(caps.check_path("/workspace/README.md").is_ok());
    }

    #[test]
    fn test_allowed_paths_restriction() {
        let caps = AgentCapabilities::read_only()
            .with_allowed_paths(vec!["src/**".into(), "tests/**".into()]);

        assert!(caps.check_path("src/main.rs").is_ok());
        assert!(caps.check_path("src/lib/utils.rs").is_ok());
        assert!(caps.check_path("tests/integration.rs").is_ok());
        assert!(caps.check_path("config/settings.toml").is_err());
        assert!(caps.check_path("README.md").is_err());
    }

    #[test]
    fn test_denied_takes_precedence() {
        let caps = AgentCapabilities::read_only()
            .with_denied_paths(vec!["**/secret/**".into()])
            .with_allowed_paths(vec!["**".into()]);

        assert!(caps.check_path("src/main.rs").is_ok());
        assert!(caps.check_path("src/secret/key.txt").is_err());
    }

    #[test]
    fn test_client_configured_denied_commands() {
        let caps = AgentCapabilities::full_access()
            .with_denied_commands(vec![r"rm\s+-rf\s+/".into(), r"^sudo\s".into()]);

        assert!(caps.check_command("rm -rf /").is_err());
        assert!(caps.check_command("sudo rm file").is_err());

        // Common shell patterns are NOT blocked
        assert!(caps.check_command("ls -la").is_ok());
        assert!(caps.check_command("cargo build").is_ok());
        assert!(caps.check_command("unzip file.zip 2>/dev/null").is_ok());
        assert!(
            caps.check_command("python3 -m markitdown file.pptx")
                .is_ok()
        );
    }

    #[test]
    fn test_allowed_commands_restriction() {
        let caps = AgentCapabilities::full_access()
            .with_allowed_commands(vec![r"^cargo ".into(), r"^git ".into()]);

        assert!(caps.check_command("cargo build").is_ok());
        assert!(caps.check_command("git status").is_ok());
        assert!(caps.check_command("ls -la").is_err());
        assert!(caps.check_command("npm install").is_err());
    }

    #[test]
    fn test_glob_matching() {
        // Simple wildcards
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "src/main.rs"));

        // Double star for recursive matching in subdirectories
        assert!(glob_match("**/*.rs", "src/main.rs"));
        assert!(glob_match("**/*.rs", "deep/nested/file.rs"));

        // Directory patterns with /** suffix
        assert!(glob_match("src/**", "src/lib/utils.rs"));
        assert!(glob_match("src/**", "src/main.rs"));

        // Match files in any subdirectory
        assert!(glob_match("**/test*", "src/tests/test_utils.rs"));
        assert!(glob_match("**/test*.rs", "dir/test_main.rs"));

        // Root-level matches need direct pattern
        assert!(glob_match("test*", "test_main.rs"));
        assert!(glob_match("test*.rs", "test_main.rs"));

        // Absolute paths (tools resolve to absolute before checking capabilities)
        assert!(glob_match("**/.env", "/workspace/.env"));
        assert!(glob_match("**/.env.*", "/workspace/.env.local"));
        assert!(glob_match("**/secrets/**", "/workspace/secrets/key.txt"));
        assert!(glob_match("**/*.pem", "/workspace/certs/server.pem"));
        assert!(glob_match("**/*.key", "/workspace/server.key"));
        assert!(glob_match("**/id_rsa", "/home/user/.ssh/id_rsa"));
        assert!(glob_match("**/*.rs", "/Users/dev/project/src/main.rs"));

        // Absolute paths should NOT false-positive
        assert!(!glob_match("**/.env", "/workspace/src/main.rs"));
        assert!(!glob_match("**/*.pem", "/workspace/src/lib.rs"));
    }

    // =============================================
    // Diagnostic reason tests (check_* methods)
    // =============================================

    #[test]
    fn check_read_disabled_explains_reason() {
        let caps = AgentCapabilities::none();
        let err = caps.check_read("src/main.rs").unwrap_err();
        assert!(err.contains("read access is disabled"), "got: {err}");
    }

    #[test]
    fn check_write_disabled_explains_reason() {
        let caps = AgentCapabilities::read_only();
        let err = caps.check_write("src/main.rs").unwrap_err();
        assert!(err.contains("write access is disabled"), "got: {err}");
    }

    #[test]
    fn check_exec_disabled_explains_reason() {
        let caps = AgentCapabilities::read_only();
        let err = caps.check_exec("ls").unwrap_err();
        assert!(err.contains("command execution is disabled"), "got: {err}");
    }

    #[test]
    fn check_read_denied_path_explains_pattern() {
        let caps = AgentCapabilities::full_access().with_denied_paths(vec!["**/.env*".into()]);
        let err = caps.check_read("/workspace/.env.local").unwrap_err();
        assert!(err.contains("denied pattern"), "got: {err}");
        assert!(err.contains("**/.env*"), "got: {err}");
    }

    #[test]
    fn check_read_not_in_allowed_list() {
        let caps = AgentCapabilities::full_access().with_allowed_paths(vec!["src/**".into()]);
        let err = caps.check_read("/workspace/README.md").unwrap_err();
        assert!(err.contains("not in allowed list"), "got: {err}");
        assert!(err.contains("src/**"), "got: {err}");
    }

    #[test]
    fn check_exec_denied_command_explains_pattern() {
        let caps = AgentCapabilities::full_access().with_denied_commands(vec![r"^sudo\s".into()]);
        let err = caps.check_exec("sudo rm -rf /").unwrap_err();
        assert!(err.contains("denied pattern"), "got: {err}");
        assert!(err.contains("^sudo\\s"), "got: {err}");
    }

    #[test]
    fn check_exec_not_in_allowed_list() {
        let caps = AgentCapabilities::full_access()
            .with_allowed_commands(vec![r"^cargo ".into(), r"^git ".into()]);
        let err = caps.check_exec("npm install").unwrap_err();
        assert!(err.contains("not in allowed list"), "got: {err}");
        assert!(err.contains("^cargo "), "got: {err}");
    }

    #[test]
    fn check_allowed_operations_return_ok() {
        let caps = AgentCapabilities::full_access();
        assert!(caps.check_read("any/path").is_ok());
        assert!(caps.check_write("any/path").is_ok());
        assert!(caps.check_exec("any command").is_ok());
    }

    /// Verify `full_access()` never blocks common shell patterns that agents
    /// routinely emit. Each entry here was either denied in a real session
    /// or represents a pattern class that naive deny-lists would break.
    #[test]
    fn full_access_allows_common_shell_patterns() {
        let caps = AgentCapabilities::full_access();

        let commands = [
            // Heredoc with cat redirect (denied in previous session)
            "cat > /tmp/test_caps.rs << 'EOF'\nfn main() { println!(\"hello\"); }\nEOF",
            // Grep with pipe-separated OR patterns (denied in previous session)
            r#"grep -n "agent_loop\|Permission\|permission\|denied\|Denied" src/agent_loop.rs"#,
            // Multi-command chains
            "cd /workspace && cargo build && cargo test",
            "mkdir -p /tmp/test && cd /tmp/test && echo hello > file.txt",
            // Pipes and redirects
            "cargo test 2>&1 | head -50",
            "cat file.txt | grep pattern | wc -l",
            "echo 'data' >> /tmp/append.txt",
            // Subshells and grouping
            "(cd /tmp && ls -la)",
            "{ echo a; echo b; } > /tmp/out.txt",
            // Process substitution and special chars
            "diff <(sort file1) <(sort file2)",
            "find . -name '*.rs' -exec grep -l 'TODO' {} +",
            // Common dev commands
            "cargo clippy -- -D warnings",
            "cargo fmt --check",
            "git diff --stat HEAD~1",
            "npm install && npm run build",
            "python3 -c 'print(\"hello\")'",
            // Commands with special regex chars that shouldn't trip up matching
            "grep -rn 'foo(bar)' src/",
            "echo '$HOME is ~/work'",
            "ls *.rs",
        ];

        for cmd in &commands {
            assert!(
                caps.check_exec(cmd).is_ok(),
                "full_access() unexpectedly blocked command: {cmd}"
            );
        }
    }

    /// Verify `full_access()` allows reading/writing any path, including
    /// paths that a naive deny-list might block (dotfiles, tmp, etc.).
    #[test]
    fn full_access_allows_all_paths() {
        let caps = AgentCapabilities::full_access();

        let paths = [
            "src/main.rs",
            ".env",
            ".env.local",
            "/tmp/test_caps.rs",
            "/home/user/.ssh/config",
            "/workspace/secrets/api_key.txt",
            "/workspace/certs/server.pem",
            "Cargo.toml",
            "node_modules/.package-lock.json",
        ];

        for path in &paths {
            assert!(
                caps.check_read(path).is_ok(),
                "full_access() unexpectedly blocked read: {path}"
            );
            assert!(
                caps.check_write(path).is_ok(),
                "full_access() unexpectedly blocked write: {path}"
            );
        }
    }

    /// Verify `Default` is `full_access()` — the SDK is unopinionated out of the box.
    /// Consumers restrict from there, not opt-in to each capability.
    #[test]
    fn default_is_full_access() {
        let caps = AgentCapabilities::default();

        // Everything allowed by default
        assert!(caps.check_read("src/main.rs").is_ok());
        assert!(caps.check_write("src/main.rs").is_ok());
        assert!(caps.check_exec("ls").is_ok());

        // No deny lists
        assert!(caps.check_path(".env").is_ok());
        assert!(caps.check_path("/home/user/.ssh/id_rsa").is_ok());
        assert!(caps.check_command("sudo rm -rf /").is_ok());
    }
}
