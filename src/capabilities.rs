use serde::{Deserialize, Serialize};

/// Capabilities that control what the agent can do.
///
/// This provides a security model for primitive tools (Read, Write, Grep, Glob, Bash).
/// Paths are matched using glob patterns, commands using regex patterns.
///
/// By default, no paths or commands are denied — the SDK is unopinionated and leaves
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

    /// Check if a path can be read
    #[must_use]
    pub fn can_read(&self, path: &str) -> bool {
        self.read && self.path_allowed(path)
    }

    /// Check if a path can be written
    #[must_use]
    pub fn can_write(&self, path: &str) -> bool {
        self.write && self.path_allowed(path)
    }

    /// Check if a command can be executed
    #[must_use]
    pub fn can_exec(&self, command: &str) -> bool {
        self.exec && self.command_allowed(command)
    }

    /// Check if a path is allowed (not in denied list and in allowed list if specified)
    #[must_use]
    pub fn path_allowed(&self, path: &str) -> bool {
        // Check denied patterns first (takes precedence)
        for pattern in &self.denied_paths {
            if glob_match(pattern, path) {
                return false;
            }
        }

        // If allowed_paths is empty, all non-denied paths are allowed
        if self.allowed_paths.is_empty() {
            return true;
        }

        // Check if path matches any allowed pattern
        for pattern in &self.allowed_paths {
            if glob_match(pattern, path) {
                return true;
            }
        }

        false
    }

    /// Check if a command is allowed
    #[must_use]
    pub fn command_allowed(&self, command: &str) -> bool {
        // Check denied patterns first
        for pattern in &self.denied_commands {
            if regex_match(pattern, command) {
                return false;
            }
        }

        // If allowed_commands is empty, all non-denied commands are allowed
        if self.allowed_commands.is_empty() {
            return true;
        }

        // Check if command matches any allowed pattern
        for pattern in &self.allowed_commands {
            if regex_match(pattern, command) {
                return true;
            }
        }

        false
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
        assert!(caps.path_allowed("src/main.rs"));
        assert!(caps.path_allowed(".env"));
        assert!(caps.path_allowed("/workspace/secrets/key.txt"));
        assert!(caps.command_allowed("any command"));
    }

    #[test]
    fn test_full_access_allows_everything() {
        let caps = AgentCapabilities::full_access();

        assert!(caps.can_read("/any/path"));
        assert!(caps.can_write("/any/path"));
        assert!(caps.can_exec("any command"));
    }

    #[test]
    fn test_read_only_cannot_write() {
        let caps = AgentCapabilities::read_only();

        assert!(caps.can_read("src/main.rs"));
        assert!(!caps.can_write("src/main.rs"));
        assert!(!caps.can_exec("ls"));
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
        assert!(!caps.path_allowed(".env"));
        assert!(!caps.path_allowed("config/.env.local"));
        assert!(!caps.path_allowed("app/secrets/key.txt"));
        assert!(!caps.path_allowed("certs/server.pem"));

        // Denied paths (absolute — after resolve_path)
        assert!(!caps.path_allowed("/workspace/.env"));
        assert!(!caps.path_allowed("/workspace/.env.production"));
        assert!(!caps.path_allowed("/workspace/secrets/key.txt"));
        assert!(!caps.path_allowed("/workspace/certs/server.pem"));

        // Normal files still allowed
        assert!(caps.path_allowed("src/main.rs"));
        assert!(caps.path_allowed("/workspace/src/main.rs"));
        assert!(caps.path_allowed("/workspace/README.md"));
    }

    #[test]
    fn test_allowed_paths_restriction() {
        let caps = AgentCapabilities::read_only()
            .with_allowed_paths(vec!["src/**".into(), "tests/**".into()]);

        assert!(caps.path_allowed("src/main.rs"));
        assert!(caps.path_allowed("src/lib/utils.rs"));
        assert!(caps.path_allowed("tests/integration.rs"));
        assert!(!caps.path_allowed("config/settings.toml"));
        assert!(!caps.path_allowed("README.md"));
    }

    #[test]
    fn test_denied_takes_precedence() {
        let caps = AgentCapabilities::read_only()
            .with_denied_paths(vec!["**/secret/**".into()])
            .with_allowed_paths(vec!["**".into()]);

        assert!(caps.path_allowed("src/main.rs"));
        assert!(!caps.path_allowed("src/secret/key.txt"));
    }

    #[test]
    fn test_client_configured_denied_commands() {
        let caps = AgentCapabilities::full_access()
            .with_denied_commands(vec![r"rm\s+-rf\s+/".into(), r"^sudo\s".into()]);

        assert!(!caps.command_allowed("rm -rf /"));
        assert!(!caps.command_allowed("sudo rm file"));

        // Common shell patterns are NOT blocked
        assert!(caps.command_allowed("ls -la"));
        assert!(caps.command_allowed("cargo build"));
        assert!(caps.command_allowed("unzip file.zip 2>/dev/null"));
        assert!(caps.command_allowed("python3 -m markitdown file.pptx"));
    }

    #[test]
    fn test_allowed_commands_restriction() {
        let caps = AgentCapabilities::full_access()
            .with_allowed_commands(vec![r"^cargo ".into(), r"^git ".into()]);

        assert!(caps.command_allowed("cargo build"));
        assert!(caps.command_allowed("git status"));
        assert!(!caps.command_allowed("ls -la"));
        assert!(!caps.command_allowed("npm install"));
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
}
