//! URL validation and SSRF protection.

use anyhow::{Context, Result, bail};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use url::Url;

/// Default blocked hostnames for SSRF protection.
const DEFAULT_BLOCKED_HOSTS: &[&str] = &[
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "[::1]",
    "169.254.169.254",          // AWS metadata
    "metadata.google.internal", // GCP metadata
    "metadata.goog",            // GCP metadata alternate
];

/// URL validator with SSRF protection.
///
/// Validates URLs before fetching to prevent Server-Side Request Forgery attacks.
/// By default, blocks access to:
/// - Localhost and loopback addresses
/// - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
/// - Cloud metadata endpoints (AWS, GCP)
///
/// # Example
///
/// ```ignore
/// use agent_sdk::web::UrlValidator;
///
/// let validator = UrlValidator::new();
/// assert!(validator.validate("https://example.com").is_ok());
/// assert!(validator.validate("http://localhost").is_err());
/// ```
#[derive(Clone, Debug)]
pub struct UrlValidator {
    /// Only allow these domains (if Some).
    allowed_domains: Option<Vec<String>>,
    /// Block these hostnames/IPs.
    blocked_hosts: Vec<String>,
    /// Allow private IP ranges (default: false).
    allow_private_ips: bool,
    /// Maximum number of redirects to follow (default: 3).
    max_redirects: usize,
    /// Require HTTPS (default: true).
    require_https: bool,
}

impl Default for UrlValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl UrlValidator {
    /// Create a new URL validator with default security settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allowed_domains: None,
            blocked_hosts: DEFAULT_BLOCKED_HOSTS
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            allow_private_ips: false,
            max_redirects: 3,
            require_https: true,
        }
    }

    /// Only allow URLs from specific domains.
    #[must_use]
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Add additional blocked hosts.
    #[must_use]
    pub fn with_blocked_hosts(mut self, hosts: Vec<String>) -> Self {
        self.blocked_hosts.extend(hosts);
        self
    }

    /// Allow private IP ranges (dangerous - use with caution).
    #[must_use]
    pub const fn with_allow_private_ips(mut self, allow: bool) -> Self {
        self.allow_private_ips = allow;
        self
    }

    /// Set maximum redirects.
    #[must_use]
    pub const fn with_max_redirects(mut self, max: usize) -> Self {
        self.max_redirects = max;
        self
    }

    /// Allow HTTP URLs (default requires HTTPS).
    #[must_use]
    pub const fn with_allow_http(mut self) -> Self {
        self.require_https = false;
        self
    }

    /// Get the maximum number of redirects allowed.
    #[must_use]
    pub const fn max_redirects(&self) -> usize {
        self.max_redirects
    }

    /// Validate a URL string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The URL is malformed
    /// - The scheme is not HTTP or HTTPS
    /// - HTTPS is required but HTTP is used
    /// - The host is blocked
    /// - The host resolves to a private/blocked IP
    /// - The domain is not in the allowed list
    pub fn validate(&self, url_str: &str) -> Result<Url> {
        let url = Url::parse(url_str).context("Invalid URL format")?;

        // Check scheme
        match url.scheme() {
            "https" => {}
            "http" => {
                if self.require_https {
                    bail!("HTTPS required, but HTTP URL provided");
                }
            }
            scheme => bail!("Unsupported URL scheme: {scheme}"),
        }

        // Check host
        let host = url.host_str().context("URL must have a host")?;

        // Check blocked hosts
        if self.blocked_hosts.iter().any(|blocked| {
            host.eq_ignore_ascii_case(blocked) || host.ends_with(&format!(".{blocked}"))
        }) {
            bail!("Access to host '{host}' is blocked");
        }

        // Check allowed domains
        if let Some(ref allowed) = self.allowed_domains {
            let is_allowed = allowed.iter().any(|domain| {
                host.eq_ignore_ascii_case(domain) || host.ends_with(&format!(".{domain}"))
            });
            if !is_allowed {
                bail!("Host '{host}' is not in the allowed domains list");
            }
        }

        // Resolve and check IP
        self.validate_resolved_ip(host)?;

        Ok(url)
    }

    /// Validate that the resolved IP addresses are safe.
    fn validate_resolved_ip(&self, host: &str) -> Result<()> {
        // Try to resolve the hostname
        let addrs: Vec<_> = format!("{host}:80")
            .to_socket_addrs()
            .map(Iterator::collect)
            .unwrap_or_default();

        for addr in addrs {
            let ip = addr.ip();
            if !self.allow_private_ips && is_private_ip(&ip) {
                bail!("Access to private IP address {ip} is blocked");
            }
            if is_loopback(&ip) {
                bail!("Access to loopback address {ip} is blocked");
            }
            if is_link_local(&ip) {
                bail!("Access to link-local address {ip} is blocked");
            }
        }

        Ok(())
    }
}

/// Check if an IP address is private.
fn is_private_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => is_private_ipv4(*ipv4),
        IpAddr::V6(ipv6) => is_private_ipv6(ipv6),
    }
}

/// Check if an IPv4 address is private.
fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();

    // 10.0.0.0/8
    if octets[0] == 10 {
        return true;
    }

    // 172.16.0.0/12
    if octets[0] == 172 && (16..=31).contains(&octets[1]) {
        return true;
    }

    // 192.168.0.0/16
    if octets[0] == 192 && octets[1] == 168 {
        return true;
    }

    // 100.64.0.0/10 (Carrier-grade NAT)
    if octets[0] == 100 && (64..=127).contains(&octets[1]) {
        return true;
    }

    false
}

/// Check if an IPv6 address is private.
const fn is_private_ipv6(ip: &Ipv6Addr) -> bool {
    // Unique local addresses (fc00::/7)
    let segments = ip.segments();
    (segments[0] & 0xfe00) == 0xfc00
}

/// Check if an IP is a loopback address.
const fn is_loopback(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => ipv4.is_loopback(),
        IpAddr::V6(ipv6) => ipv6.is_loopback(),
    }
}

/// Check if an IP is a link-local address.
const fn is_link_local(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => ipv4.is_link_local(),
        IpAddr::V6(ipv6) => {
            // fe80::/10
            let segments = ipv6.segments();
            (segments[0] & 0xffc0) == 0xfe80
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_https_url() {
        let validator = UrlValidator::new();
        assert!(validator.validate("https://example.com").is_ok());
        assert!(validator.validate("https://example.com/path").is_ok());
        assert!(validator.validate("https://sub.example.com").is_ok());
    }

    #[test]
    fn test_http_blocked_by_default() {
        let validator = UrlValidator::new();
        let result = validator.validate("http://example.com");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("HTTPS required"));
    }

    #[test]
    fn test_http_allowed_with_flag() {
        let validator = UrlValidator::new().with_allow_http();
        assert!(validator.validate("http://example.com").is_ok());
    }

    #[test]
    fn test_localhost_blocked() {
        let validator = UrlValidator::new().with_allow_http();
        assert!(validator.validate("http://localhost").is_err());
        assert!(validator.validate("http://127.0.0.1").is_err());
        assert!(validator.validate("http://[::1]").is_err());
    }

    #[test]
    fn test_metadata_endpoints_blocked() {
        let validator = UrlValidator::new().with_allow_http();
        assert!(validator.validate("http://169.254.169.254").is_err());
        assert!(
            validator
                .validate("http://metadata.google.internal")
                .is_err()
        );
    }

    #[test]
    fn test_invalid_url() {
        let validator = UrlValidator::new();
        assert!(validator.validate("not-a-url").is_err());
        assert!(validator.validate("").is_err());
        assert!(validator.validate("ftp://example.com").is_err());
    }

    #[test]
    fn test_allowed_domains() {
        let validator = UrlValidator::new().with_allowed_domains(vec!["example.com".to_string()]);

        assert!(validator.validate("https://example.com").is_ok());
        assert!(validator.validate("https://sub.example.com").is_ok());

        let result = validator.validate("https://other.com");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in the allowed domains")
        );
    }

    #[test]
    fn test_blocked_hosts() {
        let validator = UrlValidator::new().with_blocked_hosts(vec!["blocked.com".to_string()]);

        let result = validator.validate("https://blocked.com");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("blocked"));
    }

    #[test]
    fn test_is_private_ipv4() {
        // Private ranges
        assert!(is_private_ipv4(Ipv4Addr::new(10, 0, 0, 1)));
        assert!(is_private_ipv4(Ipv4Addr::new(10, 255, 255, 255)));
        assert!(is_private_ipv4(Ipv4Addr::new(172, 16, 0, 1)));
        assert!(is_private_ipv4(Ipv4Addr::new(172, 31, 255, 255)));
        assert!(is_private_ipv4(Ipv4Addr::new(192, 168, 0, 1)));
        assert!(is_private_ipv4(Ipv4Addr::new(192, 168, 255, 255)));

        // Not private
        assert!(!is_private_ipv4(Ipv4Addr::new(8, 8, 8, 8)));
        assert!(!is_private_ipv4(Ipv4Addr::new(1, 1, 1, 1)));
        assert!(!is_private_ipv4(Ipv4Addr::new(172, 15, 0, 1)));
        assert!(!is_private_ipv4(Ipv4Addr::new(172, 32, 0, 1)));
    }

    #[test]
    fn test_max_redirects() {
        let validator = UrlValidator::new().with_max_redirects(5);
        assert_eq!(validator.max_redirects(), 5);
    }

    #[test]
    fn test_default_validator() {
        let validator = UrlValidator::default();
        assert!(!validator.allow_private_ips);
        assert!(validator.require_https);
        assert_eq!(validator.max_redirects, 3);
    }
}
