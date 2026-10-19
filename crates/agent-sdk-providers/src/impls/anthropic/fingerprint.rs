//! Claude Code inference-fingerprint forging for Anthropic OAuth requests.
//!
//! Over OAuth this provider presents not just Claude Code's request identity
//! but the integrity and attribution signals a real Claude Code client emits:
//! the Stainless SDK's telemetry headers, the `x-anthropic-billing-header`
//! system block with its `cch` body attestation, and a `metadata.user_id`
//! naming a synthetic device. Every constant and derivation mirrors omp
//! (`can1357/oh-my-pi`: `providers/claude-code-fingerprint.ts`,
//! `providers/anthropic-identity.ts`, `providers/anthropic.ts`) so the two
//! clients are indistinguishable on the wire.

use sha2::{Digest, Sha256};

/// `@anthropic-ai/sdk` version the Claude Code release this provider
/// presents as bundles. Rides the Stainless telemetry headers, not the wire
/// identity: `CLAUDE_CODE_VERSION` stays the reported CLI version.
pub(super) const STAINLESS_PACKAGE_VERSION: &str = "0.112.1";
/// Node runtime version the bundled SDK reports.
pub(super) const STAINLESS_RUNTIME_VERSION: &str = "v26.3.0";
/// Request timeout (seconds) Claude Code configures the SDK with.
pub(super) const STAINLESS_TIMEOUT: &str = "600";
/// Maps this platform's `std::env::consts::OS` to the Stainless wire value.
pub(super) fn stainless_os() -> &'static str {
    static OS: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| match std::env::consts::OS {
            "macos" => "MacOS".to_owned(),
            "windows" => "Windows".to_owned(),
            "linux" => "Linux".to_owned(),
            "freebsd" => "FreeBSD".to_owned(),
            other => format!("Other::{other}"),
        });
    &OS
}

/// Maps this platform's `std::env::consts::ARCH` to the Stainless wire value.
pub(super) fn stainless_arch() -> &'static str {
    static ARCH: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| match std::env::consts::ARCH {
            "x86_64" => "x64".to_owned(),
            "aarch64" => "arm64".to_owned(),
            "x86" => "x86".to_owned(),
            other => format!("other::{other}"),
        });
    &ARCH
}

const INSTALL_ID_ENV: &str = "AGENT_SDK_INSTALL_ID";
const INSTALL_ID_FILE: &str = "install-id";

/// Per-install UUID backing the OAuth `metadata.user_id` device id, persisted
/// at `<config dir>/agent-sdk/install-id` (0600) so it survives process and
/// agent-state restarts. `AGENT_SDK_INSTALL_ID` overrides the file for tests
/// and locked-down environments.
fn install_id() -> String {
    static INSTALL_ID: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(read_or_create_install_id);
    INSTALL_ID.clone()
}

fn read_or_create_install_id() -> String {
    if let Ok(configured) = std::env::var(INSTALL_ID_ENV) {
        let configured = configured.trim();
        if !configured.is_empty() {
            return configured.to_owned();
        }
    }
    install_id_in(&install_id_dir())
}

fn install_id_dir() -> std::path::PathBuf {
    if let Some(configured) = non_empty_env("XDG_CONFIG_HOME") {
        return std::path::PathBuf::from(configured).join("agent-sdk");
    }
    if let Some(app_data) = non_empty_env("APPDATA") {
        return std::path::PathBuf::from(app_data).join("agent-sdk");
    }
    non_empty_env("HOME").map_or_else(std::path::PathBuf::new, |home| {
        std::path::PathBuf::from(home)
            .join(".config")
            .join("agent-sdk")
    })
}

fn non_empty_env(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn read_install_id(path: &std::path::Path) -> Option<String> {
    let existing = std::fs::read_to_string(path).ok()?;
    let existing = existing.trim();
    uuid::Uuid::parse_str(existing)
        .ok()
        .map(|_| existing.to_owned())
}

/// Reads the persisted install id under `dir`, or creates one. Concurrent
/// first calls race on `create_new`; the loser re-reads the winner's id.
fn install_id_in(dir: &std::path::Path) -> String {
    let path = dir.join(INSTALL_ID_FILE);
    if let Some(existing) = read_install_id(&path) {
        return existing;
    }
    // Present but unparseable: unlink so the exclusive create does not trip
    // on the garbage (an ignored race just falls through to re-read below).
    let _ = std::fs::remove_file(&path);
    let generated = uuid::Uuid::new_v4().to_string();
    match create_install_id_file(&path, &generated) {
        Ok(()) => generated,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            read_install_id(&path).unwrap_or(generated)
        }
        Err(error) => {
            log::warn!(
                "agent-sdk: failed to persist the install id at {}: {error}",
                path.display()
            );
            generated
        }
    }
}

fn create_install_id_file(path: &std::path::Path, install_id: &str) -> std::io::Result<()> {
    use std::io::Write;

    #[cfg(unix)]
    fn open_exclusive(path: &std::path::Path) -> std::io::Result<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
    }

    #[cfg(not(unix))]
    fn open_exclusive(path: &std::path::Path) -> std::io::Result<std::fs::File> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = open_exclusive(path)?;
    file.write_all(format!("{install_id}\n").as_bytes())
}

/// Salt of omp's `device_id` derivation, kept verbatim so an install id
/// produces the same device id here and there.
const DEVICE_ID_DOMAIN: &str = "omp-claude-device-id-v1:";

/// The stable Claude device id omp reports in `metadata.user_id`.
fn device_id(install_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DEVICE_ID_DOMAIN);
    hasher.update(install_id);
    to_hex(&hasher.finalize())
}

/// The OAuth `metadata.user_id` omp reports: a JSON document naming the
/// synthetic device and the request's Claude Code session.
pub(super) fn metadata_user_id(session_id: &str) -> String {
    serde_json::json!({ "device_id": device_id(&install_id()), "session_id": session_id })
        .to_string()
}

const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";
/// Salt of Claude Code's `computeFingerprint` (`utils/fingerprint.ts`),
/// reproduced verbatim by omp's `createClaudeBillingHeader`.
const FINGERPRINT_SALT: &str = "59cf53e54c78";
const CCH_PLACEHOLDER: &str = "cch=00000";
/// Seed of the `cch` body attestation.
pub(super) const CCH_SEED: u64 = 0x4d65_9218_e32a_3268;
/// How far past the system[0] anchor the `cch` placeholder may sit.
const CCH_SEARCH_WINDOW: usize = 150;
/// Serialized prefix of the billing block: the `system` key serializes
/// before `messages` and `tools`, and JSON escaping cannot produce plain
/// quotes inside message text, so this can only match the injected
/// `system[0]`.
const BILLING_SYSTEM_MARKER: &str =
    "\"system\":[{\"type\":\"text\",\"text\":\"x-anthropic-billing-header:";

/// The `system[0]` billing block text Claude Code (and omp) prepend: the
/// reported CLI version fingerprinted over the first wire user message,
/// plus a `cch` placeholder [`patch_cch`] replaces with an attestation of
/// the serialized body.
pub(super) fn claude_billing_header(first_user_message_text: &str, version: &str) -> String {
    let mut fingerprint_input = String::new();
    for index in [4, 7, 20] {
        fingerprint_input.push(first_user_message_text.chars().nth(index).unwrap_or('0'));
    }
    let digest = Sha256::digest(format!("{FINGERPRINT_SALT}{fingerprint_input}{version}"));
    let suffix = &to_hex(&digest)[..3];
    format!(
        "{BILLING_HEADER_PREFIX} cc_version={version}.{suffix}; cc_entrypoint=cli; {CCH_PLACEHOLDER};"
    )
}

/// Outcome of a [`patch_cch`] pass over a serialized body.
pub(super) enum CchOutcome {
    /// The placeholder was replaced with the attestation hash.
    Patched,
    /// The body carries no billing block (not ours to attest).
    NoBillingHeader,
    /// The placeholder sits past the system[0] anchor — a serialization
    /// regression, sent unattested.
    Unanchored,
}

/// Replaces the [`CCH_PLACEHOLDER`] in a serialized request body with the low
/// 20 bits (5 hex chars) of xxHash64(body-with-placeholder, [`CCH_SEED`]) —
/// the attestation Claude Code's `cch` carries, hashed over the exact bytes
/// sent.
pub(super) fn patch_cch(body: &mut String) -> CchOutcome {
    let Some(marker) = body.find(BILLING_SYSTEM_MARKER) else {
        return CchOutcome::NoBillingHeader;
    };
    let search_from = marker + BILLING_SYSTEM_MARKER.len();
    let search_to = (search_from + CCH_SEARCH_WINDOW).min(body.len());
    let window = &body.as_bytes()[search_from..search_to];
    let Some(placeholder) = window
        .windows(CCH_PLACEHOLDER.len())
        .position(|candidate| candidate == CCH_PLACEHOLDER.as_bytes())
    else {
        return CchOutcome::Unanchored;
    };
    let digits = search_from + placeholder + "cch=".len();
    let cch = xxhash_rust::xxh64::xxh64(body.as_bytes(), CCH_SEED) & 0xfffff;
    body.replace_range(digits..digits + 5, &format!("{cch:05x}"));
    CchOutcome::Patched
}

const VERSION_TOO_OLD_CODE: &str = "claude_code_version_too_old";
const REQUIRED_VERSION_SUFFIX: &str = " or newer is required";

/// The minimum Claude Code version an Anthropic
/// `claude_code_version_too_old` rejection names ("... version X.Y.Z or
/// newer is required ..."), if the body carries one. Case-insensitive,
/// matching omp's pattern.
pub(super) fn required_claude_code_version(error_body: &str) -> Option<String> {
    let lowered = error_body.to_ascii_lowercase();
    if !lowered.contains(VERSION_TOO_OLD_CODE) {
        return None;
    }
    let mut search = 0;
    while let Some(found) = lowered[search..].find("version ") {
        let tail = search + found + "version ".len();
        let mut rest = &error_body[tail..];
        if let Some(version) = parse_semver(&mut rest)
            && rest
                .to_ascii_lowercase()
                .starts_with(REQUIRED_VERSION_SUFFIX)
        {
            return Some(version.to_owned());
        }
        search = tail;
    }
    None
}

/// Consumes `X.Y.Z` off the front of `rest`, leaving any trailing text.
fn parse_semver<'a>(rest: &mut &'a str) -> Option<&'a str> {
    let mut end = 0;
    for component in 0..3 {
        if component > 0 && rest.as_bytes().get(end) != Some(&b'.') {
            return None;
        }
        end += usize::from(component > 0);
        let digit_start = end;
        while rest.as_bytes().get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if end == digit_start {
            return None;
        }
    }
    let version = &rest[..end];
    *rest = &rest[end..];
    Some(version)
}

/// Ordered comparison of two `X.Y.Z` versions; `None` when either is not one
/// (never adopted, matching omp's `compareSemver` fallback).
pub(super) fn compare_semver(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    Some(semver_components(left)?.cmp(&semver_components(right)?))
}

fn semver_components(version: &str) -> Option<[u64; 3]> {
    let mut components = version.split('.');
    let mut parts = [0; 3];
    for part in &mut parts {
        *part = components.next()?.parse().ok()?;
    }
    components.next().is_none().then_some(parts)
}

fn to_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut hex, byte| {
            use std::fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256_hex(input: &str) -> String {
        to_hex(&Sha256::digest(input))
    }

    #[test]
    fn install_id_is_created_once_and_reread() {
        let dir = std::env::temp_dir().join(format!(
            "agent-sdk-install-id-test-{}",
            uuid::Uuid::new_v4()
        ));
        let first = install_id_in(&dir);
        let read_back = read_install_id(&dir.join(INSTALL_ID_FILE));
        assert_eq!(read_back.as_deref(), Some(first.as_str()));
        assert_eq!(install_id_in(&dir), first);
        assert_eq!(install_id_in(&dir), first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn install_id_overwrites_garbage() {
        let dir = std::env::temp_dir().join(format!(
            "agent-sdk-install-id-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("{error}"));
        std::fs::write(dir.join(INSTALL_ID_FILE), "not a uuid\n").unwrap_or_else(|error| {
            panic!("{error}");
        });
        let regenerated = install_id_in(&dir);
        assert_eq!(
            read_install_id(&dir.join(INSTALL_ID_FILE)).as_deref(),
            Some(regenerated.as_str())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn device_id_is_a_stable_sha256_of_the_install_domain() {
        let first = device_id("0b6f6d1a-2f88-4c25-8f66-d0d5a5cfb7e3");
        assert_eq!(first.len(), 64);
        assert_eq!(first, device_id("0b6f6d1a-2f88-4c25-8f66-d0d5a5cfb7e3"));
        assert_eq!(
            first,
            sha256_hex("omp-claude-device-id-v1:0b6f6d1a-2f88-4c25-8f66-d0d5a5cfb7e3")
        );
        assert_ne!(first, device_id("another-install"));
    }

    #[test]
    fn billing_header_fingerprints_the_first_user_message_and_version() {
        let header = claude_billing_header("abcdefghijklmnopqrstuvwx", "2.1.296");
        let expected_suffix = &sha256_hex("59cf53e54c78ehu2.1.296")[..3];
        assert_eq!(
            header,
            format!(
                "x-anthropic-billing-header: cc_version=2.1.296.{expected_suffix}; \
                 cc_entrypoint=cli; cch=00000;"
            )
        );
    }

    #[test]
    fn billing_header_pads_short_first_user_messages() {
        let header = claude_billing_header("abc", "2.1.301");
        let expected_suffix = &sha256_hex("59cf53e54c780002.1.301")[..3];
        assert!(header.starts_with(&format!(
            "x-anthropic-billing-header: cc_version=2.1.301.{expected_suffix}; cc_entrypoint=cli;"
        )));
    }

    fn patched_body() -> String {
        let mut body = String::from(
            "{\"model\":\"claude-opus-5-5\",\"system\":[{\"type\":\"text\",\"text\":\
             \"x-anthropic-billing-header: cc_version=2.1.296.abc; cc_entrypoint=cli; cch=00000;\
             \"},{\"type\":\"text\",\"text\":\"You are Claude Code.\"}]}",
        );
        assert!(matches!(patch_cch(&mut body), CchOutcome::Patched));
        body
    }

    #[test]
    fn patch_cch_embeds_the_low_20_bits_of_xxhash64() {
        let body = patched_body();
        let cch = body
            .find("cch=")
            .map_or_else(|| panic!("no cch in {body}"), |at| &body[at + 4..at + 9]);
        let expected =
            xxhash_rust::xxh64::xxh64(body.replace(cch, "00000").as_bytes(), CCH_SEED) & 0xfffff;
        assert_eq!(cch, format!("{expected:05x}"));
        assert_ne!(cch, "00000");
    }

    #[test]
    fn patch_cch_distinguishes_missing_and_unanchored_placeholders() {
        let mut no_billing =
            String::from("{\"messages\":[{\"role\":\"user\",\"content\":\"cch=00000\"}]}");
        assert!(matches!(
            patch_cch(&mut no_billing),
            CchOutcome::NoBillingHeader
        ));

        let filler = "0".repeat(200);
        let mut unanchored = format!("{BILLING_SYSTEM_MARKER}{filler}{CCH_PLACEHOLDER}");
        assert!(matches!(patch_cch(&mut unanchored), CchOutcome::Unanchored));
    }

    #[test]
    fn required_version_is_parsed_case_insensitively() {
        let body = "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\
                    \"message\":\"claude_code_version_too_old: version 2.1.301 or newer is \
                    required for claude-opus-5-5\"}}";
        assert_eq!(
            required_claude_code_version(body).as_deref(),
            Some("2.1.301")
        );
        let capitalized = "claude_code_version_too_old: Version 2.2.0 OR NEWER IS REQUIRED";
        assert_eq!(
            required_claude_code_version(capitalized).as_deref(),
            Some("2.2.0")
        );
    }

    #[test]
    fn required_version_needs_the_code_and_a_complete_phrase() {
        assert_eq!(
            required_claude_code_version("version 2.1.301 or newer is required"),
            None
        );
        assert_eq!(
            required_claude_code_version(
                "claude_code_version_too_old: version 2.1 or newer is required"
            ),
            None
        );
        assert_eq!(
            required_claude_code_version(
                "claude_code_version_too_old: version 2.1.301 or older is fine"
            ),
            None
        );
    }

    #[test]
    fn compare_semver_orders_three_components_and_rejects_garbage() {
        assert_eq!(
            compare_semver("2.1.301", "2.1.296"),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            compare_semver("2.1.296", "2.1.296"),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_semver("2.1.9", "2.1.10"),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(compare_semver("2.1", "2.1.0"), None);
        assert_eq!(compare_semver("2.1.0.1", "2.1.0"), None);
    }

    #[test]
    fn metadata_user_id_names_the_device_and_session() {
        let user_id = metadata_user_id("7d87474e-b6a2-4a70-a267-65a44e94d81e");
        let parsed: serde_json::Value =
            serde_json::from_str(&user_id).unwrap_or_else(|error| panic!("{error}: {user_id}"));
        assert_eq!(
            parsed["device_id"].as_str().map(str::len),
            Some(64),
            "{user_id}"
        );
        assert_eq!(
            parsed["session_id"].as_str(),
            Some("7d87474e-b6a2-4a70-a267-65a44e94d81e")
        );
    }
}
