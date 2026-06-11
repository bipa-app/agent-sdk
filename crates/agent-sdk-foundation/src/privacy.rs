//! PII detection and masking primitives.
//!
//! This module defines the [`PiiDetector`] trait and baseline
//! implementations that locate entity-level PII (emails, phone
//! numbers, credit cards, Brazilian identifiers) in plain text.
//! Detection returns byte-offset spans so callers can mask, tokenize,
//! or flag content without coupling to a single masking strategy.
//!
//! # Relationship to audit redaction
//!
//! This module also ships a key-name based [`RedactionPolicy`]
//! (in [`redaction`]) that matches JSON fields like `password` or
//! `api_key` and masks their values. The two layers are complementary:
//! the policy handles structural redaction by key, and a
//! [`PiiDetector`] scans any remaining string values for freeform
//! entities (a card number dropped into a prompt, a CPF mentioned
//! inside a tool response, a bearer token leaking into an error
//! message).
//!
//! Integrations typically compose the two via
//! [`redact_for_observability`].
//!
//! # Categories
//!
//! [`PiiCategory`] enumerates the well-known types plus a
//! [`Custom`](PiiCategory::Custom) escape hatch for project-specific
//! categories. Detectors emit spans tagged with a category so
//! downstream masking can preserve type information (e.g. render
//! `[REDACTED:email]` instead of a generic marker).
//!
//! # Built-in detectors
//!
//! - [`NoopDetector`] — detects nothing.
//! - [`SecretDetector`] — recognises credential prefixes (`Bearer`,
//!   `sk-`, GitHub PATs, AWS access keys, Google API keys).
//! - [`EntityDetector`] — regex + check-digit validation for emails,
//!   E.164 phones, credit card PANs (Luhn), Brazilian CPFs and CNPJs
//!   (mod-11 check digits), Pix UUID keys, IPv4 addresses, JWTs.
//! - [`CompositeDetector`] — wraps multiple detectors and
//!   deduplicates overlapping spans.
//! - [`BaselineDetector`] — convenience composite of
//!   [`SecretDetector`] + [`EntityDetector`], suitable as the SDK
//!   default detector.
//!
//! # Usage
//!
//! ```
//! use agent_sdk_foundation::privacy::{BaselineDetector, PiiDetector, mask_spans};
//!
//! let detector = BaselineDetector::new()?;
//! let text = "Pay Pix to CPF 111.444.777-35 or card 4111 1111 1111 1111.";
//! let spans = detector.detect(text);
//! let masked = mask_spans(text, &spans);
//! assert!(masked.contains("[REDACTED:cpf]"));
//! assert!(masked.contains("[REDACTED:credit_card]"));
//! # Ok::<(), regex::Error>(())
//! ```
//!
//! # Limitations
//!
//! - Detection is deterministic pattern + checksum based. It will
//!   miss entities that require semantic understanding (person
//!   names, postal addresses, document photos) — those need a
//!   neural detector wired in through the same [`PiiDetector`]
//!   trait.
//! - Spans use UTF-8 byte offsets; passing non-boundary offsets to
//!   [`mask_spans`] will silently skip those spans rather than
//!   panic.

use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

pub mod redaction;

/// Credential value prefixes recognised for *wholesale* redaction.
///
/// Single source of truth for the baseline [`RedactionPolicy`](redaction::RedactionPolicy)
/// `sensitive_value_prefixes` list, which is built from this slice. The
/// [`SecretDetector`] covers the same credential families through its richer
/// regex (each with per-prefix body and length rules); when adding a new
/// family, update both so they do not drift.
pub(crate) const SECRET_PREFIXES: &[&str] = &[
    "Bearer ",
    "sk-",
    "pk-",
    "xox",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "github_pat_",
    "AKIA",
    "AIza",
];

pub use redaction::{
    REDACTED_MARKER, RedactionLevel, RedactionPolicy, redact_error, redact_for_observability,
    redact_string, redact_value,
};

// ─────────────────────────────────────────────────────────────────────
// Category and span
// ─────────────────────────────────────────────────────────────────────

/// A well-known PII category tag.
///
/// Detectors emit spans labelled with a category so downstream
/// masking and audit logic can preserve type information.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PiiCategory {
    /// Credential or secret (API key, bearer token, password-shaped value).
    Secret,
    /// Email address.
    Email,
    /// Phone number (E.164 international format).
    Phone,
    /// Credit card PAN, Luhn-validated.
    CreditCard,
    /// Brazilian individual taxpayer ID (CPF), mod-11 validated.
    Cpf,
    /// Brazilian corporate taxpayer ID (CNPJ), mod-11 validated.
    Cnpj,
    /// Brazilian national ID (RG) — shape only, no check-digit.
    Rg,
    /// Brazilian driver's license number (CNH).
    Cnh,
    /// Pix instant-payment key in UUID form.
    PixKey,
    /// IPv4 address.
    IpAddress,
    /// JSON Web Token.
    Jwt,
    /// Custom, project-specific category.
    Custom(String),
}

impl PiiCategory {
    /// Stable machine-readable tag suitable for placeholder markers
    /// such as `[REDACTED:<tag>]`.
    #[must_use]
    pub const fn as_tag(&self) -> &str {
        match self {
            Self::Secret => "secret",
            Self::Email => "email",
            Self::Phone => "phone",
            Self::CreditCard => "credit_card",
            Self::Cpf => "cpf",
            Self::Cnpj => "cnpj",
            Self::Rg => "rg",
            Self::Cnh => "cnh",
            Self::PixKey => "pix_key",
            Self::IpAddress => "ip_address",
            Self::Jwt => "jwt",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// A span of PII located within a text buffer, expressed as half-open
/// byte offsets.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PiiSpan {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
    /// Category label assigned by the detector.
    pub category: PiiCategory,
}

impl PiiSpan {
    #[must_use]
    pub const fn new(start: usize, end: usize, category: PiiCategory) -> Self {
        Self {
            start,
            end,
            category,
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Whether this span strictly overlaps another (shared bytes,
    /// not merely touching).
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

// ─────────────────────────────────────────────────────────────────────
// PiiDetector trait
// ─────────────────────────────────────────────────────────────────────

/// Locates PII within plain text.
///
/// Implementations must be deterministic and side-effect free.
/// The order of returned spans is unspecified; callers that need
/// ordered output should sort by `start`.
///
/// [`Debug`] is a supertrait so detectors embedded in larger config
/// structs (e.g. an audit redaction policy) can derive `Debug`
/// without custom impls.
pub trait PiiDetector: Send + Sync + std::fmt::Debug {
    /// Find every PII span in `text`. Returned spans use UTF-8
    /// char-boundary-safe byte offsets.
    fn detect(&self, text: &str) -> Vec<PiiSpan>;
}

impl<T: PiiDetector + ?Sized> PiiDetector for Box<T> {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        (**self).detect(text)
    }
}

impl<T: PiiDetector + ?Sized> PiiDetector for std::sync::Arc<T> {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        (**self).detect(text)
    }
}

// ─────────────────────────────────────────────────────────────────────
// NoopDetector
// ─────────────────────────────────────────────────────────────────────

/// Detector that never reports any spans — a sentinel for paths
/// where PII detection is explicitly disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopDetector;

impl PiiDetector for NoopDetector {
    fn detect(&self, _text: &str) -> Vec<PiiSpan> {
        Vec::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// SecretDetector
// ─────────────────────────────────────────────────────────────────────

/// Detects credential-shaped tokens by prefix.
///
/// Covers Bearer tokens, OpenAI-style keys (`sk-`, `pk-`), Slack
/// (`xox…`), GitHub PATs (`ghp_`, `gho_`, `ghs_`, `ghu_`,
/// `github_pat_`), AWS access keys (`AKIA…`), and Google API keys
/// (`AIza…`).
#[derive(Debug)]
pub struct SecretDetector {
    pattern: Regex,
}

impl SecretDetector {
    /// Baseline detector covering the token shapes listed above.
    ///
    /// # Errors
    /// Returns a [`regex::Error`] if the internal pattern fails to
    /// compile — this should only occur on a corrupted build of
    /// the `regex` crate.
    pub fn baseline() -> Result<Self, regex::Error> {
        let pattern = Regex::new(
            r"(?x)
              (?:
                  \bBearer\s+[A-Za-z0-9._~+/=\-]{8,}
                | \bsk-[A-Za-z0-9_\-]{16,}
                | \bpk-[A-Za-z0-9_\-]{16,}
                | \bxox[abpsr]-[A-Za-z0-9\-]{8,}
                | \bghp_[A-Za-z0-9]{20,}
                | \bgho_[A-Za-z0-9]{20,}
                | \bghs_[A-Za-z0-9]{20,}
                | \bghu_[A-Za-z0-9]{20,}
                | \bgithub_pat_[A-Za-z0-9_]{20,}
                | \bAKIA[A-Z0-9]{16}
                | \bAIza[A-Za-z0-9_\-]{30,}
              )
            ",
        )?;
        Ok(Self { pattern })
    }
}

impl PiiDetector for SecretDetector {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        self.pattern
            .find_iter(text)
            .map(|m| PiiSpan::new(m.start(), m.end(), PiiCategory::Secret))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// EntityDetector
// ─────────────────────────────────────────────────────────────────────

/// Entity categories that [`EntityDetector`] knows how to look for.
///
/// This is the subset of [`PiiCategory`] that the entity detector
/// implements — it intentionally excludes [`Secret`](PiiCategory::Secret)
/// (handled by [`SecretDetector`]) and [`Custom`](PiiCategory::Custom)
/// (handled via user-supplied detectors).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DetectCategory {
    Email = 0,
    Phone = 1,
    CreditCard = 2,
    Cpf = 3,
    Cnpj = 4,
    PixUuid = 5,
    Ipv4 = 6,
    Jwt = 7,
}

impl DetectCategory {
    const fn mask(self) -> u16 {
        1u16 << (self as u8)
    }
}

/// Bitmask-packed set of entity categories to detect.
///
/// Construct with [`CategorySet::all`] or [`CategorySet::none`] and
/// toggle individual categories with [`with`](Self::with) and
/// [`without`](Self::without).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategorySet(u16);

impl CategorySet {
    const ALL_MASK: u16 = DetectCategory::Email.mask()
        | DetectCategory::Phone.mask()
        | DetectCategory::CreditCard.mask()
        | DetectCategory::Cpf.mask()
        | DetectCategory::Cnpj.mask()
        | DetectCategory::PixUuid.mask()
        | DetectCategory::Ipv4.mask()
        | DetectCategory::Jwt.mask();

    /// All categories enabled.
    #[must_use]
    pub const fn all() -> Self {
        Self(Self::ALL_MASK)
    }

    /// All categories disabled.
    #[must_use]
    pub const fn none() -> Self {
        Self(0)
    }

    /// Enable a single category, returning the updated set.
    #[must_use]
    pub const fn with(mut self, category: DetectCategory) -> Self {
        self.0 |= category.mask();
        self
    }

    /// Disable a single category, returning the updated set.
    #[must_use]
    pub const fn without(mut self, category: DetectCategory) -> Self {
        self.0 &= !category.mask();
        self
    }

    /// Whether the given category is enabled.
    #[must_use]
    pub const fn contains(self, category: DetectCategory) -> bool {
        self.0 & category.mask() != 0
    }
}

impl Default for CategorySet {
    fn default() -> Self {
        Self::all()
    }
}

/// Source patterns for each entity category, ordered to match the
/// [`DetectCategory`] discriminants (`Email = 0` … `Jwt = 7`).
///
/// Both the individual [`Regex`] fields and the [`RegexSet`] prefilter are
/// built from these so the two can never drift; the prefilter's match indices
/// line up 1:1 with `DetectCategory as usize`.
const ENTITY_PATTERNS: [&str; 8] = [
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    r"\+[1-9]\d{7,14}",
    r"\b(?:\d[ \-]?){12,18}\d\b",
    r"\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b",
    r"\b(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})\b",
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|1?\d\d?)\b",
    r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b",
];

/// Entity-aware detector using regex patterns plus check-digit
/// validation where applicable.
///
/// PANs are validated with Luhn; CPFs and CNPJs with the Brazilian
/// mod-11 algorithm. This sharply reduces false positives on random
/// numeric strings (invoice numbers, order IDs, etc.) that share
/// the same shape as real identifiers.
///
/// A [`RegexSet`] prefilter runs first: one multi-pattern pass identifies
/// which categories appear at all, so a clean string (the common case) pays a
/// single scan instead of eight independent `find_iter` passes.
#[derive(Debug)]
pub struct EntityDetector {
    email: Regex,
    phone: Regex,
    credit_card: Regex,
    cpf: Regex,
    cnpj: Regex,
    pix_uuid: Regex,
    ipv4: Regex,
    jwt: Regex,
    prefilter: RegexSet,
    enabled: CategorySet,
}

impl EntityDetector {
    /// Construct a detector with an explicit category toggle set.
    ///
    /// # Errors
    /// Returns a [`regex::Error`] if any internal pattern fails to
    /// compile.
    pub fn new(enabled: CategorySet) -> Result<Self, regex::Error> {
        Ok(Self {
            email: Regex::new(ENTITY_PATTERNS[DetectCategory::Email as usize])?,
            phone: Regex::new(ENTITY_PATTERNS[DetectCategory::Phone as usize])?,
            credit_card: Regex::new(ENTITY_PATTERNS[DetectCategory::CreditCard as usize])?,
            cpf: Regex::new(ENTITY_PATTERNS[DetectCategory::Cpf as usize])?,
            cnpj: Regex::new(ENTITY_PATTERNS[DetectCategory::Cnpj as usize])?,
            pix_uuid: Regex::new(ENTITY_PATTERNS[DetectCategory::PixUuid as usize])?,
            ipv4: Regex::new(ENTITY_PATTERNS[DetectCategory::Ipv4 as usize])?,
            jwt: Regex::new(ENTITY_PATTERNS[DetectCategory::Jwt as usize])?,
            prefilter: RegexSet::new(ENTITY_PATTERNS)?,
            enabled,
        })
    }

    /// Whether `category` is both enabled and reported as present by the
    /// prefilter pass — gates the per-category `find_iter`.
    fn should_scan(&self, matches: &regex::SetMatches, category: DetectCategory) -> bool {
        self.enabled.contains(category) && matches.matched(category as usize)
    }

    /// Baseline detector with all categories enabled.
    ///
    /// # Errors
    /// See [`EntityDetector::new`].
    pub fn baseline() -> Result<Self, regex::Error> {
        Self::new(CategorySet::all())
    }
}

impl PiiDetector for EntityDetector {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        let mut spans = Vec::new();

        // One multi-pattern pass tells us which categories appear at all;
        // a clean string short-circuits here without any per-category scan.
        let matches = self.prefilter.matches(text);
        if !matches.matched_any() {
            return spans;
        }

        if self.should_scan(&matches, DetectCategory::Email) {
            for m in self.email.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::Email));
            }
        }

        if self.should_scan(&matches, DetectCategory::Phone) {
            for m in self.phone.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::Phone));
            }
        }

        if self.should_scan(&matches, DetectCategory::CreditCard) {
            for m in self.credit_card.find_iter(text) {
                push_credit_card_spans(m.as_str(), m.start(), &mut spans);
            }
        }

        if self.should_scan(&matches, DetectCategory::Cpf) {
            for m in self.cpf.find_iter(text) {
                if cpf_is_valid(m.as_str()) {
                    spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::Cpf));
                }
            }
        }

        if self.should_scan(&matches, DetectCategory::Cnpj) {
            for m in self.cnpj.find_iter(text) {
                if cnpj_is_valid(m.as_str()) {
                    spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::Cnpj));
                }
            }
        }

        if self.should_scan(&matches, DetectCategory::PixUuid) {
            for m in self.pix_uuid.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::PixKey));
            }
        }

        if self.should_scan(&matches, DetectCategory::Ipv4) {
            for m in self.ipv4.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::IpAddress));
            }
        }

        if self.should_scan(&matches, DetectCategory::Jwt) {
            for m in self.jwt.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), PiiCategory::Jwt));
            }
        }

        spans
    }
}

/// Emit credit-card spans for one regex match's text.
///
/// `matched` is the matched run (it starts and ends with a digit; digits may
/// be separated by single spaces or dashes). `base` is the byte offset of
/// `matched` within the source text.
///
/// A clean PAN — the whole run passes Luhn — yields one span. Otherwise the
/// run carried extra digits (e.g. `4111 1111 1111 1111 150`, a PAN followed by
/// an amount): the greedy regex grabbed all of them and a single Luhn check on
/// the combined digits fails, which previously let the real PAN leak unmasked.
/// Instead, slide a 13-19 digit window that begins at a digit-group boundary
/// (the run start or just after a separator) and emit the leftmost-longest
/// Luhn-valid, non-overlapping sub-windows. Anchoring window starts to group
/// boundaries keeps sequential filler like `1234 5678 9012 3456` from matching
/// a coincidental interior sub-window.
fn push_credit_card_spans(matched: &str, base: usize, out: &mut Vec<PiiSpan>) {
    if luhn_is_valid(matched) {
        out.push(PiiSpan::new(
            base,
            base + matched.len(),
            PiiCategory::CreditCard,
        ));
        return;
    }

    let bytes = matched.as_bytes();
    let digit_offsets: Vec<usize> = matched
        .bytes()
        .enumerate()
        .filter(|(_, b)| b.is_ascii_digit())
        .map(|(i, _)| i)
        .collect();
    let n = digit_offsets.len();

    let is_group_start = |di: usize| -> bool {
        let off = digit_offsets[di];
        off == 0 || !bytes[off - 1].is_ascii_digit()
    };

    let mut di = 0;
    while di < n {
        if !is_group_start(di) {
            di += 1;
            continue;
        }
        let max_len = (n - di).min(19);
        let mut emitted = None;
        if max_len >= 13 {
            for len in (13..=max_len).rev() {
                let start_off = digit_offsets[di];
                // ASCII digits are one byte, so `+ 1` lands on a char boundary.
                let end_off = digit_offsets[di + len - 1] + 1;
                if luhn_is_valid(&matched[start_off..end_off]) {
                    out.push(PiiSpan::new(
                        base + start_off,
                        base + end_off,
                        PiiCategory::CreditCard,
                    ));
                    emitted = Some(len);
                    break;
                }
            }
        }
        // Skip past an emitted window; otherwise advance one digit and retry.
        di += emitted.unwrap_or(1);
    }
}

// ─────────────────────────────────────────────────────────────────────
// CompositeDetector and BaselineDetector
// ─────────────────────────────────────────────────────────────────────

/// Aggregates multiple detectors, optionally deduplicating
/// overlapping spans.
#[derive(Debug)]
pub struct CompositeDetector {
    detectors: Vec<Box<dyn PiiDetector>>,
    dedup: bool,
}

impl CompositeDetector {
    #[must_use]
    pub fn new(detectors: Vec<Box<dyn PiiDetector>>) -> Self {
        Self {
            detectors,
            dedup: true,
        }
    }

    /// Disable overlap deduplication. Use when callers want every
    /// detector's raw output (e.g. for metrics or debugging).
    #[must_use]
    pub const fn without_dedup(mut self) -> Self {
        self.dedup = false;
        self
    }
}

impl PiiDetector for CompositeDetector {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        let mut spans: Vec<PiiSpan> = self.detectors.iter().flat_map(|d| d.detect(text)).collect();
        if self.dedup {
            dedup_overlapping(&mut spans);
        }
        spans
    }
}

/// Convenience composite of [`SecretDetector`] + [`EntityDetector`]
/// using default settings. Suitable as the SDK default detector.
#[derive(Debug)]
pub struct BaselineDetector {
    inner: CompositeDetector,
}

impl BaselineDetector {
    /// Construct the baseline detector.
    ///
    /// # Errors
    /// See [`SecretDetector::baseline`] / [`EntityDetector::baseline`].
    pub fn new() -> Result<Self, regex::Error> {
        let secrets: Box<dyn PiiDetector> = Box::new(SecretDetector::baseline()?);
        let entities: Box<dyn PiiDetector> = Box::new(EntityDetector::baseline()?);
        Ok(Self {
            inner: CompositeDetector::new(vec![secrets, entities]),
        })
    }
}

impl PiiDetector for BaselineDetector {
    fn detect(&self, text: &str) -> Vec<PiiSpan> {
        self.inner.detect(text)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Span utilities
// ─────────────────────────────────────────────────────────────────────

/// Sort spans by start (asc) then length (desc), drop empties, and **merge**
/// overlapping spans into a single covering interval.
///
/// The earlier span's category wins. Non-overlapping spans are preserved in
/// left-to-right order.
///
/// Merging (rather than dropping the later span, as a previous version did) is
/// a masking-safety requirement: a span that starts inside a kept span but
/// extends past its end would otherwise be discarded wholesale, leaving its
/// non-overlapping tail (`kept.end..span.end`) in cleartext. With custom
/// detectors composed via [`CompositeDetector`], that tail can be the bulk of
/// a credential or email — so it must be covered, not dropped.
pub fn dedup_overlapping(spans: &mut Vec<PiiSpan>) {
    spans.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.len().cmp(&a.len())));
    let mut kept: Vec<PiiSpan> = Vec::with_capacity(spans.len());
    for span in spans.drain(..) {
        if span.is_empty() {
            continue;
        }
        match kept.last_mut() {
            // Overlaps the previously kept span: extend it to cover both so no
            // tail leaks. The first span's category is retained.
            Some(prev) if prev.end > span.start => {
                prev.end = prev.end.max(span.end);
            }
            _ => kept.push(span),
        }
    }
    *spans = kept;
}

// ─────────────────────────────────────────────────────────────────────
// Masking helpers
// ─────────────────────────────────────────────────────────────────────

/// Replace every span in `text` with `[REDACTED:<category>]`.
#[must_use]
pub fn mask_spans(text: &str, spans: &[PiiSpan]) -> String {
    mask_with(text, spans, |span, _matched| {
        format!("[REDACTED:{}]", span.category.as_tag())
    })
}

/// Replace every span in `text` using a caller-provided masker.
///
/// The closure receives the span metadata plus the original matched
/// substring. Useful for format-preserving masking (e.g.
/// `****-****-****-1234` on PANs) or for reversible tokenization.
///
/// Overlapping spans are deduplicated via [`dedup_overlapping`]
/// before masking. Non-char-boundary offsets are silently skipped
/// to avoid panics.
#[must_use]
pub fn mask_with<F>(text: &str, spans: &[PiiSpan], f: F) -> String
where
    F: Fn(&PiiSpan, &str) -> String,
{
    // Fast path: the common production callers (baseline / composite
    // detectors) already hand us sorted, non-empty, non-overlapping spans, so
    // mask straight from the borrowed slice and skip the clone + sort + dedup.
    if spans_are_clean(spans) {
        return mask_sorted(text, spans, &f);
    }
    let mut sorted = spans.to_vec();
    dedup_overlapping(&mut sorted);
    mask_sorted(text, &sorted, &f)
}

/// Whether `spans` are already sorted by start, non-empty, and
/// non-overlapping (touching is allowed) — i.e. safe to mask without a
/// dedup pass.
fn spans_are_clean(spans: &[PiiSpan]) -> bool {
    spans.iter().all(|s| !s.is_empty()) && spans.windows(2).all(|w| w[0].end <= w[1].start)
}

/// Mask `text` using pre-sorted, non-overlapping `spans`.
///
/// A span whose start *or* end is not a UTF-8 char boundary is skipped
/// entirely: the matched slice is fetched **before** any output is written, so
/// a valid-start / invalid-end span never duplicates the prefix nor leaks the
/// bytes it was meant to mask.
fn mask_sorted<F>(text: &str, sorted: &[PiiSpan], f: &F) -> String
where
    F: Fn(&PiiSpan, &str) -> String,
{
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for span in sorted {
        if span.start < cursor {
            continue;
        }
        let (Some(prefix), Some(matched)) =
            (text.get(cursor..span.start), text.get(span.start..span.end))
        else {
            continue;
        };
        out.push_str(prefix);
        out.push_str(&f(span, matched));
        cursor = span.end;
    }
    if let Some(suffix) = text.get(cursor..) {
        out.push_str(suffix);
    }
    out
}

/// Format-preserving PAN mask: keeps the last four digits and
/// replaces the preceding digits with `*` grouped in fours.
#[must_use]
pub fn mask_pan(pan: &str) -> String {
    let digits: Vec<char> = pan.chars().filter(char::is_ascii_digit).collect();
    if digits.len() < 4 {
        return format!("[REDACTED:{}]", PiiCategory::CreditCard.as_tag());
    }
    let last_four: String = digits.iter().rev().take(4).rev().copied().collect();
    format!("****-****-****-{last_four}")
}

// ─────────────────────────────────────────────────────────────────────
// Check-digit validators (private)
// ─────────────────────────────────────────────────────────────────────

fn luhn_is_valid(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 0 {
                d
            } else {
                let doubled = d * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

fn cpf_is_valid(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() != 11 {
        return false;
    }
    if digits.iter().all(|&d| d == digits[0]) {
        return false;
    }
    let Some(first_nine) = digits.get(..9) else {
        return false;
    };
    let check1 = mod11_cpf_check(first_nine, 10);
    if digits.get(9) != Some(&check1) {
        return false;
    }
    let Some(first_ten) = digits.get(..10) else {
        return false;
    };
    let check2 = mod11_cpf_check(first_ten, 11);
    digits.get(10) == Some(&check2)
}

fn mod11_cpf_check(slice: &[u32], weight_start: u32) -> u32 {
    let weights = (0_u32..).map(|i| weight_start.saturating_sub(i));
    let sum: u32 = slice.iter().zip(weights).map(|(d, w)| d * w).sum();
    let rem = sum % 11;
    if rem < 2 { 0 } else { 11 - rem }
}

fn cnpj_is_valid(s: &str) -> bool {
    const WEIGHTS1: [u32; 12] = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
    const WEIGHTS2: [u32; 13] = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];

    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() != 14 {
        return false;
    }
    if digits.iter().all(|&d| d == digits[0]) {
        return false;
    }
    let Some(first_twelve) = digits.get(..12) else {
        return false;
    };
    let check1 = weighted_mod11(first_twelve, &WEIGHTS1);
    if digits.get(12) != Some(&check1) {
        return false;
    }
    let Some(first_thirteen) = digits.get(..13) else {
        return false;
    };
    let check2 = weighted_mod11(first_thirteen, &WEIGHTS2);
    digits.get(13) == Some(&check2)
}

fn weighted_mod11(slice: &[u32], weights: &[u32]) -> u32 {
    let sum: u32 = slice.iter().zip(weights.iter()).map(|(d, w)| d * w).sum();
    let rem = sum % 11;
    if rem < 2 { 0 } else { 11 - rem }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), regex::Error>;

    // ── PiiCategory ─────────────────────────────────────────────

    #[test]
    fn category_as_tag_returns_stable_strings() {
        assert_eq!(PiiCategory::Secret.as_tag(), "secret");
        assert_eq!(PiiCategory::Email.as_tag(), "email");
        assert_eq!(PiiCategory::Phone.as_tag(), "phone");
        assert_eq!(PiiCategory::CreditCard.as_tag(), "credit_card");
        assert_eq!(PiiCategory::Cpf.as_tag(), "cpf");
        assert_eq!(PiiCategory::Cnpj.as_tag(), "cnpj");
        assert_eq!(PiiCategory::PixKey.as_tag(), "pix_key");
        assert_eq!(PiiCategory::IpAddress.as_tag(), "ip_address");
        assert_eq!(PiiCategory::Jwt.as_tag(), "jwt");
        assert_eq!(PiiCategory::Custom("org_id".to_owned()).as_tag(), "org_id");
    }

    #[test]
    fn unit_category_serialises_as_snake_case_string() -> serde_json::Result<()> {
        let json = serde_json::to_string(&PiiCategory::Email)?;
        assert_eq!(json, r#""email""#);
        let back: PiiCategory = serde_json::from_str(&json)?;
        assert_eq!(back, PiiCategory::Email);
        Ok(())
    }

    #[test]
    fn custom_category_round_trips() -> serde_json::Result<()> {
        let original = PiiCategory::Custom("account_key".to_owned());
        let json = serde_json::to_string(&original)?;
        let back: PiiCategory = serde_json::from_str(&json)?;
        assert_eq!(back, original);
        Ok(())
    }

    // ── PiiSpan ─────────────────────────────────────────────────

    #[test]
    fn span_len_and_is_empty() {
        let s = PiiSpan::new(5, 10, PiiCategory::Email);
        assert_eq!(s.len(), 5);
        assert!(!s.is_empty());
        let z = PiiSpan::new(5, 5, PiiCategory::Email);
        assert!(z.is_empty());
    }

    #[test]
    fn span_overlaps_detects_shared_bytes_only() {
        let a = PiiSpan::new(0, 5, PiiCategory::Email);
        let b = PiiSpan::new(3, 8, PiiCategory::Email);
        let c = PiiSpan::new(5, 10, PiiCategory::Email);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c)); // Touching is not overlap
        assert!(!c.overlaps(&a));
    }

    // ── NoopDetector ────────────────────────────────────────────

    #[test]
    fn noop_detector_finds_nothing() {
        let d = NoopDetector;
        assert!(d.detect("sk-abc123 email a@b.co").is_empty());
    }

    // ── SecretDetector ──────────────────────────────────────────

    #[test]
    fn secret_detector_detects_common_prefixes() -> TestResult {
        let d = SecretDetector::baseline()?;
        let cases = [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
            "key=sk-abcdefghijklmnopqrstuv",
            "GH token ghp_abcdefghijklmnopqrstuvwxyz",
            "AWS AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234567890-slack",
            "GOOGLE_KEY=AIzaSyA-abcdefghijklmnopqrstuvwxyz123",
        ];
        for text in cases {
            let spans = d.detect(text);
            assert_eq!(spans.len(), 1, "expected 1 span in {text:?}, got {spans:?}");
            assert_eq!(spans[0].category, PiiCategory::Secret);
        }
        Ok(())
    }

    #[test]
    fn secret_detector_ignores_non_secret_text() -> TestResult {
        let d = SecretDetector::baseline()?;
        assert!(d.detect("just some ordinary prose").is_empty());
        assert!(d.detect("sk-short").is_empty()); // Below min body length
        Ok(())
    }

    // ── EntityDetector: email ────────────────────────────────────

    #[test]
    fn detects_email() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("please email me at ana.silva+tag@example.com tomorrow");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::Email);
        Ok(())
    }

    // ── EntityDetector: phone ────────────────────────────────────

    #[test]
    fn detects_e164_phone() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("call +5511987654321 for support");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::Phone);
        Ok(())
    }

    #[test]
    fn non_e164_phone_not_detected() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Phone))?;
        // Missing leading '+' — not E.164.
        assert!(d.detect("call 11987654321").is_empty());
        Ok(())
    }

    // ── EntityDetector: credit card (Luhn) ───────────────────────

    #[test]
    fn detects_luhn_valid_pan() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("card 4111 1111 1111 1111 expires soon");
        let pan_count = spans
            .iter()
            .filter(|s| s.category == PiiCategory::CreditCard)
            .count();
        assert_eq!(pan_count, 1);
        Ok(())
    }

    #[test]
    fn rejects_luhn_invalid_pan() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::CreditCard))?;
        // 16 digits but not Luhn-valid
        let spans = d.detect("card 1234 5678 9012 3456");
        assert!(spans.is_empty(), "Luhn-invalid PAN leaked: {spans:?}");
        Ok(())
    }

    #[test]
    fn detects_mastercard_test_pan() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::CreditCard))?;
        let spans = d.detect("5500-0000-0000-0004");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::CreditCard);
        Ok(())
    }

    // ── EntityDetector: CPF ─────────────────────────────────────

    #[test]
    fn detects_valid_cpf_formatted() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("meu CPF é 111.444.777-35 ok?");
        let cpf_count = spans
            .iter()
            .filter(|s| s.category == PiiCategory::Cpf)
            .count();
        assert_eq!(cpf_count, 1);
        Ok(())
    }

    #[test]
    fn detects_valid_cpf_unformatted() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Cpf))?;
        let spans = d.detect("cpf 11144477735 confere");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::Cpf);
        Ok(())
    }

    #[test]
    fn rejects_invalid_cpf() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Cpf))?;
        // 11 digits but wrong check digits
        assert!(d.detect("cpf 12345678900").is_empty());
        // All-same digits — rejected
        assert!(d.detect("cpf 11111111111").is_empty());
        // Invalid formatted
        assert!(d.detect("cpf 123.456.789-00").is_empty());
        Ok(())
    }

    // ── EntityDetector: CNPJ ────────────────────────────────────

    #[test]
    fn detects_valid_cnpj_formatted() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("CNPJ 11.222.333/0001-81 registered");
        let cnpj_count = spans
            .iter()
            .filter(|s| s.category == PiiCategory::Cnpj)
            .count();
        assert_eq!(cnpj_count, 1);
        Ok(())
    }

    #[test]
    fn detects_valid_cnpj_unformatted() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Cnpj))?;
        let spans = d.detect("cnpj 11222333000181 ok");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::Cnpj);
        Ok(())
    }

    #[test]
    fn rejects_invalid_cnpj() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Cnpj))?;
        assert!(d.detect("cnpj 12345678000100").is_empty());
        assert!(d.detect("cnpj 11111111111111").is_empty());
        Ok(())
    }

    // ── EntityDetector: Pix UUID ─────────────────────────────────

    #[test]
    fn detects_pix_uuid_key() -> TestResult {
        let d = EntityDetector::baseline()?;
        let spans = d.detect("pix key 123e4567-e89b-12d3-a456-426614174000 configurada");
        let pix_count = spans
            .iter()
            .filter(|s| s.category == PiiCategory::PixKey)
            .count();
        assert_eq!(pix_count, 1);
        Ok(())
    }

    // ── EntityDetector: IPv4 ────────────────────────────────────

    #[test]
    fn detects_ipv4() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Ipv4))?;
        let spans = d.detect("request from 192.168.1.100 blocked");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::IpAddress);
        Ok(())
    }

    #[test]
    fn rejects_out_of_range_ipv4_octets() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Ipv4))?;
        assert!(d.detect("999.999.999.999").is_empty());
        Ok(())
    }

    // ── EntityDetector: JWT ─────────────────────────────────────

    #[test]
    fn detects_jwt() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::Jwt))?;
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abc-_def";
        let spans = d.detect(&format!("token: {jwt} here"));
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].category, PiiCategory::Jwt);
        Ok(())
    }

    // ── EntityDetector: toggle set ──────────────────────────────

    #[test]
    fn disabled_categories_are_skipped() -> TestResult {
        let d = EntityDetector::new(CategorySet::none())?;
        assert!(d.detect("a@b.co and 111.444.777-35").is_empty());
        Ok(())
    }

    // ── CompositeDetector ──────────────────────────────────────

    #[test]
    fn composite_merges_detectors() -> TestResult {
        let secrets: Box<dyn PiiDetector> = Box::new(SecretDetector::baseline()?);
        let entities: Box<dyn PiiDetector> = Box::new(EntityDetector::baseline()?);
        let composite = CompositeDetector::new(vec![secrets, entities]);
        let text = "login=a@b.co key=sk-abcdefghijklmnopqrstuv";
        let spans = composite.detect(text);
        assert_eq!(spans.len(), 2);
        // Categories present, order by start.
        let categories: Vec<&PiiCategory> = spans.iter().map(|s| &s.category).collect();
        assert!(categories.contains(&&PiiCategory::Email));
        assert!(categories.contains(&&PiiCategory::Secret));
        Ok(())
    }

    #[test]
    fn composite_dedupes_overlapping_spans() {
        // Two detectors that both match the same bytes produce a
        // single output span after dedup.
        #[derive(Debug)]
        struct Always(PiiCategory);
        impl PiiDetector for Always {
            fn detect(&self, text: &str) -> Vec<PiiSpan> {
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![PiiSpan::new(0, text.len(), self.0.clone())]
                }
            }
        }

        let composite = CompositeDetector::new(vec![
            Box::new(Always(PiiCategory::Email)),
            Box::new(Always(PiiCategory::Secret)),
        ]);
        let spans = composite.detect("hello");
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn composite_without_dedup_preserves_overlaps() {
        #[derive(Debug)]
        struct Always(PiiCategory);
        impl PiiDetector for Always {
            fn detect(&self, text: &str) -> Vec<PiiSpan> {
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![PiiSpan::new(0, text.len(), self.0.clone())]
                }
            }
        }

        let composite = CompositeDetector::new(vec![
            Box::new(Always(PiiCategory::Email)),
            Box::new(Always(PiiCategory::Secret)),
        ])
        .without_dedup();
        let spans = composite.detect("hello");
        assert_eq!(spans.len(), 2);
    }

    // ── BaselineDetector ──────────────────────────────────────

    #[test]
    fn baseline_finds_mixed_pii() -> TestResult {
        let d = BaselineDetector::new()?;
        let text = "email: a@b.co, CPF: 111.444.777-35, key: sk-abcdefghijklmnopqrstuv";
        let mut spans = d.detect(text);
        spans.sort_by_key(|s| s.start);
        let kinds: Vec<&PiiCategory> = spans.iter().map(|s| &s.category).collect();
        assert_eq!(
            kinds,
            vec![&PiiCategory::Email, &PiiCategory::Cpf, &PiiCategory::Secret,]
        );
        Ok(())
    }

    // ── dedup_overlapping ─────────────────────────────────────

    #[test]
    fn dedup_keeps_longest_on_overlap() {
        let mut spans = vec![
            PiiSpan::new(0, 5, PiiCategory::Email),
            PiiSpan::new(0, 8, PiiCategory::Secret), // longer, same start
            PiiSpan::new(10, 15, PiiCategory::Phone),
        ];
        dedup_overlapping(&mut spans);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].category, PiiCategory::Secret);
        assert_eq!(spans[1].category, PiiCategory::Phone);
    }

    #[test]
    fn dedup_drops_empty_spans() {
        let mut spans = vec![
            PiiSpan::new(5, 5, PiiCategory::Email),
            PiiSpan::new(10, 15, PiiCategory::Phone),
        ];
        dedup_overlapping(&mut spans);
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn dedup_preserves_non_overlapping() {
        let mut spans = vec![
            PiiSpan::new(0, 3, PiiCategory::Email),
            PiiSpan::new(5, 8, PiiCategory::Phone),
            PiiSpan::new(10, 15, PiiCategory::Cpf),
        ];
        dedup_overlapping(&mut spans);
        assert_eq!(spans.len(), 3);
    }

    // ── mask_spans ────────────────────────────────────────────

    #[test]
    fn mask_spans_produces_type_tagged_markers() -> TestResult {
        let d = BaselineDetector::new()?;
        let text = "email a@b.co please";
        let spans = d.detect(text);
        let masked = mask_spans(text, &spans);
        assert_eq!(masked, "email [REDACTED:email] please");
        Ok(())
    }

    #[test]
    fn mask_spans_preserves_text_without_pii() -> TestResult {
        let d = BaselineDetector::new()?;
        let text = "no pii here, just prose";
        let masked = mask_spans(text, &d.detect(text));
        assert_eq!(masked, text);
        Ok(())
    }

    #[test]
    fn mask_spans_handles_multiple_spans_in_order() -> TestResult {
        let d = BaselineDetector::new()?;
        let text = "a@b.co then c@d.co";
        let masked = mask_spans(text, &d.detect(text));
        assert_eq!(masked, "[REDACTED:email] then [REDACTED:email]");
        Ok(())
    }

    // ── mask_with ─────────────────────────────────────────────

    #[test]
    fn mask_with_supports_format_preserving_pan_mask() -> TestResult {
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::CreditCard))?;
        let text = "card 4111 1111 1111 1111 thanks";
        let spans = d.detect(text);
        let masked = mask_with(text, &spans, |span, matched| {
            if span.category == PiiCategory::CreditCard {
                mask_pan(matched)
            } else {
                format!("[REDACTED:{}]", span.category.as_tag())
            }
        });
        assert!(masked.contains("****-****-****-1111"), "got: {masked}");
        Ok(())
    }

    #[test]
    fn mask_with_skips_non_boundary_spans_silently() {
        // Multi-byte char at bytes 0..2 (é), span landing inside
        // that char should be skipped rather than panic.
        let text = "é abc";
        let spans = vec![PiiSpan::new(1, 3, PiiCategory::Email)];
        let masked = mask_with(text, &spans, |_, _| "X".to_owned());
        // Either unchanged or prefix kept — what matters is no panic.
        assert!(masked.contains("abc"));
    }

    #[test]
    fn mask_with_skips_span_with_valid_start_invalid_end() {
        // `span.start` is a char boundary (between 'a' and 'b') but `span.end`
        // lands inside the multi-byte 'é'. The span must be skipped without
        // duplicating the already-emitted prefix or leaking PII bytes.
        let text = "ab é"; // bytes: a=0 b=1 ' '=2 é=3..5
        let spans = vec![PiiSpan::new(1, 4, PiiCategory::Email)];
        let masked = mask_with(text, &spans, |_, _| "X".to_owned());
        assert_eq!(masked, "ab é");
    }

    // ── credit-card sliding window (PAN + trailing digits) ─────

    #[test]
    fn detects_pan_followed_by_trailing_digits() -> TestResult {
        // A valid 16-digit PAN followed by an amount: the greedy regex grabs
        // all 19 digits and a single Luhn check on the run fails. The real PAN
        // is a Luhn-valid sub-window and must still be masked, not leaked.
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::CreditCard))?;
        let text = "card 4111 1111 1111 1111 150";
        let spans = d.detect(text);
        let pan_spans = spans
            .iter()
            .filter(|s| s.category == PiiCategory::CreditCard)
            .count();
        assert_eq!(pan_spans, 1, "expected the embedded PAN: {spans:?}");
        let masked = mask_spans(text, &spans);
        assert!(
            !masked.contains("4111 1111 1111 1111"),
            "PAN leaked: {masked}"
        );
        assert!(masked.contains("[REDACTED:credit_card]"), "got: {masked}");
        Ok(())
    }

    #[test]
    fn sequential_filler_digits_do_not_false_positive() -> TestResult {
        // Sliding must not flag a coincidental interior Luhn-valid sub-window
        // in obviously-sequential filler that is not a PAN.
        let d = EntityDetector::new(CategorySet::none().with(DetectCategory::CreditCard))?;
        assert!(d.detect("order 1234 5678 9012 3456 processed").is_empty());
        Ok(())
    }

    #[test]
    fn entity_detector_clean_string_via_prefilter() -> TestResult {
        // The RegexSet prefilter short-circuits a string with no PII.
        let d = EntityDetector::baseline()?;
        assert!(
            d.detect("a perfectly ordinary sentence with no pii")
                .is_empty()
        );
        Ok(())
    }

    // ── dedup_overlapping: merge (no leaked tail) ──────────────

    #[test]
    fn dedup_merges_overlapping_tail_instead_of_dropping() {
        // A later span that starts inside the kept span but extends past its
        // end must be merged (covering interval), not dropped — otherwise the
        // tail bytes stay unmasked.
        let mut spans = vec![
            PiiSpan::new(0, 6, PiiCategory::Secret),
            PiiSpan::new(4, 12, PiiCategory::Email),
        ];
        dedup_overlapping(&mut spans);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 12, "tail must be covered, not dropped");
        assert_eq!(spans[0].category, PiiCategory::Secret);
    }

    #[test]
    fn mask_with_overlapping_spans_leaks_no_tail() {
        // Concrete leak case: a secret span overlapping an email span. The
        // overlapping tail ('@example.com') must not survive masking.
        let text = "Bearer abc123def.ana@example.com";
        // 'Bearer abc123def.ana' = bytes 0..20, 'ana@example.com' = 17..32.
        let spans = vec![
            PiiSpan::new(0, 20, PiiCategory::Secret),
            PiiSpan::new(17, 32, PiiCategory::Email),
        ];
        let masked = mask_spans(text, &spans);
        assert!(!masked.contains("@example.com"), "tail leaked: {masked}");
    }

    // ── mask_pan ──────────────────────────────────────────────

    #[test]
    fn mask_pan_keeps_last_four() {
        assert_eq!(mask_pan("4111 1111 1111 1111"), "****-****-****-1111");
        assert_eq!(mask_pan("4111111111111111"), "****-****-****-1111");
        assert_eq!(mask_pan("4111-1111-1111-1234"), "****-****-****-1234");
    }

    #[test]
    fn mask_pan_falls_back_for_too_few_digits() {
        assert_eq!(mask_pan("abc"), "[REDACTED:credit_card]");
        assert_eq!(mask_pan("12"), "[REDACTED:credit_card]");
    }

    // ── Luhn, CPF, CNPJ validators (edge cases) ───────────────

    #[test]
    fn luhn_rejects_wrong_length() {
        assert!(!luhn_is_valid("1234567890"));
        assert!(!luhn_is_valid("12345678901234567890"));
    }

    #[test]
    fn cpf_validator_accepts_known_good() {
        assert!(cpf_is_valid("111.444.777-35"));
        assert!(cpf_is_valid("11144477735"));
    }

    #[test]
    fn cnpj_validator_accepts_known_good() {
        assert!(cnpj_is_valid("11.222.333/0001-81"));
        assert!(cnpj_is_valid("11222333000181"));
    }
}
