//! Sampler selection and resolution.
//!
//! Mirrors the `OTel` SDK environment-variable spec for `OTEL_TRACES_SAMPLER` /
//! `OTEL_TRACES_SAMPLER_ARG` so callers can configure sampling without writing
//! Rust code.
//!
//! See <https://opentelemetry.io/docs/languages/sdk-configuration/general-sdk-configuration/#otel_traces_sampler>.

use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use opentelemetry_sdk::trace::Sampler;

/// The set of samplers the bootstrap helper knows how to install.
///
/// Variants map 1:1 to the values the `OTel` spec defines for
/// `OTEL_TRACES_SAMPLER`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    /// Always sample. Maps to `always_on`.
    AlwaysOn,
    /// Never sample. Maps to `always_off`.
    AlwaysOff,
    /// Sample at a fixed ratio. Maps to `traceidratio`.
    TraceIdRatio,
    /// Defer to the parent context; root spans are always sampled.
    /// Maps to `parentbased_always_on` (the spec default).
    ParentBasedAlwaysOn,
    /// Defer to the parent context; root spans are never sampled.
    /// Maps to `parentbased_always_off`.
    ParentBasedAlwaysOff,
    /// Defer to the parent context; root spans use a ratio sampler.
    /// Maps to `parentbased_traceidratio`.
    #[default]
    ParentBasedTraceIdRatio,
}

impl SamplerKind {
    /// Lower-case canonical string used by the `OTel` env spec.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AlwaysOn => "always_on",
            Self::AlwaysOff => "always_off",
            Self::TraceIdRatio => "traceidratio",
            Self::ParentBasedAlwaysOn => "parentbased_always_on",
            Self::ParentBasedAlwaysOff => "parentbased_always_off",
            Self::ParentBasedTraceIdRatio => "parentbased_traceidratio",
        }
    }

    /// Whether the sampler honours the `sample_ratio` argument.
    #[must_use]
    pub const fn uses_ratio(self) -> bool {
        matches!(self, Self::TraceIdRatio | Self::ParentBasedTraceIdRatio)
    }
}

impl fmt::Display for SamplerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SamplerKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim() {
            "always_on" => Ok(Self::AlwaysOn),
            "always_off" => Ok(Self::AlwaysOff),
            "traceidratio" => Ok(Self::TraceIdRatio),
            "parentbased_always_on" => Ok(Self::ParentBasedAlwaysOn),
            "parentbased_always_off" => Ok(Self::ParentBasedAlwaysOff),
            "parentbased_traceidratio" => Ok(Self::ParentBasedTraceIdRatio),
            other => bail!(
                "unknown OTEL_TRACES_SAMPLER value `{other}`; expected one of \
                 always_on, always_off, traceidratio, parentbased_always_on, \
                 parentbased_always_off, parentbased_traceidratio"
            ),
        }
    }
}

/// Resolve a [`Sampler`] from a [`SamplerKind`] + ratio argument.
///
/// `ratio` is only consulted for ratio-based samplers; for the others it is
/// validated only well enough to surface obviously-broken configuration
/// (NaN or out-of-range values) so callers fail fast rather than silently
/// installing the wrong sampler.
///
/// # Errors
/// Returns an error if `ratio` is not in `[0.0, 1.0]` or is not finite when
/// the chosen sampler actually uses it.
pub fn resolve(kind: SamplerKind, ratio: f64) -> Result<Sampler> {
    if kind.uses_ratio() {
        validate_ratio(ratio)
            .with_context(|| format!("invalid sample ratio `{ratio}` for sampler `{kind}`"))?;
    }

    Ok(match kind {
        SamplerKind::AlwaysOn => Sampler::AlwaysOn,
        SamplerKind::AlwaysOff => Sampler::AlwaysOff,
        SamplerKind::TraceIdRatio => Sampler::TraceIdRatioBased(ratio),
        SamplerKind::ParentBasedAlwaysOn => Sampler::ParentBased(Box::new(Sampler::AlwaysOn)),
        SamplerKind::ParentBasedAlwaysOff => Sampler::ParentBased(Box::new(Sampler::AlwaysOff)),
        SamplerKind::ParentBasedTraceIdRatio => {
            Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(ratio)))
        }
    })
}

fn validate_ratio(ratio: f64) -> Result<()> {
    if !ratio.is_finite() {
        bail!("ratio must be finite (got {ratio})");
    }
    if !(0.0..=1.0).contains(&ratio) {
        bail!("ratio must be between 0.0 and 1.0 inclusive (got {ratio})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_canonical_name() -> Result<()> {
        assert_eq!(SamplerKind::from_str("always_on")?, SamplerKind::AlwaysOn);
        assert_eq!(SamplerKind::from_str("always_off")?, SamplerKind::AlwaysOff);
        assert_eq!(
            SamplerKind::from_str("traceidratio")?,
            SamplerKind::TraceIdRatio
        );
        assert_eq!(
            SamplerKind::from_str("parentbased_always_on")?,
            SamplerKind::ParentBasedAlwaysOn
        );
        assert_eq!(
            SamplerKind::from_str("parentbased_always_off")?,
            SamplerKind::ParentBasedAlwaysOff
        );
        assert_eq!(
            SamplerKind::from_str("parentbased_traceidratio")?,
            SamplerKind::ParentBasedTraceIdRatio
        );
        Ok(())
    }

    #[test]
    fn unknown_sampler_returns_error() {
        let Err(err) = SamplerKind::from_str("never") else {
            panic!("must error on unknown sampler");
        };
        let msg = format!("{err}");
        assert!(msg.contains("unknown OTEL_TRACES_SAMPLER"), "got: {msg}");
    }

    #[test]
    fn ratio_validation_rejects_out_of_range() {
        assert!(resolve(SamplerKind::TraceIdRatio, 1.5).is_err());
        assert!(resolve(SamplerKind::TraceIdRatio, -0.1).is_err());
        assert!(resolve(SamplerKind::TraceIdRatio, f64::NAN).is_err());
        assert!(resolve(SamplerKind::ParentBasedTraceIdRatio, f64::INFINITY).is_err());
    }

    #[test]
    fn ratio_validation_accepts_endpoints() -> Result<()> {
        let _ = resolve(SamplerKind::TraceIdRatio, 0.0)?;
        let _ = resolve(SamplerKind::TraceIdRatio, 1.0)?;
        let _ = resolve(SamplerKind::ParentBasedTraceIdRatio, 0.05)?;
        Ok(())
    }

    #[test]
    fn non_ratio_sampler_ignores_ratio() -> Result<()> {
        let _ = resolve(SamplerKind::AlwaysOn, f64::NAN)?;
        let _ = resolve(SamplerKind::ParentBasedAlwaysOff, -1.0)?;
        Ok(())
    }
}
