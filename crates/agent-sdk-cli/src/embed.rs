//! Compose stack and docs embedded at compile time.
//!
//! The CLI is a thin distribution channel for files that already live
//! in the SDK workspace under `dev/observability/langfuse/` and
//! `crates/agent-sdk/docs/observability/`. `include_str!` makes the
//! binary self-contained while keeping a single source of truth: a
//! `cargo install --git …` build pulls the files in via the source
//! tree at build time, and the CLI never drifts from the checked-in
//! compose.

pub const COMPOSE_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../dev/observability/langfuse/docker-compose.yml"
));

pub const COLLECTOR_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../dev/observability/langfuse/otel-collector.yaml"
));

pub const LANGFUSE_DOC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../agent-sdk/docs/observability/LANGFUSE.md"
));

pub const COMPOSE_FILENAME: &str = "docker-compose.yml";
pub const COLLECTOR_FILENAME: &str = "otel-collector.yaml";
pub const DOC_FILENAME: &str = "LANGFUSE.md";
pub const DEFAULT_DEST_REL: &str = "dev/observability/langfuse";
