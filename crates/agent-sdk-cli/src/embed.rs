//! Compose stack and docs embedded at compile time.
//!
//! The CLI is a thin distribution channel for the langfuse observability
//! stack. The files are embedded via `include_str!` so the binary is
//! self-contained. They live under this crate's `embedded/` directory
//! (NOT the workspace `dev/` tree) so they are packaged into the published
//! `.crate` tarball — `include_str!` cannot reach files outside the crate
//! root, which is what a published crate compiles against. The copies under
//! `embedded/` mirror `dev/observability/langfuse/` and
//! `crates/agent-sdk/docs/observability/LANGFUSE.md` — keep the `embedded/`
//! copies in sync with those sources when they change.

pub const COMPOSE_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/embedded/langfuse/docker-compose.yml"
));

pub const COLLECTOR_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/embedded/langfuse/otel-collector.yaml"
));

pub const LANGFUSE_DOC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/embedded/LANGFUSE.md"
));

pub const COMPOSE_FILENAME: &str = "docker-compose.yml";
pub const COLLECTOR_FILENAME: &str = "otel-collector.yaml";
pub const DOC_FILENAME: &str = "LANGFUSE.md";
pub const DEFAULT_DEST_REL: &str = "dev/observability/langfuse";
