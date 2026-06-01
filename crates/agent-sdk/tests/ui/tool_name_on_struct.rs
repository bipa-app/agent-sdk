//! `#[derive(ToolName)]` is only valid on enums; a struct must be rejected.

#[derive(agent_sdk::ToolName)]
struct NotAnEnum;

fn main() {}
