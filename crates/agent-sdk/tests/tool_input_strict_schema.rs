//! Every built-in tool must accept the payload a model sends through
//! `OpenAI` strict mode, where each optional property becomes a required,
//! nullable one and is sent as `null` when the model leaves it unset.

#![cfg(any(feature = "openai", feature = "openai-codex"))]

use std::sync::Arc;

use agent_sdk::builtin_tools::{BuiltinToolsConfig, register_builtin_tools};
use agent_sdk::todo::TodoState;
use agent_sdk::{
    AgentCapabilities, AskUserQuestionTool, InMemoryFileSystem, ToolContext, ToolRegistry,
    ToolResult,
};
use agent_sdk_providers::impls::openai_schema::normalized_strict_schema;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use tokio::sync::RwLock;

fn accepts_null(schema: &Value) -> bool {
    let type_is_null = schema.get("type").is_some_and(|kind| match kind {
        Value::String(kind) => kind == "null",
        Value::Array(kinds) => kinds.iter().any(|kind| kind == "null"),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_) => false,
    });
    type_is_null
        || schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|variants| variants.iter().any(accepts_null))
}

/// Build a value for `schema` that sets every nullable property to `null`
/// and fills everything else with the smallest valid value, recording the
/// path of each property set to `null`.
fn null_payload(schema: &Value, path: &str, nulled: &mut Vec<String>) -> Result<Value> {
    if let Some(variants) = schema.get("anyOf").and_then(Value::as_array) {
        let first = variants
            .iter()
            .find(|variant| !accepts_null(variant))
            .with_context(|| format!("{path}: anyOf has no non-null variant"))?;
        return null_payload(first, path, nulled);
    }
    if let Some(first) = schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
    {
        return Ok(first.clone());
    }
    let kind = match schema.get("type") {
        Some(Value::String(kind)) => kind.as_str(),
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .find(|kind| *kind != "null")
            .with_context(|| format!("{path}: type list has no non-null type"))?,
        other => bail!("{path}: unsupported type {other:?}"),
    };
    match kind {
        "string" => Ok(json!("x")),
        "integer" | "number" => Ok(json!(1)),
        "boolean" => Ok(json!(false)),
        "array" => {
            let items = schema
                .get("items")
                .with_context(|| format!("{path}: array without items"))?;
            Ok(Value::Array(vec![null_payload(
                items,
                &format!("{path}[]"),
                nulled,
            )?]))
        }
        "object" => {
            let mut object = Map::new();
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (name, property) in properties {
                    let property_path = format!("{path}.{name}");
                    let value = if accepts_null(property) {
                        nulled.push(property_path);
                        Value::Null
                    } else {
                        null_payload(property, &property_path, nulled)?
                    };
                    object.insert(name.clone(), value);
                }
            }
            Ok(Value::Object(object))
        }
        other => bail!("{path}: unsupported type {other}"),
    }
}

#[cfg(feature = "web")]
struct EmptySearch;

#[cfg(feature = "web")]
#[async_trait::async_trait]
impl agent_sdk::web::SearchProvider for EmptySearch {
    async fn search(
        &self,
        query: &str,
        _max_results: usize,
    ) -> Result<agent_sdk::web::SearchResponse> {
        Ok(agent_sdk::web::SearchResponse {
            query: query.to_owned(),
            results: Vec::new(),
            total_results: None,
        })
    }

    fn provider_name(&self) -> &'static str {
        "empty"
    }
}

fn builtin_registry() -> ToolRegistry<()> {
    let mut registry = ToolRegistry::new();
    register_builtin_tools(
        &mut registry,
        BuiltinToolsConfig {
            environment: Arc::new(InMemoryFileSystem::new("/workspace")),
            capabilities: AgentCapabilities::full_access(),
            todo_state: Some(Arc::new(RwLock::new(TodoState::new()))),
            #[cfg(feature = "web")]
            link_fetch: true,
        },
    );
    let (ask_user, _questions, _answers) = AskUserQuestionTool::with_channels(1);
    registry.register(ask_user);
    #[cfg(feature = "web")]
    agent_sdk::builtin_tools::register_web_search(&mut registry, EmptySearch);
    registry
}

/// Input decoding is the only place these tools produce a `serde_json`
/// error, so one anywhere in the chain means the payload was refused.
fn rejected_as_input(outcome: &Result<ToolResult>) -> bool {
    outcome.as_ref().is_err_and(|error| {
        error
            .chain()
            .any(|cause| cause.downcast_ref::<serde_json::Error>().is_some())
    })
}

#[tokio::test]
async fn every_builtin_tool_accepts_null_for_each_strict_nullable_field() -> Result<()> {
    let registry = builtin_registry();
    let ctx = ToolContext::new(());

    let read = registry.get("read").context("read is registered")?;
    let malformed = read.execute(&ctx, json!({"path": 1})).await;
    ensure!(
        rejected_as_input(&malformed),
        "a malformed payload must be detected as an input rejection"
    );

    let mut nulled = Vec::new();
    let mut failures = Vec::new();
    let mut sent_without_strict = Vec::new();
    for tool in registry.all() {
        let name = tool.name_str().to_owned();
        let Ok(schema) = normalized_strict_schema(&tool.input_schema()) else {
            sent_without_strict.push(name);
            continue;
        };
        let payload = null_payload(&schema, &name, &mut nulled)?;
        let outcome = tool.execute(&ctx, payload.clone()).await;
        if rejected_as_input(&outcome) {
            failures.push(format!("{name}: {payload} -> {outcome:?}"));
        }
    }
    ensure!(
        failures.is_empty(),
        "tools refused null optional fields:\n{}",
        failures.join("\n")
    );

    ensure!(
        sent_without_strict == ["todo_read"],
        "only the argument-less todo_read may skip strict mode, \
         which leaves its schema without nullable fields; skipped: {sent_without_strict:?}"
    );

    for expected in [
        "read.offset",
        "read.limit",
        "edit.replace_all",
        "grep.path",
        "grep.recursive",
        "grep.case_insensitive",
        "glob.path",
        "bash.timeout_ms",
        "ask_user.header",
        "ask_user.options",
        "ask_user.multi_select",
    ] {
        ensure!(
            nulled.iter().any(|path| path == expected),
            "{expected} was not exercised; nulled fields: {nulled:?}"
        );
    }
    Ok(())
}
