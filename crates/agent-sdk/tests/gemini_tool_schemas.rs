//! Gemini parses `functionDeclarations[].parameters` as its `Schema` proto,
//! which has one `type` per node and string-only `enum` values, and 400s the
//! whole request on any field the proto lacks. Every built-in tool, plus the
//! tool schemas bip sends (`tests/fixtures/bip_tool_schemas.json`, whose
//! `bash` replaces the built-in one as it does in bip), must reach the request
//! body in a shape the proto parses.

#![cfg(feature = "gemini")]

use std::sync::Arc;

use agent_sdk::builtin_tools::{BuiltinToolsConfig, register_builtin_tools};
use agent_sdk::llm::{ChatRequest, LlmProvider, Message, Tool};
use agent_sdk::todo::TodoState;
use agent_sdk::{AgentCapabilities, AskUserQuestionTool, InMemoryFileSystem, ToolRegistry};
use agent_sdk_foundation::{ToolConcurrency, ToolTier};
use agent_sdk_providers::GeminiProvider;
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};
use tokio::sync::RwLock;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

const BIP_TOOL_SCHEMAS: &str = include_str!("fixtures/bip_tool_schemas.json");

/// Fields the `Schema` proto has no slot for; each one 400s the request.
const REJECTED_KEYWORDS: [&str; 15] = [
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "definitions",
    "additionalProperties",
    "const",
    "$comment",
    "examples",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "uniqueItems",
    "prefixItems",
    "deprecated",
    "readOnly",
];

fn builtin_tools() -> Vec<Tool> {
    let mut registry: ToolRegistry<()> = ToolRegistry::new();
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
    registry.to_llm_tools()
}

fn bip_tools() -> Result<Vec<Tool>> {
    let schemas: Map<String, Value> =
        serde_json::from_str(BIP_TOOL_SCHEMAS).context("bip tool schema fixture parses")?;
    Ok(schemas
        .into_iter()
        .map(|(name, input_schema)| Tool {
            display_name: name.clone(),
            description: format!("bip `{name}` tool"),
            name,
            input_schema,
            tier: ToolTier::Observe,
            concurrency: ToolConcurrency::Exclusive,
        })
        .collect())
}

fn has_type_list(value: &Value) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, child)| (key == "type" && child.is_array()) || has_type_list(child)),
        Value::Array(items) => items.iter().any(has_type_list),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

/// Walk the schema positions of one parameters node, collecting every
/// construct the `Schema` proto rejects.
fn rejected_constructs(schema: &Value, path: &str, found: &mut Vec<String>) {
    let Value::Object(node) = schema else {
        found.push(format!("{path}: schema is not an object"));
        return;
    };
    for key in REJECTED_KEYWORDS {
        if node.contains_key(key) {
            found.push(format!("{path}.{key}"));
        }
    }
    if let Some(kind) = node.get("type")
        && !kind.is_string()
    {
        found.push(format!("{path}.type = {kind}"));
    }
    if let Some(Value::Array(values)) = node.get("enum")
        && values
            .iter()
            .any(|value| !matches!(value, Value::String(_) | Value::Null))
    {
        found.push(format!("{path}.enum has a non-string value"));
    }
    if let Some(Value::Object(properties)) = node.get("properties") {
        for (name, property) in properties {
            rejected_constructs(property, &format!("{path}.properties.{name}"), found);
        }
    }
    if let Some(items) = node.get("items") {
        rejected_constructs(items, &format!("{path}.items"), found);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(branches)) = node.get(key) {
            for (index, branch) in branches.iter().enumerate() {
                rejected_constructs(branch, &format!("{path}.{key}[{index}]"), found);
            }
        }
    }
}

async fn gemini_request_body(tools: Vec<Tool>) -> Result<Value> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "ok" }] },
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;
    let provider = GeminiProvider::new("test-key", "gemini-4-argon").with_base_url(server.uri());
    let mut request = ChatRequest::new("system", vec![Message::user("hi")]);
    request.tools = Some(tools);
    provider.chat(request).await.context("mock chat succeeds")?;
    let requests = server
        .received_requests()
        .await
        .context("wiremock records requests")?;
    let sent = requests.first().context("one request was sent")?;
    serde_json::from_slice(&sent.body).context("request body is JSON")
}

#[tokio::test]
async fn every_sent_tool_schema_parses_as_the_gemini_schema_proto() -> Result<()> {
    let bip = bip_tools()?;
    ensure!(
        bip.iter().any(|tool| has_type_list(&tool.input_schema)),
        "the bip fixtures must carry the type lists this test guards against"
    );
    let mut tools = builtin_tools();
    tools.retain(|builtin| !bip.iter().any(|tool| tool.name == builtin.name));
    tools.extend(bip);
    let expected = tools.len();

    let body = gemini_request_body(tools).await?;

    let declarations = body
        .pointer("/tools/0/functionDeclarations")
        .and_then(Value::as_array)
        .context("request carries function declarations")?;
    ensure!(
        declarations.len() == expected,
        "sent {} declarations, expected {expected}",
        declarations.len()
    );
    let mut found = Vec::new();
    for declaration in declarations {
        let name = declaration
            .get("name")
            .and_then(Value::as_str)
            .context("declaration has a name")?;
        let parameters = declaration
            .get("parameters")
            .with_context(|| format!("{name} has parameters"))?;
        rejected_constructs(parameters, name, &mut found);
    }
    ensure!(
        found.is_empty(),
        "rejected constructs reached Gemini: {found:#?}"
    );
    Ok(())
}

#[tokio::test]
async fn bip_nullable_arrays_are_sent_as_nullable() -> Result<()> {
    let body = gemini_request_body(bip_tools()?).await?;

    let declarations = body
        .pointer("/tools/0/functionDeclarations")
        .and_then(Value::as_array)
        .context("request carries function declarations")?;
    let worker = declarations
        .iter()
        .find(|declaration| declaration["name"] == "subagent_worker")
        .context("subagent_worker is declared")?;
    let inputs = worker
        .pointer("/parameters/properties/unit/properties/inputs")
        .context("unit.inputs is sent")?;
    ensure!(inputs["type"] == "array", "inputs type: {inputs}");
    ensure!(inputs["nullable"] == true, "inputs nullable: {inputs}");
    ensure!(
        inputs["items"] == json!({ "type": "string" }),
        "inputs items: {inputs}"
    );
    Ok(())
}
