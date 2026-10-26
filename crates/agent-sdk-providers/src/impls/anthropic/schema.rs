//! The schema Anthropic's JSON outputs (`output_config.format`) receive.
//!
//! JSON outputs accept a subset of JSON Schema: objects must set
//! `additionalProperties: false`, and numeric, string-length and array-size
//! constraints are refused with a 400
//! (<https://platform.claude.com/docs/en/build-with-claude/structured-outputs>).
//! Anthropic's SDKs transform a schema before sending it: they drop each
//! unsupported constraint and note it in the field's description, close every
//! object, and keep only the supported string formats. This module does the
//! same. The caller's schema is unchanged, so a caller that validates the
//! response (the structured-output runner does) still enforces every
//! constraint.

use std::fmt;

use serde_json::{Map, Value};

/// Keywords JSON outputs refuse, which the transform drops and describes.
const UNSUPPORTED_CONSTRAINTS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
];

/// String formats JSON outputs accept.
const SUPPORTED_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Keywords whose value is one subschema.
const SUBSCHEMA_KEYWORDS: &[&str] = &["items", "not", "if", "then", "else", "contains"];

/// Keywords whose value is a list of subschemas.
const SUBSCHEMA_LIST_KEYWORDS: &[&str] = &["prefixItems", "anyOf", "allOf", "oneOf"];

/// Keywords whose value maps names to subschemas.
const SUBSCHEMA_MAP_KEYWORDS: &[&str] = &["properties", "$defs", "definitions"];

/// Why a schema cannot reach JSON outputs without changing its meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JsonOutputSchemaError {
    /// An object accepts property names it does not list
    /// (`additionalProperties` other than `false`). JSON outputs accept only
    /// closed objects.
    OpenObject,
    /// An object describes property names with `patternProperties`.
    PatternProperties,
}

impl fmt::Display for JsonOutputSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenObject => formatter.write_str(
                "Anthropic JSON outputs accept only closed objects (additionalProperties: false)",
            ),
            Self::PatternProperties => {
                formatter.write_str("Anthropic JSON outputs do not accept patternProperties")
            }
        }
    }
}

impl std::error::Error for JsonOutputSchemaError {}

/// Return the schema to send as `output_config.format.schema`.
///
/// # Errors
///
/// Returns a [`JsonOutputSchemaError`] when an object is open or uses
/// `patternProperties`: closing it would change what it accepts.
pub fn json_output_schema(schema: &Value) -> Result<Value, JsonOutputSchemaError> {
    let mut wire = schema.clone();
    transform(&mut wire)?;
    Ok(wire)
}

fn transform(node: &mut Value) -> Result<(), JsonOutputSchemaError> {
    let Value::Object(map) = node else {
        return Ok(());
    };
    close_object(map)?;
    let mut dropped = Map::new();
    for keyword in UNSUPPORTED_CONSTRAINTS {
        if let Some(value) = map.remove(*keyword) {
            dropped.insert((*keyword).to_owned(), value);
        }
    }
    if map
        .get("minItems")
        .and_then(Value::as_u64)
        .is_some_and(|min| min > 1)
        && let Some(value) = map.remove("minItems")
    {
        dropped.insert("minItems".to_owned(), value);
    }
    if map
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|format| !SUPPORTED_FORMATS.contains(&format))
        && let Some(value) = map.remove("format")
    {
        dropped.insert("format".to_owned(), value);
    }
    if !dropped.is_empty() {
        describe(map, &Value::Object(dropped));
    }
    for keyword in SUBSCHEMA_KEYWORDS {
        match map.get_mut(*keyword) {
            Some(Value::Array(list)) => list.iter_mut().try_for_each(transform)?,
            Some(subschema) => transform(subschema)?,
            None => {}
        }
    }
    for keyword in SUBSCHEMA_LIST_KEYWORDS {
        if let Some(Value::Array(list)) = map.get_mut(*keyword) {
            list.iter_mut().try_for_each(transform)?;
        }
    }
    for keyword in SUBSCHEMA_MAP_KEYWORDS {
        if let Some(Value::Object(subschemas)) = map.get_mut(*keyword) {
            subschemas.values_mut().try_for_each(transform)?;
        }
    }
    Ok(())
}

fn close_object(map: &mut Map<String, Value>) -> Result<(), JsonOutputSchemaError> {
    let is_object = map.contains_key("properties")
        || match map.get("type") {
            Some(Value::String(kind)) => kind == "object",
            Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind == "object"),
            _ => false,
        };
    if !is_object {
        return Ok(());
    }
    if map.contains_key("patternProperties") {
        return Err(JsonOutputSchemaError::PatternProperties);
    }
    match map.get("additionalProperties") {
        None => {
            map.insert("additionalProperties".to_owned(), Value::Bool(false));
            Ok(())
        }
        Some(Value::Bool(false)) => Ok(()),
        Some(_) => Err(JsonOutputSchemaError::OpenObject),
    }
}

fn describe(map: &mut Map<String, Value>, dropped: &Value) {
    let note = dropped.to_string();
    let description = match map.get("description").and_then(Value::as_str) {
        Some(existing) if !existing.is_empty() => format!("{existing}\n\n{note}"),
        _ => note,
    };
    map.insert("description".to_owned(), Value::String(description));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unsupported_constraints_move_into_the_description() -> anyhow::Result<()> {
        let schema = json!({
            "type": "object",
            "properties": {
                "amount_cents": { "type": ["integer", "null"], "minimum": 1 },
                "label": { "type": ["string", "null"], "maxLength": 80, "description": "A label" },
                "due_on": { "type": ["string", "null"], "format": "date" },
                "code": { "type": "string", "format": "iban" },
                "options": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": { "text": { "type": "string", "minLength": 1 } },
                        "required": ["text"]
                    }
                },
                "pairs": { "type": "array", "minItems": 2, "items": { "type": "integer" } }
            },
            "required": ["amount_cents", "label", "due_on", "code", "options", "pairs"],
            "additionalProperties": false
        });

        let wire = json_output_schema(&schema)?;

        assert_eq!(
            wire,
            json!({
                "type": "object",
                "properties": {
                    "amount_cents": { "type": ["integer", "null"], "description": "{\"minimum\":1}" },
                    "label": {
                        "type": ["string", "null"],
                        "description": "A label\n\n{\"maxLength\":80}"
                    },
                    "due_on": { "type": ["string", "null"], "format": "date" },
                    "code": { "type": "string", "description": "{\"format\":\"iban\"}" },
                    "options": {
                        "type": "array",
                        "minItems": 1,
                        "description": "{\"maxItems\":8}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": { "type": "string", "description": "{\"minLength\":1}" }
                            },
                            "required": ["text"],
                            "additionalProperties": false
                        }
                    },
                    "pairs": {
                        "type": "array",
                        "description": "{\"minItems\":2}",
                        "items": { "type": "integer" }
                    }
                },
                "required": ["amount_cents", "label", "due_on", "code", "options", "pairs"],
                "additionalProperties": false
            })
        );
        assert_eq!(schema["properties"]["amount_cents"]["minimum"], 1);
        Ok(())
    }

    #[test]
    fn nested_schemas_in_unions_and_definitions_are_transformed() -> anyhow::Result<()> {
        let schema = json!({
            "$defs": { "money": { "type": "object", "properties": { "cents": { "type": "integer", "maximum": 10 } } } },
            "anyOf": [ { "$ref": "#/$defs/money" }, { "type": "string", "maxLength": 3 } ]
        });

        let wire = json_output_schema(&schema)?;

        assert_eq!(wire["$defs"]["money"]["additionalProperties"], false);
        assert_eq!(
            wire["$defs"]["money"]["properties"]["cents"],
            json!({ "type": "integer", "description": "{\"maximum\":10}" })
        );
        assert_eq!(
            wire["anyOf"][1],
            json!({ "type": "string", "description": "{\"maxLength\":3}" })
        );
        assert_eq!(wire["anyOf"][0], json!({ "$ref": "#/$defs/money" }));
        Ok(())
    }

    #[test]
    fn an_object_that_accepts_unlisted_names_is_refused() {
        for (schema, expected) in [
            (
                json!({ "type": "object", "additionalProperties": true }),
                JsonOutputSchemaError::OpenObject,
            ),
            (
                json!({ "type": "object", "additionalProperties": { "type": "string" } }),
                JsonOutputSchemaError::OpenObject,
            ),
            (
                json!({
                    "type": "object",
                    "properties": { "inner": { "type": "object", "patternProperties": { "^x": {} } } }
                }),
                JsonOutputSchemaError::PatternProperties,
            ),
        ] {
            assert_eq!(json_output_schema(&schema), Err(expected), "{schema}");
        }
    }
}
