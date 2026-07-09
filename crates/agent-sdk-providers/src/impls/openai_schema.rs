//! Shared JSON-Schema normalization for `OpenAI` strict mode.

use std::fmt;

/// Why a schema cannot be sent through `OpenAI` strict mode without losing
/// meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictSchemaError {
    /// An object leaves its property names or values unconstrained.
    FreeFormObject,
    /// An object maps otherwise unknown property names to a schema.
    TypedAdditionalProperties,
    /// An object uses regular expressions to describe otherwise unknown
    /// property names.
    PatternProperties,
}

impl fmt::Display for StrictSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FreeFormObject => formatter.write_str("free-form object"),
            Self::TypedAdditionalProperties => {
                formatter.write_str("typed additionalProperties mapping")
            }
            Self::PatternProperties => formatter.write_str("patternProperties mapping"),
        }
    }
}

impl std::error::Error for StrictSchemaError {}

/// Return the schema `OpenAI` strict mode will receive without modifying the
/// caller-owned schema.
///
/// `OpenAI` strict mode requires each object to be closed and each property to
/// be required. Optional properties therefore become nullable required
/// properties. A schema with dynamic property names cannot be converted this
/// way without changing its contract, so it is rejected before any mutation.
pub fn normalized_strict_schema(
    schema: &serde_json::Value,
) -> Result<serde_json::Value, StrictSchemaError> {
    validate_strict_compatibility(schema)?;

    let mut normalized = schema.clone();
    normalize_schema_node(&mut normalized);
    Ok(normalized)
}

/// Normalize a JSON Schema for `OpenAI` strict mode.
///
/// Returns `false` when the schema contains dynamic object properties that
/// cannot be represented in strict mode without changing their meaning.
/// Otherwise every object is closed with `additionalProperties: false`, every
/// property is made required, and properties that were optional remain
/// semantically optional by accepting `null`.
pub fn normalize_strict_schema(schema: &mut serde_json::Value) -> bool {
    let Ok(normalized) = normalized_strict_schema(schema) else {
        return false;
    };
    *schema = normalized;
    true
}

fn validate_strict_compatibility(schema: &serde_json::Value) -> Result<(), StrictSchemaError> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };

    let has_properties = object
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|properties| !properties.is_empty());
    let is_object = is_object_schema(object);

    if is_object {
        if object.contains_key("patternProperties") {
            return Err(StrictSchemaError::PatternProperties);
        }

        match object.get("additionalProperties") {
            Some(serde_json::Value::Bool(false)) => {}
            Some(serde_json::Value::Bool(true)) => {
                return Err(StrictSchemaError::FreeFormObject);
            }
            Some(serde_json::Value::Object(_)) => {
                return Err(StrictSchemaError::TypedAdditionalProperties);
            }
            Some(_) => return Err(StrictSchemaError::FreeFormObject),
            None if !has_properties => return Err(StrictSchemaError::FreeFormObject),
            None => {}
        }
    }

    if let Some(properties) = object
        .get("properties")
        .and_then(serde_json::Value::as_object)
    {
        for property in properties.values() {
            validate_strict_compatibility(property)?;
        }
    }

    if let Some(items) = object.get("items") {
        validate_schema_or_array(items)?;
    }

    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = object.get(keyword).and_then(serde_json::Value::as_array) {
            for variant in variants {
                validate_strict_compatibility(variant)?;
            }
        }
    }

    for keyword in ["$defs", "definitions"] {
        if let Some(definitions) = object.get(keyword).and_then(serde_json::Value::as_object) {
            for definition in definitions.values() {
                validate_strict_compatibility(definition)?;
            }
        }
    }

    Ok(())
}

fn validate_schema_or_array(value: &serde_json::Value) -> Result<(), StrictSchemaError> {
    match value {
        serde_json::Value::Array(schemas) => {
            for schema in schemas {
                validate_strict_compatibility(schema)?;
            }
            Ok(())
        }
        schema => validate_strict_compatibility(schema),
    }
}

fn normalize_schema_node(schema: &mut serde_json::Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };

    let is_object = is_object_schema(object);

    if is_object {
        object.insert(
            "additionalProperties".to_owned(),
            serde_json::Value::Bool(false),
        );

        let originally_required: std::collections::HashSet<String> = object
            .get("required")
            .and_then(serde_json::Value::as_array)
            .map(|required| {
                required
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        object
            .entry("properties".to_owned())
            .or_insert_with(|| serde_json::json!({}));

        if let Some(properties) = object
            .get_mut("properties")
            .and_then(serde_json::Value::as_object_mut)
        {
            for (name, property) in properties.iter_mut() {
                normalize_schema_node(property);
                if !originally_required.contains(name) {
                    make_nullable(property);
                }
            }

            let required = properties
                .keys()
                .cloned()
                .map(serde_json::Value::String)
                .collect();
            object.insert("required".to_owned(), serde_json::Value::Array(required));
        }
    }

    if let Some(items) = object.get_mut("items") {
        normalize_schema_or_array(items);
    }

    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = object
            .get_mut(keyword)
            .and_then(serde_json::Value::as_array_mut)
        {
            for variant in variants {
                normalize_schema_node(variant);
            }
        }
    }

    for keyword in ["$defs", "definitions"] {
        if let Some(definitions) = object
            .get_mut(keyword)
            .and_then(serde_json::Value::as_object_mut)
        {
            for definition in definitions.values_mut() {
                normalize_schema_node(definition);
            }
        }
    }
}

fn normalize_schema_or_array(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(schemas) => {
            for schema in schemas {
                normalize_schema_node(schema);
            }
        }
        schema => normalize_schema_node(schema),
    }
}

fn is_object_type(value: &serde_json::Value) -> bool {
    value.as_str() == Some("object")
        || value
            .as_array()
            .is_some_and(|types| types.iter().any(|value| value.as_str() == Some("object")))
}

fn is_object_schema(object: &serde_json::Map<String, serde_json::Value>) -> bool {
    object.get("type").is_some_and(is_object_type)
        || object
            .get("properties")
            .is_some_and(serde_json::Value::is_object)
        || object.contains_key("additionalProperties")
        || object.contains_key("patternProperties")
}

fn make_nullable(schema: &mut serde_json::Value) {
    if accepts_null(schema) {
        return;
    }

    if let Some(any_of) = schema
        .as_object_mut()
        .and_then(|object| object.get_mut("anyOf"))
        .and_then(serde_json::Value::as_array_mut)
    {
        any_of.push(serde_json::json!({"type": "null"}));
        return;
    }

    let original = schema.clone();
    *schema = serde_json::json!({
        "anyOf": [original, {"type": "null"}]
    });
}

fn accepts_null(schema: &serde_json::Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };

    if object.get("type").is_some_and(|value| {
        value.as_str() == Some("null")
            || value
                .as_array()
                .is_some_and(|types| types.iter().any(|value| value.as_str() == Some("null")))
    }) {
        return true;
    }

    object
        .get("anyOf")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|variants| variants.iter().any(accepts_null))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

    #[test]
    fn normalizes_nested_optional_objects_and_definitions() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "required_name": {"type": "string"},
                "optional_profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"}
                    }
                }
            },
            "required": ["required_name"],
            "$defs": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        });

        assert!(normalize_strict_schema(&mut schema));
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema["required"].as_array().is_some_and(|required| {
            required.len() == 2
                && required.contains(&serde_json::json!("required_name"))
                && required.contains(&serde_json::json!("optional_profile"))
        }));
        assert_eq!(
            schema["properties"]["optional_profile"]["anyOf"][0]["additionalProperties"],
            false
        );
        assert_eq!(
            schema["properties"]["optional_profile"]["anyOf"][1]["type"],
            "null"
        );
        assert_eq!(schema["$defs"]["address"]["additionalProperties"], false);
    }

    #[test]
    fn preserves_already_nullable_properties() -> anyhow::Result<()> {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "nickname": {"type": ["string", "null"]}
            }
        });

        assert!(normalize_strict_schema(&mut schema));
        let nickname = schema["properties"]
            .get("nickname")
            .context("nickname schema missing")?;
        assert!(nickname.get("anyOf").is_none());
        assert_eq!(nickname["type"], serde_json::json!(["string", "null"]));
        Ok(())
    }

    #[test]
    fn rejects_freeform_objects_without_mutating_them() {
        let mut schema = serde_json::json!({"type": "object"});
        let original = schema.clone();

        assert!(!normalize_strict_schema(&mut schema));
        assert_eq!(schema, original);
    }

    #[test]
    fn rejects_dynamic_property_mappings_without_mutating_them() {
        for mut schema in [
            serde_json::json!({
                "type": "object",
                "properties": {"fixed": {"type": "string"}},
                "additionalProperties": {"type": "string"}
            }),
            serde_json::json!({
                "type": "object",
                "properties": {"fixed": {"type": "string"}},
                "patternProperties": {"^x-": {"type": "string"}}
            }),
            serde_json::json!({
                "patternProperties": {"^x-": {"type": "string"}}
            }),
            serde_json::json!({
                "type": "object",
                "properties": {"fixed": {"type": "string"}},
                "additionalProperties": true
            }),
            serde_json::json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"fixed": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                }
            }),
        ] {
            let original = schema.clone();
            assert!(!normalize_strict_schema(&mut schema));
            assert_eq!(schema, original);
        }
    }

    #[test]
    fn accepts_closed_empty_objects() {
        let mut schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false
        });

        assert!(normalize_strict_schema(&mut schema));
        assert_eq!(schema["properties"], serde_json::json!({}));
        assert_eq!(schema["required"], serde_json::json!([]));
    }

    #[test]
    fn returns_normalized_copy_without_mutating_source_schema() -> anyhow::Result<()> {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"nickname": {"type": "string"}}
        });

        let normalized = normalized_strict_schema(&schema)?;
        assert_eq!(schema.get("required"), None);
        assert_eq!(normalized["required"], serde_json::json!(["nickname"]));
        assert_eq!(
            normalized["properties"]["nickname"]["anyOf"][1]["type"],
            "null"
        );
        Ok(())
    }
}
