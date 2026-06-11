//! Hand-rolled `#[tool(...)]` / `#[tool_name(...)]` attribute parsing.
//!
//! We parse the attributes directly with `syn` (rather than via a derive like
//! `darling`'s `FromDeriveInput`) so that *all* generated code is ours: a
//! third-party attribute-deriving macro emits code that trips the workspace's
//! `nursery`/`pedantic` clippy lints under `-D warnings`, which we cannot fix
//! without an `#[allow]` (forbidden by the project's lint policy).

use std::collections::HashSet;

use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, ExprLit, Lit, Token, Type};

/// The interpreted value of a `key = value` entry.
///
/// Type-valued keys (`input`, `context`) are parsed directly as a [`Type`] so
/// generic syntax like `input = Vec<Args>` or `context = Arc<Ctx>` parses
/// cleanly. Such syntax is **not** a valid Rust expression — `syn`'s `Expr`
/// parser rejects it with a confusing "comparison operators cannot be chained"
/// error — so the old "parse everything as `Expr`, reinterpret later" approach
/// broke on real-world generic context/input types.
enum MetaValue {
    Type(Box<Type>),
    Expr(Box<Expr>),
}

/// One `key = value` (or bare `key`) entry inside a `#[tool(...)]` list.
struct MetaEntry {
    key: syn::Ident,
    value: Option<MetaValue>,
}

/// Keys whose value is a (possibly generic) type rather than an expression.
fn is_type_valued_key(key: &syn::Ident) -> bool {
    key == "input" || key == "context"
}

impl syn::parse::Parse for MetaEntry {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let key: syn::Ident = input.parse()?;
        let value = if input.peek(Token![=]) {
            let _: Token![=] = input.parse()?;
            if is_type_valued_key(&key) {
                Some(MetaValue::Type(Box::new(input.parse()?)))
            } else {
                Some(MetaValue::Expr(Box::new(input.parse()?)))
            }
        } else {
            None
        };
        Ok(Self { key, value })
    }
}

/// The parsed contents of every `#[tool(...)]` attribute on a derive input.
pub struct ToolAttr {
    entries: Vec<MetaEntry>,
}

/// Valid keys accepted under a given attribute name.
///
/// Returns `None` for attribute names this crate does not own, in which case
/// key validation is skipped (we never reject keys we don't understand).
fn allowed_keys(ident: &str) -> Option<&'static [&'static str]> {
    match ident {
        // Shared by `#[derive(Tool)]` and `#[derive(TypedTool)]`. `input` is
        // only meaningful for `TypedTool`, but it is accepted under both so the
        // single `#[tool(...)]` grammar stays uniform; an `input` on a plain
        // `Tool` is simply unused rather than a hard error.
        "tool" => Some(&[
            "name",
            "description",
            "display_name",
            "context",
            "tier",
            "schema",
            "input",
        ]),
        "tool_name" => Some(&["rename_all"]),
        _ => None,
    }
}

impl ToolAttr {
    /// Collect every `#[ident(...)]` attribute (e.g. all `#[tool(...)]`) into a
    /// single flat list of entries.
    ///
    /// # Errors
    /// Returns an error if an attribute's contents are not a comma-separated
    /// `key = value` list, or if any entry uses an unknown or duplicate key.
    pub fn collect(attrs: &[Attribute], ident: &str) -> syn::Result<Self> {
        let mut entries = Vec::new();
        for attr in attrs {
            if !attr.path().is_ident(ident) {
                continue;
            }
            let parsed =
                attr.parse_args_with(Punctuated::<MetaEntry, Token![,]>::parse_terminated)?;
            entries.extend(parsed);
        }
        let this = Self { entries };
        this.validate_keys(ident)?;
        Ok(this)
    }

    /// Reject unknown and duplicate keys.
    ///
    /// A silently-ignored key is a security footgun: a typo like
    /// `teir = ToolTier::Confirm` would otherwise compile cleanly and leave
    /// `tier()` at the always-allowed `ToolTier::Observe` default, silently
    /// downgrading a confirmation-gated tool to auto-execute. Duplicate keys
    /// (previously resolved first-wins) are likewise rejected so the effective
    /// value is never ambiguous.
    fn validate_keys(&self, ident: &str) -> syn::Result<()> {
        let Some(allowed) = allowed_keys(ident) else {
            return Ok(());
        };
        let mut seen = HashSet::new();
        for entry in &self.entries {
            let key = entry.key.to_string();
            if !allowed.contains(&key.as_str()) {
                return Err(syn::Error::new_spanned(
                    &entry.key,
                    format!(
                        "unknown `#[{ident}(...)]` key `{key}`; valid keys are: {}",
                        allowed.join(", ")
                    ),
                ));
            }
            if !seen.insert(key.clone()) {
                return Err(syn::Error::new_spanned(
                    &entry.key,
                    format!("duplicate `#[{ident}(...)]` key `{key}`"),
                ));
            }
        }
        Ok(())
    }

    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries
            .iter()
            .find(|e| e.key == key)
            .and_then(|e| e.value.as_ref())
    }

    /// A required string-literal value, e.g. `name = "foo"`.
    ///
    /// # Errors
    /// Returns an error if the key is absent or its value is not a string
    /// literal.
    pub fn require_string(&self, key: &str, span: proc_macro2::Span) -> syn::Result<String> {
        self.get(key).map_or_else(
            || {
                Err(syn::Error::new(
                    span,
                    format!("#[tool(...)] is missing the required `{key} = \"...\"`"),
                ))
            },
            |value| as_string(value, key),
        )
    }

    /// An optional string-literal value.
    ///
    /// # Errors
    /// Returns an error if present but not a string literal.
    pub fn opt_string(&self, key: &str) -> syn::Result<Option<String>> {
        self.get(key).map(|value| as_string(value, key)).transpose()
    }

    /// A required (possibly generic) type, e.g. `input = Vec<MyArgs>`.
    ///
    /// # Errors
    /// Returns an error if absent or not parseable as a type.
    pub fn require_type(&self, key: &str, span: proc_macro2::Span) -> syn::Result<Type> {
        self.get(key).map_or_else(
            || {
                Err(syn::Error::new(
                    span,
                    format!("#[tool(...)] is missing the required `{key} = <Type>`"),
                ))
            },
            |value| as_type(value, key),
        )
    }

    /// An optional (possibly generic) type, e.g. `context = Arc<MyCtx>`.
    ///
    /// # Errors
    /// Returns an error if present but not parseable as a type.
    pub fn opt_type(&self, key: &str) -> syn::Result<Option<Type>> {
        self.get(key).map(|value| as_type(value, key)).transpose()
    }

    /// An optional raw expression value, e.g. `schema = json!({...})`.
    #[must_use]
    pub fn opt_expr(&self, key: &str) -> Option<Expr> {
        self.get(key).and_then(|value| match value {
            MetaValue::Expr(expr) => Some((**expr).clone()),
            MetaValue::Type(_) => None,
        })
    }
}

fn as_string(value: &MetaValue, key: &str) -> syn::Result<String> {
    match value {
        MetaValue::Expr(expr) => match &**expr {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Ok(s.value()),
            other => Err(syn::Error::new_spanned(
                other,
                format!("`{key}` must be a string literal"),
            )),
        },
        MetaValue::Type(ty) => Err(syn::Error::new_spanned(
            ty,
            format!("`{key}` must be a string literal"),
        )),
    }
}

fn as_type(value: &MetaValue, key: &str) -> syn::Result<Type> {
    match value {
        // Type-valued keys are parsed as a `Type` up front, so generics and
        // paths (`input = Foo<Bar>`, `context = Arc<Ctx>`) just work.
        MetaValue::Type(ty) => Ok((**ty).clone()),
        // Defensive fallback: a non-type key read via `*_type`. Re-parse the
        // expression's tokens as a type so simple paths still resolve.
        MetaValue::Expr(expr) => syn::parse2(quote::quote!(#expr))
            .map_err(|_| syn::Error::new_spanned(expr, format!("`{key}` must be a type"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn collect(ident: &str, tokens: proc_macro2::TokenStream) -> syn::Result<ToolAttr> {
        let input: syn::DeriveInput = syn::parse2(tokens)?;
        ToolAttr::collect(&input.attrs, ident)
    }

    #[test]
    fn parses_generic_input_type() -> syn::Result<()> {
        let attr = collect(
            "tool",
            quote! {
                #[tool(name = "x", description = "y", input = Vec<Args>)]
                struct Foo;
            },
        )?;
        let ty = attr.require_type("input", proc_macro2::Span::call_site())?;
        assert_eq!(quote!(#ty).to_string(), quote!(Vec<Args>).to_string());
        Ok(())
    }

    #[test]
    fn parses_arc_context_type() -> syn::Result<()> {
        let attr = collect(
            "tool",
            quote! {
                #[tool(name = "x", description = "y", context = Arc<Ctx>)]
                struct Foo;
            },
        )?;
        let ty = attr
            .opt_type("context")?
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "context missing"))?;
        assert_eq!(quote!(#ty).to_string(), quote!(Arc<Ctx>).to_string());
        Ok(())
    }

    #[test]
    fn parses_nested_generic_type_with_inner_comma() -> syn::Result<()> {
        let attr = collect(
            "tool",
            quote! {
                #[tool(
                    name = "x",
                    description = "y",
                    context = std::sync::Arc<std::collections::HashMap<String, u8>>
                )]
                struct Foo;
            },
        )?;
        let ty = attr
            .opt_type("context")?
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "context missing"))?;
        let rendered = quote!(#ty).to_string();
        assert!(rendered.contains("HashMap"), "got: {rendered}");
        assert!(rendered.contains("String"), "got: {rendered}");
        Ok(())
    }

    #[test]
    fn simple_path_input_type_still_parses() -> syn::Result<()> {
        let attr = collect(
            "tool",
            quote! {
                #[tool(name = "x", description = "y", input = Value)]
                struct Foo;
            },
        )?;
        let ty = attr.require_type("input", proc_macro2::Span::call_site())?;
        assert_eq!(quote!(#ty).to_string(), quote!(Value).to_string());
        Ok(())
    }

    #[test]
    fn unknown_key_is_a_hard_error() {
        // A typo'd security-relevant key (`teir`) must fail to compile rather
        // than be silently dropped (which would downgrade the tool to Observe).
        let result = collect(
            "tool",
            quote! {
                #[tool(name = "pay", description = "d", teir = ToolTier::Confirm)]
                struct Foo;
            },
        );
        match result {
            Ok(_) => panic!("unknown key `teir` must be rejected"),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("teir"), "got: {msg}");
                assert!(msg.contains("valid keys"), "got: {msg}");
            }
        }
    }

    #[test]
    fn duplicate_key_is_a_hard_error() {
        let result = collect(
            "tool",
            quote! {
                #[tool(name = "a", name = "b", description = "d")]
                struct Foo;
            },
        );
        match result {
            Ok(_) => panic!("duplicate key `name` must be rejected"),
            Err(err) => assert!(err.to_string().contains("duplicate"), "got: {err}"),
        }
    }

    #[test]
    fn all_known_tool_keys_are_accepted() -> syn::Result<()> {
        let attr = collect(
            "tool",
            quote! {
                #[tool(
                    name = "x",
                    description = "y",
                    display_name = "X",
                    input = Value,
                    context = Arc<Ctx>,
                    tier = ToolTier::Observe,
                    schema = json!({})
                )]
                struct Foo;
            },
        )?;
        assert_eq!(
            attr.require_string("name", proc_macro2::Span::call_site())?,
            "x"
        );
        assert!(attr.opt_expr("tier").is_some());
        assert!(attr.opt_expr("schema").is_some());
        Ok(())
    }

    #[test]
    fn unknown_tool_name_key_is_rejected() {
        let result = collect(
            "tool_name",
            quote! {
                #[tool_name(renameall = "snake_case")]
                enum Foo { A }
            },
        );
        match result {
            Ok(_) => panic!("unknown key `renameall` must be rejected"),
            Err(err) => assert!(err.to_string().contains("renameall"), "got: {err}"),
        }
    }
}
