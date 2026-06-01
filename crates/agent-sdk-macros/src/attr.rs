//! Hand-rolled `#[tool(...)]` / `#[tool_name(...)]` attribute parsing.
//!
//! We parse the attributes directly with `syn` (rather than via a derive like
//! `darling`'s `FromDeriveInput`) so that *all* generated code is ours: a
//! third-party attribute-deriving macro emits code that trips the workspace's
//! `nursery`/`pedantic` clippy lints under `-D warnings`, which we cannot fix
//! without an `#[allow]` (forbidden by the project's lint policy).

use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, ExprLit, Lit, Token, Type};

/// One `key = value` (or bare `key`) entry inside a `#[tool(...)]` list, with
/// the value kept as raw tokens so callers decide how to interpret it
/// (string literal, bare type, arbitrary expression).
struct MetaEntry {
    key: syn::Ident,
    value: Option<Expr>,
}

impl syn::parse::Parse for MetaEntry {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let key: syn::Ident = input.parse()?;
        let value = if input.peek(Token![=]) {
            let _: Token![=] = input.parse()?;
            Some(input.parse()?)
        } else {
            None
        };
        Ok(Self { key, value })
    }
}

/// The parsed contents of every `#[tool(...)]` attribute on a derive input.
///
/// Values are stored as raw [`Expr`]s; the typed accessors below reinterpret
/// them (e.g. a path expression as a [`Type`]).
pub struct ToolAttr {
    entries: Vec<MetaEntry>,
}

impl ToolAttr {
    /// Collect every `#[ident(...)]` attribute (e.g. all `#[tool(...)]`) into a
    /// single flat list of entries.
    ///
    /// # Errors
    /// Returns an error if an attribute's contents are not a comma-separated
    /// `key = value` list.
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
        Ok(Self { entries })
    }

    fn get(&self, key: &str) -> Option<&Expr> {
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
            |expr| as_string(expr, key),
        )
    }

    /// An optional string-literal value.
    ///
    /// # Errors
    /// Returns an error if present but not a string literal.
    pub fn opt_string(&self, key: &str) -> syn::Result<Option<String>> {
        self.get(key).map(|e| as_string(e, key)).transpose()
    }

    /// A required bare type, e.g. `input = MyArgs`.
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
            as_type,
        )
    }

    /// An optional bare type, e.g. `context = MyCtx`.
    ///
    /// # Errors
    /// Returns an error if present but not parseable as a type.
    pub fn opt_type(&self, key: &str) -> syn::Result<Option<Type>> {
        self.get(key).map(as_type).transpose()
    }

    /// An optional raw expression value, e.g. `schema = json!({...})`.
    #[must_use]
    pub fn opt_expr(&self, key: &str) -> Option<Expr> {
        self.get(key).cloned()
    }
}

fn as_string(expr: &Expr, key: &str) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        other => Err(syn::Error::new_spanned(
            other,
            format!("`{key}` must be a string literal"),
        )),
    }
}

fn as_type(expr: &Expr) -> syn::Result<Type> {
    // Re-parse the expression's tokens as a type so paths and generics work
    // (`input = Foo<Bar>` etc.).
    syn::parse2(quote::quote!(#expr))
}
