//! `#[derive(TypedTool)]` — generates a [`TypedTool`] impl with a typed
//! `Input`, optionally auto-deriving the JSON schema from `Input` via
//! `schemars` (`schema = "derive"`).

use crate::attr::ToolAttr;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, ExprLit, Lit, Type, parse2};

/// Distinguishes the three schema modes the `schema = ...` attribute can take.
enum SchemaMode {
    /// `schema = "derive"` — generate from `Input` via `schemars`.
    Derive,
    /// `schema = <expr>` — a hand-written `serde_json::Value`.
    Provided(Expr),
    /// No `schema` attribute — fall back to `{"type":"object"}`.
    Default,
}

fn classify_schema(schema: Option<Expr>) -> SchemaMode {
    match schema {
        None => SchemaMode::Default,
        Some(Expr::Lit(ExprLit {
            lit: Lit::Str(ref s),
            ..
        })) if s.value() == "derive" => SchemaMode::Derive,
        Some(expr) => SchemaMode::Provided(expr),
    }
}

pub fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let ident = &input.ident;
    let span = ident.span();
    let attr = ToolAttr::collect(&input.attrs, "tool")?;

    let name = attr.require_string("name", span)?;
    let description = attr.require_string("description", span)?;
    let display = attr.opt_string("display_name")?.unwrap_or_default();
    let input_ty = attr.require_type("input", span)?;
    let ctx = attr
        .opt_type("context")?
        .unwrap_or_else(|| syn::parse_quote!(()));

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let tier_body = attr.opt_expr("tier").map_or_else(
        || quote! { ::agent_sdk::ToolTier::Observe },
        |t| quote! { #t },
    );

    let schema_body = match classify_schema(attr.opt_expr("schema")) {
        SchemaMode::Default => {
            quote! { ::agent_sdk::__macro_support::json!({ "type": "object" }) }
        }
        SchemaMode::Provided(expr) => quote! { #expr },
        SchemaMode::Derive => derive_schema_body(&input_ty),
    };

    Ok(quote! {
        impl #impl_generics ::agent_sdk::TypedTool<#ctx> for #ident #ty_generics #where_clause {
            type Input = #input_ty;

            fn name(&self) -> &'static str {
                #name
            }

            fn display_name(&self) -> &'static str {
                #display
            }

            fn description(&self) -> &'static str {
                #description
            }

            fn input_schema(&self) -> ::agent_sdk::__macro_support::Value {
                #schema_body
            }

            fn tier(&self) -> ::agent_sdk::ToolTier {
                #tier_body
            }

            fn execute(
                &self,
                ctx: &::agent_sdk::ToolContext<#ctx>,
                input: Self::Input,
            ) -> impl ::core::future::Future<
                Output = ::agent_sdk::__macro_support::Result<::agent_sdk::ToolResult>,
            > + Send {
                // Delegate to the user-written `ToolLogic::execute` (a trait
                // method, so a non-awaiting body stays lint-clean).
                <Self as ::agent_sdk::ToolLogic<#ctx>>::execute(self, ctx, input)
            }
        }
    })
}

/// The `input_schema` body for `schema = "derive"`.
///
/// Gated on this crate's `schema-derive` feature: only when it is enabled do we
/// emit the `schemars` call. Without the feature we emit a hard `compile_error!`
/// so the misconfiguration is caught at the derive site (rather than failing
/// later on a missing `schemars` dependency).
#[cfg(feature = "schema-derive")]
fn derive_schema_body(input_ty: &Type) -> TokenStream {
    quote! {
        // `schemars` produces a typed `Schema`; the SDK's tool contract speaks
        // `serde_json::Value`, so we serialize through it. The schema is a
        // self-contained object literal, so this never fails in practice.
        ::agent_sdk::__macro_support::to_value(
            ::agent_sdk::__macro_support::schema_for::<#input_ty>()
        )
        .unwrap_or_else(|_| ::agent_sdk::__macro_support::json!({ "type": "object" }))
    }
}

#[cfg(not(feature = "schema-derive"))]
fn derive_schema_body(_input_ty: &Type) -> TokenStream {
    quote! {
        ::core::compile_error!(
            "`#[tool(schema = \"derive\")]` requires the `agent-sdk` `macros-schema` \
             feature (which enables `agent-sdk-macros/schema-derive`). Either enable \
             that feature or provide a `schema = <serde_json::Value>` expression."
        )
    }
}
