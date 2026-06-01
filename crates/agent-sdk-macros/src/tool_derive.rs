//! `#[derive(Tool)]` — generates a [`SimpleTool`] impl from `#[tool(...)]`.

use crate::attr::ToolAttr;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse2};

pub fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let ident = &input.ident;
    let span = ident.span();
    let attr = ToolAttr::collect(&input.attrs, "tool")?;

    let name = attr.require_string("name", span)?;
    let description = attr.require_string("description", span)?;
    let display = attr.opt_string("display_name")?.unwrap_or_default();
    let ctx = attr
        .opt_type("context")?
        .unwrap_or_else(|| syn::parse_quote!(()));

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let tier_body = attr.opt_expr("tier").map_or_else(
        || quote! { ::agent_sdk::ToolTier::Observe },
        |t| quote! { #t },
    );

    // The untyped `Tool` derive always uses a hand-provided (or default)
    // schema — there is no typed `Input` to derive one from.
    let schema_body = attr.opt_expr("schema").map_or_else(
        || quote! { ::agent_sdk::__macro_support::json!({ "type": "object" }) },
        |s| quote! { #s },
    );

    Ok(quote! {
        impl #impl_generics ::agent_sdk::SimpleTool<#ctx> for #ident #ty_generics #where_clause {
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
                input: ::agent_sdk::__macro_support::Value,
            ) -> impl ::core::future::Future<
                Output = ::agent_sdk::__macro_support::Result<::agent_sdk::ToolResult>,
            > + Send {
                // Delegate to the user-written `ToolLogic::execute`. Targeting a
                // trait method (rather than an inherent fn) keeps a fully
                // synchronous tool body lint-clean under `clippy::unused_async`.
                <Self as ::agent_sdk::ToolLogic<#ctx>>::execute(self, ctx, input)
            }
        }
    })
}
