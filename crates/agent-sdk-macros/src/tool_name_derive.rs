//! `#[derive(ToolName)]` — derive `Serialize` + `Deserialize` + the
//! [`ToolName`] marker for a tool-name enum in one shot.

use crate::attr::ToolAttr;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse2};

pub fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let ident = &input.ident;

    // `ToolName` is only meaningful for enums (string-tag serialization).
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            ident,
            "#[derive(ToolName)] can only be applied to enums",
        ));
    };

    // Only unit variants are supported (tool names are simple string tags).
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "#[derive(ToolName)] only supports unit (fieldless) enum variants",
            ));
        }
    }

    let attr = ToolAttr::collect(&input.attrs, "tool_name")?;
    // Default to `snake_case`, matching the SDK's built-in `PrimitiveToolName`.
    let rename = attr
        .opt_string("rename_all")?
        .unwrap_or_else(|| "snake_case".to_string());

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mirror = quote::format_ident!("__{}ToolNameMirror", ident);

    // `quote!` consumes each interpolated iterator, so build the variant token
    // streams once and reuse the owned `Vec`s (which `quote!` borrows).
    let decl_variants: Vec<_> = data.variants.iter().map(|v| &v.ident).collect();
    let to_mirror_arms: Vec<_> = data
        .variants
        .iter()
        .map(|v| {
            let id = &v.ident;
            quote! { #ident::#id => #mirror::#id }
        })
        .collect();
    let from_mirror_arms: Vec<_> = data
        .variants
        .iter()
        .map(|v| {
            let id = &v.ident;
            quote! { #mirror::#id => #ident::#id }
        })
        .collect();

    Ok(quote! {
        // A derive cannot re-emit `#[derive(Serialize, Deserialize)]` on the
        // user's enum, so we forward through a hidden mirror enum that *does*
        // carry the serde derive (with the `rename_all` policy). This keeps
        // `#[derive(ToolName)]` the single derive a user needs while still
        // relying on serde's own codegen for the wire format.
        const _: () = {
            #[derive(
                ::agent_sdk::__macro_support::Serialize,
                ::agent_sdk::__macro_support::Deserialize,
            )]
            #[serde(crate = "::agent_sdk::__macro_support::serde", rename_all = #rename)]
            enum #mirror {
                #( #decl_variants ),*
            }

            impl ::core::convert::From<& #ident> for #mirror {
                fn from(value: & #ident) -> Self {
                    match value {
                        #( #to_mirror_arms ),*
                    }
                }
            }

            impl ::core::convert::From<#mirror> for #ident {
                fn from(value: #mirror) -> Self {
                    match value {
                        #( #from_mirror_arms ),*
                    }
                }
            }

            impl ::agent_sdk::__macro_support::Serialize for #ident {
                fn serialize<S>(&self, serializer: S) -> ::core::result::Result<S::Ok, S::Error>
                where
                    S: ::agent_sdk::__macro_support::Serializer,
                {
                    #mirror::from(self).serialize(serializer)
                }
            }

            impl<'de> ::agent_sdk::__macro_support::Deserialize<'de> for #ident {
                fn deserialize<D>(deserializer: D) -> ::core::result::Result<Self, D::Error>
                where
                    D: ::agent_sdk::__macro_support::Deserializer<'de>,
                {
                    #mirror::deserialize(deserializer).map(#ident::from)
                }
            }
        };

        impl #impl_generics ::agent_sdk::ToolName for #ident #ty_generics #where_clause {}
    })
}
