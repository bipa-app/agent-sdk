use anyhow::{Context, Result};

pub async fn fetch_model_list_body(
    builder: reqwest::RequestBuilder,
    provider: &str,
) -> Result<String> {
    let response = builder
        .send()
        .await
        .with_context(|| format!("{provider} list_models request failed"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.with_context(|| {
            format!("failed to read {provider} list_models error response body")
        })?;
        anyhow::bail!("{provider} list_models returned HTTP {status}: {body}");
    }

    response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} list_models success response body"))
}
