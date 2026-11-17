//! Credentials for signing Bedrock requests.
//!
//! The types are AWS's (`aws-credential-types`): any [`ProvideCredentials`]
//! source works, including the providers of `aws-config`. This module adds
//! one source those do not cover, web identity federation from a token file
//! or from the Google Compute Engine metadata server (GCE, GKE with Workload
//! Identity, Cloud Run). It trades the OIDC token for temporary credentials
//! through STS `AssumeRoleWithWebIdentity`
//! (<https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html>)
//! and caches them until five minutes before they expire.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aws_credential_types::Credentials;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider, future};
use aws_smithy_types::date_time::{DateTime, Format};

const REFRESH_MARGIN: Duration = Duration::from_mins(5);
const STS_TIMEOUT: Duration = Duration::from_secs(30);
const METADATA_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_METADATA_HOST: &str = "metadata.google.internal";
const PROVIDER_NAME: &str = "agent-sdk-web-identity";

/// Where a web identity token comes from.
#[derive(Clone, Debug)]
pub enum WebIdentityToken {
    /// A file the platform keeps current (`AWS_WEB_IDENTITY_TOKEN_FILE`).
    /// Read again at every refresh, since platforms rotate it.
    File(PathBuf),
    /// A Google ID token for the workload's service account, minted by the
    /// Compute Engine metadata server with this audience. The host is
    /// `GCE_METADATA_HOST` when set, else `metadata.google.internal`.
    GcpMetadata { audience: String },
}

/// Temporary credentials from STS `AssumeRoleWithWebIdentity`, cached until
/// five minutes before they expire.
///
/// A source that refuses (an unreadable token file, or an identity provider
/// answering 4xx) fails with [`CredentialsError::InvalidConfiguration`],
/// which retrying will not fix; a network failure or a 5xx fails with
/// [`CredentialsError::ProviderError`].
#[derive(Debug)]
pub struct WebIdentityCredentials {
    client: reqwest::Client,
    role_arn: String,
    session_name: String,
    token: WebIdentityToken,
    sts_endpoint: String,
    metadata_endpoint: String,
    duration: Option<Duration>,
    cached: tokio::sync::Mutex<Option<Credentials>>,
}

impl WebIdentityCredentials {
    /// Assume `role_arn` through the regional STS endpoint of `sts_region`.
    #[must_use]
    pub fn new(
        role_arn: impl Into<String>,
        session_name: impl Into<String>,
        token: WebIdentityToken,
        sts_region: &str,
    ) -> Self {
        let metadata_host =
            non_empty_env("GCE_METADATA_HOST").unwrap_or_else(|| DEFAULT_METADATA_HOST.to_owned());
        Self {
            client: crate::tls_roots::http_client_builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_else(|error| {
                    log::error!(
                        "failed to build the STS HTTP client, falling back to reqwest defaults: {error}"
                    );
                    reqwest::Client::default()
                }),
            role_arn: role_arn.into(),
            session_name: session_name.into(),
            token,
            sts_endpoint: format!("https://sts.{sts_region}.amazonaws.com/"),
            metadata_endpoint: format!("http://{metadata_host}"),
            duration: None,
            cached: tokio::sync::Mutex::new(None),
        }
    }

    /// `AWS_ROLE_ARN` with `AWS_WEB_IDENTITY_TOKEN_FILE` and the optional
    /// `AWS_ROLE_SESSION_NAME`; `None` when either of the first two is unset.
    #[must_use]
    pub fn from_env(sts_region: &str) -> Option<Self> {
        let role_arn = non_empty_env("AWS_ROLE_ARN")?;
        let token_file = non_empty_env("AWS_WEB_IDENTITY_TOKEN_FILE")?;
        let session_name = non_empty_env("AWS_ROLE_SESSION_NAME")
            .unwrap_or_else(|| "agent-sdk-bedrock".to_owned());
        Some(Self::new(
            role_arn,
            session_name,
            WebIdentityToken::File(PathBuf::from(token_file)),
            sts_region,
        ))
    }

    /// Ask STS for sessions of this length (900 seconds up to the role's
    /// maximum). STS defaults to one hour.
    #[must_use]
    pub const fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Send STS calls to another endpoint (tests, VPC endpoints).
    #[must_use]
    pub fn with_sts_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.sts_endpoint = endpoint.into();
        self
    }

    /// Fetch Google ID tokens from another metadata server (tests).
    #[must_use]
    pub fn with_metadata_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.metadata_endpoint = endpoint.into();
        self
    }

    /// Cached credentials while they stay fresh, else new ones from STS.
    ///
    /// # Errors
    ///
    /// Returns a [`CredentialsError`] when the token or STS fails.
    pub async fn credentials(&self) -> Result<Credentials, CredentialsError> {
        let mut cached = self.cached.lock().await;
        if let Some(credentials) = cached.as_ref()
            && is_fresh(credentials, SystemTime::now())
        {
            return Ok(credentials.clone());
        }
        let fresh = self.assume_role().await?;
        *cached = Some(fresh.clone());
        drop(cached);
        Ok(fresh)
    }

    /// Forget the cached credentials after AWS rejected them as expired.
    /// Returns whether anything was dropped, which is when retrying can help.
    pub async fn invalidate(&self) -> bool {
        self.cached.lock().await.take().is_some()
    }

    async fn web_identity_token(&self) -> Result<String, CredentialsError> {
        match &self.token {
            WebIdentityToken::File(path) => tokio::fs::read_to_string(path)
                .await
                .map(|token| token.trim().to_owned())
                .map_err(|error| {
                    CredentialsError::invalid_configuration(format!(
                        "reading the web identity token file {}: {error}",
                        path.display()
                    ))
                }),
            WebIdentityToken::GcpMetadata { audience } => {
                let url = format!(
                    "{}/computeMetadata/v1/instance/service-accounts/default/identity",
                    self.metadata_endpoint
                );
                let response = self
                    .client
                    .get(url)
                    .query(&[("audience", audience.as_str()), ("format", "full")])
                    .header("Metadata-Flavor", "Google")
                    .timeout(METADATA_TIMEOUT)
                    .send()
                    .await
                    .map_err(CredentialsError::provider_error)?;
                let status = response.status();
                let body = response
                    .text()
                    .await
                    .map_err(CredentialsError::provider_error)?;
                if status.is_success() {
                    return Ok(body.trim().to_owned());
                }
                Err(refusal(
                    status,
                    format!("the metadata server answered {status} for an ID token: {body}"),
                ))
            }
        }
    }

    async fn assume_role(&self) -> Result<Credentials, CredentialsError> {
        let token = self.web_identity_token().await?;
        let form = {
            let mut form = url::form_urlencoded::Serializer::new(String::new());
            form.append_pair("Action", "AssumeRoleWithWebIdentity")
                .append_pair("Version", "2011-06-15")
                .append_pair("RoleArn", &self.role_arn)
                .append_pair("RoleSessionName", &self.session_name)
                .append_pair("WebIdentityToken", &token);
            if let Some(duration) = self.duration {
                form.append_pair("DurationSeconds", &duration.as_secs().to_string());
            }
            form.finish()
        };
        let response = self
            .client
            .post(&self.sts_endpoint)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .timeout(STS_TIMEOUT)
            .body(form)
            .send()
            .await
            .map_err(CredentialsError::provider_error)?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(CredentialsError::provider_error)?;
        if !status.is_success() {
            return Err(refusal(
                status,
                format!(
                    "STS AssumeRoleWithWebIdentity answered {status}: {}",
                    sts_error(&body)
                ),
            ));
        }
        parse_assume_role_response(&body)
    }
}

impl ProvideCredentials for WebIdentityCredentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(self.credentials())
    }
}

/// How a Bedrock request authenticates.
#[derive(Clone)]
pub enum BedrockAuth {
    /// A Bedrock API key, sent as `Authorization: Bearer`
    /// (<https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html>).
    ApiKey(String),
    /// `SigV4` with credentials from any AWS credentials provider.
    Credentials(SharedCredentialsProvider),
    /// `SigV4` with web identity credentials, which the provider drops and
    /// fetches again once when AWS rejects them as expired.
    WebIdentity(Arc<WebIdentityCredentials>),
}

impl BedrockAuth {
    /// The conventional environment variable holding a Bedrock API key.
    pub const API_KEY_ENV: &'static str = "AWS_BEARER_TOKEN_BEDROCK";

    /// The first source the environment configures, in this order: a Bedrock
    /// API key ([`API_KEY_ENV`](Self::API_KEY_ENV)), static keys
    /// (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and the optional
    /// `AWS_SESSION_TOKEN`), web identity from a token file
    /// ([`WebIdentityCredentials::from_env`], through STS in `region`).
    ///
    /// Shared config files (`AWS_PROFILE`, SSO), EC2 instance profiles and ECS
    /// or EKS Pod Identity are not read here; pass `aws-config`'s provider
    /// as [`BedrockAuth::Credentials`] for those.
    ///
    /// # Errors
    ///
    /// Returns an error when the environment configures none of them.
    pub fn from_env(region: &str) -> anyhow::Result<Self> {
        if let Some(api_key) = non_empty_env(Self::API_KEY_ENV) {
            return Ok(Self::ApiKey(api_key));
        }
        if let (Some(access_key_id), Some(secret_access_key)) = (
            non_empty_env("AWS_ACCESS_KEY_ID"),
            non_empty_env("AWS_SECRET_ACCESS_KEY"),
        ) {
            return Ok(Self::Credentials(SharedCredentialsProvider::new(
                Credentials::new(
                    access_key_id,
                    secret_access_key,
                    non_empty_env("AWS_SESSION_TOKEN"),
                    None,
                    "environment",
                ),
            )));
        }
        if let Some(web_identity) = WebIdentityCredentials::from_env(region) {
            return Ok(Self::WebIdentity(Arc::new(web_identity)));
        }
        anyhow::bail!(
            "no Bedrock credentials in the environment: set {}, or AWS_ACCESS_KEY_ID and \
             AWS_SECRET_ACCESS_KEY, or AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE",
            Self::API_KEY_ENV
        )
    }
}

impl std::fmt::Debug for BedrockAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => formatter.write_str("BedrockAuth::ApiKey(<redacted>)"),
            Self::Credentials(_) => formatter.write_str("BedrockAuth::Credentials"),
            Self::WebIdentity(_) => formatter.write_str("BedrockAuth::WebIdentity"),
        }
    }
}

fn is_fresh(credentials: &Credentials, now: SystemTime) -> bool {
    credentials
        .expiry()
        .is_none_or(|expiry| now + REFRESH_MARGIN < expiry)
}

/// A 4xx is a refusal retrying will not fix; anything else may pass later.
fn refusal(status: reqwest::StatusCode, message: String) -> CredentialsError {
    if status.is_client_error() {
        CredentialsError::invalid_configuration(message)
    } else {
        CredentialsError::provider_error(message)
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// The text of the first `<tag>` element inside `scope`, entity-decoded.
fn xml_text<'a>(scope: &'a str, tag: &str) -> Option<std::borrow::Cow<'a, str>> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = scope.find(&open)? + open.len();
    let end = start + scope[start..].find(&close)?;
    let raw = &scope[start..end];
    if raw.contains('&') {
        Some(std::borrow::Cow::Owned(
            raw.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&apos;", "'")
                .replace("&amp;", "&"),
        ))
    } else {
        Some(std::borrow::Cow::Borrowed(raw))
    }
}

fn parse_assume_role_response(body: &str) -> Result<Credentials, CredentialsError> {
    let credentials = xml_text(body, "Credentials").ok_or_else(|| {
        CredentialsError::unhandled("STS AssumeRoleWithWebIdentity response carries no Credentials")
    })?;
    let field = |tag: &str| {
        xml_text(&credentials, tag)
            .map(std::borrow::Cow::into_owned)
            .ok_or_else(|| CredentialsError::unhandled(format!("STS credentials carry no {tag}")))
    };
    let expiration = field("Expiration")?;
    let expires_at = DateTime::from_str(&expiration, Format::DateTime)
        .ok()
        .and_then(|expiry| SystemTime::try_from(expiry).ok())
        .ok_or_else(|| {
            CredentialsError::unhandled(format!("STS Expiration {expiration:?} is not a date-time"))
        })?;
    Ok(Credentials::new(
        field("AccessKeyId")?,
        field("SecretAccessKey")?,
        Some(field("SessionToken")?),
        Some(expires_at),
        PROVIDER_NAME,
    ))
}

/// `Code: Message` from an STS error body, or the body itself.
fn sts_error(body: &str) -> String {
    match (xml_text(body, "Code"), xml_text(body, "Message")) {
        (Some(code), Some(message)) => format!("{code}: {message}"),
        (Some(code), None) => code.into_owned(),
        _ => body.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sts_response(expiration: &str) -> String {
        // AWS's sample response (STS API reference), with an expiry the test sets.
        format!(
            r#"<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleWithWebIdentityResult>
    <SubjectFromWebIdentityToken>amzn1.account.AF6RHO7KZU5XRVQJGXK6HB56KR2A</SubjectFromWebIdentityToken>
    <Audience>client.5498841531868486423.1548@apps.example.com</Audience>
    <AssumedRoleUser>
      <Arn>arn:aws:sts::123456789012:assumed-role/FederatedWebIdentityRole/app1</Arn>
      <AssumedRoleId>AROACLKWSDQRAOEXAMPLE:app1</AssumedRoleId>
    </AssumedRoleUser>
    <Credentials>
      <SessionToken>AQoDYXdzEE0a8ANXXXXXXXXNO1ewxE5TijQyp+IEXAMPLE</SessionToken>
      <SecretAccessKey>wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY</SecretAccessKey>
      <Expiration>{expiration}</Expiration>
      <AccessKeyId>ASgeIAIOSFODNN7EXAMPLE</AccessKeyId>
    </Credentials>
    <SourceIdentity>SourceIdentityValue</SourceIdentity>
    <Provider>www.amazon.com</Provider>
  </AssumeRoleWithWebIdentityResult>
  <ResponseMetadata>
    <RequestId>ad4156e9-bce1-11e2-82e6-6b6efEXAMPLE</RequestId>
  </ResponseMetadata>
</AssumeRoleWithWebIdentityResponse>"#
        )
    }

    fn rfc3339_in(duration: Duration) -> anyhow::Result<String> {
        Ok(DateTime::from(SystemTime::now() + duration).fmt(Format::DateTime)?)
    }

    #[test]
    fn parses_the_sts_sample_response() -> anyhow::Result<()> {
        let credentials = parse_assume_role_response(&sts_response("2014-10-24T23:00:23Z"))?;
        assert_eq!(credentials.access_key_id(), "ASgeIAIOSFODNN7EXAMPLE");
        assert_eq!(
            credentials.secret_access_key(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY"
        );
        assert_eq!(
            credentials.session_token(),
            Some("AQoDYXdzEE0a8ANXXXXXXXXNO1ewxE5TijQyp+IEXAMPLE")
        );
        assert_eq!(
            credentials.expiry(),
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_414_191_623))
        );
        Ok(())
    }

    #[test]
    fn debug_never_prints_secrets() {
        let auth = format!(
            "{:?}",
            BedrockAuth::ApiKey("bedrock-api-key-value".to_owned())
        );
        assert!(!auth.contains("bedrock-api-key-value"));
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("token-value".to_owned()),
            None,
            "test",
        );
        let printed = format!("{credentials:?}");
        assert!(!printed.contains("EXAMPLEKEY") && !printed.contains("token-value"));
    }

    #[test]
    fn an_sts_error_names_its_code() {
        let body = r#"<ErrorResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><Error><Type>Sender</Type><Code>InvalidIdentityToken</Code><Message>Incorrect token audience</Message></Error></ErrorResponse>"#;
        assert_eq!(
            sts_error(body),
            "InvalidIdentityToken: Incorrect token audience"
        );
    }

    #[tokio::test]
    async fn exchanges_a_metadata_id_token_and_caches_until_near_expiry() -> anyhow::Result<()> {
        let metadata = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(
                "/computeMetadata/v1/instance/service-accounts/default/identity",
            ))
            .and(query_param("audience", "sts-audience"))
            .and(header("Metadata-Flavor", "Google"))
            .respond_with(ResponseTemplate::new(200).set_body_string("google.id.token\n"))
            .expect(1)
            .mount(&metadata)
            .await;
        let sts = MockServer::start().await;
        let expiration = rfc3339_in(Duration::from_hours(1))?;
        Mock::given(method("POST"))
            .and(body_string_contains("Action=AssumeRoleWithWebIdentity"))
            .and(body_string_contains("WebIdentityToken=google.id.token"))
            .and(body_string_contains(
                "RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fexample",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string(sts_response(&expiration)))
            .expect(1)
            .mount(&sts)
            .await;

        let source = WebIdentityCredentials::new(
            "arn:aws:iam::123456789012:role/example",
            "session",
            WebIdentityToken::GcpMetadata {
                audience: "sts-audience".to_owned(),
            },
            "us-east-1",
        )
        .with_sts_endpoint(sts.uri())
        .with_metadata_endpoint(metadata.uri());

        let first = source.provide_credentials().await?;
        let second = source.provide_credentials().await?;
        assert_eq!(first.access_key_id(), "ASgeIAIOSFODNN7EXAMPLE");
        assert_eq!(second.access_key_id(), first.access_key_id());
        Ok(())
    }

    #[tokio::test]
    async fn refreshes_credentials_inside_the_margin_and_after_invalidation() -> anyhow::Result<()>
    {
        let token_file = tempfile::NamedTempFile::new()?;
        std::fs::write(token_file.path(), "file.token")?;
        let sts = MockServer::start().await;
        let near_expiry = rfc3339_in(Duration::from_mins(2))?;
        Mock::given(method("POST"))
            .and(body_string_contains("WebIdentityToken=file.token"))
            .respond_with(ResponseTemplate::new(200).set_body_string(sts_response(&near_expiry)))
            .expect(3)
            .mount(&sts)
            .await;

        let source = WebIdentityCredentials::new(
            "arn:aws:iam::123456789012:role/example",
            "session",
            WebIdentityToken::File(token_file.path().to_path_buf()),
            "us-east-1",
        )
        .with_sts_endpoint(sts.uri());

        // Expiring inside the margin: every call refreshes.
        source.credentials().await?;
        source.credentials().await?;
        assert!(source.invalidate().await);
        source.credentials().await?;
        Ok(())
    }

    #[tokio::test]
    async fn an_sts_rejection_is_an_invalid_configuration() -> anyhow::Result<()> {
        let token_file = tempfile::NamedTempFile::new()?;
        std::fs::write(token_file.path(), "file.token")?;
        let sts = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                "<ErrorResponse><Error><Type>Sender</Type><Code>AccessDenied</Code>\
                 <Message>Not authorized to perform sts:AssumeRoleWithWebIdentity</Message>\
                 </Error></ErrorResponse>",
            ))
            .mount(&sts)
            .await;
        let source = WebIdentityCredentials::new(
            "arn:aws:iam::123456789012:role/example",
            "session",
            WebIdentityToken::File(token_file.path().to_path_buf()),
            "us-east-1",
        )
        .with_sts_endpoint(sts.uri());
        let error = source
            .credentials()
            .await
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            matches!(error, CredentialsError::InvalidConfiguration(_)),
            "{error:?}"
        );
        assert!(format!("{error:?}").contains("AccessDenied"), "{error:?}");
        Ok(())
    }
}
