//! `SigV4` signing of the Bedrock runtime requests, by AWS's `aws-sigv4`.
//!
//! The request has no query string and is signed over `host` and the body
//! only, so signing another header or a query needs a change here. The
//! signer adds `x-amz-date`, `authorization` and, with temporary credentials,
//! `x-amz-security-token`. The URL carries the model id percent-encoded once,
//! and the signer encodes each path segment again, as services other than S3
//! require.

use std::time::SystemTime;

use anyhow::Context;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;

/// One request, as it goes on the wire.
pub struct SigningInput<'a> {
    pub method: &'a str,
    /// The full URL, path already percent-encoded once.
    pub url: &'a str,
    /// The `Host` header value: the URL host, with its port when the URL names
    /// one.
    pub host: &'a str,
    pub payload: &'a [u8],
    pub region: &'a str,
    pub service: &'a str,
    pub time: SystemTime,
}

/// The headers that sign the request, as `(name, value)` pairs.
pub fn signed_headers(
    input: &SigningInput<'_>,
    credentials: Credentials,
) -> anyhow::Result<Vec<(&'static str, String)>> {
    let identity = Identity::from(credentials);
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(input.region)
        .name(input.service)
        .time(input.time)
        .settings(SigningSettings::default())
        .build()
        .context("SigV4 signing parameters")?
        .into();
    let request = SignableRequest::new(
        input.method,
        input.url,
        [("host", input.host)].into_iter(),
        SignableBody::Bytes(input.payload),
    )
    .context("a SigV4 signable request")?;
    let (instructions, _signature) = sign(request, &params)
        .context("SigV4 signing")?
        .into_parts();
    let (headers, _params) = instructions.into_parts();
    Ok(headers
        .into_iter()
        .map(|header| (header.name(), header.value().to_owned()))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    // AWS's published example credentials, used by the SigV4 test suite.
    const ACCESS_KEY_ID: &str = "AKIDEXAMPLE";
    const SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    /// 2015-08-30T12:36:00Z, the suite's timestamp.
    const SUITE_TIME: u64 = 1_440_938_160;
    /// 2026-11-17T12:00:00Z.
    const BEDROCK_TIME: u64 = 1_794_916_800;

    fn credentials(access_key_id: &str, secret: &str, token: Option<&str>) -> Credentials {
        Credentials::new(
            access_key_id,
            secret,
            token.map(str::to_owned),
            None,
            "test",
        )
    }

    fn header<'a>(headers: &'a [(&'static str, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(header, _)| *header == name)
            .map(|(_, value)| value.as_str())
    }

    fn suite_input<'a>(method: &'a str, host: &'a str, url: &'a str) -> SigningInput<'a> {
        SigningInput {
            method,
            url,
            host,
            payload: b"",
            region: "us-east-1",
            service: "service",
            time: SystemTime::UNIX_EPOCH + Duration::from_secs(SUITE_TIME),
        }
    }

    /// The suite's `post-vanilla` vector.
    #[test]
    fn signs_the_post_vanilla_vector() -> anyhow::Result<()> {
        let headers = signed_headers(
            &suite_input(
                "POST",
                "example.amazonaws.com",
                "https://example.amazonaws.com/",
            ),
            credentials(ACCESS_KEY_ID, SECRET_ACCESS_KEY, None),
        )?;
        assert_eq!(header(&headers, "x-amz-date"), Some("20150830T123600Z"));
        assert_eq!(
            header(&headers, "authorization"),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
                 SignedHeaders=host;x-amz-date, \
                 Signature=5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b"
            )
        );
        assert_eq!(header(&headers, "x-amz-security-token"), None);
        Ok(())
    }

    /// The suite's `get-vanilla-with-session-token` vector: the token is a
    /// signed header.
    #[test]
    fn signs_the_session_token_vector() -> anyhow::Result<()> {
        let token = "6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267";
        let headers = signed_headers(
            &suite_input(
                "GET",
                "example.amazonaws.com",
                "https://example.amazonaws.com/",
            ),
            credentials(ACCESS_KEY_ID, SECRET_ACCESS_KEY, Some(token)),
        )?;
        assert_eq!(
            header(&headers, "authorization"),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
                 SignedHeaders=host;x-amz-date;x-amz-security-token, \
                 Signature=07ec1639c89043aa0e3e2de82b96708f198cceab042d4a97044c66dd9f74e7f8"
            )
        );
        assert_eq!(header(&headers, "x-amz-security-token"), Some(token));
        Ok(())
    }

    /// The suite's `double-encode-path` vector: a path the URL already
    /// carries percent-encoded is encoded again, the case of a `...-v1:0`
    /// model id or an inference profile ARN in a Bedrock path.
    #[test]
    fn signs_the_double_encode_path_vector() -> anyhow::Result<()> {
        let headers = signed_headers(
            &suite_input(
                "POST",
                "tj9n5r0m12.execute-api.us-east-1.amazonaws.com",
                "https://tj9n5r0m12.execute-api.us-east-1.amazonaws.com/test/@connections/JBDvjfGEIAMCERw%3D",
            ),
            credentials("ANOTREAL", "notrealrnrELgWzOk3IfjzDKtFBhDby", None),
        )?;
        let authorization = header(&headers, "authorization").context("no authorization")?;
        assert!(
            authorization.ends_with(
                "Signature=57d157672191bac40bae387e48bbe14b15303c001fdbb01f4abf295dccb09705"
            ),
            "{authorization}"
        );
        Ok(())
    }

    /// Bedrock-shaped requests: the signing name, the region in the scope,
    /// and a model id and an inference profile ARN encoded in the path.
    #[test]
    fn signs_bedrock_requests_for_the_bedrock_service() -> anyhow::Result<()> {
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
        let cases = [
            (
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-opus-4-5-20251101-v1%3A0/invoke",
                None,
                "b933752df706cfa5bacb3260fe3bef1d4deb0a73c2b142156500f45112983e27",
            ),
            (
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-sonnet-5-5/invoke-with-response-stream",
                Some("session-token-example"),
                "d6ab3484151d5fad7e393720ece3f0ca0e7401a947229fdfa2705cd2a784685f",
            ),
        ];
        for (url, token, expected) in cases {
            let headers = signed_headers(
                &SigningInput {
                    method: "POST",
                    url,
                    host: "bedrock-runtime.us-east-1.amazonaws.com",
                    payload: body,
                    region: "us-east-1",
                    service: "bedrock",
                    time: SystemTime::UNIX_EPOCH + Duration::from_secs(BEDROCK_TIME),
                },
                credentials(ACCESS_KEY_ID, SECRET_ACCESS_KEY, token),
            )?;
            let authorization = header(&headers, "authorization").context("no authorization")?;
            assert!(
                authorization.starts_with(
                    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261117/us-east-1/bedrock/aws4_request"
                ) && authorization.ends_with(&format!("Signature={expected}")),
                "{url}: {authorization}"
            );
            assert_eq!(header(&headers, "x-amz-security-token"), token);
        }
        Ok(())
    }
}
