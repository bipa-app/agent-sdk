//! Provider HTTPS on hosts without a system CA store.
//!
//! A minimal container (debian-slim without `ca-certificates`, distroless,
//! `scratch`) ships no CA certificates, so every provider handshake fails
//! with "unable to get local issuer certificate". When the system store
//! yields no certificate at all, provider clients trust the bundled Mozilla
//! roots (`webpki-root-certs`) instead, and the process logs that once.
//!
//! "The system store" is what the selected backend itself would load:
//!
//! - native-tls: openssl-probe's `SSL_CERT_FILE` (else the first well-known
//!   bundle file) plus `SSL_CERT_DIR` and every well-known cert directory
//!   that exists. openssl-probe adds those directories even when
//!   `SSL_CERT_DIR` is set.
//! - rustls (when native-tls is off): rustls-native-certs, which
//!   rustls-platform-verifier uses. There `SSL_CERT_FILE` / `SSL_CERT_DIR`
//!   replace the platform paths.
//!
//! A custom CA configured through those variables keeps working, because
//! the store is then not empty and nothing is added. The decision counts
//! certificates that parse, not paths that exist, because a slim image
//! still has an empty `/etc/ssl/certs`.
//!
//! On native-tls the bundled roots are merged into what OpenSSL loads by
//! itself (`set_default_verify_paths`: the compiled-in `OPENSSLDIR`, a
//! colon-separated `SSL_CERT_DIR`), which openssl-probe does not see. On a
//! store that is really empty that is the bundled roots alone, and a CA kept
//! only there still works. rustls uses the bundled roots alone, because its
//! platform verifier refuses to build on an empty store.
//!
//! Only OpenSSL-style unix targets fall back. macOS, iOS, Windows and Android
//! keep their platform stores, which native-tls reads without these
//! variables, so an empty `SSL_CERT_FILE` says nothing about them.

/// A `reqwest` client builder for provider HTTPS. Use it in place of
/// `reqwest::Client::builder()`: it adds the bundled roots when the system
/// store is empty and is a plain builder otherwise.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    let builder = reqwest::Client::builder();
    #[cfg(all(
        any(feature = "tls-native-tls", feature = "tls-rustls"),
        unix,
        not(target_vendor = "apple"),
        not(target_os = "android"),
    ))]
    let builder = with_roots(builder, bundled::reqwest_roots());
    builder
}

/// `builder` trusting `roots` when there are any: merged on native-tls,
/// alone on rustls (see the module docs).
#[cfg(all(
    any(feature = "tls-native-tls", feature = "tls-rustls"),
    unix,
    not(target_vendor = "apple"),
    not(target_os = "android"),
))]
fn with_roots(
    builder: reqwest::ClientBuilder,
    roots: Option<&[reqwest::Certificate]>,
) -> reqwest::ClientBuilder {
    match roots {
        #[cfg(feature = "tls-native-tls")]
        Some(roots) => builder.tls_certs_merge(roots.iter().cloned()),
        #[cfg(not(feature = "tls-native-tls"))]
        Some(roots) => builder.tls_certs_only(roots.iter().cloned()),
        None => builder,
    }
}

/// The TLS connector for the Codex WebSocket: the bundled roots when the
/// system store is empty, otherwise `None` (tokio-tungstenite's default,
/// which reads the system store).
#[cfg(all(
    feature = "openai-codex",
    any(feature = "tls-native-tls", feature = "tls-rustls")
))]
pub fn websocket_connector() -> Option<tokio_tungstenite::Connector> {
    #[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
    {
        bundled::websocket_connector()
    }
    #[cfg(not(all(unix, not(target_vendor = "apple"), not(target_os = "android"))))]
    {
        None
    }
}

#[cfg(all(
    any(feature = "tls-native-tls", feature = "tls-rustls"),
    unix,
    not(target_vendor = "apple"),
    not(target_os = "android"),
))]
mod bundled {
    use std::sync::LazyLock;

    /// Whether this process trusts the bundled roots, decided once.
    static IN_USE: LazyLock<bool> = LazyLock::new(|| {
        let empty = system_store_is_empty();
        if empty {
            log::warn!(
                "the system CA store has no certificates; provider HTTPS trusts the \
                 {} bundled Mozilla roots instead (install ca-certificates or set \
                 SSL_CERT_FILE / SSL_CERT_DIR to use the system store)",
                webpki_root_certs::TLS_SERVER_ROOT_CERTS.len()
            );
        }
        empty
    });

    static REQWEST_ROOTS: LazyLock<Vec<reqwest::Certificate>> = LazyLock::new(|| {
        webpki_root_certs::TLS_SERVER_ROOT_CERTS
            .iter()
            .filter_map(|der| {
                reqwest::Certificate::from_der(der)
                    .inspect_err(|error| log::debug!("skipping a bundled root: {error}"))
                    .ok()
            })
            .collect()
    });

    #[cfg(feature = "tls-native-tls")]
    pub(super) fn system_store_is_empty() -> bool {
        system_store_is_empty_from(&openssl_probe::probe())
    }

    /// Whether native-tls finds no certificate in what `probe` names.
    #[cfg(feature = "tls-native-tls")]
    pub(super) fn system_store_is_empty_from(probe: &openssl_probe::ProbeResult) -> bool {
        !openssl_paths_hold_a_certificate(probe.cert_file.as_deref(), &probe.cert_dir)
    }

    #[cfg(not(feature = "tls-native-tls"))]
    pub(super) fn system_store_is_empty() -> bool {
        let found = rustls_native_certs::load_native_certs();
        for error in &found.errors {
            log::debug!("reading the system CA store: {error}");
        }
        found.certs.is_empty()
    }

    /// Whether `file` or any file in `dirs` holds a PEM certificate that
    /// parses, so OpenSSL has at least one root to verify against.
    #[cfg(feature = "tls-native-tls")]
    fn openssl_paths_hold_a_certificate(
        file: Option<&std::path::Path>,
        dirs: &[std::path::PathBuf],
    ) -> bool {
        file.is_some_and(holds_a_certificate)
            || dirs.iter().any(|dir| {
                std::fs::read_dir(dir).is_ok_and(|entries| {
                    entries
                        .flatten()
                        .any(|entry| holds_a_certificate(&entry.path()))
                })
            })
    }

    #[cfg(feature = "tls-native-tls")]
    fn holds_a_certificate(path: &std::path::Path) -> bool {
        use rustls_pki_types::pem::PemObject as _;
        std::fs::read(path).is_ok_and(|pem| {
            rustls_pki_types::CertificateDer::pem_slice_iter(&pem).any(|cert| cert.is_ok())
        })
    }

    pub(super) fn in_use() -> bool {
        *IN_USE
    }

    pub(super) fn reqwest_roots() -> Option<&'static [reqwest::Certificate]> {
        in_use().then(|| REQWEST_ROOTS.as_slice())
    }

    #[cfg(feature = "openai-codex")]
    pub(super) fn websocket_connector() -> Option<tokio_tungstenite::Connector> {
        if !in_use() {
            return None;
        }
        // native-tls wins when both backends are on, as in reqwest and
        // tokio-tungstenite.
        #[cfg(feature = "tls-native-tls")]
        {
            static CONNECTOR: LazyLock<Option<native_tls::TlsConnector>> = LazyLock::new(|| {
                // Merged into what OpenSSL loads by itself, as for reqwest.
                let mut builder = native_tls::TlsConnector::builder();
                for der in webpki_root_certs::TLS_SERVER_ROOT_CERTS {
                    if let Ok(root) = native_tls::Certificate::from_der(der) {
                        builder.add_root_certificate(root);
                    }
                }
                builder
                    .build()
                    .inspect_err(|error| {
                        log::warn!("building the bundled-root WebSocket TLS connector: {error}");
                    })
                    .ok()
            });
            CONNECTOR
                .clone()
                .map(tokio_tungstenite::Connector::NativeTls)
        }
        #[cfg(not(feature = "tls-native-tls"))]
        {
            static CONFIG: LazyLock<Option<std::sync::Arc<rustls::ClientConfig>>> =
                LazyLock::new(|| {
                    let mut roots = rustls::RootCertStore::empty();
                    roots.add_parsable_certificates(
                        webpki_root_certs::TLS_SERVER_ROOT_CERTS.iter().cloned(),
                    );
                    rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
                        rustls::crypto::aws_lc_rs::default_provider(),
                    ))
                    .with_safe_default_protocol_versions()
                    .inspect_err(|error| {
                        log::warn!("building the bundled-root WebSocket TLS config: {error}");
                    })
                    .ok()
                    .map(|config| {
                        std::sync::Arc::new(
                            config.with_root_certificates(roots).with_no_client_auth(),
                        )
                    })
                });
            CONFIG.clone().map(tokio_tungstenite::Connector::Rustls)
        }
    }
}

/// The decision reads process environment once per process, so each case
/// runs in a child process with its own `SSL_CERT_FILE` / `SSL_CERT_DIR`:
/// the parent test re-runs this binary on one ignored `child_*` test, which
/// does nothing unless [`CHILD_ENV`] is set.
#[cfg(all(
    test,
    any(feature = "tls-native-tls", feature = "tls-rustls"),
    unix,
    not(target_vendor = "apple"),
    not(target_os = "android"),
))]
mod tests {
    use super::{bundled, http_client_builder};
    use anyhow::{Context as _, Result, ensure};
    use std::io::{Read as _, Write as _};
    use std::path::Path;

    const CHILD_ENV: &str = "AGENT_SDK_TLS_ROOTS_CHILD";
    const RAN: &str = "tls-roots child ran";
    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls");

    /// Re-run this test binary on `child` with `envs`, and fail unless the
    /// child passed and actually ran its body.
    fn run_child(child: &str, envs: &[(&str, &std::ffi::OsStr)]) -> Result<()> {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args([
                &format!("tls_roots::tests::{child}"),
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, "1")
            .env_remove("SSL_CERT_FILE")
            .env_remove("SSL_CERT_DIR")
            .envs(envs.iter().copied())
            .output()
            .context("run the child test")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(
            output.status.success() && stdout.contains(RAN),
            "child {child} failed ({}):\n{stdout}\n{stderr}",
            output.status
        );
        Ok(())
    }

    fn is_child() -> bool {
        let child = std::env::var_os(CHILD_ENV).is_some();
        if child {
            println!("{RAN}");
        }
        child
    }

    /// One HTTPS request answered `204` by a server presenting the
    /// fixtures' `localhost` leaf, which only the fixtures' test CA signs.
    fn serve_one_request() -> Result<(u16, std::thread::JoinHandle<()>)> {
        let cert = std::fs::read(Path::new(FIXTURES).join("localhost.pem"))?;
        let key = std::fs::read(Path::new(FIXTURES).join("localhost-key.pem"))?;
        let acceptor =
            native_tls::TlsAcceptor::new(native_tls::Identity::from_pkcs8(&cert, &key)?)?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || {
            let Ok((socket, _)) = listener.accept() else {
                return;
            };
            // A client that rejects the certificate aborts the handshake.
            let Ok(mut stream) = acceptor.accept(socket) else {
                return;
            };
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => return,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }
            let _ = stream.write_all(
                b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
        });
        Ok((port, server))
    }

    #[cfg(not(feature = "tls-native-tls"))]
    fn empty_store(dir: &Path) -> Result<[(&'static str, std::ffi::OsString); 2]> {
        let file = dir.join("empty.pem");
        std::fs::write(&file, b"")?;
        let certs = dir.join("certs");
        std::fs::create_dir(&certs)?;
        Ok([
            ("SSL_CERT_FILE", file.into_os_string()),
            ("SSL_CERT_DIR", certs.into_os_string()),
        ])
    }

    // openssl-probe always adds the well-known cert dirs, so only rustls
    // sees an empty store through these variables on a host that has one;
    // native-tls is covered by `openssl_paths_*` below.
    #[cfg(not(feature = "tls-native-tls"))]
    #[test]
    fn an_empty_system_store_falls_back_to_the_bundled_roots() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let [file, certs] = empty_store(dir.path())?;
        run_child(
            "child_empty_store_uses_the_bundled_roots",
            &[(file.0, &file.1), (certs.0, &certs.1)],
        )
    }

    #[cfg(not(feature = "tls-native-tls"))]
    #[tokio::test]
    #[ignore = "run by an_empty_system_store_falls_back_to_the_bundled_roots in a child process"]
    async fn child_empty_store_uses_the_bundled_roots() -> Result<()> {
        if !is_child() {
            return Ok(());
        }
        ensure!(bundled::system_store_is_empty(), "the store was not empty");
        let roots = bundled::reqwest_roots().context("no bundled roots on an empty store")?;
        ensure!(roots.len() > 100, "only {} bundled roots", roots.len());
        #[cfg(feature = "openai-codex")]
        ensure!(
            super::websocket_connector().is_some(),
            "the WebSocket kept the empty system store"
        );
        // The client builds with them, and a CA outside them is still refused.
        let (port, server) = serve_one_request()?;
        let refused = http_client_builder()
            .no_proxy()
            .build()?
            .get(format!("https://localhost:{port}/"))
            .send()
            .await;
        ensure!(
            refused.is_err(),
            "a CA outside the bundled roots was trusted"
        );
        let _ = server.join();
        Ok(())
    }

    #[test]
    fn a_custom_ca_in_ssl_cert_file_wins_over_the_bundled_roots() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let ca = Path::new(FIXTURES).join("test-ca.pem").into_os_string();
        run_child(
            "child_custom_ca_is_trusted_and_nothing_is_added",
            &[
                ("SSL_CERT_FILE", &ca),
                ("SSL_CERT_DIR", dir.path().as_os_str()),
            ],
        )
    }

    #[tokio::test]
    #[ignore = "run by a_custom_ca_in_ssl_cert_file_wins_over_the_bundled_roots in a child process"]
    async fn child_custom_ca_is_trusted_and_nothing_is_added() -> Result<()> {
        if !is_child() {
            return Ok(());
        }
        ensure!(
            !bundled::system_store_is_empty() && bundled::reqwest_roots().is_none(),
            "the bundled roots were added beside a custom CA"
        );
        #[cfg(feature = "openai-codex")]
        ensure!(
            super::websocket_connector().is_none(),
            "the WebSocket replaced a custom CA"
        );
        let (port, server) = serve_one_request()?;
        let response = http_client_builder()
            .no_proxy()
            .build()?
            .get(format!("https://localhost:{port}/"))
            .send()
            .await
            .context("the custom CA was not trusted")?;
        ensure!(
            response.status() == reqwest::StatusCode::NO_CONTENT,
            "unexpected status {}",
            response.status()
        );
        let _ = server.join();
        Ok(())
    }

    /// By hand, rustls only: `cargo test -p agent-sdk-providers
    /// --no-default-features --features tls-rustls tls_roots -- --ignored`.
    #[cfg(not(feature = "tls-native-tls"))]
    #[test]
    #[ignore = "live: reaches api.anthropic.com over the network"]
    fn live_an_empty_store_still_reaches_a_provider() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let [file, certs] = empty_store(dir.path())?;
        run_child(
            "child_live_empty_store_reaches_a_provider",
            &[(file.0, &file.1), (certs.0, &certs.1)],
        )
    }

    #[cfg(not(feature = "tls-native-tls"))]
    #[tokio::test]
    #[ignore = "run by live_an_empty_store_still_reaches_a_provider in a child process"]
    async fn child_live_empty_store_reaches_a_provider() -> Result<()> {
        if !is_child() {
            return Ok(());
        }
        ensure!(bundled::in_use(), "the store was not empty");
        // Any HTTP status means the handshake verified.
        let response = http_client_builder()
            .build()?
            .get("https://api.anthropic.com/v1/models")
            .send()
            .await
            .context("HTTPS to api.anthropic.com over the bundled roots")?;
        println!("api.anthropic.com answered {}", response.status());
        Ok(())
    }

    #[cfg(feature = "tls-native-tls")]
    #[test]
    fn openssl_paths_without_a_certificate_are_an_empty_store() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("ca-certificates.crt");
        std::fs::write(&file, b"")?;
        let certs = dir.path().join("certs");
        std::fs::create_dir(&certs)?;
        // A slim image's leftovers: an empty bundle, a README, a dangling link.
        std::fs::write(certs.join("README"), b"no certificates here")?;
        std::os::unix::fs::symlink(dir.path().join("gone.pem"), certs.join("0123abcd.0"))?;
        ensure!(
            bundled::system_store_is_empty_from(&openssl_probe::ProbeResult {
                cert_file: Some(file),
                cert_dir: vec![certs],
            }),
            "an empty bundle and cert dir counted as a store"
        );
        ensure!(
            bundled::system_store_is_empty_from(&openssl_probe::ProbeResult {
                cert_file: None,
                cert_dir: vec![dir.path().join("missing")],
            }),
            "a missing cert dir counted as a store"
        );
        Ok(())
    }

    #[cfg(feature = "tls-native-tls")]
    #[test]
    fn one_parsable_certificate_in_a_bundle_or_dir_is_a_store() -> Result<()> {
        let ca = Path::new(FIXTURES).join("test-ca.pem");
        let dir = tempfile::tempdir()?;
        std::os::unix::fs::symlink(&ca, dir.path().join("5ed36f99.0"))?;
        ensure!(
            !bundled::system_store_is_empty_from(&openssl_probe::ProbeResult {
                cert_file: Some(ca),
                cert_dir: Vec::new(),
            }),
            "a bundle with a certificate counted as empty"
        );
        ensure!(
            !bundled::system_store_is_empty_from(&openssl_probe::ProbeResult {
                cert_file: None,
                cert_dir: vec![dir.path().to_path_buf()],
            }),
            "a cert dir with a certificate counted as empty"
        );
        Ok(())
    }

    /// The fallback roots reach the client the backend builds. The fixture
    /// CA stands in for the bundled roots, which no offline server chains
    /// to: a client given it as fallback roots trusts the fixture server,
    /// and a plain one, on the host's own store, does not.
    #[tokio::test]
    async fn fallback_roots_are_trusted_by_the_built_client() -> Result<()> {
        let pem = std::fs::read(Path::new(FIXTURES).join("test-ca.pem"))?;
        let roots = [reqwest::Certificate::from_pem(&pem)?];

        let (port, server) = serve_one_request()?;
        let response = super::with_roots(reqwest::Client::builder().no_proxy(), Some(&roots))
            .build()?
            .get(format!("https://localhost:{port}/"))
            .send()
            .await
            .context("the fallback roots did not reach the client")?;
        ensure!(
            response.status() == reqwest::StatusCode::NO_CONTENT,
            "unexpected status {}",
            response.status()
        );
        let _ = server.join();

        let (port, server) = serve_one_request()?;
        let refused = super::with_roots(reqwest::Client::builder().no_proxy(), None)
            .build()?
            .get(format!("https://localhost:{port}/"))
            .send()
            .await;
        ensure!(
            refused.is_err(),
            "the host store trusted the fixture CA, so this test proves nothing"
        );
        let _ = server.join();
        Ok(())
    }
}
