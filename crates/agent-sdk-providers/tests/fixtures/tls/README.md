Test-only TLS fixtures for the bundled-root fallback tests (`src/tls_roots.rs`).

- `test-ca.pem`: a self-signed CA ("agent-sdk test CA"). Its key was discarded after signing.
- `localhost.pem` / `localhost-key.pem`: a leaf for `localhost` and `127.0.0.1`, signed by that CA, with its PKCS#8 key.

All three are valid for 100 years and trust nothing outside these tests.
