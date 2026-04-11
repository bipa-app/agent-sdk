fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = std::path::Path::new("../agent-service-host/proto");
    println!("cargo:rerun-if-changed={}", proto_root.display());

    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    let mut config = tonic_build::Config::new();
    config.protoc_executable(protoc);

    let proto_files = [
        proto_root.join("agent/service/v1/control.proto"),
        proto_root.join("agent/service/v1/events.proto"),
        proto_root.join("agent/service/v1/errors.proto"),
    ];

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos_with_config(
            config,
            &proto_files
                .iter()
                .map(std::path::PathBuf::as_path)
                .collect::<Vec<_>>(),
            &[proto_root],
        )?;

    Ok(())
}
