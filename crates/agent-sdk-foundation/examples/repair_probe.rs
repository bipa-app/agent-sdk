fn main() {
    let path = std::env::args().nth(1).expect("path");
    let raw = std::fs::read_to_string(path).expect("read");
    let messages: Vec<agent_sdk_foundation::llm::Message> =
        serde_json::from_str(&raw).expect("parse");
    eprintln!(
        "input: {} messages, valid={}",
        messages.len(),
        agent_sdk_foundation::llm::is_provider_valid_tool_sequence(&messages)
    );
    let repaired =
        agent_sdk_foundation::llm::repair_tool_sequence_in_place(&messages, "[cancelled]");
    let valid = agent_sdk_foundation::llm::is_provider_valid_tool_sequence(&repaired);
    eprintln!("repaired: {} messages, valid={}", repaired.len(), valid);
    assert!(valid, "founder projection must repair to a valid sequence");
    eprintln!("PROOF OK");
}
