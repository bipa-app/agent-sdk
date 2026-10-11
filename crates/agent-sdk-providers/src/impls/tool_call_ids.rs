//! Durable tool-call ids for `OpenAI`-compatible responses.
//!
//! A tool-use id keys the pairing of a call with its result for the whole
//! life of a thread, so it must never repeat. Gateways serving open-weight
//! models such as Kimi return the id the model wrote itself: `<name>:<index>`,
//! `<name>_<index>` or `functions.<name>:<index>`, where the index restarts
//! whenever the model loses count, for example after compaction shrinks the
//! visible history. Other gateways number calls with generic ids such as
//! `call_0` or `0`. Those ids repeat across turns, so on any route other than
//! `api.openai.com` they are replaced here, before anything is streamed or
//! persisted.

use std::collections::HashSet;

use agent_sdk_foundation::llm::{ChatOutcome, Content, ContentBlock, Message};

/// `OpenAI` Chat Completions rejects a tool-call id longer than this.
const MAX_TOOL_CALL_ID_LEN: usize = 40;
/// Kimi's tool-call parsers only accept a decimal index after the stem.
const MINTED_INDEX_DIGITS: usize = 12;
const MINTED_INDEX_MODULUS: u64 = 10_u64.pow(12);
/// Provider ids shorter than this carry too little entropy to stay unique.
const MIN_OPAQUE_ID_LEN: usize = 8;
/// A trailing number this long or shorter counts a call rather than naming it.
const MAX_INDEX_DIGITS: usize = 9;
/// Generic words gateways put in front of a call index, compared lowercase.
const INDEX_PREFIXES: &[&str] = &[
    "call",
    "calls",
    "fc",
    "func",
    "function",
    "functions",
    "id",
    "tc",
    "tool",
    "tool_call",
    "tool-call",
    "toolcall",
    "tool_use",
    "tooluse",
];

/// Who wrote the tool-call ids a route returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolCallIdOrigin {
    /// `api.openai.com`, whose ids are opaque and unique: only an empty or
    /// repeated id is replaced.
    FirstParty,
    /// Any other `OpenAI`-compatible endpoint: ids that only number the call
    /// are replaced as well.
    #[default]
    Gateway,
}

impl ToolCallIdOrigin {
    pub fn of_base_url(base_url: &str) -> Self {
        if super::openai::is_official_openai_base_url(base_url) {
            Self::FirstParty
        } else {
            Self::Gateway
        }
    }
}

/// Give every tool use in a successful decoded response its durable id.
pub fn with_unique_tool_ids(
    history: &[Message],
    origin: ToolCallIdOrigin,
    mut outcome: ChatOutcome,
) -> ChatOutcome {
    if let ChatOutcome::Success(response) = &mut outcome {
        ToolCallIdAssigner::for_history(history, origin).assign_blocks(&mut response.content);
    }
    outcome
}

/// Assigns the id each tool call in one response is stored under.
///
/// Chat Completions and the stateless Responses input carry the stored id in
/// both the assistant tool call and its result, so a replaced id never has to
/// be translated back to the provider's original.
#[derive(Debug, Default)]
pub struct ToolCallIdAssigner {
    taken: HashSet<String>,
    origin: ToolCallIdOrigin,
}

impl ToolCallIdAssigner {
    /// Start from every tool-use and tool-result id already in the history.
    pub fn for_history(messages: &[Message], origin: ToolCallIdOrigin) -> Self {
        let mut taken = HashSet::new();
        for message in messages {
            let Content::Blocks(blocks) = &message.content else {
                continue;
            };
            for block in blocks {
                if let ContentBlock::ToolUse { id, .. } = block {
                    taken.insert(id.clone());
                } else if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                    taken.insert(tool_use_id.clone());
                }
            }
        }
        Self { taken, origin }
    }

    /// Return the id to store for one provider tool call.
    ///
    /// The provider's id is kept when it is present, not yet used, and opaque.
    /// A `<name>:<index>` id keeps its `<name>:` stem and gets a random index:
    /// Kimi's chat template shows the model a past call by its id alone, so the
    /// stem is how it still sees which function ran.
    pub fn assign(&mut self, provider_id: &str, tool_name: &str) -> String {
        let shape = match self.origin {
            ToolCallIdOrigin::FirstParty => ProviderIdShape::Opaque,
            ToolCallIdOrigin::Gateway => ProviderIdShape::of(provider_id, tool_name),
        };
        let id = match shape {
            ProviderIdShape::Opaque
                if !provider_id.is_empty() && !self.taken.contains(provider_id) =>
            {
                provider_id.to_owned()
            }
            ProviderIdShape::Opaque | ProviderIdShape::Counter => self.mint(None),
            ProviderIdShape::NamedIndex(stem) => self.mint(Some(stem)),
        };
        if id != provider_id {
            log::debug!(
                "Replaced provider tool-call id provider_id={provider_id:?} tool={tool_name} id={id}"
            );
        }
        self.taken.insert(id.clone());
        id
    }

    /// Replace the id of every tool use in a decoded response.
    pub fn assign_blocks(&mut self, blocks: &mut [ContentBlock]) {
        for block in blocks {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                *id = self.assign(id, name);
            }
        }
    }

    fn mint(&self, stem: Option<&str>) -> String {
        loop {
            let candidate = fresh_id(stem);
            if !self.taken.contains(&candidate) {
                return candidate;
            }
        }
    }
}

enum ProviderIdShape<'a> {
    /// Unique by construction.
    Opaque,
    /// `<name>:<n>`, `<name>_<n>` or `functions.<name>:<n>`, with its stem.
    NamedIndex(&'a str),
    /// Empty, short, all digits, or a generic word and a small index.
    Counter,
}

impl<'a> ProviderIdShape<'a> {
    fn of(id: &'a str, tool_name: &str) -> Self {
        if id.is_empty() {
            return Self::Counter;
        }
        let stem = id.trim_end_matches(|c: char| c.is_ascii_digit());
        let index_digits = id.len() - stem.len();
        if index_digits > 0
            && let Some(prefix) = stem.strip_suffix([':', '_'])
            && prefix.strip_prefix("functions.").unwrap_or(prefix) == tool_name
        {
            return Self::NamedIndex(stem);
        }
        if id.len() < MIN_OPAQUE_ID_LEN {
            return Self::Counter;
        }
        let prefix = stem
            .strip_suffix([':', '_', '-', '.'])
            .unwrap_or(stem)
            .to_ascii_lowercase();
        let counts_calls = prefix.is_empty()
            || prefix.starts_with("functions.")
            || INDEX_PREFIXES.contains(&prefix.as_str());
        if index_digits > 0 && index_digits <= MAX_INDEX_DIGITS && counts_calls {
            Self::Counter
        } else {
            Self::Opaque
        }
    }
}

fn fresh_id(stem: Option<&str>) -> String {
    let random = uuid::Uuid::new_v4();
    match stem {
        Some(stem) if stem.len() + MINTED_INDEX_DIGITS <= MAX_TOOL_CALL_ID_LEN => {
            format!(
                "{stem}{:0width$}",
                random.as_u64_pair().0 % MINTED_INDEX_MODULUS,
                width = MINTED_INDEX_DIGITS
            )
        }
        Some(_) | None => format!("call_{}", random.simple()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(ids: &[(&str, &str)]) -> Vec<Message> {
        let mut messages = vec![Message::user("start")];
        for (id, name) in ids {
            messages.push(Message::assistant_with_content(vec![
                ContentBlock::ToolUse {
                    id: (*id).to_owned(),
                    name: (*name).to_owned(),
                    input: serde_json::json!({}),
                    thought_signature: None,
                },
            ]));
            messages.push(Message::tool_result(*id, "ok", false));
        }
        messages
    }

    fn gateway(history: &[Message]) -> ToolCallIdAssigner {
        ToolCallIdAssigner::for_history(history, ToolCallIdOrigin::Gateway)
    }

    fn assert_reminted_with_stem(minted: &str, original: &str, stem: &str) {
        assert_ne!(minted, original);
        let index = minted.strip_prefix(stem).unwrap_or_default();
        assert!(
            !index.is_empty() && index.chars().all(|c| c.is_ascii_digit()),
            "{minted} should keep the {stem} stem with a decimal index"
        );
        assert!(minted.len() <= MAX_TOOL_CALL_ID_LEN, "{minted} is too long");
    }

    fn assert_fresh_call_id(minted: &str, original: &str) {
        assert_ne!(minted, original);
        assert!(
            minted.starts_with("call_") && minted.len() > "call_".len(),
            "{minted} should be a fresh call_ id"
        );
        assert!(minted.len() <= MAX_TOOL_CALL_ID_LEN, "{minted} is too long");
    }

    #[test]
    fn positional_id_that_repeats_history_is_replaced() {
        let mut ids = gateway(&history(&[("bash:0", "bash"), ("bash:1", "bash")]));

        let first = ids.assign("bash:0", "bash");
        let second = ids.assign("bash:1", "bash");

        assert_reminted_with_stem(&first, "bash:0", "bash:");
        assert_reminted_with_stem(&second, "bash:1", "bash:");
        assert_ne!(first, second);
    }

    #[test]
    fn every_positional_shape_is_replaced_even_without_a_visible_collision() {
        let mut ids = ToolCallIdAssigner::default();

        for (provider_id, name, stem) in [
            ("bash:0", "bash", "bash:"),
            ("bash_8", "bash", "bash_"),
            ("read:25", "read", "read:"),
            ("functions.read:3", "read", "functions.read:"),
            ("link_fetch:1", "link_fetch", "link_fetch:"),
            (
                "functions.link_fetch:1",
                "link_fetch",
                "functions.link_fetch:",
            ),
        ] {
            let minted = ids.assign(provider_id, name);
            assert_reminted_with_stem(&minted, provider_id, stem);
        }
    }

    #[test]
    fn counter_ids_get_fresh_call_ids_even_without_a_visible_collision() {
        let mut ids = ToolCallIdAssigner::default();

        for (provider_id, name) in [
            ("0", "read"),
            ("17", "read"),
            ("call_0", "read"),
            ("call_123", "read_file"),
            ("toolcall-42", "read"),
            ("tool_1", "bash"),
            ("tool_call_12", "bash"),
            ("Tool_Call:2", "bash"),
            ("fc_3", "bash"),
            ("functions.read:3", "bash"),
            ("x1", "bash"),
            ("abc", "bash"),
        ] {
            let minted = ids.assign(provider_id, name);
            assert_fresh_call_id(&minted, provider_id);
        }
    }

    #[test]
    fn opaque_unique_ids_pass_through() {
        let mut ids = gateway(&history(&[("call_history", "read")]));

        for (provider_id, name) in [
            ("call_a31qKPFi7qB2YK3WxyXVjTuC", "read"),
            ("call_a31qKPFi7qB2YK3WxyXVj123", "read"),
            ("call_8473629104857362", "read"),
            ("toolu_01ABC", "bash"),
            ("lookup_v2", "lookup_v2"),
            ("D681PevKs", "bash"),
        ] {
            assert_eq!(ids.assign(provider_id, name), provider_id);
        }
    }

    #[test]
    fn first_party_ids_are_replaced_only_when_missing_or_repeated() {
        let mut ids = ToolCallIdAssigner::for_history(
            &history(&[("call_history", "read")]),
            ToolCallIdOrigin::FirstParty,
        );

        assert_eq!(ids.assign("call_0", "read"), "call_0");
        assert_eq!(ids.assign("bash:0", "bash"), "bash:0");
        assert_fresh_call_id(&ids.assign("", "read"), "");
        assert_fresh_call_id(&ids.assign("call_0", "read"), "call_0");
        assert_fresh_call_id(&ids.assign("call_history", "read"), "call_history");
    }

    #[test]
    fn missing_and_duplicate_ids_get_fresh_unique_ids() {
        let mut ids = gateway(&history(&[("call_history", "read")]));

        let empty = ids.assign("", "read");
        let first = ids.assign("call_same", "read");
        let duplicate = ids.assign("call_same", "read");
        let historical = ids.assign("call_history", "read");

        assert_eq!(first, "call_same");
        for minted in [&empty, &duplicate, &historical] {
            assert_fresh_call_id(minted, "call_same");
        }
        let all: HashSet<&String> = [&empty, &first, &duplicate, &historical]
            .into_iter()
            .collect();
        assert_eq!(all.len(), 4);
        assert_ne!(historical, "call_history");
    }

    #[test]
    fn minted_ids_never_repeat() {
        let mut ids = ToolCallIdAssigner::default();
        let minted: HashSet<String> = (0..2_000).map(|_| ids.assign("bash:0", "bash")).collect();
        assert_eq!(minted.len(), 2_000);
        assert!(!minted.contains("bash:0"));
    }

    #[test]
    fn minted_ids_fit_the_chat_completions_length_limit() {
        let long_name = "x".repeat(64);
        let long_positional = format!("functions.{long_name}:0");
        let mut ids = ToolCallIdAssigner::default();

        for _ in 0..200 {
            for (provider_id, name) in [
                ("functions.link_fetch:0", "link_fetch"),
                ("call_0", "read"),
                ("", "read"),
                (long_positional.as_str(), long_name.as_str()),
            ] {
                let minted = ids.assign(provider_id, name);
                assert!(minted.len() <= MAX_TOOL_CALL_ID_LEN, "{minted} is too long");
            }
        }
    }

    #[test]
    fn assign_blocks_rewrites_only_tool_uses() {
        let mut ids = ToolCallIdAssigner::default();
        let mut blocks = vec![
            ContentBlock::Text {
                text: "bash:0".to_owned(),
            },
            ContentBlock::ToolUse {
                id: "bash:0".to_owned(),
                name: "bash".to_owned(),
                input: serde_json::json!({}),
                thought_signature: None,
            },
        ];

        ids.assign_blocks(&mut blocks);

        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "bash:0"));
        assert!(
            matches!(&blocks[1], ContentBlock::ToolUse { id, .. } if id != "bash:0" && id.starts_with("bash:"))
        );
    }
}
