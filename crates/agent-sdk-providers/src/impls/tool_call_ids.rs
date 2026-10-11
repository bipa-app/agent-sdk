//! Durable tool-call ids for `OpenAI`-compatible responses.
//!
//! A tool-use id keys the pairing of a call with its result for the whole
//! life of a thread, so it must never repeat. Gateways serving open-weight
//! models such as Kimi return the id the model wrote itself: `<name>:<index>`,
//! `<name>_<index>` or `functions.<name>:<index>`, where the index restarts
//! whenever the model loses count, for example after compaction shrinks the
//! visible history. Those ids repeat across turns, so they are replaced here,
//! before anything is streamed or persisted.

use std::collections::HashSet;

use agent_sdk_foundation::llm::{ChatOutcome, Content, ContentBlock, Message};

/// Give every tool use in a successful decoded response its durable id.
pub fn with_unique_tool_ids(history: &[Message], mut outcome: ChatOutcome) -> ChatOutcome {
    if let ChatOutcome::Success(response) = &mut outcome {
        ToolCallIdAssigner::for_history(history).assign_blocks(&mut response.content);
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
}

impl ToolCallIdAssigner {
    /// Start from every tool-use and tool-result id already in the history.
    pub fn for_history(messages: &[Message]) -> Self {
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
        Self { taken }
    }

    /// Return the id to store for one provider tool call.
    ///
    /// The provider's id is kept when it is present, not yet used, and not
    /// positional. A positional id keeps its `<name>:` stem and gets a random
    /// index: Kimi's chat template shows the model a past call by its id
    /// alone, so the stem is how it still sees which function ran.
    pub fn assign(&mut self, provider_id: &str, tool_name: &str) -> String {
        let stem = positional_stem(provider_id, tool_name);
        let id = if stem.is_none() && !provider_id.is_empty() && !self.taken.contains(provider_id) {
            provider_id.to_owned()
        } else {
            let minted = loop {
                let candidate = stem.map_or_else(
                    || format!("call_{}", uuid::Uuid::new_v4().simple()),
                    |stem| format!("{stem}{}", uuid::Uuid::new_v4().as_u64_pair().0),
                );
                if !self.taken.contains(&candidate) {
                    break candidate;
                }
            };
            log::debug!(
                "Replaced provider tool-call id provider_id={provider_id:?} tool={tool_name} id={minted}"
            );
            minted
        };
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
}

/// The `<name>:` / `<name>_` / `functions.<name>:` stem of an id that only
/// numbers the call, or `None` when the id is not derived from the name.
fn positional_stem<'a>(id: &'a str, tool_name: &str) -> Option<&'a str> {
    let stem = id.trim_end_matches(|c: char| c.is_ascii_digit());
    if stem.len() == id.len() {
        return None;
    }
    let prefix = stem.strip_suffix(':').or_else(|| stem.strip_suffix('_'))?;
    let name = prefix.strip_prefix("functions.").unwrap_or(prefix);
    (name == tool_name).then_some(stem)
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

    fn assert_reminted_with_stem(minted: &str, original: &str, stem: &str) {
        assert_ne!(minted, original);
        let index = minted.strip_prefix(stem).unwrap_or_default();
        assert!(
            !index.is_empty() && index.chars().all(|c| c.is_ascii_digit()),
            "{minted} should keep the {stem} stem with a numeric index"
        );
    }

    #[test]
    fn positional_id_that_repeats_history_is_replaced() {
        let mut ids =
            ToolCallIdAssigner::for_history(&history(&[("bash:0", "bash"), ("bash:1", "bash")]));

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
        ] {
            let minted = ids.assign(provider_id, name);
            assert_reminted_with_stem(&minted, provider_id, stem);
        }
    }

    #[test]
    fn opaque_unique_ids_pass_through() {
        let mut ids = ToolCallIdAssigner::for_history(&history(&[("call_history", "read")]));

        for (provider_id, name) in [
            ("call_a31qKPFi7qB2YK3WxyXVjTuC", "read"),
            ("call_123", "read_file"),
            ("toolu_01ABC", "bash"),
            ("read_v2", "read_v2"),
        ] {
            assert_eq!(ids.assign(provider_id, name), provider_id);
        }
    }

    #[test]
    fn missing_and_duplicate_ids_get_fresh_unique_ids() {
        let mut ids = ToolCallIdAssigner::for_history(&history(&[("call_history", "read")]));

        let empty = ids.assign("", "read");
        let first = ids.assign("call_same", "read");
        let duplicate = ids.assign("call_same", "read");
        let historical = ids.assign("call_history", "read");

        assert_eq!(first, "call_same");
        for minted in [&empty, &duplicate, &historical] {
            assert!(minted.starts_with("call_") && minted.len() > "call_".len());
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
