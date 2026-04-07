//! User interaction types and tools.
//!
//! This module provides types and tools for agent-user interaction:
//!
//! - [`ConfirmationRequest`] / [`ConfirmationResponse`] - For tool confirmations
//! - [`QuestionRequest`] / [`QuestionResponse`] - For agent-initiated questions
//! - [`AskUserQuestionTool`] - Tool that allows agents to ask questions
//!
//! # Confirmation Flow
//!
//! When an agent needs to execute a tool that requires confirmation:
//!
//! 1. Agent hooks create a [`ConfirmationRequest`]
//! 2. UI displays the request to the user
//! 3. User responds with [`ConfirmationResponse`]
//! 4. Agent proceeds based on the response
//!
//! # Question Flow
//!
//! When an agent needs to ask the user a question:
//!
//! 1. Agent calls `ask_user` tool with a [`QuestionRequest`]
//! 2. UI displays the question to the user
//! 3. User responds with [`QuestionResponse`]
//! 4. Agent receives the answer and continues
//!
//! # Example
//!
//! ```no_run
//! use agent_sdk::user_interaction::{AskUserQuestionTool, QuestionRequest, QuestionResponse};
//! use tokio::sync::mpsc;
//!
//! // Create channels for communication
//! let (tool, mut request_rx, response_tx) = AskUserQuestionTool::with_channels(10);
//!
//! // The tool can be registered with the agent's tool registry
//! // The UI handles requests and responses through the channels
//! ```

use crate::{PrimitiveToolName, Tool, ToolContext, ToolResult, ToolTier};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;

/// Request for user confirmation of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmationRequest {
    /// Name of the tool requiring confirmation.
    pub tool_name: String,

    /// Human-readable description of the action.
    pub description: String,

    /// Preview of the tool input (JSON formatted).
    pub input_preview: String,

    /// The tier of the tool (serialized as string).
    pub tier: String,

    /// Agent's recent reasoning text that led to this tool call.
    pub context: Option<String>,
}

impl ConfirmationRequest {
    /// Creates a new confirmation request.
    #[must_use]
    pub fn new(
        tool_name: impl Into<String>,
        description: impl Into<String>,
        input_preview: impl Into<String>,
        tier: ToolTier,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            description: description.into(),
            input_preview: input_preview.into(),
            tier: format!("{tier:?}"),
            context: None,
        }
    }

    /// Adds context to the request.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Response to a confirmation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfirmationResponse {
    /// User approved the tool execution.
    Approved,

    /// User denied the tool execution.
    Denied,

    /// User wants to approve all future requests for this tool.
    ApproveAll,
}

/// Request for user to answer a question from the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionRequest {
    /// The question text to display.
    pub question: String,

    /// Optional header/category for the question.
    pub header: Option<String>,

    /// Available options (if multiple choice).
    /// Empty means free-form text input.
    pub options: Vec<QuestionOption>,

    /// Whether multiple options can be selected.
    pub multi_select: bool,
}

impl QuestionRequest {
    /// Creates a new free-form question request.
    #[must_use]
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            header: None,
            options: Vec::new(),
            multi_select: false,
        }
    }

    /// Creates a new multiple-choice question request.
    #[must_use]
    pub fn with_options(question: impl Into<String>, options: Vec<QuestionOption>) -> Self {
        Self {
            question: question.into(),
            header: None,
            options,
            multi_select: false,
        }
    }

    /// Adds a header to the question.
    #[must_use]
    pub fn with_header(mut self, header: impl Into<String>) -> Self {
        self.header = Some(header.into());
        self
    }

    /// Enables multi-select mode.
    #[must_use]
    pub const fn with_multi_select(mut self) -> Self {
        self.multi_select = true;
        self
    }
}

/// An option in a multiple-choice question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    /// The display label for this option.
    pub label: String,

    /// Optional description explaining this option.
    pub description: Option<String>,
}

impl QuestionOption {
    /// Creates a new option with just a label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
        }
    }

    /// Creates a new option with label and description.
    #[must_use]
    pub fn with_description(label: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: Some(description.into()),
        }
    }
}

/// Response to a question request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResponse {
    /// The user's answer (text or selected option labels).
    pub answer: String,

    /// Whether the user cancelled/skipped the question.
    pub cancelled: bool,
}

impl QuestionResponse {
    /// Creates a new successful response.
    #[must_use]
    pub fn success(answer: impl Into<String>) -> Self {
        Self {
            answer: answer.into(),
            cancelled: false,
        }
    }

    /// Creates a cancelled response.
    #[must_use]
    pub const fn cancelled() -> Self {
        Self {
            answer: String::new(),
            cancelled: true,
        }
    }
}

/// Tool that allows the agent to ask questions to the user.
///
/// This is essential for:
/// - Clarifying ambiguous requirements
/// - Offering choices between approaches
/// - Getting user preferences
/// - Confirming before major changes
pub struct AskUserQuestionTool {
    /// Channel to send questions to the UI.
    question_tx: mpsc::Sender<QuestionRequest>,

    /// Channel to receive answers from the UI.
    question_rx: tokio::sync::Mutex<mpsc::Receiver<QuestionResponse>>,
}

impl AskUserQuestionTool {
    /// Creates a new tool with the given channels.
    #[must_use]
    pub fn new(
        question_tx: mpsc::Sender<QuestionRequest>,
        question_rx: mpsc::Receiver<QuestionResponse>,
    ) -> Self {
        Self {
            question_tx,
            question_rx: tokio::sync::Mutex::new(question_rx),
        }
    }

    /// Creates a new tool with fresh channels.
    ///
    /// Returns `(tool, request_receiver, response_sender)` where:
    /// - `tool` is the `AskUserQuestionTool` instance
    /// - `request_receiver` receives questions from the agent
    /// - `response_sender` sends user answers back to the agent
    #[must_use]
    pub fn with_channels(
        buffer_size: usize,
    ) -> (
        Self,
        mpsc::Receiver<QuestionRequest>,
        mpsc::Sender<QuestionResponse>,
    ) {
        let (request_tx, request_rx) = mpsc::channel(buffer_size);
        let (response_tx, response_rx) = mpsc::channel(buffer_size);

        let tool = Self::new(request_tx, response_rx);
        (tool, request_rx, response_tx)
    }
}

/// Input schema for the `AskUserQuestion` tool.
#[derive(Debug, Deserialize, Serialize)]
struct AskUserInput {
    /// The question to ask the user.
    question: String,

    /// Optional header/category for the question.
    #[serde(default)]
    header: Option<String>,

    /// Optional list of choices for multiple-choice questions.
    #[serde(default)]
    options: Vec<OptionInput>,

    /// Whether multiple options can be selected.
    #[serde(default)]
    multi_select: bool,
}

/// Input for a single option.
#[derive(Debug, Deserialize, Serialize)]
struct OptionInput {
    /// The display label.
    label: String,

    /// Optional description.
    #[serde(default)]
    description: Option<String>,
}

impl<Ctx: Send + Sync + 'static> Tool<Ctx> for AskUserQuestionTool {
    type Name = PrimitiveToolName;

    fn name(&self) -> PrimitiveToolName {
        PrimitiveToolName::AskUser
    }

    fn display_name(&self) -> &'static str {
        "Ask User"
    }

    fn description(&self) -> &'static str {
        "Ask the user a question to get clarification, preferences, or choices. \
         Use this when you need user input before proceeding. For yes/no confirmations \
         of dangerous operations, tool confirmation will be shown automatically - \
         use this tool for open-ended questions or when offering choices."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["question"],
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user. Be clear and specific."
                },
                "header": {
                    "type": "string",
                    "description": "Optional short header/category (e.g., 'Auth method', 'Library choice')"
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of choices for multiple-choice questions",
                    "items": {
                        "type": "object",
                        "required": ["label"],
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "The option text to display"
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional explanation of this option"
                            }
                        }
                    }
                },
                "multi_select": {
                    "type": "boolean",
                    "description": "Whether multiple options can be selected (default: false)"
                }
            }
        })
    }

    fn tier(&self) -> ToolTier {
        // Questions don't modify anything, but they do require user interaction
        ToolTier::Observe
    }

    async fn execute(&self, _ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
        // Parse input
        let input: AskUserInput =
            serde_json::from_value(input).context("Invalid input for ask_user tool")?;

        // Build request
        let request = QuestionRequest {
            question: input.question.clone(),
            header: input.header,
            options: input
                .options
                .into_iter()
                .map(|o| QuestionOption {
                    label: o.label,
                    description: o.description,
                })
                .collect(),
            multi_select: input.multi_select,
        };

        // Send question to UI
        self.question_tx
            .send(request)
            .await
            .context("Failed to send question to UI - channel closed")?;

        // Wait for response
        let response = {
            let mut rx = self.question_rx.lock().await;
            rx.recv()
                .await
                .context("Failed to receive answer from UI - channel closed")?
        };

        if response.cancelled {
            Ok(ToolResult::error(
                "User cancelled the question without providing an answer.",
            ))
        } else {
            Ok(ToolResult::success(format!(
                "User answered: {}",
                response.answer
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;

    #[test]
    fn test_confirmation_request_new() {
        let req =
            ConfirmationRequest::new("write", "Write to file: foo.txt", "{}", ToolTier::Confirm);
        assert_eq!(req.tool_name, "write");
        assert!(req.context.is_none());
    }

    #[test]
    fn test_confirmation_request_with_context() {
        let req = ConfirmationRequest::new("write", "Write to file", "{}", ToolTier::Confirm)
            .with_context("Agent was fixing a bug");
        assert!(req.context.is_some());
        assert_eq!(req.context.unwrap(), "Agent was fixing a bug");
    }

    #[test]
    fn test_confirmation_response_serialization() {
        assert_eq!(
            serde_json::to_string(&ConfirmationResponse::Approved).unwrap(),
            "\"approved\""
        );
        assert_eq!(
            serde_json::to_string(&ConfirmationResponse::Denied).unwrap(),
            "\"denied\""
        );
        assert_eq!(
            serde_json::to_string(&ConfirmationResponse::ApproveAll).unwrap(),
            "\"approve_all\""
        );
    }

    #[test]
    fn test_question_request_new() {
        let req = QuestionRequest::new("What color?");
        assert_eq!(req.question, "What color?");
        assert!(req.options.is_empty());
        assert!(!req.multi_select);
    }

    #[test]
    fn test_question_request_with_options() {
        let req = QuestionRequest::with_options(
            "Which framework?",
            vec![
                QuestionOption::new("React"),
                QuestionOption::with_description("Vue", "Progressive framework"),
            ],
        )
        .with_header("Framework")
        .with_multi_select();

        assert_eq!(req.options.len(), 2);
        assert!(req.multi_select);
        assert_eq!(req.header.unwrap(), "Framework");
    }

    #[test]
    fn test_question_response() {
        let success = QuestionResponse::success("Blue");
        assert!(!success.cancelled);
        assert_eq!(success.answer, "Blue");

        let cancelled = QuestionResponse::cancelled();
        assert!(cancelled.cancelled);
    }

    #[tokio::test]
    async fn test_ask_user_tool_creation() {
        let (tool, _rx, _tx) = AskUserQuestionTool::with_channels(10);

        // Use Tool<()> explicitly to satisfy type inference
        assert_eq!(Tool::<()>::name(&tool), PrimitiveToolName::AskUser);
        assert_eq!(Tool::<()>::tier(&tool), ToolTier::Observe);
    }

    #[tokio::test]
    async fn test_ask_user_tool_execute() {
        let (tool, mut request_rx, response_tx) = AskUserQuestionTool::with_channels(10);

        // Spawn task to handle the question
        let handle = tokio::spawn(async move {
            if let Some(request) = request_rx.recv().await {
                assert_eq!(request.question, "What color?");
                response_tx
                    .send(QuestionResponse::success("Blue"))
                    .await
                    .unwrap();
            }
        });

        let ctx = ToolContext::new(());
        let result = tool
            .execute(
                &ctx,
                json!({
                    "question": "What color?"
                }),
            )
            .await
            .unwrap();

        handle.await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("Blue"));
    }

    #[tokio::test]
    async fn test_ask_user_with_options() {
        let (tool, mut request_rx, response_tx) = AskUserQuestionTool::with_channels(10);

        let handle = tokio::spawn(async move {
            if let Some(request) = request_rx.recv().await {
                assert_eq!(request.options.len(), 2);
                assert_eq!(request.options[0].label, "Option A");
                response_tx
                    .send(QuestionResponse::success("Option A"))
                    .await
                    .unwrap();
            }
        });

        let ctx = ToolContext::new(());
        let result = tool
            .execute(
                &ctx,
                json!({
                    "question": "Which option?",
                    "options": [
                        {"label": "Option A", "description": "First choice"},
                        {"label": "Option B", "description": "Second choice"}
                    ]
                }),
            )
            .await
            .unwrap();

        handle.await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_ask_user_cancelled() {
        let (tool, mut request_rx, response_tx) = AskUserQuestionTool::with_channels(10);

        let handle = tokio::spawn(async move {
            if request_rx.recv().await.is_some() {
                response_tx
                    .send(QuestionResponse::cancelled())
                    .await
                    .unwrap();
            }
        });

        let ctx = ToolContext::new(());
        let result = tool
            .execute(
                &ctx,
                json!({
                    "question": "Continue?"
                }),
            )
            .await
            .unwrap();

        handle.await.unwrap();
        assert!(!result.success);
        assert!(result.output.contains("cancelled"));
    }
}
