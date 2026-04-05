use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest,
    MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
    Usage,
};

use super::{Provider, ProviderFuture};

pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    http_client: reqwest::Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub response: String,
    pub created_at: String,
    pub done: bool,
    #[serde(default)]
    pub context: Vec<i32>,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: u32,
    #[serde(default)]
    pub eval_duration: u64,
}

impl OllamaClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self::new_with_url(DEFAULT_OLLAMA_URL, model)
    }

    pub fn new_with_url(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            http_client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self, ApiError> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
        
        let model = std::env::var("OLLAMA_MODEL")
            .map_err(|_| ApiError::missing_credentials(
                "Ollama",
                &["OLLAMA_MODEL"],
            ))?;

        Ok(Self::new_with_url(base_url, model))
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let prompt = self.messages_to_prompt(&request.messages, request.system.as_deref());
        
        let ollama_req = OllamaRequest {
            model: self.model.clone(),
            prompt,
            system: request.system.clone(),
            stream: false,
            temperature: Some(0.7),
        };

        let endpoint = format!("{}/api/generate", self.base_url);
        let response = self
            .http_client
            .post(&endpoint)
            .json(&ollama_req)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Api {
                status,
                error_type: None,
                message: Some("Ollama API error".to_string()),
                body,
                retryable: status.is_server_error(),
            });
        }

        let ollama_resp: OllamaResponse = response.json().await?;

        Ok(self.ollama_response_to_message_response(&ollama_resp))
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        let prompt = self.messages_to_prompt(&request.messages, request.system.as_deref());
        
        let ollama_req = OllamaRequest {
            model: self.model.clone(),
            prompt,
            system: request.system.clone(),
            stream: true,
            temperature: Some(0.7),
        };

        let endpoint = format!("{}/api/generate", self.base_url);
        let response = self
            .http_client
            .post(&endpoint)
            .json(&ollama_req)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Api {
                status,
                error_type: None,
                message: Some("Ollama API error".to_string()),
                body,
                retryable: status.is_server_error(),
            });
        }

        Ok(MessageStream {
            response,
            buffer: String::new(),
            message_started: false,
            text_started: false,
            done: false,
            accumulated_text: String::new(),
            prompt_eval_count: 0,
            eval_count: 0,
        })
    }

    fn messages_to_prompt(&self, messages: &[InputMessage], system: Option<&str>) -> String {
        let mut prompt = String::new();

        if let Some(sys) = system {
            prompt.push_str("System: ");
            prompt.push_str(sys);
            prompt.push_str("\n\n");
        }

        for message in messages {
            match message.role.as_str() {
                "user" => {
                    prompt.push_str("User: ");
                    for block in &message.content {
                        if let InputContentBlock::Text { text } = block {
                            prompt.push_str(text);
                        }
                    }
                    prompt.push_str("\n\n");
                }
                "assistant" => {
                    prompt.push_str("Assistant: ");
                    for block in &message.content {
                        if let InputContentBlock::Text { text } = block {
                            prompt.push_str(text);
                        }
                    }
                    prompt.push_str("\n\n");
                }
                _ => {}
            }
        }

        prompt.push_str("Assistant: ");
        prompt
    }

    fn ollama_response_to_message_response(&self, resp: &OllamaResponse) -> MessageResponse {
        let input_tokens = resp.prompt_eval_count;
        let output_tokens = resp.eval_count;

        MessageResponse {
            id: format!("ollama-{}", uuid::Uuid::new_v4()),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::Text {
                text: resp.response.clone(),
            }],
            model: resp.model.clone(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens,
            },
            request_id: None,
        }
    }
}

#[derive(Debug)]
pub struct MessageStream {
    response: reqwest::Response,
    buffer: String,
    message_started: bool,
    text_started: bool,
    done: bool,
    accumulated_text: String,
    prompt_eval_count: u32,
    eval_count: u32,
}

impl MessageStream {
    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            // Handle end-of-stream sequence
            if self.done && self.buffer.is_empty() {
                if self.text_started {
                    self.text_started = false;
                    return Ok(Some(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                        index: 0,
                    })));
                }
                if self.message_started {
                    self.message_started = false;
                    return Ok(Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                        delta: MessageDelta {
                            stop_reason: Some("end_turn".to_string()),
                            stop_sequence: None,
                        },
                        usage: Usage {
                            input_tokens: self.prompt_eval_count,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                            output_tokens: self.eval_count,
                        },
                    })));
                }
                return Ok(Some(StreamEvent::MessageStop(MessageStopEvent {})));
            }

            // Read more data if buffer is empty
            if self.buffer.is_empty() && !self.done {
                if let Some(chunk) = self.response.chunk().await? {
                    self.buffer = String::from_utf8(chunk.to_vec())
                        .map_err(|_| ApiError::InvalidSseFrame("invalid utf8 in ollama response"))?;
                } else {
                    self.done = true;
                    continue;
                }
            }

            // Parse one line from buffer
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer.drain(..=newline_pos).collect::<String>();
                let trimmed = line.trim();
                
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<OllamaResponse>(trimmed) {
                    Ok(resp) => {
                        self.prompt_eval_count = resp.prompt_eval_count;
                        self.eval_count = resp.eval_count;

                        // Start message if not already started
                        if !self.message_started {
                            self.message_started = true;
                            let msg = MessageResponse {
                                id: format!("ollama-{}", uuid::Uuid::new_v4()),
                                kind: "message".to_string(),
                                role: "assistant".to_string(),
                                content: Vec::new(),
                                model: resp.model.clone(),
                                stop_reason: None,
                                stop_sequence: None,
                                usage: Usage {
                                    input_tokens: 0,
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                    output_tokens: 0,
                                },
                                request_id: None,
                            };
                            return Ok(Some(StreamEvent::MessageStart(MessageStartEvent {
                                message: msg,
                            })));
                        }

                        // Emit text delta if response has content
                        if !resp.response.is_empty() {
                            if !self.text_started {
                                self.text_started = true;
                                self.accumulated_text.clear();
                                return Ok(Some(StreamEvent::ContentBlockStart(
                                    ContentBlockStartEvent {
                                        index: 0,
                                        content_block: OutputContentBlock::Text {
                                            text: String::new(),
                                        },
                                    },
                                )));
                            }

                            self.accumulated_text.push_str(&resp.response);
                            return Ok(Some(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                                index: 0,
                                delta: ContentBlockDelta::TextDelta {
                                    text: resp.response,
                                },
                            })));
                        }

                        if resp.done {
                            self.done = true;
                        }

                        continue;
                    }
                    Err(_) => {
                        continue;
                    }
                }
            } else if self.done {
                continue;
            } else {
                return Ok(None);
            }
        }
    }
}

impl Provider for OllamaClient {
    type Stream = MessageStream;

    fn send_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(async move { self.send_message(request).await })
    }

    fn stream_message<'a>(
        &'a self,
        request: &'a MessageRequest,
    ) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move { self.stream_message(request).await })
    }
}

pub fn has_ollama_available() -> bool {
    std::process::Command::new("curl")
        .arg("-s")
        .arg(format!("{}/api/tags", DEFAULT_OLLAMA_URL))
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}