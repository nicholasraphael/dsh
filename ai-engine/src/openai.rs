use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::io::Write;

#[derive(Debug, Deserialize)]
pub struct ChatChunkDelta {
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChunkChoice {
    pub delta: ChatChunkDelta,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: usize,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}
