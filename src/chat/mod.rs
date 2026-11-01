//! Candle-backed chat inference over Safetensors checkpoints.

mod engine;

pub(crate) use engine::{ChatMessage, ChatParams, Engine, Generation, TextSink, ToolChoice};

pub fn chat(
    path: &std::path::Path,
    options: crate::options::ChatOptions,
    archive: crate::context::ContextArchive,
    store: crate::context::ContextStore,
) -> Result<(), String> {
    engine::chat(path, options, archive, store)
}
