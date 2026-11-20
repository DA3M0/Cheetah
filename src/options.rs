//! Shared options for local chat generation.
//!
//! [`ChatOptions`] is the generation-parameter struct shared by the CLI
//! chat (`cheetah chat`) and the HTTP server (`cheetah serve`): sampling
//! policy (temperature / top-p / top-k / repeat penalty), context length,
//! per-turn token budget, random seed, and the system prompt. It is also
//! serialized into conversation archives (`chat/context.rs`), so the
//! fields must stay backward compatible — new fields need
//! `#[serde(default)]`.

use serde::{Deserialize, Serialize};

/// Default cap on tokens generated per conversational turn.
pub const DEFAULT_MAX_TOKENS_PER_TURN: usize = 256;
/// Default context window (tokens); older history is dropped as a sliding
/// window once exceeded.
pub const DEFAULT_CONTEXT_TOKENS: usize = 8192;
/// Default temperature 0: deterministic greedy output; enable sampling
/// explicitly when more varied answers are wanted.
pub const DEFAULT_TEMPERATURE: f64 = 0.0;
/// Default nucleus-sampling probability threshold; 1.0 disables truncation.
pub const DEFAULT_TOP_P: f64 = 0.9;
/// Default repeat penalty; 1.0 disables it.
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.0;
/// System prompt injected when `--system` is not given.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise assistant. Respond in English.";

// Serde uses this function when loading archives written before ctx_size was introduced.
fn default_context_tokens() -> usize {
    DEFAULT_CONTEXT_TOKENS
}

/// All tunable parameters for one chat generation.
///
/// Most generation switches visible via `cheetah chat --help` land here,
/// and the corresponding serve-request fields (temperature / top_p / ...)
/// are merged into it as well.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Inference thread count; `None` lets the backend decide from the CPU
    /// core count.
    pub threads: Option<usize>,
    #[serde(default = "default_context_tokens")]
    pub ctx_size: usize,
    pub max_tokens: usize,
    /// Sampling temperature; 0 disables sampling and uses deterministic
    /// greedy decoding.
    pub temperature: f64,
    /// Nucleus-sampling probability threshold; 1 samples from the full
    /// distribution.
    pub top_p: f64,
    /// Top-k candidate count for sampling (truncates the full vocabulary
    /// only); 0 disables it.
    /// Note: the per-token MoE expert count is controlled by the model
    /// config's `num_experts_per_tok` and is unrelated to this field.
    pub top_k: usize,
    /// Repeat penalty; 1 disables it.
    pub repeat_penalty: f32,
    /// Random seed; `None` derives one from the clock.
    pub seed: Option<u64>,
    /// System prompt, injected on the first turn only.
    pub system: Option<String>,
}

impl Default for ChatOptions {
    /// The only difference from [`DEFAULT_CHAT_OPTIONS`]: interactive chat
    /// injects the system prompt by default (the HTTP server does not).
    fn default() -> Self {
        Self {
            system: Some(DEFAULT_SYSTEM_PROMPT.to_string()),
            ..DEFAULT_CHAT_OPTIONS
        }
    }
}

/// Const-friendly copy of the default options (safe to expand in `const`
/// contexts).
pub const DEFAULT_CHAT_OPTIONS: ChatOptions = ChatOptions {
    threads: None,
    ctx_size: DEFAULT_CONTEXT_TOKENS,
    max_tokens: DEFAULT_MAX_TOKENS_PER_TURN,
    temperature: DEFAULT_TEMPERATURE,
    top_p: DEFAULT_TOP_P,
    top_k: 0,
    repeat_penalty: DEFAULT_REPEAT_PENALTY,
    seed: None,
    system: None,
};
