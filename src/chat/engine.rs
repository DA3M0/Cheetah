//! Generation engine for Cheetah's chat and CLI inference path.
//!
//! This module is the orchestration layer between the command-line entry
//! points and the model backends. It owns the logic that is common to every
//! architecture: resolving model files, loading the configuration and
//! tokenizer, selecting a backend, rendering chat prompts, running the
//! prefill and decode loop, sampling tokens, applying stop conditions, and
//! streaming decoded text through a `TextSink`.
//!
//! # Model loading
//!
//! `Engine::load` resolves the model directory (or a single Safetensors
//! shard), reads the Hugging Face-style `config.json`, and falls back to the
//! snapshot stored in `cheetah_manifest.json` when the config was not copied
//! into a prepared directory. Weight shards are discovered from the manifest,
//! from a Safetensors index, or from a lone `model.safetensors`. Actual
//! backend construction is delegated to `chat::registry`: the native,
//! mmap-backed CPU backend is preferred when it is registered and enabled,
//! while non-CPU `CHEETAH_DEVICE` values such as Vulkan route through the
//! accelerator loader. An Ant residency plan is computed during loading and
//! can narrow the set of weight files that the loader maps.
//!
//! # Tokenization and prompt rendering
//!
//! The tokenizer is a Hugging Face `tokenizers` instance loaded from
//! `tokenizer.json`. Chat messages are rendered into a model-specific prompt
//! by a function selected from the architecture family. For Llama 3-style
//! architectures the header-based renderer is used when the vocabulary and
//! manifest stop markers indicate that the special header tokens exist;
//! otherwise the classic instruction renderer is used.
//!
//! # Prefill and decode
//!
//! `Engine::generate_inner` performs one prefill call over the full prompt,
//! then runs an autoregressive decode loop that feeds a single token back into
//! the model together with its absolute position. The backend owns the KV
//! cache between calls; the engine clears that cache before prefill and after
//! generation. Each step samples from the returned logits, checks the
//! end-of-sequence token and textual stop markers, and emits only the newly
//! decoded text so streaming callers observe incremental deltas.
//!
//! # Sampling
//!
//! Sampling supports greedy decoding (temperature zero), temperature scaling,
//! a repetition penalty over recent history, top-k truncation, and nucleus
//! (top-p) filtering. A seeded `StdRng` makes runs reproducible; when the
//! caller does not supply a seed, a wall-clock seed is used instead.
//!
//! # Context handling
//!
//! `Engine::prepare` renders and tokenizes the conversation, then drops the
//! oldest turns until the prompt plus the reserved generation budget fits the
//! effective context limit. That limit is the smallest of the configured
//! context size, the model's `max_position_embeddings`, and the architecture
//! family's cap. The interactive loop keeps a `ContextArchive` of completed
//! turns so conversations can be saved and reloaded.
//!
//! # Streaming callbacks
//!
//! Callers provide a `TextSink`. The engine pushes incremental decoded text
//! into it, polls it for cancellation between tokens, and calls `finish` once
//! at the end. The CLI uses `StdoutSink`; other front ends can implement the
//! trait to forward text to an HTTP stream, a logger, or a user interface.
//!
//! # Ant tiering
//!
//! When Ant is enabled, the engine tracks which atomic weight units are
//! resident in VRAM or RAM, promotes queued RAM units into VRAM, pages evicted
//! units back in before a request, and enforces the configured pool budget.
//! On the native backend these transitions also fault or release the real
//! mmap pages; on host-resident or accelerator backends only the bookkeeping
//! is updated.

// Engine orchestration depends on the chat model registry (backend selection
// and prompt templates), Ant's residency manager, terminal color and progress
// helpers, the context archive used by the interactive loop, and the shared
// chat options.
use super::registry::{Device, KvSnapshot, ModelForward};
use crate::ant::{Ant, AntLayerResidency, AntSettings, AntUnit, ant_swap_fit};
use crate::chat::toolcalls::{self, ResolvedToolChoice, ToolDefinition, ToolResponse};
use crate::color::{self, Color};
use crate::context::{ContextArchive, ContextStore, ConversationTurn};
use crate::mcp::McpRegistry;
use crate::options::ChatOptions;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

/// Generic fallback context length when no architecture family declares one
/// (unregistered architectures / missing family metadata).
///
/// This value is only used while synthesizing a configuration from a manifest
/// snapshot. Real loads prefer the model's `max_position_embeddings` and then
/// clamp it against the family cap and the user's `--ctx-size`.
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 4096;

/// One turn of a chat conversation, independent of the wire format used by
/// the CLI, HTTP server, or tool-calling front end.
///
/// The role is a plain string ("system", "user", "assistant", ...) because
/// prompt templates are architecture-specific and may need arbitrary roles.
/// Message content is already flattened to text by the time a renderer sees
/// it; tool-only payloads are converted before reaching this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Conversation role, such as "system", "user", or "assistant".
    pub role: String,
    /// Textual message body. Multi-part content is concatenated by the caller.
    pub content: String,
}

impl ChatMessage {
    /// Builds a message from any string-like role and content.
    ///
    /// Taking `impl Into<String>` keeps call sites ergonomic for both string
    /// literals and owned strings without requiring an extra allocation when
    /// an owned `String` is already available.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Formats the current wall-clock time as a bracketed seconds-since-epoch
/// timestamp for CLI log lines.
///
/// The fractional part has millisecond precision. If the system clock is set
/// before the Unix epoch, the duration conversion fails and a stable zero is
/// used instead, because logging must never fail the process.
fn log_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    format!("[{:.3}]", now)
}

/// Creates the progress spinner shown while a model is loading.
///
/// When stdout is not a terminal (for example in tests or when piping output)
/// the draw target is hidden so escape sequences do not pollute redirected
/// output. The steady tick keeps the spinner animated during long weight
/// loads without requiring the caller to drive it manually.
fn create_spinner(message: &'static str) -> ProgressBar {
    let draw_target = if std::io::stdout().is_terminal() {
        ProgressDrawTarget::stdout()
    } else {
        ProgressDrawTarget::hidden()
    };
    let bar = ProgressBar::new_spinner();
    bar.set_draw_target(draw_target);
    bar.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}").expect("valid spinner style"),
    );
    bar.enable_steady_tick(Duration::from_millis(80));
    bar.set_message(message);
    bar
}

/// Loads an engine while showing CLI progress and a post-load summary.
///
/// This is the CLI-facing wrapper around `Engine::load`. It deliberately keeps
/// user-interface concerns (timestamps, colored output, spinner lifecycle) out
/// of the loading code so tests and other callers can load an engine without
/// touching the terminal. On failure the spinner is abandoned rather than
/// finished, and the original error string is returned unchanged.
pub(crate) fn load_engine_with_progress(
    path: &Path,
    threads: Option<usize>,
    ctx_size: usize,
    ant: Option<&AntSettings>,
) -> Result<Engine, String> {
    // Announce the path before the long blocking load so the user sees which
    // model is being opened even if loading later fails.
    println!(
        "{} {}",
        log_timestamp(),
        color::stdout(&format!("loading model: {}", path.display()), Color::Cyan,)
    );

    let spinner = create_spinner("loading model");
    let result = Engine::load(path, threads, ctx_size, ant);
    match result {
        Ok(engine) => {
            // Replace the animated spinner with a one-line success message;
            // the detailed device/backend summary follows immediately.
            spinner.finish_with_message(format!("loaded: {}", engine.model_id()));
            print_model_info(&engine);
            Ok(engine)
        }
        Err(error) => {
            // Keep the terminal readable: mark the spinner as failed and let
            // the caller decide how to report the returned error text.
            spinner.abandon_with_message("model load failed");
            Err(error)
        }
    }
}

/// Prints the human-readable load summary: architecture, backends, device,
/// and (when Ant is active) residency and budget information.
///
/// This function is purely diagnostic; failures to lock runtime state are
/// ignored so that printing cannot turn a successful load into an error.
fn print_model_info(engine: &Engine) {
    println!();
    println!(
        "{} {}",
        log_timestamp(),
        color::stdout(&format!("model loaded: {}", engine.model_id()), Color::Cyan,),
    );
    // Architecture is optional for unregistered models, in which case only
    // the device line is printed below.
    if let Some(architecture) = engine.model_architecture() {
        println!("  architecture: {architecture}");
        // Backend names come from the registry rather than from a hard-coded
        // list, so newly registered native/Vulkan backends show up here.
        println!(
            "  backends:     {}",
            super::registry::registered_backends(architecture).join(", ")
        );
    }
    println!("  device:       {}", engine.device_name());
    // Ant-specific lines only make sense when a residency plan was built.
    if !engine.ant_plan.is_empty() {
        println!("  ant plan:     {} resident unit(s)", engine.ant_plan.len());
        println!(
            "  ant resident: {} bytes within budget {}",
            engine.ant_resident_bytes, engine.ant_budget_bytes
        );
        // Layer residency is a coarser, per-layer view derived from the
        // atomic-unit plan; count each state for a compact summary.
        if !engine.ant_layer_summary.is_empty() {
            let state_of = |wanted: &str| {
                engine
                    .ant_layer_summary
                    .iter()
                    .filter(|layer| layer.state() == wanted)
                    .count()
            };
            println!(
                "  ant layers:   {} full-vram, {} partial, {} ram, {} disk",
                state_of("full_vram"),
                state_of("partial"),
                state_of("full_ram"),
                state_of("full_disk")
            );
        }
        // try_lock keeps the summary best-effort: at this point no generation
        // is running, but avoiding a blocking lock guarantees diagnostics
        // can never hang the load path.
        if let Ok(state) = engine.ant_runtime.try_lock() {
            println!(
                "  ant swap:     {} queued for VRAM promotion, {} evicted at load",
                state.promotion_queue.len(),
                state.evicted.len()
            );
        }
        // Surface budget overruns even though they are usually resolved by a
        // later prefetch/promotion pass; this warns operators early.
        if !engine.ant_within_budget() {
            println!(
                "  ant warning:  resident bytes {} exceed vram budget {}",
                engine.ant_resident_bytes, engine.ant_budget_bytes
            );
        }
    }
}

/// Receives streamed generation output and provides cancellation feedback.
///
/// The engine is deliberately sink-agnostic: the CLI writes to stdout, while
/// server front ends may forward chunks over HTTP/SSE or collect them into a
/// buffer. Implementations must be cheap to call once per token because the
/// generation loop is latency-sensitive.
pub trait TextSink {
    /// Receives the next incremental slice of decoded text.
    ///
    /// The slice only contains characters that have not been emitted before.
    /// Returning an error aborts generation and propagates the message to the
    /// caller; implementors should reserve errors for real I/O failures.
    fn push(&mut self, text: &str) -> Result<(), String>;

    /// Called once after the loop ends, whether by EOS, stop marker, length,
    /// or cancellation. The default is a no-op.
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Whether the engine should accumulate the full answer in
    /// `Generation::text`.
    ///
    /// Streaming callers that discard the final text can return false to
    /// avoid growing a large `String`; the default keeps the text because
    /// most callers need it for history or non-streaming responses.
    fn capture_text(&self) -> bool {
        true
    }

    /// Polled before each decode step to request cooperative cancellation.
    ///
    /// Returning true ends generation with `FinishReason::Stop` after the
    /// previously emitted text has been delivered. The default never
    /// interrupts.
    fn interrupted(&mut self) -> bool {
        false
    }
}

/// Streaming sink for tool rounds: text deltas stream live with hold-back.
///
/// A tool call begins with exactly the markers plain text rarely starts a
/// line with — a `{` (the tool JSON) or a code fence — so everything from the
/// first such marker stays buffered until the round decision: plain answers
/// flush the remainder at round end (brace-bearing prose arrives as one
/// slightly-late delta), tool-call rounds drop it (the calls go to the
/// executor instead and must never leak as content). The wrapper is what
/// makes the agentic loop stream honestly instead of buffering whole rounds.
struct ToolStreamSink<'a> {
    inner: &'a mut dyn TextSink,
    buffer: String,
    emitted: usize,
}

impl<'a> ToolStreamSink<'a> {
    fn new(inner: &'a mut dyn TextSink) -> Self {
        Self {
            inner,
            buffer: String::new(),
            emitted: 0,
        }
    }

    /// Emit everything up to the first hold marker, if any.
    fn stream_safe(&mut self) -> Result<(), String> {
        let rest = &self.buffer[self.emitted..];
        let hold = match (rest.find('{'), rest.find("```")) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let safe_end = self.emitted + hold.unwrap_or(rest.len());
        if safe_end > self.emitted {
            let delta = self.buffer[self.emitted..safe_end].to_owned();
            self.emitted = safe_end;
            self.inner.push(&delta)?;
        }
        Ok(())
    }

    /// Round decided as a tool call: discard the held remainder.
    fn drop_round(&mut self) {
        self.emitted = self.buffer.len();
    }

    /// Round decided as a plain answer: flush everything still held (the
    /// held tail is prose containing braces, not a tool call).
    fn flush_round(&mut self) -> Result<(), String> {
        if self.buffer.len() > self.emitted {
            let delta = self.buffer[self.emitted..].to_owned();
            self.emitted = self.buffer.len();
            self.inner.push(&delta)?;
        }
        Ok(())
    }
}

impl TextSink for ToolStreamSink<'_> {
    fn push(&mut self, text: &str) -> Result<(), String> {
        self.buffer.push_str(text);
        self.stream_safe()
    }

    /// Per-round boundary inside the tool loop — the round decision (flush
    /// vs drop) is made by the loop after parsing, never here.
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// The loop parses `generation.text`, so the transcript must be captured
    /// even though the safe prefix already streamed.
    fn capture_text(&self) -> bool {
        true
    }

    fn interrupted(&mut self) -> bool {
        self.inner.interrupted()
    }
}

/// Why the decode loop stopped producing tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A natural end: an EOS token, a textual stop marker, or cancellation.
    Stop,
    /// The reserved generation budget (`--max-tokens`) was exhausted.
    Length,
}

impl FinishReason {
    /// Returns the API-facing string used in responses and logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Result of one generation request, including text and timing metadata.
#[derive(Debug, Clone)]
pub struct Generation {
    /// Full generated text. Empty when the sink disabled text capture.
    pub text: String,
    /// Number of prompt tokens fed through prefill.
    pub prompt_tokens: usize,
    /// Number of tokens produced by the decode loop.
    pub generated_tokens: usize,
    /// Time spent before the first decode step: Ant hooks, model setup and
    /// the prompt forward pass — the inference-side time-to-first-token
    /// component (queue wait and tokenization happen upstream of this).
    pub prefill_duration: Duration,
    /// Time spent in the decode loop, excluding prefill and cleanup.
    pub decode_duration: Duration,
    /// Why the loop terminated.
    pub finish_reason: FinishReason,
}

impl Generation {
    /// Decode throughput in tokens per second.
    ///
    /// Returns `None` for empty generations or a zero measured duration, so
    /// callers can distinguish "no measurable speed" from a real zero rate.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let seconds = self.decode_duration.as_secs_f64();
        if self.generated_tokens == 0 || seconds == 0.0 {
            return None;
        }
        Some(self.generated_tokens as f64 / seconds)
    }
}

/// Tool-calling policy requested by an OpenAI-compatible API call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    /// Let the model decide whether a tool call is needed.
    Auto,
    /// Forbid tool calls and answer directly.
    None,
    /// Require the model to call one of the provided tools.
    Required,
}

/// Extra per-request parameters attached to an already-tokenized prompt.
///
/// The struct is intentionally JSON-friendly so HTTP handlers can pass
/// template metadata through without depending on engine internals.
#[derive(Debug, Clone, Default)]
pub struct ChatParams {
    /// JSON array of additional stop markers contributed by the chat template.
    pub additional_stops_json: String,
}

impl ChatParams {
    /// Validates and returns the model's raw text output for non-streaming
    /// tool-call responses.
    ///
    /// Tool-call parsing is currently a pass-through: the `_partial` flag is
    /// accepted for API compatibility but does not change behavior.
    pub fn parse(&self, text: &str, _partial: bool) -> Result<String, String> {
        Ok(text.to_owned())
    }
}

/// Owns the dynamically dispatched backend model and records the device it
/// was loaded on.
///
/// The device field is kept for diagnostics and future routing decisions even
/// when the backend itself reports its device name; the attribute suppresses
/// dead-code warnings without affecting layout or behavior.
struct LoadedModel {
    /// Backend model implementation shared behind the engine mutex.
    model: Box<dyn ModelForward>,
    /// Device selected during backend construction.
    #[allow(dead_code)]
    device: Device,
}

/// Opaque per-request inference session placeholder.
///
/// KV-cache state currently lives inside the backend and is cleared by the
/// engine around each generation, so the context carries no fields yet. The
/// type exists to keep the public API stable if per-request state is added.
pub struct InferenceContext;

/// Loaded model plus everything needed to turn chat messages into tokens and
/// tokens back into streamed text.
///
/// The engine is shared (`&self`) by request handlers. The backend model is
/// protected by a mutex because it may contain a mutable KV cache; tokenization
/// and prompt rendering are immutable and can run concurrently with unrelated
/// work. Ant bookkeeping uses a second mutex so residency checks do not
/// contend with the model lock while a request is being prepared.
/// Cross-generation KV prefix reuse store: the engine keeps the most recent
/// post-prefill KV snapshots keyed by their exact prompt tokens, so a request
/// whose history extends a cached prompt (multi-turn chat, agentic tool
/// rounds, identical retries) skips re-prefilling the shared prefix. Newest
/// entries live at the back; eviction is LRU under both an entry-count and a
/// total-byte budget (KV state is O(context) per entry, so unbounded caching
/// would balloon memory).
struct PrefixCache {
    entries: VecDeque<PrefixCacheEntry>,
}

struct PrefixCacheEntry {
    /// The exact prompt tokens the snapshot was captured for.
    tokens: Vec<u32>,
    /// KV state taken right after that prompt's prefill (before decode).
    snapshot: Arc<KvSnapshot>,
    /// Snapshot size in bytes, kept for the eviction budget.
    bytes: usize,
}

/// At most this many recent prompts are cached.
const PREFIX_CACHE_MAX_ENTRIES: usize = 4;
/// Total snapshot bytes the cache may hold before LRU eviction kicks in.
const PREFIX_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Prompts shorter than this are not cached — too little history to reuse.
const PREFIX_CACHE_MIN_TOKENS: usize = 16;
/// Reuse below this many shared positions costs more than it saves.
const PREFIX_MIN_REUSE_POSITIONS: usize = 8;

impl PrefixCache {
    /// Longest cached prefix of `prompt`: `(positions, snapshot)` where the
    /// entry's tokens match `prompt[..positions]` and at least one prompt
    /// token remains to forward (a full hit has no stored logits to return,
    /// and re-forwarding the final token is cheap). Touches the entry (LRU)
    /// on hit.
    fn longest_prefix(&mut self, prompt: &[u32]) -> Option<(usize, Arc<KvSnapshot>)> {
        // (shared positions, entry index) of the best match so far.
        let mut best: Option<(usize, usize)> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let shared = entry
                .tokens
                .iter()
                .zip(prompt.iter())
                .take_while(|(left, right)| left == right)
                .count()
                .min(prompt.len().saturating_sub(1));
            if shared >= PREFIX_MIN_REUSE_POSITIONS
                && best.is_none_or(|(best_shared, _)| shared > best_shared)
            {
                best = Some((shared, index));
            }
        }
        let (shared, index) = best?;
        let entry = self.entries.remove(index)?;
        self.entries.push_back(entry);
        let entry = self.entries.back()?;
        Some((shared, Arc::clone(&entry.snapshot)))
    }

    /// Inserts a post-prefill snapshot for `tokens`, evicting
    /// least-recently-used entries until both budgets hold.
    fn insert(&mut self, tokens: Vec<u32>, snapshot: KvSnapshot) {
        let bytes = snapshot
            .kv_k
            .iter()
            .chain(snapshot.kv_v.iter())
            .map(|layer| layer.len() * std::mem::size_of::<f32>())
            .sum();
        self.entries.push_back(PrefixCacheEntry {
            tokens,
            snapshot: Arc::new(snapshot),
            bytes,
        });
        let mut total: usize = self.entries.iter().map(|entry| entry.bytes).sum();
        while self.entries.len() > PREFIX_CACHE_MAX_ENTRIES
            || (total > PREFIX_CACHE_MAX_BYTES && self.entries.len() > 1)
        {
            let evicted = self
                .entries
                .pop_front()
                .map(|entry| entry.bytes)
                .unwrap_or(0);
            total = total.saturating_sub(evicted);
        }
    }
}

pub struct Engine {
    /// Backend model and its selected device, guarded because decoding
    /// mutates the KV cache.
    model: Mutex<LoadedModel>,
    /// Hugging Face tokenizer used for both prompt encoding and output
    /// decoding.
    tokenizer: Tokenizer,
    /// Human-readable model identifier (usually the directory or file name).
    model_id: String,
    /// Architecture name used for backend and template lookup.
    architecture: Option<String>,
    /// Function that renders `ChatMessage`s into the model's prompt format.
    prompt_renderer: super::registry::PromptRenderer,
    /// Effective context cap: model limit, family cap, and `--ctx-size`
    /// combined.
    max_position_embeddings: usize,
    /// Token ids that terminate generation, including stop-marker tokens
    /// resolved through the vocabulary.
    eos_token_ids: HashSet<u32>,
    /// Textual markers stripped from output when encountered.
    stop_markers: Vec<String>,
    /// Planned residency tier ("vram", "ram", or "disk") per atomic unit.
    ant_plan: std::collections::BTreeMap<String, String>,
    /// Bytes counted as resident immediately after loading.
    ant_resident_bytes: u64,
    /// VRAM pool budget used by Ant decisions.
    ant_budget_bytes: u64,
    /// Original Ant configuration, retained so runtime swaps can re-create a
    /// manager with the same settings.
    ant_settings: Option<AntSettings>,
    /// Atomic weight units discovered for this model.
    ant_units: Vec<AntUnit>,
    /// Model directory, needed by Ant for paging files in and out.
    ant_model_dir: Option<PathBuf>,
    /// Whether the selected device is an accelerator, which gates budget
    /// enforcement behavior.
    ant_device_accelerator: bool,
    /// Per-layer residency summary used for diagnostics.
    ant_layer_summary: Vec<AntLayerResidency>,
    /// Static per-unit metadata resolved at load time.
    ant_unit_index: std::collections::BTreeMap<String, AntUnitRuntime>,
    /// Mutable cross-request Ant bookkeeping.
    ant_runtime: Mutex<AntRuntimeState>,
    /// Cross-generation KV prefix cache (see [`PrefixCache`]); consulted and
    /// refilled by `generate_inner`. Backends without KV snapshot support
    /// simply never hit it.
    prefix_cache: Mutex<PrefixCache>,
}

/// Deserialized subset of a Hugging Face `config.json` needed by the engine.
#[derive(Debug, Deserialize)]
struct ModelConfig {
    /// Model family identifier, used as the architecture fallback.
    model_type: String,
    /// Explicit architecture class names, when present.
    architectures: Option<Vec<String>>,
    /// EOS token id(s), which may be a single number or an array.
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    /// Maximum trained context length; defaults when the field is absent.
    #[serde(default = "default_context_size")]
    max_position_embeddings: usize,
}

/// Serde default for models whose config omits `max_position_embeddings`.
fn default_context_size() -> usize {
    4096
}

/// Ranks a residency tier for eviction ordering.
///
/// Lower values are more valuable (VRAM before RAM); unknown tiers sort last
/// so they are evicted first. The numeric order is inverted by callers that
/// sort descending when choosing victims.
fn ant_tier_priority(tier: &str) -> u64 {
    match tier {
        "vram" => 0,
        "ram" => 1,
        _ => 2,
    }
}

/// Selects the weight shards that the loader should map for the current Ant
/// plan.
///
/// A shard is kept when at least one atomic unit sourced from it is neither
/// disk-tier nor evicted at load time. Several units normally share a shard,
/// so the result is a de-duplicated set of shard paths derived from the
/// original ordered weight list. Disk-only and evicted units are excluded so
/// the backend does not fault their pages in during loading.
fn ant_resident_weight_paths(
    weight_paths: &[PathBuf],
    units: &[AntUnit],
    plan: &std::collections::BTreeMap<String, String>,
    evicted: &[String],
) -> Vec<PathBuf> {
    // Collect source file names (not full paths) because the manifest stores
    // shard-relative names while the loader receives parent-joined paths.
    let mut keep: std::collections::HashSet<String> = std::collections::HashSet::new();
    for unit in units {
        // Units missing from the plan are treated as VRAM residents, matching
        // the loader's default assumption elsewhere.
        let tier = plan
            .get(&unit.unit_id)
            .map(|tier| tier.as_str())
            .unwrap_or("vram");
        if tier == "disk" || evicted.iter().any(|e| e == &unit.unit_id) {
            continue;
        }
        keep.insert(unit.source_file.clone());
    }
    // Preserve the caller's shard order; only the membership test is
    // name-based. `file_name` guards against a manifest entry that embeds a
    // directory, and non-UTF-8 names simply cannot match a manifest entry.
    let mut paths = weight_paths.to_vec();
    paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| keep.contains(name))
    });
    paths
}

/// Applies the VRAM budget to the initial residency plan and evicts units that
/// do not fit.
///
/// Only accelerator devices enforce the budget here: on CPU the "VRAM" pool
/// is conceptual and real memory pressure is handled by the OS. Eviction
/// walks the lowest-value tiers first (RAM before VRAM, and higher layer ids
/// before lower ones within a tier). Evictions are best-effort; a failed
/// `evict_unit` leaves the unit resident and keeps counting against the
/// budget. Returns the remaining resident byte count and the successfully
/// evicted unit ids so callers can update their bookkeeping.
fn ant_apply_budget(
    manager: &Ant,
    dst_dir: &Path,
    units: &[AntUnit],
    plan: &std::collections::BTreeMap<String, String>,
    budget_bytes: u64,
    device_is_accelerator: bool,
) -> (u64, Vec<String>) {
    let mut resident_bytes = 0u64;
    let mut evicted: Vec<String> = Vec::new();
    // Tuple layout: (tier priority, byte size, unit id). Priority is kept so
    // the sort can order victims without looking the unit up again.
    let mut resident: Vec<(u64, u64, String)> = Vec::new();
    for unit in units {
        let tier = plan
            .get(&unit.unit_id)
            .map(|tier| tier.as_str())
            .unwrap_or("vram");
        if tier != "disk" {
            // Saturating arithmetic avoids overflow on adversarial manifests
            // with absurdly large declared sizes.
            resident_bytes = resident_bytes.saturating_add(unit.byte_size());
            resident.push((
                ant_tier_priority(tier),
                unit.byte_size(),
                unit.unit_id.clone(),
            ));
        }
    }
    // A zero budget means "unlimited" and skips the eviction pass entirely.
    if device_is_accelerator && budget_bytes > 0 && resident_bytes > budget_bytes {
        // Descending priority evicts RAM-tier units before VRAM-tier ones.
        resident.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, bytes, unit_id) in resident {
            if resident_bytes <= budget_bytes {
                break;
            }
            if manager.evict_unit(dst_dir, &unit_id).is_ok() {
                resident_bytes = resident_bytes.saturating_sub(bytes);
                evicted.push(unit_id);
            }
        }
    }
    (resident_bytes, evicted)
}

/// Runtime residency facts for one atomic unit, resolved once at load time.
///
/// Keeping these values in a map indexed by unit id avoids re-parsing the
/// manifest or re-querying metadata during the hot generation path, while
/// still allowing swaps to update byte accounting cheaply.
#[derive(Debug, Clone)]
struct AntUnitRuntime {
    /// Serialized size of the unit in bytes.
    bytes: u64,
    /// Transformer layer the unit belongs to; -1 typically marks embeddings
    /// or other non-layer tensors.
    layer_id: i32,
    /// Planned tier at load time ("vram", "ram", or "disk").
    tier: String,
    /// Pinned units are exempt from eviction (for example the output head).
    pinned: bool,
}

/// Phase 3: cross-generation Ant bookkeeping. Tracks VRAM/RAM occupancy, the
/// units evicted at runtime, and the in-memory (RAM-tier) units that are next
/// in line for promotion into VRAM, so every swap re-checks the budget before
/// it executes and only swaps in as memory becomes available.
///
/// The state is protected by `Engine::ant_runtime` rather than living inside
/// the model lock, which lets a request plan its residency changes before it
/// needs exclusive access to the backend.
#[derive(Default)]
struct AntRuntimeState {
    /// Bytes currently accounted to the VRAM pool.
    vram_used_bytes: u64,
    /// Bytes currently accounted to the RAM pool.
    ram_used_bytes: u64,
    /// Units that were evicted and are therefore not resident.
    evicted: HashSet<String>,
    /// Disk-tier units paged in at runtime; they are resident although the
    /// plan places them on disk.
    runtime_resident: HashSet<String>,
    /// RAM-tier resident units, in layer order: the in-memory portion next in
    /// line for promotion / swapping into VRAM.
    promotion_queue: Vec<String>,
    /// Number of successful swap-ins since load, for diagnostics.
    swaps_in: u64,
    /// Number of successful swap-outs since load, for diagnostics.
    swaps_out: u64,
    /// Number of swap attempts rejected because the budget did not fit.
    swap_skips: u64,
}

/// Construction, residency management, prompt preparation, and generation for
/// a loaded model.
///
/// Methods that only inspect immutable metadata take `&self`; decoding takes
/// `&self` as well and relies on the model mutex for interior mutability so a
/// single `Engine` can be shared across threads or requests.
impl Engine {
    /// Loads a model directory or Safetensors file and prepares it for chat.
    ///
    /// The method performs the following phases in order: validate thread and
    /// context arguments, read configuration (falling back to the manifest
    /// snapshot), load the tokenizer, resolve weight shards, choose a prompt
    /// renderer, build the Ant residency plan, select and construct the
    /// backend, then derive EOS ids, stop markers, and the effective context
    /// limit. All failures are reported as human-readable strings because
    /// this API is consumed directly by CLI and server error paths.
    pub fn load(
        path: &Path,
        threads: Option<usize>,
        ctx_size: usize,
        ant: Option<&AntSettings>,
    ) -> Result<Self, String> {
        // A zero context cannot hold even an empty prompt, and downstream
        // token budget arithmetic would underflow semantically.
        if ctx_size == 0 {
            return Err("--ctx-size must be greater than 0.".into());
        }
        // Fall back to the number of physical cores for the Rayon pool. The
        // `.max(1)` protects unusual platforms that report zero CPUs.
        let threads = threads.unwrap_or_else(|| num_cpus::get_physical().max(1));
        if threads == 0 {
            return Err("--threads must be greater than 0.".into());
        }
        // Reuse the existing global pool when one is already initialized
        // (e.g. in-process callers such as the test harness). A second
        // `build_global` fails, but that is expected and intentionally ignored
        // so repeated engine loads share one pool instead of panicking.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
        // Both a directory and a direct shard path are accepted; for a file we
        // use its parent as the root for config, tokenizer, and manifest.
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let config_path = model_dir.join("config.json");
        let config_bytes = match fs::read(&config_path) {
            Ok(bytes) => bytes,
            Err(read_error) => {
                // Ant: a prepared directory is self-describing — rebuild a
                // minimal config from the manifest snapshot so a missing
                // config.json does not gate loading.
                synth_config_from_manifest(model_dir, &read_error)?
            }
        };
        // Keep both a raw JSON tree (passed to backends that need extra
        // architecture-specific fields) and a typed view for the common fields.
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        let config: ModelConfig = serde_json::from_value(raw_config.clone())
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        // Prefer the first declared architecture class name; fall back to the
        // model_type (for example "llama" or "qwen2") when the list is absent.
        let architecture = config
            .architectures
            .as_ref()
            .and_then(|architectures| architectures.first())
            .map(String::as_str)
            .unwrap_or(config.model_type.as_str())
            .to_owned();
        // Template lookup doubles as the architecture support check: an
        // unregistered architecture cannot be rendered or loaded.
        let template = super::registry::chat_template_for_architecture(&architecture)
            .ok_or_else(|| format!("Unsupported model architecture '{architecture}'."))?;

        // Tokenizer loading is independent of weight loading and fails early
        // with a path-specific message to make setup errors easy to diagnose.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer_bytes = fs::read(&tokenizer_path).map_err(|error| {
            format!(
                "Failed to read tokenizer '{}': {error}",
                tokenizer_path.display()
            )
        })?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| {
            format!(
                "Failed to load tokenizer '{}': {error}",
                tokenizer_path.display()
            )
        })?;
        // Weight discovery also returns manifest-provided stop markers, which
        // are more authoritative than the architecture defaults.
        let (weight_paths, manifest_stop_markers) = resolve_model_files(path)?;
        let vocabulary = tokenizer.get_vocab(true);
        // Llama 3-style header templates must not be used with a tokenizer
        // that lacks the special header tokens, or the model would see literal
        // text instead of control tokens. Manifest markers take precedence;
        // without a manifest we probe the vocabulary directly.
        let use_header_renderer = template.header_renderer.is_some()
            && (manifest_stop_markers
                .as_ref()
                .is_some_and(|markers| markers.iter().any(|marker| marker == "<|eot_id|>"))
                || manifest_stop_markers.is_none()
                    && vocabulary.contains_key("<|start_header_id|>")
                    && vocabulary.contains_key("<|eot_id|>"));
        let prompt_renderer = if use_header_renderer {
            // `header_renderer` was checked for `Some` above; `unwrap_or`
            // merely avoids an unwrap and keeps the classic renderer as a
            // defensive fallback.
            template.header_renderer.unwrap_or(template.render)
        } else {
            template.render
        };
        // Any non-CPU device string is treated as an accelerator request. The
        // exact backend is resolved later by the model registry.
        let device_is_accelerator =
            std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into()) != "cpu";
        let mut effective_weight_paths = weight_paths.to_vec();
        let mut ant_plan = std::collections::BTreeMap::new();
        let mut ant_resident_bytes = 0u64;
        let mut ant_budget_bytes = AntSettings::default().vram_pool_bytes;
        let mut ant_settings = ant.cloned();
        let mut ant_units: Vec<AntUnit> = Vec::new();
        let mut ant_evicted: Vec<String> = Vec::new();
        let mut ant_layer_summary: Vec<AntLayerResidency> = Vec::new();
        let mut ant_unit_index: std::collections::BTreeMap<String, AntUnitRuntime> =
            std::collections::BTreeMap::new();
        let mut ant_runtime = AntRuntimeState::default();
        if let Some(settings) = ant {
            ant_budget_bytes = settings.vram_pool_bytes;
            if settings.enabled {
                // Ant: derive the atomic-unit residency plan from the units manifest.
                //  Ant: 从 units manifest 派生原子单元驻留计划。
                if let Ok(manager) = Ant::with_settings(settings) {
                    if let Ok(units) = manager.units_for_model(model_dir) {
                        let plan = manager.plan_residency(&units);
                        for unit in &units {
                            let tier = plan
                                .get(&unit.unit_id)
                                .map(|tier| tier.as_str())
                                .unwrap_or("vram");
                            ant_plan.insert(unit.unit_id.clone(), tier.to_string());
                        }
                        let (resident_bytes, evicted_units) = ant_apply_budget(
                            &manager,
                            model_dir,
                            &units,
                            &plan,
                            ant_budget_bytes,
                            device_is_accelerator,
                        );
                        ant_resident_bytes = resident_bytes;
                        // Phase 2: track partial-layer residency for observability.
                        ant_layer_summary = Ant::layer_residency_summary(&units, &plan);
                        // Phase 3: seed the runtime bookkeeping from the load plan.
                        let mut promotion_queue: Vec<(i32, String)> = Vec::new();
                        for unit in &units {
                            let tier = plan
                                .get(&unit.unit_id)
                                .map(String::as_str)
                                .unwrap_or("vram");
                            let evicted = evicted_units.iter().any(|e| e == &unit.unit_id);
                            ant_unit_index.insert(
                                unit.unit_id.clone(),
                                AntUnitRuntime {
                                    bytes: unit.byte_size(),
                                    layer_id: unit.layer_id,
                                    tier: tier.to_string(),
                                    pinned: settings
                                        .resident_units
                                        .iter()
                                        .any(|pinned| pinned == &unit.unit_id),
                                },
                            );
                            if evicted {
                                ant_runtime.evicted.insert(unit.unit_id.clone());
                                continue;
                            }
                            match tier {
                                "vram" => {
                                    ant_runtime.vram_used_bytes = ant_runtime
                                        .vram_used_bytes
                                        .saturating_add(unit.byte_size());
                                }
                                "ram" => {
                                    ant_runtime.ram_used_bytes =
                                        ant_runtime.ram_used_bytes.saturating_add(unit.byte_size());
                                    promotion_queue.push((unit.layer_id, unit.unit_id.clone()));
                                }
                                _ => {}
                            }
                        }
                        promotion_queue.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                        ant_runtime.promotion_queue = promotion_queue
                            .into_iter()
                            .map(|(_, unit_id)| unit_id)
                            .collect();
                        ant_units = units;
                        ant_evicted = evicted_units;
                    }
                    ant_settings = Some(settings.clone());
                }
            }
        }
        if let Some(settings) = ant
            && settings.enabled
            && !ant_units.is_empty()
        {
            effective_weight_paths =
                ant_resident_weight_paths(&weight_paths, &ant_units, &ant_plan, &ant_evicted);
        }
        // Ant: resolve the inference device from the residency plan. Resident VRAM
        // units load into the accelerator device; resident RAM-only units load into
        // the CPU device; disk units are excluded before the loader runs.
        let mut ant_resident_device: Option<Device> = None;
        let mut ant_resident_device_is_accelerator = device_is_accelerator;
        if let Some(settings) = ant_settings.as_ref()
            && settings.enabled
            && !ant_units.is_empty()
        {
            let mut resident_tiers: Vec<&str> = Vec::new();
            for unit in &ant_units {
                let tier = ant_plan
                    .get(&unit.unit_id)
                    .map(|tier| tier.as_str())
                    .unwrap_or("vram");
                if tier == "disk" || ant_evicted.iter().any(|e| e == &unit.unit_id) {
                    continue;
                }
                resident_tiers.push(tier);
            }
            if !resident_tiers.is_empty() {
                if resident_tiers.contains(&"vram") && device_is_accelerator {
                    let selected = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into());
                    ant_resident_device = Some(super::registry::backend_device(&selected)?);
                } else if !resident_tiers.contains(&"vram") {
                    ant_resident_device = Some(Device::Cpu);
                    ant_resident_device_is_accelerator = false;
                }
            }
        }
        // Ant: prefer the native mmap-backed CPU backend for architectures that
        // support it — its tiers act on real pages.
        let native_enabled = std::env::var("CHEETAH_NATIVE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let native_model = if device_is_accelerator || !native_enabled {
            None
        } else {
            match super::registry::backend_loader_for_architecture(&architecture, "native") {
                Some(super::registry::BackendLoader::Native(loader)) => {
                    let family = super::registry::family_for_architecture(&architecture)
                        .ok_or_else(|| {
                            format!(
                                "Architecture '{architecture}' registers a native backend without family parameters."
                            )
                        })?;
                    Some(
                        loader(model_dir, &config_bytes, &raw_config, &ant_plan, family)
                            .map(|runtime| (Device::Cpu, runtime))?,
                    )
                }
                _ => None,
            }
        };
        let (device, model) = if let Some((device, model)) = native_model {
            (device, model)
        } else if let Some(resident_device) = ant_resident_device {
            super::registry::load_model_for_device(
                &effective_weight_paths,
                &config_bytes,
                raw_config.clone(),
                &architecture,
                resident_device,
                &ant_plan,
            )?
        } else {
            super::registry::load_model(
                &effective_weight_paths,
                &config_bytes,
                raw_config,
                &architecture,
                &ant_plan,
            )?
        };
        let mut eos_token_ids = HashSet::new();
        if let Some(value) = config.eos_token_id {
            match value {
                serde_json::Value::Number(number) => {
                    if let Some(id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) {
                        eos_token_ids.insert(id);
                    }
                }
                serde_json::Value::Array(ids) => {
                    for id in ids {
                        if let Some(id) = id.as_u64().and_then(|id| u32::try_from(id).ok()) {
                            eos_token_ids.insert(id);
                        }
                    }
                }
                _ => {}
            }
        }
        let stop_markers = manifest_stop_markers.unwrap_or_else(|| {
            super::registry::stop_markers_for_architecture(&architecture)
                .iter()
                .map(|marker| (*marker).to_owned())
                .collect()
        });
        for marker in &stop_markers {
            if let Some(id) = vocabulary.get(marker) {
                eos_token_ids.insert(*id);
            }
        }
        let family_context_cap = super::registry::family_for_architecture(&architecture)
            .and_then(|family| family.max_position_embeddings)
            .unwrap_or(usize::MAX);
        Ok(Self {
            model: Mutex::new(LoadedModel { model, device }),
            tokenizer,
            model_id: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model")
                .to_owned(),
            architecture: Some(architecture),
            prompt_renderer,
            max_position_embeddings: config
                .max_position_embeddings
                .min(ctx_size)
                .min(family_context_cap),
            eos_token_ids,
            stop_markers,
            ant_plan,
            ant_resident_bytes,
            ant_budget_bytes,
            ant_settings,
            ant_units,
            ant_model_dir: Some(model_dir.to_path_buf()),
            ant_device_accelerator: ant_resident_device_is_accelerator,
            ant_layer_summary,
            ant_unit_index,
            ant_runtime: Mutex::new(ant_runtime),
            prefix_cache: Mutex::new(PrefixCache {
                entries: VecDeque::new(),
            }),
        })
    }

    /// The loaded model's display id (file stem or directory name).
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The resolved architecture name from the model config; `None` when nothing was named.
    pub fn model_architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    /// Backend device label forwarded from the boxed [`ModelForward`] runtime.
    pub fn device_name(&self) -> &'static str {
        let loaded = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loaded.model.device_name()
    }

    /// Ant: whether current runtime residency (VRAM + RAM occupancy tracked by
    /// the Phase 3 bookkeeping) stays within the configured budget.
    pub fn ant_within_budget(&self) -> bool {
        let state = self.ant_lock_runtime();
        state.vram_used_bytes.saturating_add(state.ram_used_bytes) <= self.ant_budget_bytes
    }

    /// Ant: push tier transitions into the model's real memory pages when the
    /// backend owns mmap-backed weights (native CPU backend); host-resident
    /// models report no tier control and only keep the C-side bookkeeping.
    fn ant_apply_real_tier(&self, page_ins: &[(String, i32)], evictions: &[String]) {
        if page_ins.is_empty() && evictions.is_empty() {
            return;
        }
        let Ok(mut loaded) = self.model.lock() else {
            return;
        };
        let Some(control) = loaded.model.as_ant_tier_control() else {
            return;
        };
        for (unit_id, tier) in page_ins {
            control.page_in_unit(unit_id, *tier);
        }
        for unit_id in evictions {
            control.evict_unit(unit_id);
        }
    }

    /// Locks the Ant bookkeeping mutex, recovering poisoned locks instead of
    /// panicking the server or REPL.
    fn ant_lock_runtime(&self) -> std::sync::MutexGuard<'_, AntRuntimeState> {
        self.ant_runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Phase 3: page one unit in at `tier_value`, re-checking the configured
    /// budget before the swap executes. VRAM-targeted swaps evict residents to
    /// make room only when both VRAM swap and `allow_evict` permit it; the
    /// budget is re-checked before each eviction and once more before the
    /// swap-in itself, so units only swap in as memory becomes available.
    /// Returns true when the swap-in executed.
    fn ant_swap_in_unit(
        &self,
        manager: &Ant,
        unit_id: &str,
        tier_value: i32,
        allow_evict: bool,
    ) -> bool {
        let Some(settings) = self.ant_settings.as_ref() else {
            return false;
        };
        let Some(runtime) = self.ant_unit_index.get(unit_id) else {
            return false;
        };
        let mut state = self.ant_lock_runtime();
        if tier_value == 0 {
            // Eviction candidates run from the highest layer backwards; pinned
            // units, already-evicted units and the swap-in target never move.
            let mut evictable: Vec<(u64, i32, String)> = self
                .ant_unit_index
                .iter()
                .filter(|(candidate_id, candidate)| {
                    candidate_id.as_str() != unit_id
                        && candidate.tier == "vram"
                        && !candidate.pinned
                        && !state.evicted.contains(*candidate_id)
                })
                .map(|(candidate_id, candidate)| {
                    (candidate.bytes, candidate.layer_id, candidate_id.clone())
                })
                .collect();
            evictable.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
            let evictable: Vec<(u64, String)> = evictable
                .into_iter()
                .map(|(bytes, _, candidate_id)| (bytes, candidate_id))
                .collect();
            let Some(evictions) = ant_swap_fit(
                state.vram_used_bytes,
                settings.vram_pool_bytes,
                runtime.bytes,
                settings.allow_vram_swap && allow_evict,
                &evictable,
            ) else {
                state.swap_skips = state.swap_skips.saturating_add(1);
                return false;
            };
            if !evictions.is_empty() {
                let Some(model_dir) = self.ant_model_dir.as_deref() else {
                    return false;
                };
                for eviction in &evictions {
                    if manager.evict_unit(model_dir, eviction).is_ok() {
                        if let Some(evicted_runtime) = self.ant_unit_index.get(eviction) {
                            state.vram_used_bytes =
                                state.vram_used_bytes.saturating_sub(evicted_runtime.bytes);
                        }
                        state.evicted.insert(eviction.clone());
                        state.swaps_out = state.swaps_out.saturating_add(1);
                        // Ant: drop the unit's real pages on mmap backends.
                        self.ant_apply_real_tier(&[], std::slice::from_ref(eviction));
                    }
                }
                // Re-check before the swap: only proceed once the pool fits.
                if state.vram_used_bytes.saturating_add(runtime.bytes) > settings.vram_pool_bytes {
                    state.swap_skips = state.swap_skips.saturating_add(1);
                    return false;
                }
            }
            if manager.page_in(unit_id, 0).is_err() {
                return false;
            }
            // Ant: fault the unit's real pages in on mmap backends.
            self.ant_apply_real_tier(&[(unit_id.to_string(), 0)], &[]);
            state.vram_used_bytes = state.vram_used_bytes.saturating_add(runtime.bytes);
            state.swaps_in = state.swaps_in.saturating_add(1);
            state.evicted.remove(unit_id);
            // A VRAM swap-in leaves the promotion line.
            state.promotion_queue.retain(|queued| queued != unit_id);
            true
        } else {
            // RAM-targeted swap: enforce the RAM budget and skip when it does
            // not fit (evicting RAM residents to disk to make room thrashes).
            if state.ram_used_bytes.saturating_add(runtime.bytes) > settings.ram_budget_bytes {
                state.swap_skips = state.swap_skips.saturating_add(1);
                return false;
            }
            if manager.page_in(unit_id, 1).is_err() {
                return false;
            }
            // Ant: fault the unit's real pages in on mmap backends.
            self.ant_apply_real_tier(&[(unit_id.to_string(), 1)], &[]);
            state.swaps_in = state.swaps_in.saturating_add(1);
            if state.evicted.remove(unit_id) || state.runtime_resident.insert(unit_id.to_string()) {
                state.ram_used_bytes = state.ram_used_bytes.saturating_add(runtime.bytes);
            }
            // A newly RAM-resident unit joins the promotion line.
            if runtime.tier == "ram"
                && !state.promotion_queue.iter().any(|queued| queued == unit_id)
            {
                state.promotion_queue.push(unit_id.to_string());
            }
            true
        }
    }

    /// Generation hook: applies per-request resident-unit overrides — units the
    /// request pins that sit on the disk tier or were evicted are swapped back
    /// in (budget-checked, evictions allowed).
    fn apply_ant_request_overrides(&self, overrides: &HashSet<String>) -> Result<(), String> {
        let Some(settings) = self.ant_settings.as_ref() else {
            return Ok(());
        };
        if !settings.enabled || overrides.is_empty() || self.ant_units.is_empty() {
            return Ok(());
        }
        let Ok(manager) = Ant::with_settings(settings) else {
            return Ok(());
        };
        let targets: Vec<(String, i32)> = {
            let state = self.ant_lock_runtime();
            self.ant_units
                .iter()
                .filter(|unit| overrides.contains(&unit.unit_id))
                .filter(|unit| {
                    let tier = self
                        .ant_plan
                        .get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram");
                    if tier == "disk" {
                        return true;
                    }
                    state.evicted.contains(&unit.unit_id)
                })
                .map(|unit| {
                    let tier = self
                        .ant_plan
                        .get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram");
                    (unit.unit_id.clone(), if tier == "vram" { 0 } else { 1 })
                })
                .collect()
        };
        for (unit_id, tier_value) in targets {
            let _ = self.ant_swap_in_unit(&manager, &unit_id, tier_value, true);
        }
        Ok(())
    }

    /// Generation hook: swaps up to `prefetch_stages` evicted resident-tier
    /// units back in, ordered by tier priority then layer id — every swap
    /// re-checks the budget before it executes.
    fn prefetch_ant_units(&self) -> Result<(), String> {
        let Some(settings) = self.ant_settings.as_ref() else {
            return Ok(());
        };
        if !settings.enabled || settings.prefetch_stages <= 0 || self.ant_units.is_empty() {
            return Ok(());
        }
        let Ok(manager) = Ant::with_settings(settings) else {
            return Ok(());
        };
        let mut candidates: Vec<(u64, i32, String, i32)> = {
            let state = self.ant_lock_runtime();
            if state.evicted.is_empty() {
                return Ok(());
            }
            self.ant_units
                .iter()
                .filter(|unit| state.evicted.contains(&unit.unit_id))
                .filter(|unit| {
                    self.ant_plan
                        .get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram")
                        != "disk"
                })
                .map(|unit| {
                    let tier = self
                        .ant_plan
                        .get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram");
                    (
                        ant_tier_priority(tier),
                        unit.layer_id,
                        unit.unit_id.clone(),
                        if tier == "vram" { 0 } else { 1 },
                    )
                })
                .collect()
        };
        candidates.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        for (_, _, unit_id, tier_value) in candidates
            .into_iter()
            .take(settings.prefetch_stages.max(0) as usize)
        {
            // Budget-checked swap-in per prefetch stage.
            let _ = self.ant_swap_in_unit(&manager, &unit_id, tier_value, true);
        }
        Ok(())
    }

    /// Phase 3: promote the in-memory (RAM-tier) units that are next in line
    /// into VRAM. Each swap re-checks the VRAM pool first and only executes
    /// when the unit fits without evicting anyone; the walk stops at the first
    /// unit that no longer fits, so later layers are never promoted at the
    /// expense of earlier ones.
    fn promote_ant_units(&self) {
        let Some(settings) = self.ant_settings.as_ref() else {
            return;
        };
        if !settings.enabled || self.ant_units.is_empty() {
            return;
        }
        let Ok(manager) = Ant::with_settings(settings) else {
            return;
        };
        let queue: Vec<String> = self.ant_lock_runtime().promotion_queue.clone();
        for unit_id in queue {
            if !self.ant_swap_in_unit(&manager, &unit_id, 0, false) {
                break;
            }
        }
    }

    /// Generation hook: pages evicted resident-tier units back in, then evicts
    /// lowest-priority residents (RAM first, highest layers first; pinned units
    /// never move) until the VRAM+RAM occupancy holds the configured budget.
    fn enforce_ant_budget(&self) {
        let Some(settings) = self.ant_settings.as_ref() else {
            return;
        };
        if !settings.enabled || settings.vram_pool_bytes == 0 {
            return;
        }
        let Ok(manager) = Ant::with_settings(settings) else {
            return;
        };
        let units = &self.ant_units;
        let plan = &self.ant_plan;

        let evicted_targets: Vec<(String, i32)> = {
            let state = self.ant_lock_runtime();
            if state.evicted.is_empty()
                && state.vram_used_bytes.saturating_add(state.ram_used_bytes)
                    <= settings.vram_pool_bytes
            {
                return;
            }
            units
                .iter()
                .filter(|unit| state.evicted.contains(&unit.unit_id))
                .filter(|unit| {
                    plan.get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram")
                        != "disk"
                })
                .map(|unit| {
                    let tier = plan
                        .get(&unit.unit_id)
                        .map(String::as_str)
                        .unwrap_or("vram");
                    (unit.unit_id.clone(), if tier == "vram" { 0 } else { 1 })
                })
                .collect()
        };

        // Ant: page evicted resident-tier units back in before enforcing the
        // memory budget again — each swap re-checks the budget first.
        for (unit_id, tier_value) in evicted_targets {
            self.ant_swap_in_unit(&manager, &unit_id, tier_value, true);
        }

        let mut state = self.ant_lock_runtime();
        let mut resident_bytes = state.vram_used_bytes.saturating_add(state.ram_used_bytes);
        if resident_bytes <= settings.vram_pool_bytes || !self.ant_device_accelerator {
            return;
        }
        let Some(model_dir) = self.ant_model_dir.as_deref() else {
            return;
        };
        // Ant: still over budget — evict lowest-priority residents (RAM tier
        // first, then the highest VRAM layers; pinned units never move) and
        // update the runtime bookkeeping with every eviction.
        let mut evictable: Vec<(u64, i32, String)> = self
            .ant_unit_index
            .iter()
            .filter(|(unit_id, runtime)| {
                if runtime.pinned || state.evicted.contains(*unit_id) {
                    return false;
                }
                match runtime.tier.as_str() {
                    "vram" => true,
                    "ram" => true,
                    "disk" => state.runtime_resident.contains(*unit_id),
                    _ => false,
                }
            })
            .map(|(unit_id, runtime)| {
                (
                    ant_tier_priority(runtime.tier.as_str()),
                    runtime.layer_id,
                    unit_id.clone(),
                )
            })
            .collect();
        evictable.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        for (_, _, unit_id) in evictable {
            if resident_bytes <= settings.vram_pool_bytes {
                break;
            }
            let Some(runtime) = self.ant_unit_index.get(&unit_id) else {
                continue;
            };
            if manager.evict_unit(model_dir, &unit_id).is_ok() {
                resident_bytes = resident_bytes.saturating_sub(runtime.bytes);
                match runtime.tier.as_str() {
                    "vram" => {
                        state.vram_used_bytes = state.vram_used_bytes.saturating_sub(runtime.bytes);
                    }
                    _ => {
                        state.ram_used_bytes = state.ram_used_bytes.saturating_sub(runtime.bytes);
                        state.runtime_resident.remove(&unit_id);
                    }
                }
                state.evicted.insert(unit_id.clone());
                state.promotion_queue.retain(|queued| queued != &unit_id);
                state.swaps_out = state.swaps_out.saturating_add(1);
                // Ant: drop the unit's real pages on mmap backends.
                self.ant_apply_real_tier(&[], std::slice::from_ref(&unit_id));
            }
        }
    }

    /// Creates an inference context. The context is a stateless placeholder —
    /// the KV state lives inside the model runtime — but the signature keeps
    /// the generation API backend-shaped.
    pub fn new_context(&self) -> Result<InferenceContext, String> {
        Ok(InferenceContext)
    }

    /// Warms the model up with a single-token forward pass: faults in every
    /// weight page (mmap → OS page cache) and sweeps the LM head, so the
    /// first real request does not pay the cold start. The KV cache is
    /// cleared afterwards, leaving no state behind.
    pub fn warmup(&self) -> Result<(), String> {
        let loaded = self
            .model
            .lock()
            .map_err(|_| "Model mutex was poisoned.".to_string())?;
        let mut model = loaded.model.clone();
        drop(loaded);
        model
            .forward(&[0], 0)
            .map_err(|error| format!("Model warmup failed: {error}"))?;
        model.clear_cache();
        Ok(())
    }

    /// Renders and tokenizes one chat turn, truncating oldest turns (with a
    /// warning) until the input plus reserved generation tokens fits the
    /// context limit; a single turn that cannot fit is a hard error.
    ///
    /// Returns the prompt tokens and how many turns were dropped.
    pub fn prepare(
        &self,
        messages: &mut Vec<ChatMessage>,
        system: Option<&str>,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, usize), String> {
        let mut dropped_turns = 0;
        loop {
            let prompt = (self.prompt_renderer)(messages, system);
            let tokens = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|error| format!("Failed to tokenize input: {error}"))?
                .get_ids()
                .to_vec();
            if tokens.len().saturating_add(max_tokens) <= self.max_position_embeddings {
                return Ok((tokens, dropped_turns));
            }
            if messages.len() <= 1 {
                return Err(format!(
                    "Input plus reserved generation tokens exceeds the model context limit of {}; shorten the input or reduce --max-tokens.",
                    self.max_position_embeddings
                ));
            }
            let remove_count = if messages.len() > 2 { 2 } else { 1 };
            messages.drain(..remove_count);
            dropped_turns += 1;
            eprintln!(
                "{}",
                color::stderr(
                    "Context limit nearly reached; earlier conversation turns were truncated.",
                    Color::Yellow
                )
            );
        }
    }

    /// Builds the tool-calling prompt: injects the JSON tool instruction per
    /// `ToolChoice` (none / auto / required) ahead of the messages, maps the
    /// `developer` role to `system`, and tokenizes through [`Engine::prepare`].
    /// Returns the tokens plus the chat params carrying the template stop list.
    pub fn prepare_tool_chat(
        &self,
        messages_json: &str,
        tools_json: &str,
        choice: ToolChoice,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, ChatParams), String> {
        let mut messages: Vec<serde_json::Value> = serde_json::from_str(messages_json)
            .map_err(|error| format!("Failed to parse tool request messages: {error}"))?;
        let tool_instruction = match choice {
            ToolChoice::None => "Do not call tools; answer the user directly.".to_owned(),
            ToolChoice::Auto => format!(
                "If a tool is needed, respond only with JSON: {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. Available tools: {tools_json}"
            ),
            ToolChoice::Required => format!(
                "Call one of the available tools. Respond only with JSON: {{\"tool_calls\":[{{\"name\":\"function_name\",\"arguments\":{{}}}}]}}. Available tools: {tools_json}"
            ),
        };
        messages.insert(
            0,
            serde_json::json!({
                "role": "system",
                "content": tool_instruction
            }),
        );
        let mut chat_messages = messages
            .iter()
            .filter_map(|message| {
                let content = match message.get("content") {
                    Some(serde_json::Value::String(text)) => text.clone(),
                    Some(serde_json::Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                Some(ChatMessage::new(
                    match message.get("role")?.as_str()? {
                        "developer" => "system",
                        role => role,
                    },
                    content,
                ))
            })
            .collect::<Vec<_>>();
        let (tokens, _) = self.prepare(&mut chat_messages, None, max_tokens)?;
        Ok((
            tokens,
            ChatParams {
                additional_stops_json: "[]".into(),
            },
        ))
    }

    /// Upper bound on MCP tool rounds per generation: an agent loop that
    /// reaches it returns the last tool calls to the caller (the OpenAI
    /// client or the REPL user) instead of continuing silently.
    const MAX_TOOL_ROUNDS: usize = 8;

    /// The agentic tool loop for MCP-backed tools.
    ///
    /// Each round: render the tool instruction + serialize the conversation
    /// ([`Engine::prepare_tool_chat`]) → generate → parse the model output
    /// into tool calls ([`toolcalls::parse_tool_response`]). When every
    /// parsed call belongs to a connected MCP server, the calls execute
    /// through the registry, their results join the conversation as
    /// `role: "tool"` messages, and the loop continues; the first round with
    /// a plain answer — or any call the MCP registry does not own — returns,
    /// so client-declared tools keep their existing client-driven contract.
    /// Bounded by [`MAX_TOOL_ROUNDS`].
    pub fn generate_tool_chat(
        &self,
        messages_json: &str,
        tools_json: &str,
        tools: &[ToolDefinition],
        choice: ToolChoice,
        resolved: &ResolvedToolChoice,
        max_tokens: usize,
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        resident_overrides: &HashSet<String>,
        mcp: Option<&McpRegistry>,
    ) -> Result<(Generation, ToolResponse), String> {
        let mut messages = messages_json.to_owned();
        let mut context = self.new_context()?;
        let mut rounds = 0usize;
        let mut hold = ToolStreamSink::new(sink);
        loop {
            let (tokens, chat_params) =
                self.prepare_tool_chat(&messages, tools_json, choice, max_tokens)?;
            let generation = self.generate_with_chat_params(
                &mut context,
                &tokens,
                options,
                extra_stop,
                &mut hold,
                &chat_params,
                resident_overrides,
            )?;
            let parsed = chat_params
                .parse(&generation.text, false)
                .unwrap_or_else(|_| generation.text.clone());
            let response =
                toolcalls::parse_tool_response(&parsed, &generation.text, tools, resolved);
            if let ToolResponse::Calls(calls) = &response
                && let Some(mcp) = mcp
                && !calls.is_empty()
                && calls.iter().all(|call| mcp.has_tool(&call.function.name))
                && rounds < Self::MAX_TOOL_ROUNDS
            {
                hold.drop_round();
                rounds += 1;
                let mut results = Vec::with_capacity(calls.len());
                for call in calls {
                    eprintln!(
                        "{}",
                        color::stderr(
                            &format!(
                                "[mcp] calling tool '{}' on server round {}",
                                call.function.name, rounds
                            ),
                            Color::Cyan
                        )
                    );
                    let text = mcp
                        .call(&call.function.name, &call.function.arguments)
                        .unwrap_or_else(|error| format!("[error] {error}"));
                    results.push((call.id.clone(), text));
                }
                messages = toolcalls::next_tool_round(&messages, calls, &results)?;
                continue;
            }
            hold.flush_round()?;
            return Ok((generation, response));
        }
    }

    /// Public generation entry (the context placeholder is ignored); delegates
    /// to [`Engine::generate_inner`].
    pub fn generate(
        &self,
        _context: &mut InferenceContext,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        resident_overrides: &HashSet<String>,
    ) -> Result<Generation, String> {
        self.generate_inner(prompt, options, extra_stop, sink, resident_overrides)
    }

    /// Generation entry that extends the stop list with the chat template's own
    /// stop markers carried in [`ChatParams`].
    pub fn generate_with_chat_params(
        &self,
        context: &mut InferenceContext,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        params: &ChatParams,
        resident_overrides: &HashSet<String>,
    ) -> Result<Generation, String> {
        let mut stops = extra_stop.to_vec();
        let template_stops: Vec<String> = serde_json::from_str(&params.additional_stops_json)
            .map_err(|error| format!("Failed to parse template stop markers: {error}"))?;
        stops.extend(template_stops);
        self.generate(context, prompt, options, &stops, sink, resident_overrides)
    }

    /// The core generation loop.
    ///
    /// Validates the context/token limits, runs the four Ant hooks (request
    /// overrides → prefetch → promote → enforce), clones the model out of the
    /// mutex, clears the KV cache, prefills the prompt, then samples token by
    /// token: stop markers are matched on the incrementally decoded text with
    /// character-boundary guards, each accepted token feeds back at its absolute
    /// position, and emitted deltas stream through the sink.
    fn generate_inner(
        &self,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        resident_overrides: &HashSet<String>,
    ) -> Result<Generation, String> {
        if prompt.is_empty() {
            return Err("Input is empty after tokenization.".into());
        }
        if options.max_tokens == 0 {
            return Err("--max-tokens must be greater than 0.".into());
        }
        if prompt.len().saturating_add(options.max_tokens) > self.max_position_embeddings {
            return Err(format!(
                "Input plus reserved generation tokens exceeds the model context limit of {}.",
                self.max_position_embeddings
            ));
        }

        // The prefill clock covers everything before the first decode step:
        // Ant hooks, model setup and the prompt forward pass.
        let started = Instant::now();
        // Ant: apply per-request residency overrides, prefetch evicted units,
        // promote queued RAM residents into VRAM while the pool has room, then
        // enforce the resolved memory budget before generation — every swap
        // re-checks the budget before it executes.
        self.apply_ant_request_overrides(resident_overrides)?;
        self.prefetch_ant_units()?;
        self.promote_ant_units();
        self.enforce_ant_budget();

        let loaded = self
            .model
            .lock()
            .map_err(|_| "Model mutex was poisoned.".to_string())?;
        let mut model = loaded.model.clone();
        drop(loaded);
        model.clear_cache();
        // KV prefix reuse: restore a cached prefix's KV state and prefill only
        // the suffix (multi-turn history, tool rounds, identical retries).
        let mut reused_positions = 0usize;
        let cached_prefix = self
            .prefix_cache
            .lock()
            .map_err(|_| "Prefix cache mutex was poisoned.".to_string())?
            .longest_prefix(prompt);
        if let Some((positions, snapshot)) = cached_prefix
            && model.restore_kv_prefix(&snapshot, positions)
        {
            reused_positions = positions;
        }
        let mut logits_values = model
            .forward(&prompt[reused_positions..], reused_positions)
            .map_err(|error| format!("Model inference failed: {error}"))?;
        let prefill_duration = started.elapsed();
        // Cache the post-prefill KV state (before the decode loop appends
        // generated tokens) so future prompts sharing this history skip its
        // prefill entirely.
        if prompt.len() >= PREFIX_CACHE_MIN_TOKENS
            && let Some(snapshot) = model.kv_snapshot()
            && snapshot.positions == prompt.len()
            && let Ok(mut cache) = self.prefix_cache.lock()
        {
            cache.insert(prompt.to_vec(), snapshot);
        }
        let mut rng = StdRng::seed_from_u64(options.seed.unwrap_or_else(random_seed));
        let mut history = prompt.to_vec();
        let mut generated_tokens = Vec::new();
        let mut answer = String::new();
        let capture_text = sink.capture_text();
        let mut emitted_text_len = 0usize;
        let mut finish_reason = FinishReason::Length;
        let mut stop_markers = self.stop_markers.clone();
        stop_markers.extend(
            extra_stop
                .iter()
                .filter(|marker| !marker.is_empty())
                .cloned(),
        );

        let decoding_started = Instant::now();
        for _ in 0..options.max_tokens {
            if sink.interrupted() {
                finish_reason = FinishReason::Stop;
                break;
            }
            let token = sample_token(&logits_values, &history, options, &mut rng)?;
            if self.eos_token_ids.contains(&token) {
                finish_reason = FinishReason::Stop;
                break;
            }
            history.push(token);
            generated_tokens.push(token);
            let decoded = self
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|error| format!("Failed to decode model output: {error}"))?;
            let stop_at = stop_markers
                .iter()
                .filter_map(|marker| decoded.find(marker))
                .min();
            let visible_end = stop_at.unwrap_or(decoded.len());
            if !decoded.is_char_boundary(visible_end) {
                return Err("Stop sequence split a decoded character.".into());
            }
            if visible_end < emitted_text_len {
                finish_reason = FinishReason::Stop;
                break;
            }
            let delta = &decoded[emitted_text_len..visible_end];
            if capture_text {
                answer.push_str(delta);
            }
            sink.push(delta)?;
            emitted_text_len = visible_end;
            if stop_at.is_some() {
                finish_reason = FinishReason::Stop;
                break;
            }
            logits_values = model
                .forward(&[token], history.len() - 1)
                .map_err(|error| format!("Model inference failed: {error}"))?;
        }
        let decode_duration = decoding_started.elapsed();
        sink.finish()?;
        model.clear_cache();
        tracing::debug!(
            prompt_tokens = prompt.len(),
            generated_tokens = generated_tokens.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "generation complete"
        );
        Ok(Generation {
            text: answer,
            prompt_tokens: prompt.len(),
            generated_tokens: generated_tokens.len(),
            prefill_duration,
            decode_duration,
            finish_reason,
        })
    }
}

/// Resolves the weight paths and manifest stop markers for a model path
/// (directory or safetensors file).
fn resolve_model_files(path: &Path) -> Result<(Vec<PathBuf>, Option<Vec<String>>), String> {
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let manifest = read_model_manifest(model_dir)?;
    let stop_markers = manifest
        .as_ref()
        .map(parse_manifest_stop_markers)
        .transpose()?;
    let weight_paths = resolve_weight_files(path, manifest.as_ref())?;
    Ok((weight_paths, stop_markers))
}

/// Ant: rebuild a minimal config from the manifest's model-parameter snapshot
/// when the copied config.json is missing, so a prepared model directory
/// stays loadable on its own (parameters were snapshotted at `cheetah init`).
/// Family-specific extra fields (hidden_act / sliding_window / ...) required
/// by the backend-native inference paths are merged from the family constants.
fn synth_config_from_manifest(
    model_dir: &Path,
    read_error: &std::io::Error,
) -> Result<Vec<u8>, String> {
    let manifest = read_model_manifest(model_dir)?;
    let model = manifest
        .as_ref()
        .and_then(|manifest| manifest.get("model"))
        .and_then(|model| model.as_object())
        .ok_or_else(|| {
            format!(
                "Failed to read {}: {read_error}. The directory also has no manifest model-parameter snapshot; run `cheetah init` first.",
                model_dir.join("config.json").display()
            )
        })?;
    let required = [
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
    ];
    for name in required {
        if !model.contains_key(name) {
            return Err(format!(
                "Failed to read {}: {read_error}. The manifest snapshot is missing '{name}'; re-run `cheetah init`.",
                model_dir.join("config.json").display()
            ));
        }
    }
    let model_type = model
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut synthesized = model.clone();
    synthesized.insert(
        "architectures".into(),
        serde_json::Value::Array(vec![serde_json::Value::String(model_type.clone())]),
    );
    let default_max_position_embeddings = super::registry::family_for_architecture(&model_type)
        .and_then(|family| family.max_position_embeddings)
        .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS);
    if !synthesized.contains_key("max_position_embeddings") {
        synthesized.insert(
            "max_position_embeddings".into(),
            serde_json::json!(default_max_position_embeddings),
        );
    }
    // Merge the family's fallback parameters and extra config fields, when
    // registered: family values fill any gap the manifest snapshot does not
    // cover (e.g. manifests written before the snapshot was extended).
    if let Some(family) = super::registry::family_for_architecture(&model_type) {
        if !synthesized.contains_key("rope_theta") {
            synthesized.insert("rope_theta".into(), serde_json::json!(family.rope_theta));
        }
        if !synthesized.contains_key("rms_norm_eps") {
            synthesized.insert(
                "rms_norm_eps".into(),
                serde_json::json!(family.rms_norm_eps),
            );
        }
        if !synthesized.contains_key("tie_word_embeddings") {
            synthesized.insert(
                "tie_word_embeddings".into(),
                serde_json::json!(family.tie_word_embeddings),
            );
        }
        if !family.extra_config_json.trim().is_empty() {
            let extra: serde_json::Value =
                serde_json::from_str(family.extra_config_json).map_err(|error| {
                    format!("Invalid family extra_config_json for '{model_type}': {error}")
                })?;
            if let Some(source) = extra.as_object() {
                for (key, value) in source {
                    synthesized.insert(key.clone(), value.clone());
                }
            }
        }
    }
    serde_json::to_vec(&synthesized)
        .map_err(|error| format!("Failed to synthesize config.json from the manifest: {error}"))
}

/// Reads `cheetah_manifest.json` when present (`None` otherwise); parse
/// errors are loud rather than silently ignored.
fn read_model_manifest(model_dir: &Path) -> Result<Option<serde_json::Value>, String> {
    let manifest_path = model_dir.join("cheetah_manifest.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .map_err(|error| format!("Failed to read {}: {error}", manifest_path.display()))?,
    )
    .map_err(|error| format!("Invalid Cheetah manifest: {error}"))?;
    Ok(Some(manifest))
}

/// Extracts the manifest's `stop_markers` list; a missing list or a
/// non-string marker is an error.
fn parse_manifest_stop_markers(manifest: &serde_json::Value) -> Result<Vec<String>, String> {
    let markers = manifest
        .get("stop_markers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "Cheetah manifest is missing its stop_markers list.".to_string())?;
    markers
        .iter()
        .map(|marker| {
            marker
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "Cheetah manifest contains a non-string stop marker.".to_string())
        })
        .collect()
}

/// Weight resolution chain: direct `.safetensors` file → manifest shards
/// (path-traversal checked, sorted, deduplicated) →
/// `model.safetensors.index.json` weight map → single `model.safetensors`.
fn resolve_weight_files(
    path: &Path,
    manifest: Option<&serde_json::Value>,
) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
            return Err(format!(
                "Cheetah inference requires Safetensors weights: {}",
                path.display()
            ));
        }
        return Ok(vec![path.to_path_buf()]);
    }
    if let Some(manifest) = manifest {
        let shards = manifest
            .get("shards")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "Cheetah manifest is missing its shards list.".to_string())?;
        let mut files = HashSet::new();
        for shard in shards {
            let file = shard
                .get("file")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "Cheetah manifest contains a shard without a file path.".to_string()
                })?;
            let relative = Path::new(file);
            if relative.as_os_str().is_empty()
                || !relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
                || relative.extension().and_then(|ext| ext.to_str()) != Some("safetensors")
            {
                return Err(format!(
                    "Cheetah manifest contains an unsafe or unsupported shard path: {file}"
                ));
            }
            let shard_path = path.join(relative);
            if !shard_path.is_file() {
                return Err(format!(
                    "Cheetah manifest shard does not exist: {}",
                    shard_path.display()
                ));
            }
            files.insert(shard_path);
        }
        let mut paths: Vec<_> = files.into_iter().collect();
        paths.sort();
        if paths.is_empty() {
            return Err("Cheetah manifest contains no weight shards.".into());
        }
        return Ok(paths);
    }
    let index_path = path.join("model.safetensors.index.json");
    if index_path.is_file() {
        let index: serde_json::Value = serde_json::from_slice(
            &fs::read(&index_path)
                .map_err(|error| format!("Failed to read {}: {error}", index_path.display()))?,
        )
        .map_err(|error| format!("Invalid Safetensors index: {error}"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| "Safetensors index is missing its weight_map.".to_string())?;
        let files: HashSet<_> = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .collect();
        let mut paths = files
            .into_iter()
            .map(|file| path.join(file))
            .collect::<Vec<_>>();
        paths.sort();
        if paths.is_empty() {
            return Err("Safetensors index contains no weight files.".into());
        }
        return Ok(paths);
    }
    let single = path.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    Err(format!(
        "No model.safetensors or model.safetensors.index.json found in '{}'.",
        path.display()
    ))
}

/// Clock-derived sampling seed (nanoseconds since epoch; 0 pre-epoch) for
/// runs that did not pin `--seed`.
fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default()
}

/// Sampling pipeline over the full vocabulary: repeat penalty on the last 64
/// tokens → greedy argmax when temperature is 0 → temperature scaling →
/// optional top-k → top-p nucleus truncation → weighted sample.
///
/// Note: this top-k is generation sampling, unrelated to the MoE expert
/// routing top-k.
fn sample_token(
    logits: &[f32],
    history: &[u32],
    options: &ChatOptions,
    rng: &mut StdRng,
) -> Result<u32, String> {
    let mut scores = logits.to_vec();
    if options.repeat_penalty > 1.0 {
        for token in history.iter().rev().take(64) {
            if let Some(score) = scores.get_mut(*token as usize) {
                if *score > 0.0 {
                    *score /= options.repeat_penalty;
                } else {
                    *score *= options.repeat_penalty;
                }
            }
        }
    }
    if options.temperature == 0.0 {
        return scores
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| "Model returned an empty vocabulary.".to_string());
    }
    for score in &mut scores {
        *score /= options.temperature as f32;
    }
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    // Generation sampling top-k over the full vocabulary (NOT MoE expert routing).
    if options.top_k > 0 && options.top_k < order.len() {
        // Partial selection: park the k highest-scoring indices at the front
        // in O(V), then sort only those — a full-vocabulary sort dominated
        // per-token sampling cost at Qwen-scale vocabularies.
        let k = options.top_k;
        order.select_nth_unstable_by(k, |left, right| scores[*right].total_cmp(&scores[*left]));
        order[..k].sort_unstable_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        order.truncate(k);
    } else if options.top_p < 1.0 {
        // top-p over the whole vocabulary: only this rare branch needs the
        // full descending order.
        order.sort_unstable_by(|left, right| scores[*right].total_cmp(&scores[*left]));
    }
    // else (top_k == 0 && top_p >= 1.0): every token is a candidate anyway
    // and the weighted draw is order-independent, so no sort at all.
    if scores.is_empty() {
        return Err("Model returned an empty vocabulary.".to_string());
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = order
        .iter()
        .map(|index| (scores[*index] - max).exp())
        .collect::<Vec<_>>();
    let total: f32 = weights.iter().sum();
    let mut candidates = Vec::new();
    let mut cumulative = 0.0;
    for (index, weight) in order.into_iter().zip(weights) {
        if cumulative < options.top_p as f32 || candidates.is_empty() {
            cumulative += weight / total;
            candidates.push((index, weight));
        } else {
            break;
        }
    }
    let distribution = WeightedIndex::new(candidates.iter().map(|(_, weight)| *weight))
        .map_err(|error| format!("Failed to create token sampling distribution: {error}"))?;
    Ok(candidates[distribution.sample(rng)].0 as u32)
}

/// Interactive chat entry: validates that the path is a model directory or
/// safetensors file, loads the engine, prints the banner, and enters the
/// REPL loop.
pub(super) fn chat(
    path: &Path,
    options: ChatOptions,
    mut archive: ContextArchive,
    store: ContextStore,
    ant: Option<&AntSettings>,
    mcp: Option<&crate::mcp::McpRegistry>,
) -> Result<(), String> {
    if !path.is_dir() && path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        return Err(format!(
            "Chat inference requires a model directory or Safetensors file: {}",
            path.display()
        ));
    }
    let engine = load_engine_with_progress(path, options.threads, options.ctx_size, ant)?;
    // Warm up so the first turn does not pay the mmap cold start.
    engine.warmup()?;
    println!(
        "{}",
        color::stdout(
            &format!(
                "Loaded {} model ({}). Enter /exit to quit.",
                engine.model_id(),
                engine.device_name()
            ),
            Color::Green
        )
    );
    chat_loop(&engine, options, &mut archive, store, mcp)
}

struct StdoutSink;

impl TextSink for StdoutSink {
    /// Prints one streamed chunk and flushes stdout; write failures surface as generation errors.
    fn push(&mut self, text: &str) -> Result<(), String> {
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|error| format!("Failed to write response: {error}"))
    }

    /// Ends the turn with a newline.
    fn finish(&mut self) -> Result<(), String> {
        println!();
        Ok(())
    }
}

/// Formats the post-turn stats line (tokens + tok/s, plus the prefill window,
/// or "speed unavailable" when the decode window could not be measured).
fn format_generation_stats(generation: &Generation) -> String {
    let speed = match generation.tokens_per_second() {
        Some(tokens_per_second) => format!("{tokens_per_second:.2} tokens/s"),
        None => "speed unavailable".to_string(),
    };
    format!(
        "Generated {} tokens ({speed}). Prefill {} tokens in {} ms.",
        generation.generated_tokens,
        generation.prompt_tokens,
        generation.prefill_duration.as_millis()
    )
}

/// The interactive REPL: slash commands (`/exit`, `/help`, `/clear`,
/// `/save <name>`), history replay from the archive, and per-turn
/// prepare → generate → stats with archive bookkeeping.
fn chat_loop(
    engine: &Engine,
    options: ChatOptions,
    archive: &mut ContextArchive,
    store: ContextStore,
    mcp: Option<&crate::mcp::McpRegistry>,
) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut messages = Vec::new();
    for turn in &archive.turns {
        messages.push(ChatMessage::new("user", turn.user.clone()));
        messages.push(ChatMessage::new("assistant", turn.assistant.clone()));
    }
    loop {
        print!("{}", color::stdout("> ", Color::Cyan));
        std::io::stdout()
            .flush()
            .map_err(|error| format!("Failed to show chat prompt: {error}"))?;
        let mut line = String::new();
        if input
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read input: {error}"))?
            == 0
        {
            println!();
            return Ok(());
        }
        let text = line.trim();
        match text {
            "/exit" | "/quit" => return Ok(()),
            "/help" => {
                println!(
                    "{}",
                    color::stdout(
                        "/exit, /quit  Exit the chat\n/clear        Clear conversation\n/save <name>  Save conversation",
                        Color::Cyan
                    )
                );
                continue;
            }
            "/clear" => {
                messages.clear();
                archive.turns.clear();
                archive.updated_at = random_seed();
                println!("{}", color::stdout("Conversation cleared.", Color::Green));
                continue;
            }
            value if value.starts_with("/save ") => {
                let name = value.trim_start_matches("/save ").trim();
                if name.is_empty() {
                    println!("{}", color::stdout("Usage: /save <name>", Color::Yellow));
                    continue;
                }
                archive.name = name.to_owned();
                archive.turns = messages
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| ConversationTurn {
                        user: pair[0].content.clone(),
                        assistant: pair[1].content.clone(),
                    })
                    .collect();
                store.save(archive)?;
                println!(
                    "{}",
                    color::stdout(&format!("Saved context '{name}'."), Color::Green)
                );
                continue;
            }
            _ if text.is_empty() => continue,
            _ => {}
        }
        let user = text.to_owned();
        messages.push(ChatMessage::new("user", user.clone()));
        let mut sink = StdoutSink;
        let generation = match mcp {
            // MCP configured: the tools flow runs the agentic loop (tool
            // calls execute server-side and the model answers with their
            // results); the history keeps only the final text.
            Some(mcp) => {
                let tools = mcp.tool_definitions();
                let tools_json = serde_json::to_string(&tools)
                    .map_err(|error| format!("Failed to serialize MCP tools: {error}"))?;
                let history_json = serde_json::to_string(
                    &messages
                        .iter()
                        .map(|message| {
                            serde_json::json!({"role": message.role, "content": message.content})
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| format!("Failed to serialize chat history: {error}"))?;
                engine
                    .generate_tool_chat(
                        &history_json,
                        &tools_json,
                        &tools,
                        ToolChoice::Auto,
                        &toolcalls::ResolvedToolChoice::Auto,
                        options.max_tokens,
                        &options,
                        &[],
                        &mut sink,
                        &HashSet::new(),
                        Some(mcp),
                    )
                    .map(|(generation, _)| generation)
            }
            None => {
                let (tokens, _) =
                    engine.prepare(&mut messages, options.system.as_deref(), options.max_tokens)?;
                let mut context = engine.new_context()?;
                engine
                    .generate(
                        &mut context,
                        &tokens,
                        &options,
                        &[],
                        &mut sink,
                        &HashSet::new(),
                    )
                    .map(|generation| generation)
            }
        };
        match generation {
            Ok(generation) => {
                println!(
                    "{}",
                    color::stdout(&format_generation_stats(&generation), Color::Green)
                );
                let assistant = generation.text;
                messages.push(ChatMessage::new("assistant", assistant.clone()));
                archive.turns.push(ConversationTurn { user, assistant });
            }
            Err(error) => {
                messages.pop();
                return Err(error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("cheetah_{name}_{}_{}", std::process::id(), nonce));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn resolves_safetensors_from_cheetah_manifest() {
        let directory = temporary_directory("manifest_weights");
        let weights = directory.join(".cheetah/weights");
        fs::create_dir_all(&weights).unwrap();
        fs::write(weights.join("layer_0.safetensors"), []).unwrap();
        fs::write(weights.join("layer_1.safetensors"), []).unwrap();
        fs::write(directory.join("model.safetensors"), []).unwrap();
        fs::write(
            directory.join("cheetah_manifest.json"),
            r#"{"stop_markers":["<|eot_id|>"],"shards":[{"file":".cheetah/weights/layer_1.safetensors"},{"file":".cheetah/weights/layer_0.safetensors"}]}"#,
        )
        .unwrap();

        let (files, stop_markers) = resolve_model_files(&directory).unwrap();
        assert_eq!(
            files,
            vec![
                weights.join("layer_0.safetensors"),
                weights.join("layer_1.safetensors")
            ]
        );
        assert_eq!(stop_markers, Some(vec!["<|eot_id|>".to_string()]));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_unsafe_manifest_shard_paths() {
        let directory = temporary_directory("unsafe_manifest");
        fs::write(
            directory.join("cheetah_manifest.json"),
            r#"{"stop_markers":[],"shards":[{"file":"../outside.safetensors"}]}"#,
        )
        .unwrap();

        let error = resolve_model_files(&directory).unwrap_err();
        assert!(error.contains("unsafe or unsupported shard path"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resolves_direct_weight_file_and_parent_manifest_metadata() {
        let directory = temporary_directory("direct_manifest_metadata");
        let weights = directory.join("model.safetensors");
        fs::write(&weights, []).unwrap();
        fs::write(
            directory.join("cheetah_manifest.json"),
            r#"{"stop_markers":["<|stop|>"],"shards":[]}"#,
        )
        .unwrap();

        let (files, stop_markers) = resolve_model_files(&weights).unwrap();
        assert_eq!(files, vec![weights]);
        assert_eq!(stop_markers, Some(vec!["<|stop|>".to_string()]));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn renders_qwen_chatml_prompt_with_system_message() {
        let template = super::super::registry::chat_template_for_architecture("qwen2")
            .expect("qwen2 chat template");
        let prompt = (template.render)(&[ChatMessage::new("user", "hello")], Some("Be brief."));
        assert!(prompt.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn renders_llama_instruction_prompt() {
        let template = super::super::registry::chat_template_for_architecture("llama")
            .expect("llama chat template");
        let prompt = (template.render)(&[ChatMessage::new("user", "hello")], Some("Be brief."));
        assert!(prompt.contains("<<SYS>>\nBe brief.\n<</SYS>>"));
        assert!(prompt.ends_with("hello [/INST]"));
    }

    #[test]
    fn renders_mistral_instruction_prompt() {
        let template = super::super::registry::chat_template_for_architecture("MistralForCausalLM")
            .expect("mistral chat template");
        let prompt = (template.render)(
            &[
                ChatMessage::new("user", "hello"),
                ChatMessage::new("assistant", "hi"),
            ],
            Some("Be brief."),
        );
        assert!(prompt.starts_with("<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nhello [/INST]"));
        assert!(prompt.ends_with(" hi </s>"));
    }

    #[test]
    fn renders_llama3_header_prompt() {
        let template = super::super::registry::chat_template_for_architecture("llama")
            .expect("llama chat template");
        let prompt = (template.header_renderer.unwrap())(
            &[ChatMessage::new("user", "hello")],
            Some("Be brief."),
        );
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n"));
    }

    #[test]
    fn ant_resident_weight_paths_keep_only_resident_units() {
        let units = vec![
            AntUnit {
                unit_id: "model.embed_tokens.weight".to_string(),
                name: "embed_tokens".to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 4096],
                bytes: 1024,
                source_file: "model-00001-of-00002.safetensors".to_string(),
                source_offset: 0,
                shard: "model-00001-of-00002.safetensors".to_string(),
                layer_id: 0,
            },
            AntUnit {
                unit_id: "model.layers.0.weight".to_string(),
                name: "layers_0".to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 4096],
                bytes: 2048,
                source_file: "model-00001-of-00002.safetensors".to_string(),
                source_offset: 0,
                shard: "model-00001-of-00002.safetensors".to_string(),
                layer_id: 0,
            },
            AntUnit {
                unit_id: "model.layers.1.weight".to_string(),
                name: "layers_1".to_string(),
                dtype: "F32".to_string(),
                shape: vec![1, 4096],
                bytes: 4096,
                source_file: "model-00002-of-00002.safetensors".to_string(),
                source_offset: 0,
                shard: "model-00002-of-00002.safetensors".to_string(),
                layer_id: 1,
            },
        ];
        let mut plan = std::collections::BTreeMap::new();
        plan.insert("model.embed_tokens.weight".to_string(), "ram".to_string());
        plan.insert("model.layers.0.weight".to_string(), "vram".to_string());
        plan.insert("model.layers.1.weight".to_string(), "disk".to_string());
        let evicted = vec!["model.layers.0.weight".to_string()];

        let paths = ant_resident_weight_paths(
            &[
                PathBuf::from("/tmp/model-00001-of-00002.safetensors"),
                PathBuf::from("/tmp/model-00002-of-00002.safetensors"),
            ],
            &units,
            &plan,
            &evicted,
        );

        // The RAM-resident embedding and the evicted VRAM layer share the first
        // shard; the disk-tier layer is dropped entirely.
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0].file_name().and_then(|name| name.to_str()),
            Some("model-00001-of-00002.safetensors")
        );
    }

    #[test]
    fn ant_resident_weight_paths_keep_everything_when_ant_disabled() {
        let units = vec![AntUnit {
            unit_id: "model.layers.0.weight".to_string(),
            name: "layers_0".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1, 4096],
            bytes: 2048,
            source_file: "model-00001-of-00002.safetensors".to_string(),
            source_offset: 0,
            shard: "model-00001-of-00002.safetensors".to_string(),
            layer_id: 0,
        }];
        let plan = std::collections::BTreeMap::from([(
            "model.layers.0.weight".to_string(),
            "vram".to_string(),
        )]);
        let evicted: Vec<String> = Vec::new();

        let paths = ant_resident_weight_paths(
            &[PathBuf::from("/tmp/model-00001-of-00002.safetensors")],
            &units,
            &plan,
            &evicted,
        );

        // Resident-tier units are kept unless they were evicted.
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn sampling_is_deterministic_for_fixed_seed() {
        let options = ChatOptions {
            temperature: 0.7,
            top_p: 0.9,
            ..ChatOptions::default()
        };
        let mut first = StdRng::seed_from_u64(42);
        let mut second = StdRng::seed_from_u64(42);
        let logits = [0.1, 0.7, 0.3];
        assert_eq!(
            sample_token(&logits, &[], &options, &mut first).unwrap(),
            sample_token(&logits, &[], &options, &mut second).unwrap()
        );
    }

    #[test]
    fn generation_stats_report_decode_speed_and_handle_unavailable_speed() {
        let generation = Generation {
            text: String::new(),
            prompt_tokens: 0,
            generated_tokens: 6,
            prefill_duration: Duration::from_millis(1500),
            decode_duration: Duration::from_secs(2),
            finish_reason: FinishReason::Stop,
        };
        assert_eq!(generation.tokens_per_second(), Some(3.0));
        assert_eq!(
            format_generation_stats(&generation),
            "Generated 6 tokens (3.00 tokens/s). Prefill 0 tokens in 1500 ms."
        );

        let no_tokens = Generation {
            generated_tokens: 0,
            ..generation.clone()
        };
        assert_eq!(no_tokens.tokens_per_second(), None);
        assert_eq!(
            format_generation_stats(&no_tokens),
            "Generated 0 tokens (speed unavailable). Prefill 0 tokens in 1500 ms."
        );

        let zero_duration = Generation {
            decode_duration: Duration::ZERO,
            ..generation
        };
        assert_eq!(zero_duration.tokens_per_second(), None);
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::*;

    /// Builds a synthetic snapshot covering `positions` positions on
    /// `layers` layers, one f32 per (position, dim) with dim = 4.
    fn snapshot(positions: usize, layers: usize) -> KvSnapshot {
        KvSnapshot {
            positions,
            kv_k: (0..layers)
                .map(|layer| {
                    (0..positions * 4)
                        .map(|i| (layer * 1000 + i) as f32)
                        .collect()
                })
                .collect(),
            kv_v: (0..layers)
                .map(|layer| {
                    (0..positions * 4)
                        .map(|i| -((layer * 1000 + i) as f32))
                        .collect()
                })
                .collect(),
        }
    }

    fn entry_cache(tokens: &[u32], positions: usize) -> PrefixCache {
        let mut cache = PrefixCache {
            entries: VecDeque::new(),
        };
        cache.insert(tokens.to_vec(), snapshot(positions, 2));
        cache
    }

    /// Longest-prefix selection: an entry that strictly prefixes the prompt
    /// matches at its full length; a longer entry clamps to prompt.len()-1 so
    /// at least one token stays to forward.
    #[test]
    fn longest_prefix_matches_entry_prefix_and_clamps_full_hits() {
        // Strict prefix entry (16 tokens) against a 20-token prompt.
        let tokens: Vec<u32> = (1..=16).collect();
        let mut cache = entry_cache(&tokens, 16);
        let prompt: Vec<u32> = (1..=20).collect();
        let (shared, snap) = cache
            .longest_prefix(&prompt)
            .expect("extended history must hit");
        assert_eq!(shared, 16);
        assert_eq!(snap.positions, 16);

        // Identical repeat (full hit): clamped so the last token re-forwards.
        let (shared, _) = cache
            .longest_prefix(&tokens)
            .expect("identical prompt must hit");
        assert_eq!(shared, 15, "full hits keep one token to forward");

        // Disjoint history must not match.
        assert!(
            cache
                .longest_prefix(&(100..104).collect::<Vec<u32>>())
                .is_none()
        );
    }

    /// Short shared prefixes and below-threshold prompts are rejected: reuse
    /// must not churn the cache for nearly-disjoint conversations.
    #[test]
    fn reuse_requires_meaningful_prefix() {
        let tokens: Vec<u32> = (1..=16).collect();
        let mut cache = entry_cache(&tokens, 16);
        // Only 2 shared positions (< PREFIX_MIN_REUSE_POSITIONS).
        assert!(cache.longest_prefix(&[1, 2, 9, 10]).is_none());
        // Short prompt (entry prefix match, but too few shared positions).
        assert!(cache.longest_prefix(&[1, 2, 3]).is_none());
    }

    /// LRU + budgets: the oldest entry is evicted beyond the entry cap, and
    /// a single oversized snapshot is still retained (budget evicts only
    /// while more than one entry exists).
    #[test]
    fn insert_evicts_lru_within_entry_budget() {
        let mut cache = PrefixCache {
            entries: VecDeque::new(),
        };
        let rounds = (PREFIX_CACHE_MAX_ENTRIES + 2) as u32;
        for round in 0..rounds {
            let tokens: Vec<u32> = (0..PREFIX_CACHE_MIN_TOKENS as u32)
                .map(|i| i + round * 100)
                .collect();
            cache.insert(tokens, snapshot(4, 2));
        }
        assert_eq!(
            cache.entries.len(),
            PREFIX_CACHE_MAX_ENTRIES,
            "entry count must be capped"
        );
        // Only the newest PREFIX_CACHE_MAX_ENTRIES rounds survive; the front
        // holds the oldest of those.
        assert_eq!(
            cache.entries.front().unwrap().tokens[0],
            (rounds - PREFIX_CACHE_MAX_ENTRIES as u32) * 100,
            "oldest entries must be evicted first"
        );
    }
}

#[cfg(test)]
mod tool_stream_tests {
    use super::*;

    /// Recording sink: captures what the wrapper streamed.
    #[derive(Default)]
    struct RecordingSink {
        chunks: Vec<String>,
        captured: String,
    }

    impl TextSink for RecordingSink {
        fn push(&mut self, text: &str) -> Result<(), String> {
            self.chunks.push(text.to_owned());
            self.captured.push_str(text);
            Ok(())
        }

        fn capture_text(&self) -> bool {
            true
        }
    }

    /// Plain text streams through immediately (no hold markers).
    #[test]
    fn plain_text_streams_through() -> Result<(), String> {
        let mut inner = RecordingSink::default();
        {
            let mut sink = ToolStreamSink::new(&mut inner);
            sink.push("The weather is sunny today.")?;
        }
        assert_eq!(inner.chunks, vec!["The weather is sunny today."]);
        Ok::<(), String>(())
    }

    /// Once a `{` appears, everything from it is held; the round decision
    /// either flushes (prose) or drops it (tool call).
    #[test]
    fn brace_holds_and_round_decision_decides() -> Result<(), String> {
        let mut inner = RecordingSink::default();
        {
            let mut sink = ToolStreamSink::new(&mut inner);
            sink.push("Let me check: {\"tool_calls\":[")?;
            sink.drop_round();
        }
        assert_eq!(inner.chunks, vec!["Let me check: "]);
        Ok::<(), String>(())
    }

    /// A flush at round end releases the held prose (brace-bearing text that
    /// turned out to be a plain answer).
    #[test]
    fn flush_round_releases_held_prose() -> Result<(), String> {
        let mut inner = RecordingSink::default();
        {
            let mut sink = ToolStreamSink::new(&mut inner);
            sink.push("JSON looks like {\"a\": 1} in Rust syntax.")?;
            sink.flush_round()?;
        }
        assert_eq!(
            inner.chunks,
            vec!["JSON looks like ", "{\"a\": 1} in Rust syntax."]
        );
        assert_eq!(inner.captured, "JSON looks like {\"a\": 1} in Rust syntax.");
        Ok::<(), String>(())
    }

    /// Code fences hold exactly like braces (the fenced tool-call shape).
    #[test]
    fn fence_holds_like_brace() -> Result<(), String> {
        let mut inner = RecordingSink::default();
        {
            let mut sink = ToolStreamSink::new(&mut inner);
            sink.push("```json")?;
        }
        assert!(inner.chunks.is_empty());
        Ok::<(), String>(())
    }

    /// Deltas across pushes stay stitched: the hold marker may span pushes.
    #[test]
    fn hold_marker_spanning_pushes_holds() -> Result<(), String> {
        let mut inner = RecordingSink::default();
        {
            let mut sink = ToolStreamSink::new(&mut inner);
            sink.push("text with no marker yet")?;
            sink.push(" then {\"tool")?;
            sink.flush_round()?;
        }
        assert_eq!(
            inner.chunks,
            vec!["text with no marker yet", " then ", "{\"tool"]
        );
        assert_eq!(inner.captured, "text with no marker yet then {\"tool");
        Ok::<(), String>(())
    }
}
