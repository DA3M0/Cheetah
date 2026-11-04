use super::models::ModelRuntime;
use crate::context::{ContextArchive, ContextStore, ConversationTurn};
use crate::options::ChatOptions;
use candle_core::{Device, Tensor};
use candle_transformers::models::llama;
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub trait TextSink {
    fn push(&mut self, text: &str) -> Result<(), String>;

    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn capture_text(&self) -> bool {
        true
    }

    fn interrupted(&mut self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub decode_duration: Duration,
    pub finish_reason: FinishReason,
}

impl Generation {
    pub fn tokens_per_second(&self) -> Option<f64> {
        let seconds = self.decode_duration.as_secs_f64();
        if self.generated_tokens == 0 || seconds == 0.0 {
            return None;
        }
        Some(self.generated_tokens as f64 / seconds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Default)]
pub struct ChatParams {
    pub additional_stops_json: String,
}

impl ChatParams {
    pub fn parse(&self, text: &str, _partial: bool) -> Result<String, String> {
        Ok(text.to_owned())
    }
}

struct LoadedModel {
    model: Box<dyn ModelRuntime>,
    device: Device,
}

pub struct InferenceContext;

pub struct Engine {
    model: Mutex<LoadedModel>,
    tokenizer: Tokenizer,
    model_id: String,
    prompt_renderer: super::models::PromptRenderer,
    max_position_embeddings: usize,
    eos_token_ids: HashSet<u32>,
    stop_markers: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default = "default_context_size")]
    max_position_embeddings: usize,
}

fn default_context_size() -> usize {
    4096
}

impl Engine {
    pub fn load(path: &Path, threads: Option<usize>, ctx_size: usize) -> Result<Self, String> {
        if ctx_size == 0 {
            return Err("--ctx-size must be greater than 0.".into());
        }
        let threads = threads.unwrap_or_else(|| num_cpus::get_physical().max(1));
        if threads == 0 {
            return Err("--threads must be greater than 0.".into());
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("Failed to initialize CPU thread pool: {error}"))?;
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let config_path = model_dir.join("config.json");
        let config_bytes = fs::read(&config_path)
            .map_err(|error| format!("Failed to read {}: {error}", config_path.display()))?;
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        let config: ModelConfig = serde_json::from_value(raw_config.clone())
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        let architecture = config
            .architectures
            .as_ref()
            .and_then(|architectures| architectures.first())
            .map(String::as_str)
            .unwrap_or(config.model_type.as_str());
        let registration = super::models::registration_for_architecture(architecture)?;

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
        let (weight_paths, manifest_stop_markers) = resolve_model_files(path)?;
        let vocabulary = tokenizer.get_vocab(true);
        let use_header_renderer = registration.chat_template.header_renderer.is_some()
            && (manifest_stop_markers
                .as_ref()
                .is_some_and(|markers| markers.iter().any(|marker| marker == "<|eot_id|>"))
                || manifest_stop_markers.is_none()
                    && vocabulary.contains_key("<|start_header_id|>")
                    && vocabulary.contains_key("<|eot_id|>"));
        let prompt_renderer = if use_header_renderer {
            registration
                .chat_template
                .header_renderer
                .unwrap_or(registration.chat_template.render)
        } else {
            registration.chat_template.render
        };
        let (device, model) =
            super::models::load_model(&weight_paths, &config_bytes, raw_config, architecture)?;
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
            registration
                .stop_markers
                .iter()
                .map(|marker| (*marker).to_owned())
                .collect()
        });
        for marker in &stop_markers {
            if let Some(id) = vocabulary.get(marker) {
                eos_token_ids.insert(*id);
            }
        }
        Ok(Self {
            model: Mutex::new(LoadedModel { model, device }),
            tokenizer,
            model_id: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model")
                .to_owned(),
            prompt_renderer,
            max_position_embeddings: config.max_position_embeddings.min(ctx_size).min(
                if config.model_type == "llama" {
                    llama::MAX_SEQ_LEN
                } else {
                    usize::MAX
                },
            ),
            eos_token_ids,
            stop_markers,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn device_name(&self) -> &'static str {
        let loaded = self
            .model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loaded.model.device_name()
    }

    pub fn new_context(&self) -> Result<InferenceContext, String> {
        Ok(InferenceContext)
    }

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
            eprintln!("Context limit nearly reached; earlier conversation turns were truncated.");
        }
    }

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

    pub fn generate(
        &self,
        _context: &mut InferenceContext,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
    ) -> Result<Generation, String> {
        self.generate_inner(prompt, options, extra_stop, sink)
    }

    pub fn generate_with_chat_params(
        &self,
        context: &mut InferenceContext,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
        params: &ChatParams,
    ) -> Result<Generation, String> {
        let mut stops = extra_stop.to_vec();
        let template_stops: Vec<String> = serde_json::from_str(&params.additional_stops_json)
            .map_err(|error| format!("Failed to parse template stop markers: {error}"))?;
        stops.extend(template_stops);
        self.generate(context, prompt, options, &stops, sink)
    }

    fn generate_inner(
        &self,
        prompt: &[u32],
        options: &ChatOptions,
        extra_stop: &[String],
        sink: &mut dyn TextSink,
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

        let loaded = self
            .model
            .lock()
            .map_err(|_| "Model mutex was poisoned.".to_string())?;
        let mut model = loaded.model.clone();
        let device = loaded.device.clone();
        drop(loaded);
        model.clear_cache();
        let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &device)
            .map_err(|error| format!("Failed to prepare model input: {error}"))?;
        let started = Instant::now();
        let mut logits = model
            .forward(&input, 0)
            .map_err(|error| format!("Model inference failed: {error}"))?;
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
            let logits_values = logits
                .reshape((logits.elem_count(),))
                .and_then(|logits| logits.to_vec1::<f32>())
                .map_err(|error| format!("Failed to read model logits: {error}"))?;
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
            let input = Tensor::from_vec(vec![token], (1, 1), &device)
                .map_err(|error| format!("Failed to prepare generated token: {error}"))?;
            logits = model
                .forward(&input, history.len() - 1)
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
            decode_duration,
            finish_reason,
        })
    }
}

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

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default()
}

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
    order.sort_unstable_by(|left, right| scores[*right].total_cmp(&scores[*left]));
    if options.top_k > 0 && options.top_k < order.len() {
        order.truncate(options.top_k);
    }
    let max = order
        .first()
        .map(|index| scores[*index])
        .ok_or_else(|| "Model returned an empty vocabulary.".to_string())?;
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

pub(super) fn chat(
    path: &Path,
    options: ChatOptions,
    mut archive: ContextArchive,
    store: ContextStore,
) -> Result<(), String> {
    if !path.is_dir() && path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        return Err(format!(
            "Chat inference requires a model directory or Safetensors file: {}",
            path.display()
        ));
    }
    eprintln!("Loading model: {}", path.display());
    let engine = Engine::load(path, options.threads, options.ctx_size)?;
    println!(
        "Loaded {} model ({}). Enter /exit to quit.",
        engine.model_id(),
        engine.device_name()
    );
    chat_loop(&engine, options, &mut archive, store)
}

struct StdoutSink;

impl TextSink for StdoutSink {
    fn push(&mut self, text: &str) -> Result<(), String> {
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout())
            .map_err(|error| format!("Failed to write response: {error}"))
    }

    fn finish(&mut self) -> Result<(), String> {
        println!();
        Ok(())
    }
}

fn format_generation_stats(generation: &Generation) -> String {
    match generation.tokens_per_second() {
        Some(tokens_per_second) => format!(
            "Generated {} tokens ({tokens_per_second:.2} tokens/s).",
            generation.generated_tokens
        ),
        None => format!(
            "Generated {} tokens (speed unavailable).",
            generation.generated_tokens
        ),
    }
}

fn chat_loop(
    engine: &Engine,
    options: ChatOptions,
    archive: &mut ContextArchive,
    store: ContextStore,
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
        print!("> ");
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
                    "/exit, /quit  Exit the chat\n/clear        Clear conversation\n/save <name>  Save conversation"
                );
                continue;
            }
            "/clear" => {
                messages.clear();
                archive.turns.clear();
                archive.updated_at = random_seed();
                println!("Conversation cleared.");
                continue;
            }
            value if value.starts_with("/save ") => {
                let name = value.trim_start_matches("/save ").trim();
                if name.is_empty() {
                    println!("Usage: /save <name>");
                    continue;
                }
                archive.name = name.to_owned();
                archive.turns = messages
                    .chunks_exact(2)
                    .map(|pair| ConversationTurn {
                        user: pair[0].content.clone(),
                        assistant: pair[1].content.clone(),
                    })
                    .collect();
                store.save(archive)?;
                println!("Saved context '{name}'.");
                continue;
            }
            _ if text.is_empty() => continue,
            _ => {}
        }
        let user = text.to_owned();
        messages.push(ChatMessage::new("user", user.clone()));
        let (tokens, _) =
            engine.prepare(&mut messages, options.system.as_deref(), options.max_tokens)?;
        let mut context = engine.new_context()?;
        let mut sink = StdoutSink;
        match engine.generate(&mut context, &tokens, &options, &[], &mut sink) {
            Ok(generation) => {
                println!("{}", format_generation_stats(&generation));
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
        let template = super::super::models::registration_for_architecture("qwen2")
            .unwrap()
            .chat_template;
        let prompt = (template.render)(&[ChatMessage::new("user", "hello")], Some("Be brief."));
        assert!(prompt.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn renders_llama_instruction_prompt() {
        let template = super::super::models::registration_for_architecture("llama")
            .unwrap()
            .chat_template;
        let prompt = (template.render)(&[ChatMessage::new("user", "hello")], Some("Be brief."));
        assert!(prompt.contains("<<SYS>>\nBe brief.\n<</SYS>>"));
        assert!(prompt.ends_with("hello [/INST]"));
    }

    #[test]
    fn renders_mistral_instruction_prompt() {
        let template = super::super::models::registration_for_architecture("MistralForCausalLM")
            .unwrap()
            .chat_template;
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
        let template = super::super::models::registration_for_architecture("llama")
            .unwrap()
            .chat_template;
        let prompt = (template.header_renderer.unwrap())(
            &[ChatMessage::new("user", "hello")],
            Some("Be brief."),
        );
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n"));
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
            decode_duration: Duration::from_secs(2),
            finish_reason: FinishReason::Stop,
        };
        assert_eq!(generation.tokens_per_second(), Some(3.0));
        assert_eq!(
            format_generation_stats(&generation),
            "Generated 6 tokens (3.00 tokens/s)."
        );

        let no_tokens = Generation {
            generated_tokens: 0,
            ..generation.clone()
        };
        assert_eq!(no_tokens.tokens_per_second(), None);
        assert_eq!(
            format_generation_stats(&no_tokens),
            "Generated 0 tokens (speed unavailable)."
        );

        let zero_duration = Generation {
            decode_duration: Duration::ZERO,
            ..generation
        };
        assert_eq!(zero_duration.tokens_per_second(), None);
    }
}
