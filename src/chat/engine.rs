use super::models::ModelRuntime;
use crate::ant::{ant_swap_fit, Ant, AntLayerResidency, AntSettings, AntUnit};
use crate::color::{self, Color};
use crate::context::{ContextArchive, ContextStore, ConversationTurn};
use crate::options::ChatOptions;
use candle_core::Device;
use candle_transformers::models::llama;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::io::IsTerminal;
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

fn log_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    format!("[{:.3}]", now)
}

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

pub(crate) fn load_engine_with_progress(
    path: &Path,
    threads: Option<usize>,
    ctx_size: usize,
    ant: Option<&AntSettings>,
) -> Result<Engine, String> {
    println!(
        "{} {}",
        log_timestamp(),
        color::stdout(&format!("loading model: {}", path.display()), Color::Cyan,)
    );

    let spinner = create_spinner("loading model");
    let result = Engine::load(path, threads, ctx_size, ant);
    match result {
        Ok(engine) => {
            spinner.finish_with_message(format!("loaded: {}", engine.model_id()));
            print_model_info(&engine);
            Ok(engine)
        }
        Err(error) => {
            spinner.abandon_with_message("model load failed");
            Err(error)
        }
    }
}

fn print_model_info(engine: &Engine) {
    println!();
    println!(
        "{} {}",
        log_timestamp(),
        color::stdout(&format!("model loaded: {}", engine.model_id()), Color::Cyan,),
    );
    if let Some(architecture) = engine.model_architecture() {
        println!("  architecture: {architecture}");
        println!(
            "  backends:     {}",
            super::models::registered_backend_names(architecture).join(", ")
        );
    }
    println!("  device:       {}", engine.device_name());
    if !engine.ant_plan.is_empty() {
        println!("  ant plan:     {} resident unit(s)", engine.ant_plan.len());
        println!(
            "  ant resident: {} bytes within budget {}",
            engine.ant_resident_bytes, engine.ant_budget_bytes
        );
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
        if let Ok(state) = engine.ant_runtime.try_lock() {
            println!(
                "  ant swap:     {} queued for VRAM promotion, {} evicted at load",
                state.promotion_queue.len(),
                state.evicted.len()
            );
        }
        if !engine.ant_within_budget() {
            println!(
                "  ant warning:  resident bytes {} exceed vram budget {}",
                engine.ant_resident_bytes, engine.ant_budget_bytes
            );
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
    architecture: Option<String>,
    prompt_renderer: super::models::PromptRenderer,
    max_position_embeddings: usize,
    eos_token_ids: HashSet<u32>,
    stop_markers: Vec<String>,
    ant_plan: std::collections::BTreeMap<String, String>,
    ant_resident_bytes: u64,
    ant_budget_bytes: u64,
    ant_settings: Option<AntSettings>,
    ant_units: Vec<AntUnit>,
    ant_model_dir: Option<PathBuf>,
    ant_device_accelerator: bool,
    ant_layer_summary: Vec<AntLayerResidency>,
    ant_unit_index: std::collections::BTreeMap<String, AntUnitRuntime>,
    ant_runtime: Mutex<AntRuntimeState>,
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

fn ant_tier_priority(tier: &str) -> u64 {
    match tier {
        "vram" => 0,
        "ram" => 1,
        _ => 2,
    }
}

fn ant_resident_weight_paths(
    weight_paths: &[PathBuf],
    units: &[AntUnit],
    plan: &std::collections::BTreeMap<String, String>,
    evicted: &[String],
) -> Vec<PathBuf> {
    let mut keep: std::collections::HashSet<String> = std::collections::HashSet::new();
    for unit in units {
        let tier = plan
            .get(&unit.unit_id)
            .and_then(|tier| Some(tier.as_str()))
            .unwrap_or("vram");
        if tier == "disk" || evicted.iter().any(|e| e == &unit.unit_id) {
            continue;
        }
        keep.insert(unit.source_file.clone());
    }
    let mut paths = weight_paths.to_vec();
    paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| keep.contains(name))
    });
    paths
}

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
    let mut resident: Vec<(u64, u64, String)> = Vec::new();
    for unit in units {
        let tier = plan
            .get(&unit.unit_id)
            .and_then(|tier| Some(tier.as_str()))
            .unwrap_or("vram");
        if tier != "disk" {
            resident_bytes = resident_bytes.saturating_add(unit.byte_size());
            resident.push((ant_tier_priority(tier), unit.byte_size(), unit.unit_id.clone()));
        }
    }
    if device_is_accelerator && budget_bytes > 0 && resident_bytes > budget_bytes {
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
#[derive(Debug, Clone)]
struct AntUnitRuntime {
    bytes: u64,
    layer_id: i32,
    tier: String,
    pinned: bool,
}

/// Phase 3: cross-generation Ant bookkeeping. Tracks VRAM/RAM occupancy, the
/// units evicted at runtime, and the in-memory (RAM-tier) units that are next
/// in line for promotion into VRAM, so every swap re-checks the budget before
/// it executes and only swaps in as memory becomes available.
#[derive(Default)]
struct AntRuntimeState {
    vram_used_bytes: u64,
    ram_used_bytes: u64,
    evicted: HashSet<String>,
    /// Disk-tier units paged in at runtime; they are resident although the
    /// plan places them on disk.
    runtime_resident: HashSet<String>,
    /// RAM-tier resident units, in layer order: the in-memory portion next in
    /// line for promotion / swapping into VRAM.
    promotion_queue: Vec<String>,
    swaps_in: u64,
    swaps_out: u64,
    swap_skips: u64,
}

impl Engine {
    pub fn load(
        path: &Path,
        threads: Option<usize>,
        ctx_size: usize,
        ant: Option<&AntSettings>,
    ) -> Result<Self, String> {
        if ctx_size == 0 {
            return Err("--ctx-size must be greater than 0.".into());
        }
        let threads = threads.unwrap_or_else(|| num_cpus::get_physical().max(1));
        if threads == 0 {
            return Err("--threads must be greater than 0.".into());
        }
        // Reuse the existing global pool when one is already initialized
        // (e.g. in-process callers such as the test harness).
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
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
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        let config: ModelConfig = serde_json::from_value(raw_config.clone())
            .map_err(|error| format!("Invalid model configuration: {error}"))?;
        let architecture = config
            .architectures
            .as_ref()
            .and_then(|architectures| architectures.first())
            .map(String::as_str)
            .unwrap_or(config.model_type.as_str())
            .to_owned();
        let registration = super::models::registration_for_architecture(&architecture)?;

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
        let device_is_accelerator = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into()) != "cpu";
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
                    if let Ok(units) = manager.units_for_model(&model_dir) {
                        let plan = manager.plan_residency(&units);
                        for unit in &units {
                            let tier = plan
                                .get(&unit.unit_id)
                                .and_then(|tier| Some(tier.as_str()))
                                .unwrap_or("vram");
                            ant_plan.insert(unit.unit_id.clone(), tier.to_string());
                        }
                        let (resident_bytes, evicted_units) = ant_apply_budget(
                            &manager,
                            &model_dir,
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
                                    ant_runtime.ram_used_bytes = ant_runtime
                                        .ram_used_bytes
                                        .saturating_add(unit.byte_size());
                                    promotion_queue.push((unit.layer_id, unit.unit_id.clone()));
                                }
                                _ => {}
                            }
                        }
                        promotion_queue.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                        ant_runtime.promotion_queue =
                            promotion_queue.into_iter().map(|(_, unit_id)| unit_id).collect();
                        ant_units = units;
                        ant_evicted = evicted_units;
                    }
                    ant_settings = Some(settings.clone());
                }
            }
        }
        if let Some(settings) = ant {
            if settings.enabled && !ant_units.is_empty() {
                effective_weight_paths = ant_resident_weight_paths(
                    &weight_paths,
                    &ant_units,
                    &ant_plan,
                    &ant_evicted,
                );
            }
        }
        // Ant: resolve the inference device from the residency plan. Resident VRAM
        // units load into the accelerator device; resident RAM-only units load into
        // the CPU device; disk units are excluded before the loader runs.
        let mut ant_resident_device: Option<Device> = None;
        let mut ant_resident_device_is_accelerator = device_is_accelerator;
        if let Some(settings) = ant_settings.as_ref() {
            if settings.enabled && !ant_units.is_empty() {
                let mut resident_tiers: Vec<&str> = Vec::new();
                for unit in &ant_units {
                    let tier = ant_plan
                        .get(&unit.unit_id)
                        .and_then(|tier| Some(tier.as_str()))
                        .unwrap_or("vram");
                    if tier == "disk" || ant_evicted.iter().any(|e| e == &unit.unit_id) {
                        continue;
                    }
                    resident_tiers.push(tier);
                }
                if !resident_tiers.is_empty() {
                    if resident_tiers.iter().any(|tier| *tier == "vram") && device_is_accelerator {
                        let selected = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into());
                        ant_resident_device = Some(super::models::backend_device(&selected)?);
                    } else if !resident_tiers.iter().any(|tier| *tier == "vram") {
                        ant_resident_device = Some(Device::Cpu);
                        ant_resident_device_is_accelerator = false;
                    }
                }
            }
        }
        // Ant: prefer the native mmap-backed CPU backend for architectures that
        // register a native loader — its tiers act on real pages.
        // Set CHEETAH_NATIVE=0 to fall back to the Candle CPU path.
        let native_enabled = std::env::var("CHEETAH_NATIVE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let native_model = if device_is_accelerator || !native_enabled {
            None
        } else {
            match super::models::backend_loader_for_architecture(&architecture, "native") {
                Some(super::models::BackendLoader::Native(loader)) => {
                    let family = super::models::family_for_architecture(&architecture)
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
            super::models::load_model_for_device(
                &effective_weight_paths,
                &config_bytes,
                raw_config.clone(),
                &architecture,
                resident_device,
                &ant_plan,
            )?
        } else {
            super::models::load_model(&effective_weight_paths, &config_bytes, raw_config, &architecture, &ant_plan)?
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
            architecture: Some(architecture),
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
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

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
        state
            .vram_used_bytes
            .saturating_add(state.ram_used_bytes)
            <= self.ant_budget_bytes
    }

    /// Ant: push tier transitions into the model's real memory pages when the
    /// backend owns mmap-backed weights (native CPU backend); candle-backed
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
                            state.vram_used_bytes = state
                                .vram_used_bytes
                                .saturating_sub(evicted_runtime.bytes);
                        }
                        state.evicted.insert(eviction.clone());
                        state.swaps_out = state.swaps_out.saturating_add(1);
                        // Ant: drop the unit's real pages on mmap backends.
                        self.ant_apply_real_tier(&[], std::slice::from_ref(eviction));
                    }
                }
                // Re-check before the swap: only proceed once the pool fits.
                if state.vram_used_bytes.saturating_add(runtime.bytes)
                    > settings.vram_pool_bytes
                {
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
            if state.evicted.remove(unit_id)
                || state.runtime_resident.insert(unit_id.to_string())
            {
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

    fn apply_ant_request_overrides(
        &self,
        overrides: &HashSet<String>,
    ) -> Result<(), String> {
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

    fn prefetch_ant_units(&self) -> Result<(), String> {
        let Some(settings) = self.ant_settings.as_ref() else {
            return Ok(());
        };
        if !settings.enabled
            || settings.prefetch_stages <= 0
            || self.ant_units.is_empty()
        {
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
                && state
                    .vram_used_bytes
                    .saturating_add(state.ram_used_bytes)
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
        let mut resident_bytes = state
            .vram_used_bytes
            .saturating_add(state.ram_used_bytes);
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
                        state.vram_used_bytes =
                            state.vram_used_bytes.saturating_sub(runtime.bytes);
                    }
                    _ => {
                        state.ram_used_bytes =
                            state.ram_used_bytes.saturating_sub(runtime.bytes);
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
            eprintln!(
                "{}",
                color::stderr(
                    "Context limit nearly reached; earlier conversation turns were truncated.",
                    Color::Yellow
                )
            );
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
        resident_overrides: &HashSet<String>,
    ) -> Result<Generation, String> {
        self.generate_inner(prompt, options, extra_stop, sink, resident_overrides)
    }

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
        let started = Instant::now();
        let mut logits_values = model
            .forward(prompt, 0)
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

/// Ant: rebuild a minimal config from the manifest's model-parameter snapshot
/// when the copied config.json is missing, so a prepared model directory
/// stays loadable on its own (parameters were snapshotted at `cheetah init`).
/// Family-specific extra fields (hidden_act / sliding_window / ...) required
/// by the Candle transformers structs are merged from the registration.
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
    if !synthesized.contains_key("max_position_embeddings") {
        synthesized.insert("max_position_embeddings".into(), serde_json::json!(4096));
    }
    // Merge the family's fallback parameters and extra config fields, when
    // registered: family values fill any gap the manifest snapshot does not
    // cover (e.g. manifests written before the snapshot was extended).
    if let Some(family) = super::models::family_for_architecture(&model_type) {
        if !synthesized.contains_key("rope_theta") {
            synthesized.insert("rope_theta".into(), serde_json::json!(family.rope_theta));
        }
        if !synthesized.contains_key("rms_norm_eps") {
            synthesized.insert("rms_norm_eps".into(), serde_json::json!(family.rms_norm_eps));
        }
        if !synthesized.contains_key("tie_word_embeddings") {
            synthesized.insert(
                "tie_word_embeddings".into(),
                serde_json::json!(family.tie_word_embeddings),
            );
        }
        if !family.extra_config_json.trim().is_empty() {
            let extra: serde_json::Value = serde_json::from_str(family.extra_config_json)
                .map_err(|error| {
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
    ant: Option<&AntSettings>,
) -> Result<(), String> {
    if !path.is_dir() && path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        return Err(format!(
            "Chat inference requires a model directory or Safetensors file: {}",
            path.display()
        ));
    }
    let engine = load_engine_with_progress(path, options.threads, options.ctx_size, ant)?;
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
                    .chunks_exact(2)
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
        let (tokens, _) =
            engine.prepare(&mut messages, options.system.as_deref(), options.max_tokens)?;
        let mut context = engine.new_context()?;
        let mut sink = StdoutSink;
        match engine.generate(&mut context, &tokens, &options, &[], &mut sink, &HashSet::new()) {
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
