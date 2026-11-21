//! The model registry: the backend-agnostic runtime boundary, the
//! declarative registration types, and the derived lookups.
//!
//! Every file in `src/chat/registrations/` declares one family's capability
//! (architectures, chat template, stop markers, backends, family constants)
//! AND the family's forward assemblies: the registration implements
//! `ModelForward` by composing the backend toolboxes (native:
//! `chat::native::ops::OpsToolbox`; vulkan: `chat::vulkan` executors) in the
//! order its architecture prescribes. build.rs collects the declarative
//! entries into the generated `REGISTRY` (included in `chat/mod.rs`, so the
//! `<family>_registration` modules are children of `chat`) and every runtime
//! lookup derives from that table — adding a family is adding a registration
//! file, no table edits and no core-code branches.

use crate::color::{self, Color};
use serde_json::Value;
use std::path::PathBuf;

/// Backend device descriptor for engine bookkeeping and backend dispatch.
/// Host-resident backends (the native mmap backend, the Vulkan
/// executor) run against the CPU device; real accelerator handles live inside
/// backend executors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device {
    Cpu,
}

#[derive(Clone, Copy)]
pub(crate) struct ChatTemplate {
    pub(super) id: &'static str,
    pub(super) render: PromptRenderer,
    pub(super) header_renderer: Option<PromptRenderer>,
}

pub(crate) type PromptRenderer = fn(&[super::ChatMessage], Option<&str>) -> String;

/// The backend-agnostic inference boundary: every model family's forward
/// assembly implements this trait, and the engine (plus the registry's
/// `BackendLoader` table) only ever sees `Box<dyn ModelForward>`. The trait
/// carries no backend types and no family types — control flow vs. operators
/// split — so a model's layer stacking lives in its registration file and
/// the main program stays family-free.
pub(crate) trait ModelForward: ModelForwardClone + Send {
    /// Run the model over `tokens` starting at absolute position `offset` and
    /// return the logits of the final position. Backend-agnostic inference boundary.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String>;
    fn clear_cache(&mut self);
    fn device_name(&self) -> &'static str;
    /// Ant: real tier transitions on backends that own mmap-backed weights
    /// (the native CPU backend); host-resident backends report None.
    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        None
    }
}

/// Ant: tier mutations that act on real memory pages, not just bookkeeping.
pub(crate) trait AntTierControl {
    fn page_in_unit(&mut self, unit_id: &str, tier: i32) -> bool;
    fn evict_unit(&mut self, unit_id: &str) -> bool;
}

pub(crate) trait ModelForwardClone {
    fn clone_box(&self) -> Box<dyn ModelForward>;
}

impl<T> ModelForwardClone for T
where
    T: ModelForward + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ModelForward> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelForward> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// One inference backend implementation per architecture. Backend capability
/// is resolved by architecture name through the derived lookups below;
/// selection happens by backend name.
#[derive(Clone, Copy)]
pub(crate) enum BackendLoader {
    /// Vulkan compute path: receives the aligned shard paths. Only built when
    /// the `vulkan` feature is compiled in (`vulkan_loader_for_architecture`).
    #[allow(dead_code)]
    Vulkan(
        fn(
            &[PathBuf],
            &[u8],
            Value,
            &std::collections::BTreeMap<String, String>,
        ) -> Result<Box<dyn ModelForward>, String>,
    ),
    /// Native mmap + Ant tiering: receives the prepared model directory, the
    /// resolved residency plan, and the family constants for parameter
    /// fallbacks.
    Native(
        fn(
            &std::path::Path,
            &[u8],
            &Value,
            &std::collections::BTreeMap<String, String>,
            ModelFamily,
        ) -> Result<Box<dyn ModelForward>, String>,
    ),
}

/// Family-level inference constants for a model family: used as fallbacks when
/// the model directory does not carry them (missing or incomplete
/// config.json), keeping the native backend independent of directory
/// completeness.
#[derive(Clone, Copy)]
pub(crate) struct ModelFamily {
    pub(super) rope_theta: f64,
    pub(super) rms_norm_eps: f64,
    pub(super) tie_word_embeddings: bool,
    /// Known size profiles for this family, keyed by `(hidden_size, vocab_size)`:
    /// `(q_heads, kv_heads, head_dim)`. The q/kv/head-dim split cannot be
    /// derived from tensor shapes alone (and hidden size alone can be ambiguous
    /// within a family), so it is declared here.
    pub(super) profiles: &'static [(usize, usize, usize, usize, usize)],
    /// Family-specific config.json fields
    /// require (e.g. hidden_act / sliding_window), merged into a synthesized
    /// config when the model directory has none. Empty JSON object or "" = none.
    pub(super) extra_config_json: &'static str,
    /// Family-level default context length (`max_position_embeddings`). `None`
    /// means the family declares no default; config.json / the units manifest
    /// then govern (with a generic fallback in the engine).
    pub(super) max_position_embeddings: Option<usize>,
    /// Sparse-MoE expert naming layout for this family (Mixtral-compatible
    /// defaults). Drives router + expert weight slicing so the native forward
    /// never hardcodes one architecture's naming.
    pub(super) moe_layout: crate::chat::native::decoder::MoEExpertLayout,
    /// Dense-architecture deltas over the llama-shaped baseline (norm kind,
    /// activation, projection naming, residual topology, partial rope) for the
    /// native config resolution.
    pub(super) features: crate::chat::native::decoder::DenseFeatures,
}

impl ModelFamily {
    pub(super) fn profile_for(
        &self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<(usize, usize, usize)> {
        self.profiles
            .iter()
            .find(|(hidden, vocab, _, _, _)| *hidden == hidden_size && *vocab == vocab_size)
            .map(|(_, _, q_heads, kv_heads, head_dim)| (*q_heads, *kv_heads, *head_dim))
    }
}

/// One declarative model registration: what a family serves (architecture
/// names, chat template, stop markers), which backends back it, and the
/// family constants for native parameter fallbacks. Entries live in
/// `src/chat/registrations/<family>.rs` — each file also hosts that family's
/// runtime forward assemblies (the `ModelForward` implementations the
/// backend loader functions construct); build.rs assembles the declarative
/// constants into `REGISTRY` (generated in `chat/mod.rs`).
#[derive(Clone, Copy)]
pub(crate) struct ModelRegistration {
    /// Architecture names this registration serves (`Qwen2ForCausalLM` and
    /// `qwen2` style names alike; lookups normalize case + suffix).
    pub(super) architectures: &'static [&'static str],
    pub(super) chat_template: ChatTemplate,
    /// Generation stop markers for this family.
    pub(super) stop_markers: &'static [&'static str],
    /// Backend loaders keyed by backend name (`native`, `vulkan`), in
    /// declared order — the order `registered_backend_names` reports.
    pub(super) backends: &'static [(&'static str, BackendLoader)],
    /// Family constants for the native backend's parameter fallback chain;
    /// `None` for families the native backend does not serve.
    pub(super) family: Option<ModelFamily>,
}

impl ModelRegistration {
    /// Whether this registration serves the architecture name (normalized:
    /// case-insensitive, `ForCausalLM` suffix stripped on both sides).
    pub(super) fn serves(&self, architecture: &str) -> bool {
        let normalized = normalize_architecture(architecture);
        self.architectures
            .iter()
            .any(|name| normalize_architecture(name) == normalized)
    }

    /// Backend loader by name, case-insensitive.
    pub(super) fn loader_for(&self, backend: &str) -> Option<BackendLoader> {
        self.backends
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(backend))
            .map(|(_, loader)| *loader)
    }

    /// Registered backend names, in declared order.
    pub(super) fn backend_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backends.iter().map(|(name, _)| *name)
    }
}

// ---------------------------------------------------------------------------
// Derived lookups — all of these derive from the generated REGISTRY; none of
// them hardcode family data.
// ---------------------------------------------------------------------------

/// The declarative registration serving an architecture name (normalized:
/// case-insensitive, `ForCausalLM` suffix stripped), or a listing error.
pub(crate) fn registration_for_architecture(
    architecture: &str,
) -> Result<&'static ModelRegistration, String> {
    super::REGISTRY
        .iter()
        .find(|registration| registration.serves(architecture))
        .copied()
        .ok_or_else(|| {
            let registered: Vec<&str> = super::ALL_ARCHITECTURES
                .iter()
                .flat_map(|names| names.iter().copied())
                .collect();
            format!(
                "Architecture '{architecture}' is not registered; registered families: {registered:?}."
            )
        })
}

/// Architectures served by the native mmap + Ant tiering backend, derived
/// from the declarative registrations (every registration that carries a
/// native loader). Test-support view: the load path consults the
/// registration table directly.
#[cfg(test)]
pub(super) fn native_backends() -> Vec<&'static str> {
    architectures_serving("native")
}

/// Architectures served by the Vulkan compute path, derived the same way —
/// every registration that carries a Vulkan loader. Test-support view.
#[cfg(test)]
pub(super) fn vulkan_backends() -> Vec<&'static str> {
    architectures_serving("vulkan")
}

#[cfg(test)]
fn architectures_serving(backend: &str) -> Vec<&'static str> {
    super::REGISTRY
        .iter()
        .filter(|registration| registration.loader_for(backend).is_some())
        .flat_map(|registration| registration.architectures.iter().copied())
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn architecture_has_backend(architecture: &str, backend: &str) -> bool {
    registration_for_architecture(architecture)
        .map(|registration| registration.loader_for(backend).is_some())
        .unwrap_or(false)
}

/// Lowercase and strip the `ForCausalLM` suffix so `Qwen2ForCausalLM` and
/// `qwen2` resolve to the same architecture identity.
pub(super) fn normalize_architecture(architecture: &str) -> String {
    let mut normalized = architecture.to_lowercase();
    if let Some(stripped) = normalized.strip_suffix("forcausallm") {
        normalized = stripped.to_string();
    }
    normalized
}

/// Backend names registered for an architecture, in the registration's
/// declared order (e.g. `["native", "vulkan"]`).
pub(crate) fn registered_backends(architecture: &str) -> Vec<&'static str> {
    registration_for_architecture(architecture)
        .map(|registration| registration.backend_names().collect())
        .unwrap_or_default()
}

pub(crate) fn backend_loader_for_architecture(
    architecture: &str,
    backend: &str,
) -> Option<BackendLoader> {
    registration_for_architecture(architecture)
        .ok()
        .and_then(|registration| registration.loader_for(backend))
}

/// Family constants for a native backend family, from the registration file.
pub(crate) fn family_for_architecture(architecture: &str) -> Option<ModelFamily> {
    registration_for_architecture(architecture)
        .ok()
        .and_then(|registration| registration.family)
}

/// Chat prompt renderer resolved from the architecture name through the
/// declarative registrations (init-time metadata keys prompt families by
/// architecture).
pub(crate) fn chat_template_for_architecture(architecture: &str) -> Option<ChatTemplate> {
    registration_for_architecture(architecture)
        .ok()
        .map(|registration| registration.chat_template)
}

/// Stop marker token names resolved from the architecture name through the
/// declarative registrations; an unknown architecture stops on nothing.
pub(crate) fn stop_markers_for_architecture(architecture: &str) -> &'static [&'static str] {
    registration_for_architecture(architecture)
        .map(|registration| registration.stop_markers)
        .unwrap_or(&[])
}

pub(crate) fn load_model(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<(Device, Box<dyn ModelForward>), String> {
    let selected = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into());
    let selected_lower = selected.to_lowercase();
    let registration = BACKENDS
        .iter()
        .find(|registration| registration.name == selected_lower)
        .ok_or_else(|| format!("Unsupported CHEETAH_DEVICE '{selected}'; choose cpu or vulkan."))?;

    match registration.backend {
        Backend::Native => Err(no_native_backend_message(architecture)),
        Backend::Vulkan => {
            load_vulkan_or_fallback(weight_paths, config_bytes, raw_config, architecture, plan)
        }
    }
}

/// Guidance error: the architecture has no native backend available in this
/// build, and no CPU backend is registered for it.
fn no_native_backend_message(architecture: &str) -> String {
    format!(
        "Architecture '{architecture}' has no native CPU backend available. Use CHEETAH_DEVICE=vulkan, or run a model family with a registered native backend."
    )
}

pub(crate) fn backend_device(selected: &str) -> Result<Device, String> {
    let registration = BACKENDS
        .iter()
        .find(|registration| registration.name == selected.to_lowercase().as_str())
        .ok_or_else(|| format!("Unsupported CHEETAH_DEVICE '{selected}'; choose cpu or vulkan."))?;
    match &registration.backend {
        Backend::Native => Ok(Device::Cpu),
        Backend::Vulkan => Ok(Device::Cpu),
    }
}

pub(crate) fn load_model_for_device(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
    device: Device,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<(Device, Box<dyn ModelForward>), String> {
    if std::env::var("CHEETAH_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .eq_ignore_ascii_case("vulkan")
    {
        return load_vulkan_or_fallback(weight_paths, config_bytes, raw_config, architecture, plan);
    }
    let _ = device;
    Err(no_native_backend_message(architecture))
}

/// Loader for one architecture's Vulkan executor, resolved from the
/// registration's backend table. Every architecture a registration claims
/// for Vulkan must resolve here — the loader lives in the same table, so a
/// miss is a registration-file bug, not a dispatch gap.
#[cfg(feature = "vulkan")]
pub(super) fn vulkan_loader_for_architecture(architecture: &str) -> Option<BackendLoader> {
    backend_loader_for_architecture(architecture, "vulkan")
}

#[cfg_attr(not(feature = "vulkan"), allow(unused_variables, unused_mut))]
fn load_vulkan_or_fallback(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<(Device, Box<dyn ModelForward>), String> {
    #[cfg(feature = "vulkan")]
    {
        if !architecture_has_backend(architecture, "vulkan") {
            eprintln!(
                "{}",
                color::stderr(
                    &format!(
                        "cheetah: Vulkan inference is not registered for architecture '{architecture}'; falling back to CPU."
                    ),
                    Color::Yellow
                )
            );
            return Err(no_native_backend_message(architecture));
        }
        let vulkan_model = match vulkan_loader_for_architecture(architecture) {
            Some(BackendLoader::Vulkan(load)) => {
                load(weight_paths, config_bytes, raw_config.clone(), plan)
            }
            // The capability table and the loader table disagree, or the
            // architecture carries no Vulkan executor at all.
            Some(BackendLoader::Native(_)) | None => {
                Err(format!("Unsupported Vulkan architecture '{architecture}'."))
            }
        };
        match vulkan_model {
            Ok(executor) => Ok((Device::Cpu, executor)),
            Err(error) => {
                eprintln!(
                    "{}",
                    color::stderr(
                        &format!("cheetah: Vulkan model unavailable ({error})."),
                        Color::Yellow
                    )
                );
                Err(no_native_backend_message(architecture))
            }
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        eprintln!(
            "{}",
            color::stderr(
                &format!(
                    "cheetah: Vulkan support was not compiled. Rebuild with --features vulkan, or use CHEETAH_DEVICE=cpu."
                ),
                Color::Yellow
            )
        );
        Err(no_native_backend_message(architecture))
    }
}

enum Backend {
    /// Native mmap + Ant tiering backend (host-resident weights, page tiers).
    Native,
    /// Vulkan compute path.
    Vulkan,
}

struct BackendRegistration {
    name: &'static str,
    backend: Backend,
}

const BACKENDS: &[BackendRegistration] = &[
    BackendRegistration {
        name: "cpu",
        backend: Backend::Native,
    },
    BackendRegistration {
        name: "vulkan",
        backend: Backend::Vulkan,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_cpu_and_vulkan_backends() {
        let names: Vec<_> = BACKENDS.iter().map(|backend| backend.name).collect();
        assert_eq!(names, ["cpu", "vulkan"]);
    }

    #[test]
    fn backend_device_rejects_unknown_backend() {
        assert!(backend_device("does_not_exist").is_err());
        assert!(backend_device("cpu").is_ok());
        assert!(backend_device("vulkan").is_ok());
    }

    /// Registration identity: the short name and the `ForCausalLM` suffix
    /// form resolve to the same registration on every lookup surface.
    #[test]
    fn registration_lookup_normalizes_architecture_names() {
        for (short, full) in [
            ("llama", "LlamaForCausalLM"),
            ("qwen2", "Qwen2ForCausalLM"),
            ("qwen3", "Qwen3ForCausalLM"),
            ("mixtral", "MixtralForCausalLM"),
            ("mistral", "MistralForCausalLM"),
            ("gemma", "GemmaForCausalLM"),
            ("phi", "PhiForCausalLM"),
        ] {
            let short_registration = registration_for_architecture(short).unwrap();
            let full_registration = registration_for_architecture(full).unwrap();
            assert!(
                std::ptr::eq(short_registration, full_registration),
                "{short} and {full} must resolve to the same registration"
            );
            assert!(family_for_architecture(short).is_some());
            assert!(family_for_architecture(full).is_some());
        }
        assert!(registration_for_architecture("unknown").is_err());
        assert!(family_for_architecture("unknown").is_none());
    }

    #[test]
    fn native_backend_serves_llama_and_qwen_families() {
        assert!(backend_loader_for_architecture("llama", "native").is_some());
        assert!(backend_loader_for_architecture("Qwen2ForCausalLM", "native").is_some());
        assert!(backend_loader_for_architecture("qwen3", "native").is_some());
    }

    #[test]
    fn native_backend_serves_mixtral_moe_family() {
        assert!(backend_loader_for_architecture("mixtral", "native").is_some());
        assert!(backend_loader_for_architecture("MixtralForCausalLM", "native").is_some());
    }

    /// Dense coverage: the mistral / gemma / phi delta families resolve the
    /// native backend and carry their family constants.
    #[test]
    fn native_backend_serves_dense_delta_families() {
        for architecture in [
            "mistral",
            "MistralForCausalLM",
            "gemma",
            "GemmaForCausalLM",
            "phi",
            "PhiForCausalLM",
        ] {
            assert!(
                backend_loader_for_architecture(architecture, "native").is_some(),
                "{architecture} must resolve the native backend"
            );
            assert!(family_for_architecture(architecture).is_some());
        }
        assert!(backend_loader_for_architecture("llama", "native").is_some());
    }

    /// A native registration must carry family constants — the native loader's
    /// parameter fallback chain reads them. A declarative file that flips on
    /// the native backend without a family would resolve to a broken load.
    #[test]
    fn every_native_registration_carries_a_family() {
        for registration in super::super::REGISTRY {
            if registration.loader_for("native").is_some() {
                assert!(
                    registration.family.is_some(),
                    "registration for {:?} serves native but declares no family",
                    registration.architectures
                );
            }
        }
    }

    /// Architecture names must be globally unique across the registration
    /// files — a duplicated name would silently shadow one family's
    /// template/backends with whichever registration sorts first.
    #[test]
    fn registrations_cover_disjoint_architectures() {
        let mut seen = std::collections::HashSet::new();
        for names in super::super::ALL_ARCHITECTURES {
            for name in names.iter() {
                assert!(
                    seen.insert(*name),
                    "architecture '{name}' is claimed by more than one registration"
                );
            }
        }
    }

    /// Every registration that claims Vulkan must actually expose a Vulkan
    /// loader (the same table is consulted at load time), and the derived
    /// capability list must cover the published family set.
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_capability_table_and_loader_table_agree() {
        for registration in super::super::REGISTRY {
            let claims_vulkan = registration.loader_for("vulkan").is_some();
            for name in registration.architectures {
                assert_eq!(
                    architecture_has_backend(name, "vulkan"),
                    claims_vulkan,
                    "{name}: capability and loader table disagree"
                );
            }
        }
        let serving = vulkan_backends();
        for architecture in [
            "llama", "qwen2", "qwen3", "gemma", "mistral", "mixtral", "phi",
        ] {
            assert!(
                serving.iter().any(
                    |name| normalize_architecture(name) == normalize_architecture(architecture)
                ),
                "{architecture} must be reachable on the Vulkan load path"
            );
        }
        assert!(vulkan_backends().iter().all(|name| {
            super::super::ALL_ARCHITECTURES
                .iter()
                .any(|names| names.contains(name))
        }));
        assert!(!architecture_has_backend("unknown", "vulkan"));
    }

    /// Every registration's Vulkan loader must reach its own family feature
    /// selection (there is no central `model_type` dispatch anymore): the
    /// load fails on the empty weight paths, never on "no executor
    /// features".
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_loaders_reach_family_feature_selection() {
        for registration in super::super::REGISTRY {
            let Some(BackendLoader::Vulkan(load)) = registration.loader_for("vulkan") else {
                continue;
            };
            let raw = serde_json::json!({ "model_type": registration.architectures[0] });
            let error = match load(&[], &[], raw, &Default::default()) {
                Err(error) => error,
                Ok(_) => panic!("empty weight paths cannot load a model"),
            };
            assert!(
                !error.contains("no executor features"),
                "{:?} must dispatch to its family features ({error})",
                registration.architectures
            );
        }
    }
}
