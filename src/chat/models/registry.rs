use super::{BackendLoader, Device, ModelFamily, ModelRuntime};
use crate::color::{self, Color};
use serde_json::Value;
use std::path::PathBuf;

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

/// Architectures served by the native mmap + Ant tiering backend (llama /
/// mixtral / qwen2 / qwen3 shapes; mixtral carries the mixture-of-experts
/// feed-forward).
pub(super) fn native_backends() -> &'static [&'static str] {
    &["llama", "mixtral", "qwen2", "qwen3"]
}

/// Architectures served by the Vulkan compute path: every family with a Vulkan
/// executor. gemma / mistral / phi are Vulkan-only (no native backend); the
/// native-capable families serve both. Kept in step with
/// `vulkan_loader_for_architecture` (asserted by a test).
pub(super) fn vulkan_backends() -> &'static [&'static str] {
    &[
        "llama",
        "LlamaForCausalLM",
        "qwen2",
        "Qwen2ForCausalLM",
        "qwen3",
        "Qwen3ForCausalLM",
        "gemma",
        "GemmaForCausalLM",
        "mistral",
        "MistralForCausalLM",
        "mixtral",
        "MixtralForCausalLM",
        "phi",
        "PhiForCausalLM",
    ]
}

fn architecture_has_backend(architecture: &str, backend: &str) -> bool {
    let normalized = normalize_architecture(architecture);
    let names = if backend.eq_ignore_ascii_case("native") {
        native_backends()
    } else {
        vulkan_backends()
    };
    names
        .iter()
        .any(|name| normalize_architecture(name) == normalized)
}

/// Lowercase and strip the `ForCausalLM` suffix so `Qwen2ForCausalLM` and
/// `qwen2` resolve to the same architecture identity.
fn normalize_architecture(architecture: &str) -> String {
    let mut normalized = architecture.to_lowercase();
    if let Some(stripped) = normalized.strip_suffix("forcausallm") {
        normalized = stripped.to_string();
    }
    normalized
}

/// Backend names registered for an architecture: llama / mixtral / qwen2 /
/// qwen3 serve `["native", "vulkan"]`; gemma / mistral / phi serve `["vulkan"]`.
pub(crate) fn registered_backends(architecture: &str) -> Vec<&'static str> {
    if architecture_has_backend(architecture, "native") {
        return vec!["native", "vulkan"];
    }
    if architecture_has_backend(architecture, "vulkan") {
        return vec!["vulkan"];
    }
    Vec::new()
}

pub(crate) fn backend_loader_for_architecture(
    architecture: &str,
    backend: &str,
) -> Option<BackendLoader> {
    if backend.eq_ignore_ascii_case("native") && architecture_has_backend(architecture, "native") {
        return Some(BackendLoader::Native(super::native::native_family_loader));
    }
    None
}

/// Family constants for a native backend family (llama / mixtral / qwen2 /
/// qwen3).
pub(crate) fn family_for_architecture(architecture: &str) -> Option<ModelFamily> {
    match architecture {
        "llama" | "LlamaForCausalLM" => Some(llama_family()),
        "mixtral" | "MixtralForCausalLM" => Some(mixtral_family()),
        "qwen2" | "Qwen2ForCausalLM" => Some(qwen2_family()),
        "qwen3" | "Qwen3ForCausalLM" => Some(qwen3_family()),
        _ => None,
    }
}

fn mixtral_family() -> ModelFamily {
    super::ModelFamily {
        rope_theta: 8_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32768, 32, 8, 128), // Mixtral-8x7B (GQA: 32 q heads / 8 kv heads)
        ],
        extra_config_json: "",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
    }
}

fn llama_family() -> ModelFamily {
    super::ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (2048, 128256, 32, 8, 64),  // Llama-3.2-1B (tied)
            (3072, 128256, 24, 8, 128), // Llama-3.2-3B (tied)
            (4096, 128256, 32, 8, 128), // Llama-3-8B / 3.1-8B
            (4096, 32000, 32, 32, 128), // Llama-2-7B (MHA)
        ],
        extra_config_json: "",
        max_position_embeddings: Some(4096),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
    }
}

fn qwen2_family() -> ModelFamily {
    super::ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (896, 151936, 14, 2, 64),   // Qwen2-0.5B
            (1536, 151936, 12, 2, 128), // Qwen2-1.5B
            (3584, 152064, 28, 4, 128), // Qwen2-7B
        ],
        extra_config_json:
            "{\"hidden_act\":\"silu\",\"sliding_window\":4096,\"use_sliding_window\":false,\"max_window_layers\":21}",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
    }
}

fn qwen3_family() -> ModelFamily {
    super::ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (8192, 32768, 64, 8, 128), // Qwen3-32B GQA config
        ],
        extra_config_json: "{\"hidden_act\":\"silu\"}",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
    }
}

pub(crate) fn load_model(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<(Device, Box<dyn ModelRuntime>), String> {
    let selected = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into());
    let selected_lower = selected.to_lowercase();
    let registration = BACKENDS
        .iter()
        .find(|registration| registration.name == selected_lower)
        .ok_or_else(|| {
            format!("Unsupported CHEETAH_DEVICE '{selected}'; choose cpu or vulkan.")
        })?;

    match registration.backend {
        Backend::Native => Err(no_native_backend_message(architecture)),
        Backend::Vulkan => load_vulkan_or_fallback(
            weight_paths,
            config_bytes,
            raw_config,
            architecture,
            plan,
        ),
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
        .ok_or_else(|| {
            format!("Unsupported CHEETAH_DEVICE '{selected}'; choose cpu or vulkan.")
        })?;
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
) -> Result<(Device, Box<dyn ModelRuntime>), String> {
    if std::env::var("CHEETAH_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .eq_ignore_ascii_case("vulkan")
    {
        return load_vulkan_or_fallback(weight_paths, config_bytes, raw_config, architecture, plan);
    }
    let _ = device;
    Err(no_native_backend_message(architecture))
}

/// Loader for one architecture's Vulkan executor. Every architecture the
/// Vulkan capability table claims must resolve here — a registered
/// architecture without a loader would fail at load time with
/// "Unsupported Vulkan architecture".
#[cfg(feature = "vulkan")]
pub(super) fn vulkan_loader_for_architecture(architecture: &str) -> Option<super::BackendLoader> {
    let loader = match architecture {
        "llama" | "LlamaForCausalLM" => super::llama::load_vulkan,
        "qwen2" | "Qwen2ForCausalLM" => super::qwen2::load_vulkan,
        "qwen3" | "Qwen3ForCausalLM" => super::qwen3::load_vulkan,
        "gemma" | "GemmaForCausalLM" => super::gemma::load_vulkan,
        "mistral" | "MistralForCausalLM" => super::mistral::load_vulkan,
        "mixtral" | "MixtralForCausalLM" => super::mixtral::load_vulkan,
        "phi" | "PhiForCausalLM" => super::phi::load_vulkan,
        _ => return None,
    };
    Some(super::BackendLoader::Vulkan(loader))
}

#[cfg_attr(not(feature = "vulkan"), allow(unused_variables, unused_mut))]
fn load_vulkan_or_fallback(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<(Device, Box<dyn ModelRuntime>), String> {
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
            Some(super::BackendLoader::Vulkan(load)) => {
                load(weight_paths, config_bytes, raw_config.clone(), plan)
            }
            // The capability table and the loader table disagree, or the
            // architecture carries no Vulkan executor at all.
            Some(super::BackendLoader::Native(_)) | None => {
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

    #[test]
    fn native_backend_serves_llama_and_qwen_families() {
        assert!(backend_loader_for_architecture("llama", "native").is_some());
        assert!(backend_loader_for_architecture("Qwen2ForCausalLM", "native").is_some());
        assert!(backend_loader_for_architecture("qwen3", "native").is_some());
        assert!(backend_loader_for_architecture("gemma", "native").is_none());
        assert!(backend_loader_for_architecture("gemma", "vulkan").is_none());
    }

    #[test]
    fn native_backend_serves_mixtral_moe_family() {
        assert!(backend_loader_for_architecture("mixtral", "native").is_some());
        assert!(backend_loader_for_architecture("MixtralForCausalLM", "native").is_some());
        assert!(backend_loader_for_architecture("gemma", "native").is_none());
        assert!(backend_loader_for_architecture("gemma", "vulkan").is_none());
    }

    #[test]
    fn mixtral_architectures_resolve_native_and_vulkan_backends() {
        assert_eq!(registered_backends("mixtral"), vec!["native", "vulkan"]);
        assert_eq!(registered_backends("MixtralForCausalLM"), vec!["native", "vulkan"]);
    }

    /// The Vulkan capability table and the loader table must agree in both
    /// directions: a registered architecture with no loader fails at load time
    /// with "Unsupported Vulkan architecture", and a loader hidden behind a
    /// capability table miss is unreachable.
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_capability_table_and_loader_table_agree() {
        for architecture in vulkan_backends() {
            assert!(
                architecture_has_backend(architecture, "vulkan"),
                "{architecture} is missing from the Vulkan capability table"
            );
            assert!(
                vulkan_loader_for_architecture(architecture).is_some(),
                "{architecture} is registered for Vulkan but has no loader"
            );
        }
        for architecture in [
            "llama", "qwen2", "qwen3", "gemma", "mistral", "mixtral", "phi",
        ] {
            assert!(
                architecture_has_backend(architecture, "vulkan"),
                "{architecture} must be reachable on the Vulkan load path"
            );
            assert!(
                vulkan_loader_for_architecture(architecture).is_some(),
                "{architecture} is registered for Vulkan but has no loader"
            );
        }
        assert!(vulkan_loader_for_architecture("unknown").is_none());
        assert!(!architecture_has_backend("unknown", "vulkan"));
    }
}
