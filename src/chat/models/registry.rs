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

/// The declarative registration serving an architecture name (normalized:
/// case-insensitive, `ForCausalLM` suffix stripped), or a listing error.
pub(crate) fn registration_for_architecture(
    architecture: &str,
) -> Result<&'static super::ModelRegistration, String> {
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

/// Loader for one architecture's Vulkan executor, resolved from the
/// registration's backend table. Every architecture a registration claims
/// for Vulkan must resolve here — the loader lives in the same table, so a
/// miss is a registration-file bug, not a dispatch gap.
#[cfg(feature = "vulkan")]
pub(super) fn vulkan_loader_for_architecture(architecture: &str) -> Option<super::BackendLoader> {
    backend_loader_for_architecture(architecture, "vulkan")
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
        for architecture in ["mistral", "MistralForCausalLM", "gemma", "GemmaForCausalLM", "phi", "PhiForCausalLM"] {
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
                serving.iter().any(|name| normalize_architecture(name)
                    == normalize_architecture(architecture)),
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
}
