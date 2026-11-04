use super::{ModelRegistration, ModelRuntime, candle};
use crate::color::{self, Color};
use candle_core::Device;
use serde_json::Value;
use std::path::PathBuf;

type DeviceFactory = fn() -> Result<Device, String>;

enum Backend {
    Candle(DeviceFactory),
    Vulkan,
}

struct BackendRegistration {
    name: &'static str,
    backend: Backend,
}

const BACKENDS: &[BackendRegistration] = &[
    BackendRegistration {
        name: "cpu",
        backend: Backend::Candle(cpu_device),
    },
    BackendRegistration {
        name: "cuda",
        backend: Backend::Candle(cuda_device),
    },
    BackendRegistration {
        name: "metal",
        backend: Backend::Candle(metal_device),
    },
    BackendRegistration {
        name: "vulkan",
        backend: Backend::Vulkan,
    },
];

pub(in crate::chat) fn load_model(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    architecture: &str,
) -> Result<(Device, Box<dyn ModelRuntime>), String> {
    let model_registration = super::registration_for_architecture(architecture)?;
    let selected = std::env::var("CHEETAH_DEVICE").unwrap_or_else(|_| "cpu".into());
    let registration = BACKENDS
        .iter()
        .find(|registration| registration.name == selected)
        .ok_or_else(|| {
            format!("Unsupported CHEETAH_DEVICE '{selected}'; choose cpu, cuda, or metal.")
        })?;

    match registration.backend {
        Backend::Candle(create_device) => {
            let device = create_device()?;
            let model = candle::load_model(
                weight_paths,
                config_bytes,
                raw_config,
                model_registration,
                &device,
            )?;
            Ok((device, model))
        }
        Backend::Vulkan => load_vulkan_or_fallback(
            weight_paths,
            config_bytes,
            raw_config,
            architecture,
            model_registration,
        ),
    }
}

fn load_vulkan_or_fallback(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    _architecture: &str,
    model_registration: &'static ModelRegistration,
) -> Result<(Device, Box<dyn ModelRuntime>), String> {
    #[cfg(feature = "vulkan")]
    if let Some(loader) = model_registration.vulkan_loader {
        if model_registration
            .shaders
            .iter()
            .any(|shader| !super::shader_is_registered(shader))
        {
            eprintln!(
                "{}",
                color::stderr(
                    &format!(
                        "cheetah: Vulkan model '{}' references an unregistered shader; falling back to CPU.",
                        model_registration.chat_template.id
                    ),
                    Color::Yellow
                )
            );
        } else {
            let vulkan_model = (|| loader(weight_paths, config_bytes, raw_config.clone()))();
            match vulkan_model {
                Ok(executor) => return Ok((Device::Cpu, executor)),
                Err(error) => {
                    eprintln!(
                        "{}",
                        color::stderr(
                            &format!(
                                "cheetah: Vulkan model '{}' unavailable ({error}); falling back to CPU.",
                                model_registration.chat_template.id
                            ),
                            Color::Yellow
                        )
                    );
                }
            }
        }
    } else {
        eprintln!(
            "{}",
            color::stderr(
                &format!(
                    "cheetah: Vulkan inference is not registered for architecture '{_architecture}'; falling back to CPU."
                ),
                Color::Yellow
            )
        );
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _has_vulkan_registration = model_registration.vulkan_loader.is_some()
            && model_registration
                .shaders
                .iter()
                .all(|shader| super::shader_is_registered(shader));
        eprintln!(
            "{}",
            color::stderr(
                "cheetah: Vulkan support was not compiled; falling back to CPU. Rebuild with --features vulkan.",
                Color::Yellow
            )
        );
    }

    let device = Device::Cpu;
    let model = candle::load_model(
        weight_paths,
        config_bytes,
        raw_config,
        model_registration,
        &device,
    )?;
    Ok((device, model))
}

fn cpu_device() -> Result<Device, String> {
    Ok(Device::Cpu)
}

fn cuda_device() -> Result<Device, String> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).map_err(|error| format!("Failed to initialize CUDA device: {error}"))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA support was not compiled; rebuild with --features cuda.".into())
    }
}

fn metal_device() -> Result<Device, String> {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).map_err(|error| format!("Failed to initialize Metal device: {error}"))
    }
    #[cfg(not(feature = "metal"))]
    {
        Err("Metal support was not compiled; rebuild with --features metal.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_candle_and_vulkan_backends() {
        let names: Vec<_> = BACKENDS.iter().map(|backend| backend.name).collect();
        assert_eq!(names, ["cpu", "cuda", "metal", "vulkan"]);
    }
}
