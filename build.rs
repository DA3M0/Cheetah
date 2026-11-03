use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cheetah_init.c");
    println!("cargo:rerun-if-changed=cheetah_init.h");
    println!("cargo:rerun-if-changed=cheetah_vulkan.c");
    println!("cargo:rerun-if-changed=cheetah_vulkan.h");
    println!("cargo:rerun-if-changed=matmul.comp");
    println!("cargo:rerun-if-changed=matmul_bf16.comp");
    println!("cargo:rerun-if-changed=qwen_ops.comp");

    cc::Build::new()
        .file("cheetah_init.c")
        .include(".")
        .warnings(true)
        .extra_warnings(true)
        .compile("cheetah_init");

    if std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set"));
        let glslc = std::env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
        let shader_sources = [
            ("matmul.comp", "matmul.spv", "CHEETAH_DEFAULT_MATMUL_SPV"),
            (
                "matmul_bf16.comp",
                "matmul_bf16.spv",
                "CHEETAH_DEFAULT_MATMUL_BF16_SPV",
            ),
            (
                "qwen_ops.comp",
                "qwen_ops.spv",
                "CHEETAH_DEFAULT_QWEN_OPS_SPV",
            ),
        ];
        let mut build = cc::Build::new();
        let mut definitions = Vec::new();
        build
            .file("cheetah_vulkan.c")
            .include(".")
            .warnings(true)
            .extra_warnings(true);
        for (source, output, define) in shader_sources {
            let output_path = out_dir.join(output);
            match Command::new(&glslc)
                .args(["-fshader-stage=compute", source, "-o"])
                .arg(&output_path)
                .status()
            {
                Ok(status) if status.success() => {
                    let path = output_path
                        .to_string_lossy()
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"");
                    definitions.push((define, format!("\"{path}\"")));
                }
                Ok(status) => {
                    println!(
                        "cargo:warning={} failed to compile {source} (status {status}); set the corresponding SPIR-V path at runtime",
                        glslc.to_string_lossy()
                    );
                }
                Err(error) => {
                    println!(
                        "cargo:warning=unable to run {} for {source}: {error}; set the corresponding SPIR-V path at runtime",
                        glslc.to_string_lossy()
                    );
                }
            }
        }
        for (name, value) in &definitions {
            build.define(name, Some(value.as_str()));
        }
        build.compile("cheetah_vulkan");
        println!("cargo:rustc-link-lib=vulkan");
    }
}
