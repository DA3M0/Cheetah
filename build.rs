use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct ShaderSource {
    name: String,
    source: PathBuf,
    output: String,
    c_define: Option<&'static str>,
}

fn main() {
    println!("cargo:rerun-if-changed=cheetah_init.c");
    println!("cargo:rerun-if-changed=cheetah_init.h");
    println!("cargo:rerun-if-changed=cheetah_color.h");
    println!("cargo:rerun-if-changed=cheetah_vulkan.c");
    println!("cargo:rerun-if-changed=cheetah_vulkan.h");
    println!("cargo:rerun-if-changed=cheetah_memory.c");
    println!("cargo:rerun-if-changed=cheetah_memory.h");
    println!("cargo:rerun-if-changed=matmul.comp");
    println!("cargo:rerun-if-changed=matmul_bf16.comp");
    println!("cargo:rerun-if-changed=shared_ops.comp");
    println!("cargo:rerun-if-changed=src/chat/models");
    println!("cargo:rerun-if-changed=src/chat/models/registrations");

    cc::Build::new()
        .file("cheetah_init.c")
        .file("cheetah_memory.c")
        .include(".")
        .warnings(true)
        .extra_warnings(true)
        .compile("cheetah_init");

    let registration_sources = discover_registrations(Path::new("src/chat/models/registrations"));
    let shader_sources = discover_shaders(Path::new("src/chat/models"));
    generate_model_registry(&registration_sources);
    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        compile_vulkan(&shader_sources);
        println!("cargo:rustc-link-lib=vulkan");
    } else {
        generate_shader_registry(&shader_sources, &HashSet::new());
    }
}

/// Collect declarative model registrations. The directory is the contract:
/// every `.rs` file there contributes one registration — no exclusion list.
fn discover_registrations(directory: &Path) -> Vec<(String, PathBuf)> {
    let mut sources = Vec::new();
    let mut names = HashSet::new();
    for entry in fs::read_dir(directory).expect("failed to scan model registrations directory") {
        let path = entry.expect("failed to read model registration entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            panic!(
                "model registration filename is not valid UTF-8: {}",
                path.display()
            );
        };
        if !is_rust_identifier(stem) {
            panic!("model registration name '{stem}' must be a Rust identifier");
        }
        if !names.insert(stem.to_owned()) {
            panic!("duplicate model registration name '{stem}'");
        }
        println!("cargo:rerun-if-changed={}", path.display());
        sources.push((stem.to_owned(), path));
    }
    // Sort for a stable, deterministic registry order across platforms.
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    sources
}

fn discover_shaders(directory: &Path) -> Vec<ShaderSource> {
    let mut sources = vec![
        ShaderSource {
            name: "matmul".into(),
            source: PathBuf::from("matmul.comp"),
            output: "matmul.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_MATMUL_SPV"),
        },
        ShaderSource {
            name: "matmul_bf16".into(),
            source: PathBuf::from("matmul_bf16.comp"),
            output: "matmul_bf16.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_MATMUL_BF16_SPV"),
        },
        ShaderSource {
            name: "shared_ops".into(),
            source: PathBuf::from("shared_ops.comp"),
            output: "shared_ops.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_SHARED_OPS_SPV"),
        },
    ];
    let mut names: HashSet<String> = sources.iter().map(|shader| shader.name.clone()).collect();
    for entry in fs::read_dir(directory).expect("failed to scan model shader directory") {
        let path = entry.expect("failed to read model shader entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("comp") {
            continue;
        }
        let Some(stem) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_owned)
        else {
            panic!("shader filename is not valid UTF-8: {}", path.display());
        };
        if !is_rust_identifier(&stem) {
            panic!("shader name '{stem}' must be a Rust identifier");
        }
        if !names.insert(stem.clone()) {
            panic!("duplicate shader name '{stem}'");
        }
        println!("cargo:rerun-if-changed={}", path.display());
        sources.push(ShaderSource {
            name: stem.clone(),
            source: path,
            output: format!("{stem}.spv"),
            c_define: None,
        });
    }
    sources.sort_by(|left, right| left.name.cmp(&right.name));
    sources
}

fn is_rust_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let valid = matches!(chars.next(), Some('a'..='z' | 'A'..='Z' | '_'))
        && chars.all(|character| character.is_ascii_alphanumeric() || character == '_');
    valid
        && !matches!(
            value,
            "as" | "break"
                | "const"
                | "continue"
                | "crate"
                | "else"
                | "enum"
                | "extern"
                | "false"
                | "fn"
                | "for"
                | "if"
                | "impl"
                | "in"
                | "let"
                | "loop"
                | "match"
                | "mod"
                | "move"
                | "mut"
                | "pub"
                | "ref"
                | "return"
                | "self"
                | "Self"
                | "static"
                | "struct"
                | "super"
                | "trait"
                | "true"
                | "type"
                | "unsafe"
                | "use"
                | "where"
                | "while"
                | "async"
                | "await"
                | "dyn"
        )
}

fn generate_model_registry(registrations: &[(String, PathBuf)]) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut generated = String::new();
    generated.push_str("// @generated by build.rs from src/chat/models/registrations — do not edit.\n");
    generated.push_str("// Each module here is a declarative registration file; the runtime\n");
    generated.push_str("// implementations live in sibling modules of src/chat/models.\n\n");
    for (name, path) in registrations {
        let source = fs::canonicalize(path)
            .expect("failed to resolve model registration path")
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        generated.push_str(&format!("#[path = \"{source}\"] mod {name}_registration;\n"));
    }
    generated.push_str("\npub(crate) const REGISTRY: &[&ModelRegistration] = &[\n");
    for (name, _) in registrations {
        generated.push_str(&format!("    &{name}_registration::REGISTRATION,\n"));
    }
    generated.push_str("];\n\n");
    generated.push_str("/// Every architecture name across all registrations, in registry order.\n");
    generated.push_str("pub(crate) const ALL_ARCHITECTURES: &[&[&str]] = &[\n");
    for (name, _) in registrations {
        generated.push_str(&format!("    {name}_registration::ARCHITECTURES,\n"));
    }
    generated.push_str("];\n");
    fs::write(out_dir.join("model_registry.rs"), generated)
        .expect("failed to write generated model registry");
}

fn compile_vulkan(shaders: &[ShaderSource]) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let glslc = env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
    let mut compiled = HashSet::new();
    let mut build = cc::Build::new();
    let mut definitions = Vec::new();
    build
        .file("cheetah_vulkan.c")
        .include(".")
        .warnings(true)
        .extra_warnings(true);

    for shader in shaders {
        let output_path = out_dir.join(&shader.output);
        match Command::new(&glslc)
            .args(["-fshader-stage=compute"])
            .arg(&shader.source)
            .arg("-o")
            .arg(&output_path)
            .status()
        {
            Ok(status) if status.success() => {
                compiled.insert(shader.name.clone());
                if let Some(define) = shader.c_define {
                    let path = output_path
                        .to_string_lossy()
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"");
                    definitions.push((define, format!("\"{path}\"")));
                }
            }
            Ok(status) if shader.c_define.is_none() => panic!(
                "{} failed to compile model shader {} (status {status})",
                glslc.to_string_lossy(),
                shader.source.display()
            ),
            Err(error) if shader.c_define.is_none() => panic!(
                "unable to run {} for model shader {}: {error}",
                glslc.to_string_lossy(),
                shader.source.display()
            ),
            Ok(status) => println!(
                "cargo:warning={} failed to compile {} (status {status}); set the corresponding SPIR-V path at runtime",
                glslc.to_string_lossy(),
                shader.source.display()
            ),
            Err(error) => println!(
                "cargo:warning=unable to run {} for {}: {error}; set the corresponding SPIR-V path at runtime",
                glslc.to_string_lossy(),
                shader.source.display()
            ),
        }
    }

    for (name, value) in &definitions {
        build.define(name, Some(value.as_str()));
    }
    build.compile("cheetah_vulkan");
    generate_shader_registry(shaders, &compiled);
}

fn generate_shader_registry(shaders: &[ShaderSource], compiled: &HashSet<String>) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut generated = String::from("pub(super) const SHADERS: &[ShaderRegistration] = &[\n");
    for shader in shaders {
        let path = if compiled.contains(&shader.name) {
            let path = out_dir.join(&shader.output);
            format!(
                "Some(\"{}\")",
                path.to_string_lossy()
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
            )
        } else {
            "None".into()
        };
        generated.push_str(&format!(
            "    ShaderRegistration {{ name: {:?}, path: {path} }},\n",
            shader.name
        ));
    }
    generated.push_str("];\n");
    fs::write(out_dir.join("shader_registry.rs"), generated)
        .expect("failed to write generated shader registry");
}
