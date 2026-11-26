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

/// One declarative-registration directory: where the files live, which
/// generated table they assemble into, and the registration type they share.
struct RegistrySpec {
    /// Registration source directory (also the generated header comment).
    source: &'static str,
    /// Generated file name under `OUT_DIR`.
    out_file: &'static str,
    /// The registration struct type the generated `REGISTRY` points to.
    type_name: &'static str,
    /// Suffix appended to the generated module name for each registration
    /// file (`llama.rs` → `mod llama_registration;`). An empty suffix gives
    /// the file's plain stem (precision formats declare their operators and
    /// their registration in one file under their own name).
    module_suffix: &'static str,
    /// Where the runtime implementations for these registrations live (docs
    /// comment in the generated file).
    implementation_note: &'static str,
    /// Optional per-file index table: (generated const name, per-file const
    /// name) — e.g. model registrations publish `ARCHITECTURES` per file.
    index: Option<(&'static str, &'static str)>,
}

const MODEL_REGISTRY: RegistrySpec = RegistrySpec {
    source: "src/chat/registrations",
    out_file: "model_registry.rs",
    type_name: "ModelRegistration",
    module_suffix: "_registration",
    implementation_note: "the chat module tree (the registration file itself hosts the family's runtime)",
    index: Some(("ALL_ARCHITECTURES", "ARCHITECTURES")),
};

const PRECISION_REGISTRY: RegistrySpec = RegistrySpec {
    source: "src/chat/precision",
    out_file: "precision_registry.rs",
    type_name: "PrecisionRegistration",
    module_suffix: "",
    implementation_note: "this same module (each format file declares its operators and its registration)",
    index: None,
};

const OPS_REGISTRY: RegistrySpec = RegistrySpec {
    source: "src/chat/ops",
    out_file: "ops_registry.rs",
    type_name: "OpRegistration",
    module_suffix: "",
    implementation_note: "this same module (each operator file declares its operators and its registration)",
    index: None,
};

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
    println!("cargo:rerun-if-changed=matmul_q8.comp");
    println!("cargo:rerun-if-changed=matmul_q4.comp");
    println!("cargo:rerun-if-changed=shared_ops.comp");
    println!("cargo:rerun-if-changed=src/chat");
    println!("cargo:rerun-if-changed=src/chat/registrations");
    println!("cargo:rerun-if-changed=src/chat/precision");
    println!("cargo:rerun-if-changed=src/chat/ops");
    println!("cargo:rerun-if-changed=src/chat/ops/shaders");

    cc::Build::new()
        .file("cheetah_init.c")
        .file("cheetah_memory.c")
        .include(".")
        .warnings(true)
        .extra_warnings(true)
        .compile("cheetah_init");

    let registration_sources =
        discover_registrations(Path::new(MODEL_REGISTRY.source), "model registration");
    generate_registry(&registration_sources, &MODEL_REGISTRY);
    let precision_sources =
        discover_registrations(Path::new(PRECISION_REGISTRY.source), "weight precision");
    generate_registry(&precision_sources, &PRECISION_REGISTRY);
    let ops_sources = discover_registrations(Path::new(OPS_REGISTRY.source), "operator");
    generate_registry(&ops_sources, &OPS_REGISTRY);
    let shader_sources = discover_shaders(&[
        Path::new("src/chat/registrations"),
        Path::new("src/chat/ops/shaders"),
    ]);
    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        compile_vulkan(&shader_sources);
        println!("cargo:rustc-link-lib=vulkan");
    }
}

/// Collect declarative registrations from one directory. The directory is the
/// contract: every `.rs` file there contributes one registration — no
/// exclusion list. `label` names the registry kind in diagnostics.
fn discover_registrations(directory: &Path, label: &str) -> Vec<(String, PathBuf)> {
    let mut sources = Vec::new();
    let mut names = HashSet::new();
    for entry in fs::read_dir(directory).expect("failed to scan registrations directory") {
        let path = entry.expect("failed to read registration entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let Some(stem) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_owned)
        else {
            panic!("{label} filename is not valid UTF-8: {}", path.display());
        };
        if !is_rust_identifier(&stem) {
            panic!("{label} name '{stem}' must be a Rust identifier");
        }
        if !names.insert(stem.clone()) {
            panic!("duplicate {label} name '{stem}'");
        }
        println!("cargo:rerun-if-changed={}", path.display());
        sources.push((stem, path));
    }
    // Sort for a stable, deterministic registry order across platforms.
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    sources
}

/// Generate a registry file: one module per declarative registration file
/// plus the assembled `REGISTRY` (and optional index) tables the runtime
/// lookups derive from. Adding a registration is adding a file — no table
/// edits.
fn generate_registry(registrations: &[(String, PathBuf)], spec: &RegistrySpec) {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let mut generated = String::new();
    generated.push_str(&format!(
        "// @generated by build.rs from {} — do not edit.\n",
        spec.source
    ));
    generated.push_str("// Each module here is a declarative registration file; the runtime\n");
    generated.push_str(&format!(
        "// implementations live in {}.\n\n",
        spec.implementation_note
    ));
    for (name, path) in registrations {
        let source = fs::canonicalize(path)
            .expect("failed to resolve registration path")
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        generated.push_str(&format!(
            "#[path = \"{source}\"] mod {name}{};\n",
            spec.module_suffix
        ));
    }
    generated.push_str(&format!(
        "\npub(crate) const REGISTRY: &[&{}] = &[\n",
        spec.type_name
    ));
    for (name, _) in registrations {
        generated.push_str(&format!(
            "    &{name}{}::REGISTRATION,\n",
            spec.module_suffix
        ));
    }
    generated.push_str("];\n");
    if let Some((index_const, file_const)) = spec.index {
        generated
            .push_str("\n/// Every index entry across all registrations, in registry order.\n");
        generated.push_str(&format!(
            "pub(crate) const {index_const}: &[&[&str]] = &[\n"
        ));
        for (name, _) in registrations {
            generated.push_str(&format!(
                "    {name}{}::{file_const},\n",
                spec.module_suffix
            ));
        }
        generated.push_str("];\n");
    }
    fs::write(out_dir.join(spec.out_file), generated).expect("failed to write generated registry");
}

fn discover_shaders(directories: &[&Path]) -> Vec<ShaderSource> {
    let mut sources = vec![
        ShaderSource {
            name: "matmul".into(),
            source: PathBuf::from("matmul.comp"),
            output: "matmul.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_MATMUL_SPV"),
        },
        ShaderSource {
            name: "matmul_q8".into(),
            source: PathBuf::from("matmul_q8.comp"),
            output: "matmul_q8.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_MATMUL_Q8_SPV"),
        },
        ShaderSource {
            name: "matmul_q4".into(),
            source: PathBuf::from("matmul_q4.comp"),
            output: "matmul_q4.spv".into(),
            c_define: Some("CHEETAH_DEFAULT_MATMUL_Q4_SPV"),
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
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            // A shader directory may not exist yet (e.g. no custom op shaders);
            // that is not an error — it simply contributes no shaders.
            continue;
        };
        for entry in entries {
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
            Ok(status) if shader.c_define.is_none() => println!(
                "cargo:warning=custom op shader {} failed to compile (status {status}); override its SPIR-V path at runtime via the op's CustomShader.env variable",
                shader.source.display()
            ),
            Err(error) if shader.c_define.is_none() => println!(
                "cargo:warning=unable to run {} for custom op shader {}: {error}; override its SPIR-V path at runtime via the op's CustomShader.env variable",
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

    // Expose the compiled custom op shaders (c_define == None) to the Rust
    // registry so it can resolve each op's default SPIR-V path at runtime and
    // assert that every registered custom op ships a shader.
    let mut custom_list = String::from("// Generated by build.rs — do not edit.\n");
    custom_list.push_str("pub(crate) const CUSTOM_OP_SHADERS: &[(&str, &str)] = &[\n");
    for shader in shaders {
        if shader.c_define.is_some() {
            continue;
        }
        if !compiled.contains(&shader.name) {
            continue;
        }
        let path = out_dir
            .join(&shader.output)
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        custom_list.push_str(&format!("    (\"{}\", \"{path}\"),\n", shader.name));
    }
    custom_list.push_str("];\n");
    custom_list
        .push_str("pub(crate) fn custom_op_shader_default(name: &str) -> Option<&'static str> {\n");
    custom_list
        .push_str("    CUSTOM_OP_SHADERS.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)\n");
    custom_list.push_str("}\n");
    fs::write(out_dir.join("custom_op_shaders.rs"), custom_list)
        .expect("failed to write custom op shader list");

    for (name, value) in &definitions {
        build.define(name, Some(value.as_str()));
    }
    build.compile("cheetah_vulkan");
}
