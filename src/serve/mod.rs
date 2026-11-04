//! Cheetah's local OpenAI-compatible API service.

mod http;
mod openai;
mod queue;

use crate::chat::Engine;
use crate::cli::ServeOptions;
use crate::color::{self, Color};
use crate::options::ChatOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn serve(options: ServeOptions, chat_options: ChatOptions) -> Result<(), String> {
    // Refuse unauthenticated non-loopback listeners to prevent accidental network exposure.
    // 未启用鉴权时拒绝监听非回环地址，避免服务意外暴露到网络。
    if options.api_key.is_none() && !options.generate_key && !is_loopback(&options.host) {
        return Err(format!(
            "When serving on '{}', authentication must be enabled with --generate-key or --api-key.\n\
             Do not expose the server on a non-loopback address without authentication. For local use only, set --host to 127.0.0.1.",
            options.host
        ));
    }

    let (model_path, model_id) = resolve_model(&options)?;
    eprintln!(
        "{}",
        color::stderr(
            &format!("Loading model: {}", model_path.display()),
            Color::Cyan
        )
    );

    let engine = Engine::load(&model_path, chat_options.threads, chat_options.ctx_size)?;
    let api_key = resolve_api_key(&options)?;

    let addr = format!("{}:{}", options.host, options.port);
    let server = tiny_http::Server::http(&addr)
        .map_err(|error| format!("Failed to listen on {addr}: {error}"))?;
    let local_addr = server.server_addr();
    let port = match local_addr {
        tiny_http::ListenAddr::IP(address) => address.port(),
        #[cfg(unix)]
        tiny_http::ListenAddr::Unix(_) => {
            return Err("Listening on Unix sockets is not supported yet.".to_string());
        }
    };

    println!(
        "{}",
        color::stdout(
            &format!("Loaded model: {}", model_path.display()),
            Color::Green
        )
    );
    println!(
        "{}",
        color::stdout(
            &format!("Server started: {}", display_url(&options.host, port)),
            Color::Green
        )
    );
    if let Some(key) = &api_key {
        println!(
            "{}",
            color::stdout(&format!("API Key: {key}"), Color::Yellow)
        );
    } else {
        println!(
            "{}",
            color::stdout(
                "API key authentication is disabled (server is listening on a loopback address only).",
                Color::Yellow
            )
        );
    }

    ctrlc::set_handler(|| SHUTDOWN.store(true, Ordering::SeqCst))
        .map_err(|error| format!("Failed to install Ctrl-C handler: {error}"))?;

    let created = openai::now_unix();
    // Share immutable model state across request workers; the queue separately limits inference concurrency.
    // 请求线程共享不可变模型状态；生成队列独立控制推理并发上限。
    let state = Arc::new(http::ServerState {
        engine: Arc::new(engine),
        default_options: chat_options,
        model_id,
        api_key,
        generation_queue: Arc::new(queue::FifoQueue::new(options.max_concurrent_generations)?),
        created,
    });

    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(request)) => {
                let state = Arc::clone(&state);
                let ticket = http::is_generation_request(&request)
                    .then(|| Arc::clone(&state.generation_queue).enqueue());
                std::thread::spawn(move || http::handle(state, request, ticket));
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!(
                    "{}",
                    color::stderr(
                        &format!("Failed to receive connection: {error}"),
                        Color::Red
                    )
                );
            }
        }
    }

    println!("{}", color::stdout("Server stopped.", Color::Green));
    Ok(())
}

// ---- model resolution ----

fn resolve_model(options: &ServeOptions) -> Result<(PathBuf, String), String> {
    let candidate = PathBuf::from(&options.model);
    if candidate.is_file()
        && candidate
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
    {
        return Ok((candidate.clone(), model_id_from_path(&candidate)));
    }
    if candidate.is_dir() && is_model_directory(&candidate) {
        return Ok((candidate.clone(), model_id_from_path(&candidate)));
    }

    let name = model_name_for_match(&options.model);
    let mut matches = Vec::new();
    if options.model_dir.is_dir() {
        match std::fs::read_dir(&options.model_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if !entry
                        .file_type()
                        .is_ok_and(|ft| ft.is_dir() || ft.is_file())
                    {
                        continue;
                    }
                    let path = entry.path();
                    let is_safetensors = path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));
                    let is_named_model_dir = path.is_dir()
                        && is_model_directory(&path)
                        && path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .is_some_and(|stem| stem.eq_ignore_ascii_case(name));
                    let is_named_weight_file = is_safetensors
                        && path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .is_some_and(|stem| stem.eq_ignore_ascii_case(name));
                    if is_named_model_dir || is_named_weight_file {
                        matches.push(path);
                    }
                }
            }
            Err(error) => {
                return Err(format!(
                    "Failed to read model directory '{}': {error}",
                    options.model_dir.display()
                ));
            }
        }
    }

    let count = matches.len();
    let id = if count == 1 {
        let path = matches.remove(0);
        let id = model_id_from_path(&path);
        return Ok((path, id));
    } else if count == 0 {
        let listing = list_safetensors_models(&options.model_dir);
        if listing.is_empty() {
            format!(
                "Model '{}' not found: directory '{}' contains no Safetensors model directories or files.",
                options.model,
                options.model_dir.display()
            )
        } else {
            format!(
                "Model '{}' not found. Available Safetensors models in '{}':\n{}",
                options.model,
                options.model_dir.display(),
                listing
            )
        }
    } else {
        let list = matches
            .iter()
            .map(|path| format!("  {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Model name '{}' is ambiguous; found {} matches:\n{list}",
            options.model, count
        )
    };
    Err(id)
}

fn model_name_for_match(model: &str) -> &str {
    let name = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".safetensors") {
        &name[..name.len() - ".safetensors".len()]
    } else {
        name
    }
}

fn model_id_from_path(path: &Path) -> String {
    let name = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    };
    name.map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string())
}

fn is_model_directory(path: &Path) -> bool {
    path.join("config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && (path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file()
            || path.join("cheetah_manifest.json").is_file())
}

fn list_safetensors_models(dir: &Path) -> String {
    if !dir.is_dir() {
        return String::new();
    }
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if (path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
                || path.is_dir() && is_model_directory(&path))
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                names.push(format!("  {name}"));
            }
        }
    }
    names.sort();
    names.join("\n")
}

// ---- key / url helpers ----

fn resolve_api_key(options: &ServeOptions) -> Result<Option<String>, String> {
    if let Some(key) = &options.api_key {
        return Ok(Some(key.clone()));
    }
    if options.generate_key {
        return Ok(Some(openai::generate_api_key()));
    }
    Ok(None)
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

fn display_url(host: &str, port: u16) -> String {
    let display = if is_loopback(host) {
        "localhost"
    } else if host.contains(':') {
        return format!("http://[{host}]:{port}");
    } else {
        host
    };
    format!("http://{display}:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_strips_safetensors() {
        assert_eq!(model_name_for_match("qwen2.5-0.5b"), "qwen2.5-0.5b");
        assert_eq!(
            model_name_for_match("qwen2.5-0.5b.SAFETENSORS"),
            "qwen2.5-0.5b"
        );
        assert_eq!(
            model_name_for_match("path/qwen2.5-0.5b.safetensors"),
            "qwen2.5-0.5b"
        );
        assert_eq!(
            model_name_for_match("./dir/qwen2.5-0.5b.Safetensors"),
            "qwen2.5-0.5b"
        );
    }

    #[test]
    fn model_id_from_stem() {
        assert_eq!(
            model_id_from_path(Path::new("/data/qwen2.5-0.5b.safetensors")),
            "qwen2.5-0.5b"
        );
        assert_eq!(model_id_from_path(Path::new("model.safetensors")), "model");
    }

    #[test]
    fn loopback_checking() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("localhost"));
        assert!(is_loopback("::1"));
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("192.168.1.1"));
    }

    #[test]
    fn display_url_defaults_to_localhost() {
        assert_eq!(display_url("127.0.0.1", 8080), "http://localhost:8080");
        assert_eq!(display_url("::1", 9090), "http://localhost:9090");
        assert_eq!(display_url("0.0.0.0", 3000), "http://0.0.0.0:3000");
    }

    #[test]
    fn resolve_model_by_exact_file() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_file");
        let _ = std::fs::create_dir_all(&temp);
        let file = temp.join("test.safetensors");
        std::fs::write(&file, b"dummy").unwrap();

        let options = ServeOptions {
            model: file.display().to_string(),
            model_dir: PathBuf::from("."),
            ..Default::default()
        };
        let (path, id) = resolve_model(&options).unwrap();
        assert_eq!(path, file);
        assert_eq!(id, "test");
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn resolve_model_by_name_in_dir() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_name");
        let _ = std::fs::create_dir_all(&temp);
        let file = temp.join("qwen2.5-0.5b.safetensors");
        std::fs::write(&file, b"dummy").unwrap();

        let options = ServeOptions {
            model: "qwen2.5-0.5b".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let (path, id) = resolve_model(&options).unwrap();
        assert_eq!(path, file);
        assert_eq!(id, "qwen2.5-0.5b");
        std::fs::remove_file(&file).ok();
        std::fs::remove_dir(&temp).ok();
    }

    #[test]
    fn resolve_model_by_model_directory_name() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_directory");
        let model = temp.join("qwen-0.5b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(model.join("model.safetensors"), b"").unwrap();

        let options = ServeOptions {
            model: "qwen-0.5b".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let (resolved, id) = resolve_model(&options).unwrap();
        assert_eq!(resolved, model);
        assert_eq!(id, "qwen-0.5b");
        std::fs::remove_dir_all(&temp).unwrap();
    }

    #[test]
    fn resolve_model_directory_with_cheetah_manifest() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_manifest");
        let model = temp.join("prepared-model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(model.join("cheetah_manifest.json"), b"{}").unwrap();

        let options = ServeOptions {
            model: "prepared-model".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let (resolved, id) = resolve_model(&options).unwrap();
        assert_eq!(resolved, model);
        assert_eq!(id, "prepared-model");
        std::fs::remove_dir_all(&temp).unwrap();
    }

    #[test]
    fn resolve_model_ambiguous_errors() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_ambig");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("model-a.safetensors"), b"").unwrap();
        std::fs::write(temp.join("model-a.bin"), b"").unwrap(); // not Safetensors
        std::fs::write(temp.join("MODEL-A.SAFETENSORS"), b"").unwrap(); // case-insensitive match

        let options = ServeOptions {
            model: "model-a".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let model_count = std::fs::read_dir(&temp)
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
            .count();

        if model_count > 1 {
            let err = resolve_model(&options).unwrap_err();
            assert!(
                err.contains("ambiguous"),
                "expected ambiguity message, got: {err}"
            );
        } else {
            // Case-insensitive filesystems collapse these differently-cased names.
            let (_, id) = resolve_model(&options).unwrap();
            assert!(id.eq_ignore_ascii_case("model-a"), "got: {id}");
        }
        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn resolve_model_not_found_shows_listing() {
        let temp = std::env::temp_dir().join("cheetah_test_resolve_missing");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("available.safetensors"), b"").unwrap();

        let options = ServeOptions {
            model: "nonexistent".to_string(),
            model_dir: temp.clone(),
            ..Default::default()
        };
        let err = resolve_model(&options).unwrap_err();
        assert!(err.contains("available.safetensors"), "got: {err}");
        std::fs::remove_dir_all(&temp).ok();
    }
}
