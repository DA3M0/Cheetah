//! Command-line argument parsing and runtime options.
//!
//! Every subcommand's flag grammar lives here as a hand-written parser
//! (no clap dependency): `parse_chat_args` (also used by the legacy
//! bare-model form), `parse_serve_args`, `parse_inspect_args`,
//! `parse_context_args`, and `parse_ant_args` for the `--ant*` family.
//! The parsers are intentionally strict — unknown flags and repeated
//! single-value options are usage errors — so that typos surface at
//! startup instead of silently changing behavior.
//!
//! Generic value parsing helpers (`parse_positive_option`,
//! `parse_float_option`, ...) are `pub` because `main.rs` reuses the Ant
//! flags via `split_ant_args`.

use crate::ant::AntSettings;
use crate::options::ChatOptions;
use std::path::PathBuf;
use std::process::ExitCode;

/// CLI-level options for saved conversation archives (`--context*` flags).
///
/// `chat_options_specified` records whether the user passed any generation
/// flag explicitly; when resuming a named archive this flag decides
/// whether the archive's saved options are adopted as-is or validated
/// against the CLI overrides.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextCliOptions {
    /// Name of the archive to resume (`--context <name>`); `None` starts a
    /// fresh conversation.
    pub name: Option<String>,
    /// Archive directory override (`--context-dir <path>`).
    pub directory: Option<PathBuf>,
    /// True when at least one generation flag was given on the command line.
    pub chat_options_specified: bool,
}

/// `cheetah context <operation>` subcommands (list / delete).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommand {
    /// List all saved archives.
    List,
    /// Delete the archive with the given name.
    Delete(String),
}

/// Cheetah serve settings for model selection, listener, and authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeOptions {
    /// Model name, model directory, or Safetensors weight path.
    pub model: String,
    /// Directory searched when resolving a model by name.
    pub model_dir: PathBuf,
    /// Bind address; defaults to loopback so the server is not exposed
    /// unless explicitly requested.
    pub host: String,
    /// Listen port; 0 lets the OS assign a free port.
    pub port: u16,
    /// Cap on generation requests performing model inference at the same
    /// time (enforced by `serve/queue.rs` tickets).
    pub max_concurrent_generations: usize,
    /// When true, print a freshly generated API key and enable auth.
    pub generate_key: bool,
    /// Explicit API key (`--api-key`); enables auth when set.
    pub api_key: Option<String>,
}

impl Default for ServeOptions {
    /// Loopback-only listener with a single concurrent generation: the
    /// safest out-of-the-box posture.
    fn default() -> Self {
        Self {
            model: String::new(),
            model_dir: PathBuf::from("."),
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent_generations: 1,
            generate_key: false,
            api_key: None,
        }
    }
}

/// Runtime error reported by `main`: argument/usage errors exit with code 2,
/// all other runtime errors exit with code 1.
#[derive(Debug)]
pub enum RunError {
    /// Bad command-line usage (unknown flag, missing value, ...).
    Usage(String),
    /// Failure while performing the requested work.
    Runtime(String),
}

impl RunError {
    /// Convenience constructor for [`RunError::Usage`].
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    /// Maps the error kind to a process exit code (2 = usage, 1 = runtime).
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) => ExitCode::from(2),
            Self::Runtime(_) => ExitCode::FAILURE,
        }
    }
}

impl std::fmt::Display for RunError {
    /// Renders both error kinds through one presentation: the message text itself (the kind only changes the exit code).
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(message) | Self::Runtime(message) => formatter.write_str(message),
        }
    }
}

/// Parses `cheetah chat <model-path> [options]` (and the legacy bare-model
/// form, which funnels through the same parser).
///
/// Returns the model path, generation options, context/archive options, and
/// Ant CLI overrides in one pass. Every recognized generation flag also
/// sets `chat_options_specified` so archive restoration can tell explicit
/// overrides from defaults.
pub fn parse_chat_args(
    args: &[String],
) -> Result<
    (
        PathBuf,
        ChatOptions,
        ContextCliOptions,
        AntCliOptions,
        McpCliOptions,
    ),
    RunError,
> {
    let mut path = None;
    let mut options = ChatOptions::default();
    let mut context = ContextCliOptions::default();
    let mut ant = AntCliOptions::default();
    let mut mcp = McpCliOptions::default();
    let mut index = 0;
    // Track explicit inference flags separately so restored archives can
    // distinguish defaults from command-line overrides.
    while index < args.len() {
        match args[index].as_str() {
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
                context.chat_options_specified = true;
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
                context.chat_options_specified = true;
            }
            "--ctx-size" => {
                index += 1;
                options.ctx_size = parse_positive_option(args, index, "--ctx-size")?;
                context.chat_options_specified = true;
            }
            "--temperature" => {
                index += 1;
                options.temperature = parse_float_option(args, index, "--temperature", 0.0)?;
                context.chat_options_specified = true;
            }
            "--top-p" => {
                index += 1;
                options.top_p = parse_bounded_float_option(args, index, "--top-p", 0.0, 1.0)?;
                context.chat_options_specified = true;
            }
            "--top-k" => {
                index += 1;
                options.top_k = parse_nonnegative_option(args, index, "--top-k")?;
                context.chat_options_specified = true;
            }
            "--repeat-penalty" => {
                index += 1;
                options.repeat_penalty =
                    parse_bounded_float_option(args, index, "--repeat-penalty", 1.0, f64::MAX)?
                        as f32;
                context.chat_options_specified = true;
            }
            "--seed" => {
                index += 1;
                options.seed = Some(parse_u64_option(args, index, "--seed")?);
                context.chat_options_specified = true;
            }
            "--system" => {
                index += 1;
                options.system = Some(parse_text_option(args, index, "--system")?);
                context.chat_options_specified = true;
            }
            "--context" => {
                index += 1;
                let name = parse_text_option(args, index, "--context")?;
                if context.name.replace(name).is_some() {
                    return Err(RunError::usage("--context can only be specified once."));
                }
            }
            "--context-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--context-dir requires a path."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--context-dir path cannot be empty."));
                }
                if context.directory.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("--context-dir can only be specified once."));
                }
            }
            "--mcp-server" => {
                index += 1;
                mcp.add_server(parse_text_option(args, index, "--mcp-server")?.as_str());
            }
            "--mcp-config" => {
                index += 1;
                mcp.add_config(parse_text_option(args, index, "--mcp-config")?.as_str())?;
            }
            "--ant" => {
                if args
                    .get(index + 1)
                    .is_some_and(|value| is_boolean_literal(value.as_str()))
                {
                    ant.enabled = Some(parse_bool_option(args, index + 1, "--ant")?);
                    index += 1;
                } else {
                    ant.enabled = Some(true);
                }
            }
            "--ant-setup" | "--setup" => {
                let option = args[index].as_str();
                ant.interactive = Some(parse_setup_flag(args, &mut index, option)?);
            }
            "--ant-vram" => {
                index += 1;
                ant.vram_pool_bytes = Some(parse_u64_option(args, index, "--ant-vram")?);
            }
            "--ant-ram" => {
                index += 1;
                ant.ram_budget_bytes = Some(parse_u64_option(args, index, "--ant-ram")?);
            }
            "--ant-disk-dir" => {
                index += 1;
                ant.disk_dir = Some(parse_text_option(args, index, "--ant-disk-dir")?);
            }
            "--ant-resident" => {
                index += 1;
                ant.resident_units
                    .push(parse_text_option(args, index, "--ant-resident")?);
            }
            "--ant-eviction-policy" => {
                index += 1;
                ant.eviction_policy =
                    Some(parse_text_option(args, index, "--ant-eviction-policy")?);
            }
            "--ant-prefetch-stages" => {
                index += 1;
                ant.prefetch_stages =
                    Some(parse_nonnegative_option(args, index, "--ant-prefetch-stages")? as i64);
            }
            "--ant-allow-vram-swap" => {
                index += 1;
                ant.allow_vram_swap =
                    Some(parse_bool_option(args, index, "--ant-allow-vram-swap")?);
            }
            option if option.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "Unknown option '{option}'. Run `cheetah help` for usage."
                )));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("Only one model path can be specified."));
                }
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| {
        RunError::usage(
            "Specify a Safetensors model directory or file. Usage: cheetah <model-path> [options]. Run `cheetah help` for all options.",
        )
    })?;
    Ok((path, options, context, ant, mcp))
}

/// Parses `cheetah context list | delete <name> [--context-dir <path>]`.
pub fn parse_context_args(
    args: &[String],
) -> Result<(ContextCommand, ContextCliOptions), RunError> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(RunError::usage(
            "Missing context operation. Usage: cheetah context list | delete <name> [--context-dir <path>]",
        ));
    };
    let mut context = ContextCliOptions::default();
    let mut name = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--context-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--context-dir requires a path."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--context-dir path cannot be empty."));
                }
                if context.directory.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("--context-dir can only be specified once."));
                }
            }
            value if value.starts_with('-') => {
                return Err(RunError::usage(format!("Unknown option '{value}'.")));
            }
            value => {
                if name.replace(value.to_string()).is_some() {
                    return Err(RunError::usage("Only one context name can be specified."));
                }
            }
        }
        index += 1;
    }
    let operation = match command {
        "list" if name.is_none() => ContextCommand::List,
        "delete" => ContextCommand::Delete(name.ok_or_else(|| {
            RunError::usage(
                "Missing context name. Usage: cheetah context delete <name> [--context-dir <path>]",
            )
        })?),
        "list" => {
            return Err(RunError::usage(
                "cheetah context list does not accept a context name.",
            ));
        }
        other => {
            return Err(RunError::usage(format!(
                "Unknown context operation '{other}'. Available operations: list, delete."
            )));
        }
    };
    Ok((operation, context))
}

/// Parses `cheetah serve <name-or-path> [options]`.
///
/// Serve-specific flags land in [`ServeOptions`]; generation flags
/// (`--temperature`, `--max-tokens`, ...) populate the default
/// [`ChatOptions`] used as the server-wide defaults for every request
/// (requests may still override them per call).
pub fn parse_serve_args(
    args: &[String],
) -> Result<(ServeOptions, ChatOptions, McpCliOptions), RunError> {
    let mut serve = ServeOptions::default();
    let mut options = ChatOptions::default();
    let mut mcp = McpCliOptions::default();
    let mut max_concurrent_generations_specified = false;
    let mut index = 0;
    // Serve accepts both a positional model and the legacy --model flag,
    // but never both at once.
    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--model requires a model name or path."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--model cannot be empty."));
                }
                if !serve.model.is_empty() {
                    return Err(RunError::usage(
                        "Only one model can be specified; do not combine a positional argument with --model.",
                    ));
                }
                serve.model = value.clone();
            }
            "--model-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--model-dir requires a path."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--model-dir path cannot be empty."));
                }
                serve.model_dir = PathBuf::from(value);
            }
            "--host" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--host requires an address."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--host cannot be empty."));
                }
                serve.host = value.clone();
            }
            "--port" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--port requires a port number."))?;
                serve.port = value.parse::<u16>().map_err(|_| {
                    RunError::usage(format!(
                        "--port must be between 0 and 65535; received '{value}'."
                    ))
                })?;
            }
            "--max-concurrent-generations" => {
                if max_concurrent_generations_specified {
                    return Err(RunError::usage(
                        "--max-concurrent-generations can only be specified once.",
                    ));
                }
                index += 1;
                serve.max_concurrent_generations =
                    parse_positive_option(args, index, "--max-concurrent-generations")?;
                max_concurrent_generations_specified = true;
            }
            "--generate-key" => {
                if serve.generate_key {
                    return Err(RunError::usage(
                        "--generate-key can only be specified once.",
                    ));
                }
                serve.generate_key = true;
            }
            "--api-key" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| RunError::usage("--api-key requires a key."))?;
                if value.is_empty() {
                    return Err(RunError::usage("--api-key cannot be empty."));
                }
                if serve.api_key.replace(value.clone()).is_some() {
                    return Err(RunError::usage("--api-key can only be specified once."));
                }
            }
            "--threads" => {
                index += 1;
                options.threads = Some(parse_positive_option(args, index, "--threads")?);
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens = parse_positive_option(args, index, "--max-tokens")?;
            }
            "--ctx-size" => {
                index += 1;
                options.ctx_size = parse_positive_option(args, index, "--ctx-size")?;
            }
            "--temperature" => {
                index += 1;
                options.temperature = parse_float_option(args, index, "--temperature", 0.0)?;
            }
            "--top-p" => {
                index += 1;
                options.top_p = parse_bounded_float_option(args, index, "--top-p", 0.0, 1.0)?;
            }
            "--top-k" => {
                index += 1;
                options.top_k = parse_nonnegative_option(args, index, "--top-k")?;
            }
            "--repeat-penalty" => {
                index += 1;
                options.repeat_penalty =
                    parse_bounded_float_option(args, index, "--repeat-penalty", 1.0, f64::MAX)?
                        as f32;
            }
            "--seed" => {
                index += 1;
                options.seed = Some(parse_u64_option(args, index, "--seed")?);
            }
            "--system" => {
                index += 1;
                options.system = Some(parse_text_option(args, index, "--system")?);
            }
            "--mcp-server" => {
                index += 1;
                mcp.add_server(parse_text_option(args, index, "--mcp-server")?.as_str());
            }
            "--mcp-config" => {
                index += 1;
                mcp.add_config(parse_text_option(args, index, "--mcp-config")?.as_str())?;
            }
            option if option.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "Unknown option '{option}'. Run `cheetah help` for usage."
                )));
            }
            value => {
                if !serve.model.is_empty() {
                    return Err(RunError::usage(
                        "Only one model name or path can be specified for serve; use --model for the legacy syntax.",
                    ));
                }
                serve.model = value.to_string();
            }
        }
        index += 1;
    }

    if serve.model.is_empty() {
        return Err(RunError::usage(
            "Specify a model name or path. Usage: cheetah serve <name-or-path> [options]",
        ));
    }
    if serve.generate_key && serve.api_key.is_some() {
        return Err(RunError::usage(
            "--generate-key and --api-key cannot be used together.",
        ));
    }
    Ok((serve, options, mcp))
}

/// Options for `cheetah inspect <model-path>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectOptions {
    /// Safetensors file or model directory to inspect.
    pub path: PathBuf,
    /// Emit machine-readable JSON instead of the human-readable table.
    pub json: bool,
    /// Show all tensors instead of the bounded default list.
    pub show_all: bool,
    /// Write the report to a file instead of stdout.
    pub output: Option<PathBuf>,
}

/// Parses `cheetah inspect <model-path> [--all] [--json] [-o <file>]`.
pub fn parse_inspect_args(args: &[String]) -> Result<InspectOptions, RunError> {
    let mut path = None;
    let mut json = false;
    let mut show_all = false;
    let mut output = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--all" => show_all = true,
            "-o" | "--output" => {
                index += 1;
                let value = args.get(index).ok_or_else(|| {
                    RunError::usage(format!("{} requires an output file path.", args[index - 1]))
                })?;
                if value.is_empty() {
                    return Err(RunError::usage("Output file path cannot be empty."));
                }
                if output.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("-o/--output can only be specified once."));
                }
            }
            value if value.starts_with('-') => {
                return Err(RunError::usage(format!(
                    "Unknown option '{value}'. Run `cheetah help` for usage."
                )));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(RunError::usage("Only one model file can be specified."));
                }
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| {
        RunError::usage(
            "Missing model path. Usage: cheetah inspect <model-path> [--all] [--json] [-o <file>]",
        )
    })?;
    Ok(InspectOptions {
        path,
        json,
        show_all,
        output,
    })
}

/// Parses an option value that must be a strictly positive integer (> 0).
pub fn parse_positive_option(
    args: &[String],
    index: usize,
    option: &str,
) -> Result<usize, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} requires a value.")))?;
    let parsed = value.parse::<usize>().map_err(|_| {
        RunError::usage(format!(
            "{option} requires a positive integer; received '{value}'."
        ))
    })?;
    if parsed == 0 {
        return Err(RunError::usage(format!("{option} must be greater than 0.")));
    }
    Ok(parsed)
}

/// Parses an option value that must be a non-negative integer (>= 0).
pub fn parse_nonnegative_option(
    args: &[String],
    index: usize,
    option: &str,
) -> Result<usize, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} requires a value.")))?;
    value.parse::<usize>().map_err(|_| {
        RunError::usage(format!(
            "{option} requires a non-negative integer; received '{value}'."
        ))
    })
}

/// Parses an option value as an unsigned 64-bit integer (used for byte
/// budgets and seeds).
fn parse_u64_option(args: &[String], index: usize, option: &str) -> Result<u64, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} requires a value.")))?;
    value.parse::<u64>().map_err(|_| {
        RunError::usage(format!(
            "{option} requires a non-negative integer; received '{value}'."
        ))
    })
}

/// Consumes the next argument as a raw text value (paths, unit ids, prompts).
fn parse_text_option(args: &[String], index: usize, option: &str) -> Result<String, RunError> {
    args.get(index).cloned().ok_or_else(|| {
        RunError::usage(format!(
            "{option} requires a value, e.g. --system \"You are a concise assistant.\"."
        ))
    })
}

/// Recognizes boolean flag values: true/false/1/0/yes/no/on/off
/// (case-insensitive). Used to decide whether a bare flag like `--ant`
/// consumed the next token as its value.
pub fn is_boolean_literal(value: &str) -> bool {
    ["true", "false", "1", "0", "yes", "no", "on", "off"]
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(value))
}

/// Parses the next argument as a boolean (same literals as
/// [`is_boolean_literal`]); anything else is a usage error.
fn parse_bool_option(args: &[String], index: usize, option: &str) -> Result<bool, RunError> {
    let value = parse_text_option(args, index, option)?;
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(RunError::usage(format!(
            "{option} expects a boolean value (true/false); received '{value}'."
        ))),
    }
}

/// MCP (`--mcp-server` / `--mcp-config`) options shared by chat and serve.
/// Raw strings live here; the MCP registry resolves them into live server
/// connections at session start.
#[derive(Debug, Clone, Default)]
pub struct McpCliOptions {
    /// Raw `name=command args...` specifications (repeatable flag).
    pub servers: Vec<String>,
    /// Optional JSON config file: `{"servers": [{"name", "command", "args"}]}`.
    pub config: Option<PathBuf>,
}

impl McpCliOptions {
    /// Parse one `--mcp-server` value into the options.
    pub fn add_server(&mut self, spec: &str) {
        self.servers.push(spec.to_owned());
    }

    /// Parse one `--mcp-config` value into the options (once).
    pub fn add_config(&mut self, path: &str) -> Result<(), RunError> {
        if self.config.replace(PathBuf::from(path)).is_some() {
            return Err(RunError::usage("--mcp-config can only be specified once."));
        }
        Ok(())
    }
}

/// Command-line overrides for Ant memory optimization; `None` means the
/// flag was not given, deferring to the user preset / built-in defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AntCliOptions {
    /// `--ant [bool]`: enable/disable Ant entirely (bare flag = enable).
    pub enabled: Option<bool>,
    /// `--ant-vram <bytes>`: VRAM pool size override.
    pub vram_pool_bytes: Option<u64>,
    /// `--ant-ram <bytes>`: RAM budget override.
    pub ram_budget_bytes: Option<u64>,
    /// `--ant-disk-dir <path>`: disk-residency directory override.
    pub disk_dir: Option<String>,
    /// `--ant-resident <id>` (repeatable): unit ids to keep resident.
    pub resident_units: Vec<String>,
    /// `--ant-eviction-policy <policy>`: eviction policy override.
    pub eviction_policy: Option<String>,
    /// `--ant-prefetch-stages <n>`: prefetch look-ahead override.
    pub prefetch_stages: Option<i64>,
    /// `--ant-allow-vram-swap <bool>`: VRAM swap permission override.
    pub allow_vram_swap: Option<bool>,
    /// `--setup` / `--ant-setup`: run the interactive preset wizard;
    /// `None` means the flag was not given.
    pub interactive: Option<bool>,
}

impl AntCliOptions {
    /// Merges this set of overrides into a complete [`AntSettings`]
    /// (unspecified fields fall back to the built-in defaults).
    pub fn resolve(&self) -> AntSettings {
        let mut settings = AntSettings::default();
        if let Some(enabled) = self.enabled {
            settings.enabled = enabled;
        }
        if let Some(vram) = self.vram_pool_bytes {
            settings.vram_pool_bytes = vram;
        }
        if let Some(ram) = self.ram_budget_bytes {
            settings.ram_budget_bytes = ram;
        }
        if let Some(disk_dir) = self.disk_dir.clone() {
            settings.disk_dir = Some(disk_dir);
        }
        if !self.resident_units.is_empty() {
            settings.resident_units = self.resident_units.clone();
        }
        if let Some(policy) = self.eviction_policy.clone() {
            settings.eviction_policy = policy;
        }
        if let Some(prefetch) = self.prefetch_stages {
            settings.prefetch_stages = prefetch as i32;
        }
        if let Some(swap) = self.allow_vram_swap {
            settings.allow_vram_swap = swap;
        }
        settings
    }
}

/// Parse an interactive-setup flag that accepts either a bare flag or an
/// explicit boolean value (`--setup`, `--setup true`). `index` points at the
/// flag itself and is advanced past the flag and any consumed value; the
/// caller's loop still applies its trailing `index += 1`.
fn parse_setup_flag(args: &[String], index: &mut usize, option: &str) -> Result<bool, RunError> {
    if args
        .get(*index + 1)
        .is_some_and(|value| is_boolean_literal(value.as_str()))
    {
        let value = parse_bool_option(args, *index + 1, option)?;
        *index += 1;
        Ok(value)
    } else {
        Ok(true)
    }
}

/// Parses Ant memory-optimization flags: `--ant <bool>` plus the
/// budget / residency / eviction override family. Any unknown flag here is
/// a usage error, which is what makes [`crate::main::split_ant_args`]-style
/// pre-extraction safe.
pub fn parse_ant_args(args: &[String]) -> Result<AntCliOptions, RunError> {
    let mut options = AntCliOptions::default();
    let mut index = 0;
    while index < args.len() {
        let option = args[index].as_str();
        match option {
            "--ant" => {
                if args
                    .get(index + 1)
                    .is_some_and(|value| is_boolean_literal(value.as_str()))
                {
                    options.enabled = Some(parse_bool_option(args, index + 1, option)?);
                    index += 1;
                } else {
                    options.enabled = Some(true);
                }
            }
            "--ant-setup" | "--setup" => {
                let option = args[index].as_str();
                options.interactive = Some(parse_setup_flag(args, &mut index, option)?);
            }
            "--ant-vram" => {
                index += 1;
                options.vram_pool_bytes = Some(parse_u64_option(args, index, option)?);
            }
            "--ant-ram" => {
                index += 1;
                options.ram_budget_bytes = Some(parse_u64_option(args, index, option)?);
            }
            "--ant-disk-dir" => {
                index += 1;
                options.disk_dir = Some(parse_text_option(args, index, option)?);
            }
            "--ant-resident" => {
                index += 1;
                options
                    .resident_units
                    .push(parse_text_option(args, index, option)?);
            }
            "--ant-eviction-policy" => {
                index += 1;
                options.eviction_policy = Some(parse_text_option(args, index, option)?);
            }
            "--ant-prefetch-stages" => {
                index += 1;
                options.prefetch_stages =
                    Some(parse_nonnegative_option(args, index, option)? as i64);
            }
            "--ant-allow-vram-swap" => {
                index += 1;
                options.allow_vram_swap = Some(parse_bool_option(args, index, option)?);
            }
            _ => {
                return Err(RunError::usage(format!(
                    "Unknown Ant option '{option}'. Run `cheetah help` for usage."
                )));
            }
        }
        index += 1;
    }
    Ok(options)
}

/// Parses a float with a lower bound `min` (NaN rejected).
fn parse_float_option(
    args: &[String],
    index: usize,
    option: &str,
    min: f64,
) -> Result<f64, RunError> {
    let value = parse_float_raw(args, index, option)?;
    if value < min {
        return Err(RunError::usage(format!(
            "{option} must be at least {min}; received '{value}'."
        )));
    }
    Ok(value)
}

/// Parses a float constrained to the inclusive range `[min, max]`.
fn parse_bounded_float_option(
    args: &[String],
    index: usize,
    option: &str,
    min: f64,
    max: f64,
) -> Result<f64, RunError> {
    // The range check also rejects NaN, which cannot be meaningfully
    // ordered or used as a sampling parameter.
    let value = parse_float_raw(args, index, option)?;
    if !(min..=max).contains(&value) {
        return Err(RunError::usage(format!(
            "{option} must be between {min} and {max}; received '{value}'."
        )));
    }
    Ok(value)
}

/// Parses the next argument as a finite f64, rejecting NaN up front.
fn parse_float_raw(args: &[String], index: usize, option: &str) -> Result<f64, RunError> {
    let value = args
        .get(index)
        .ok_or_else(|| RunError::usage(format!("{option} requires a value.")))?;
    let parsed = value
        .parse::<f64>()
        .map_err(|_| RunError::usage(format!("{option} requires a number; received '{value}'.")))?;
    if parsed.is_nan() {
        return Err(RunError::usage(format!("{option} cannot be NaN.")));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DEFAULT_CHAT_OPTIONS;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_chat_speed_options() {
        let values = args(&[
            "model.safetensors",
            "--threads",
            "8",
            "--max-tokens",
            "64",
            "--ctx-size",
            "16384",
        ]);
        let (path, options, context, _ant, _mcp) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert!(context.chat_options_specified);
        assert_eq!(
            options,
            ChatOptions {
                threads: Some(8),
                max_tokens: 64,
                ctx_size: 16384,
                ..ChatOptions::default()
            }
        );
    }

    #[test]
    fn parses_sampling_and_system_options() {
        let values = args(&[
            "model.safetensors",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "40",
            "--repeat-penalty",
            "1.3",
            "--seed",
            "7",
            "--system",
            "You are a concise assistant.",
        ]);
        let (_, options, _, _, _mcp) = parse_chat_args(&values).unwrap();
        assert_eq!(options.temperature, 0.0);
        assert_eq!(options.top_p, 1.0);
        assert_eq!(options.top_k, 40);
        assert_eq!(options.repeat_penalty, 1.3);
        assert_eq!(options.seed, Some(7));
        assert_eq!(
            options.system.as_deref(),
            Some("You are a concise assistant.")
        );
    }

    #[test]
    fn defaults_are_deterministic() {
        assert_eq!(DEFAULT_CHAT_OPTIONS.ctx_size, 8192);
        assert_eq!(DEFAULT_CHAT_OPTIONS.temperature, 0.0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.top_p, 0.9);
        assert_eq!(DEFAULT_CHAT_OPTIONS.top_k, 0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.repeat_penalty, 1.0);
        assert_eq!(DEFAULT_CHAT_OPTIONS.seed, None);
        assert_eq!(
            ChatOptions::default().system.as_deref(),
            Some("You are a concise assistant. Respond in English.")
        );
    }

    #[test]
    fn rejects_out_of_range_sampling_values() {
        for invalid in [
            &["--temperature", "-0.1"][..],
            &["--top-p", "1.5"][..],
            &["--repeat-penalty", "0.9"][..],
            &["--temperature", "abc"][..],
            &["--top-k", "-1"][..],
            &["--system"][..],
        ] {
            let values = args(&[vec!["model.safetensors"], invalid.to_vec()].concat());
            let error = parse_chat_args(&values).unwrap_err();
            assert!(matches!(error, RunError::Usage(_)), "case: {invalid:?}");
        }
    }

    #[test]
    fn rejects_zero_and_missing_numeric_options() {
        let values = args(&["model.safetensors", "--threads", "0"]);
        assert!(
            parse_chat_args(&values)
                .unwrap_err()
                .to_string()
                .contains("must be greater than 0")
        );
        let values = args(&["model.safetensors", "--threads"]);
        assert!(
            parse_chat_args(&values)
                .unwrap_err()
                .to_string()
                .contains("requires a value")
        );
        assert!(
            parse_chat_args(&args(&["model.safetensors", "--ctx-size", "0"]))
                .unwrap_err()
                .to_string()
                .contains("must be greater than 0")
        );
    }

    #[test]
    fn rejects_missing_model_path() {
        assert!(
            parse_chat_args(&args(&["--max-tokens", "8"]))
                .unwrap_err()
                .to_string()
                .contains("Specify")
        );
    }

    #[test]
    fn parses_ant_flags_into_chat_options() {
        let ant = parse_ant_args(&args(&[
            "--ant",
            "true",
            "--ant-vram",
            "1024",
            "--ant-ram",
            "2048",
        ]))
        .unwrap();
        assert_eq!(ant.enabled, Some(true));
        assert_eq!(ant.vram_pool_bytes, Some(1024));
        assert_eq!(ant.ram_budget_bytes, Some(2048));
        assert!(ant.disk_dir.is_none());
        assert!(ant.resident_units.is_empty());

        let ant = parse_ant_args(&args(&[])).unwrap();
        assert_eq!(ant.enabled, None);
        assert_eq!(ant.vram_pool_bytes, None);
        assert_eq!(ant.ram_budget_bytes, None);
        assert_eq!(ant.disk_dir, None);
        assert!(ant.resident_units.is_empty());
    }

    #[test]
    fn parses_chat_ant_flags() {
        let values = args(&[
            "model.safetensors",
            "--ant",
            "false",
            "--ant-vram",
            "512",
            "--ant-resident",
            "model.embed_tokens.weight",
        ]);
        let (path, _options, _context, ant, _mcp) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert_eq!(ant.enabled, Some(false));
        assert_eq!(ant.vram_pool_bytes, Some(512));
        assert_eq!(
            ant.resident_units,
            vec!["model.embed_tokens.weight".to_string()]
        );
    }

    #[test]
    fn rejects_unknown_ant_flag() {
        let error = parse_ant_args(&args(&["--ant-vram", "1024", "--bogus"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
    }

    #[test]
    fn bare_ant_flag_enables_preset_ant() {
        let ant = parse_ant_args(&args(&["--ant"])).unwrap();
        assert_eq!(ant.enabled, Some(true));
        assert_eq!(ant.interactive, None);

        let values = args(&["model.safetensors", "--ant"]);
        let (path, _options, _context, ant, _mcp) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert_eq!(ant.enabled, Some(true));
    }

    #[test]
    fn ant_flag_with_boolean_value_overrides_preset() {
        let ant = parse_ant_args(&args(&["--ant", "false"])).unwrap();
        assert_eq!(ant.enabled, Some(false));
    }

    #[test]
    fn parses_ant_setup_interactive_flag() {
        let ant = parse_ant_args(&args(&["--ant-setup", "true"])).unwrap();
        assert_eq!(ant.interactive, Some(true));

        let ant = parse_ant_args(&args(&["--ant-setup", "false"])).unwrap();
        assert_eq!(ant.interactive, Some(false));
    }

    #[test]
    fn parses_setup_alias_and_bare_interactive_flags() {
        // `--setup` is the alias of `--ant-setup`; both accept a bare flag.
        let ant = parse_ant_args(&args(&["--setup"])).unwrap();
        assert_eq!(ant.interactive, Some(true));

        let ant = parse_ant_args(&args(&["--setup", "false"])).unwrap();
        assert_eq!(ant.interactive, Some(false));

        let ant = parse_ant_args(&args(&["--ant-setup"])).unwrap();
        assert_eq!(ant.interactive, Some(true));

        // A bare setup flag does not swallow a following model path.
        let (path, _, _, ant, _mcp) = parse_chat_args(&args(&[
            "model.safetensors",
            "--setup",
            "--max-tokens",
            "8",
        ]))
        .unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert_eq!(ant.interactive, Some(true));
    }

    #[test]
    fn reports_usage_errors_with_usage_variant() {
        let error = parse_chat_args(&args(&["--bogus"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
    }

    #[test]
    fn parses_context_options_and_management_commands() {
        let (_, _, context, _, _mcp) = parse_chat_args(&args(&[
            "model.safetensors",
            "--context",
            "work",
            "--context-dir",
            "/tmp/cheetah-contexts",
        ]))
        .unwrap();
        assert_eq!(context.name.as_deref(), Some("work"));
        assert_eq!(
            context.directory,
            Some(PathBuf::from("/tmp/cheetah-contexts"))
        );
        assert!(!context.chat_options_specified);
        let (_, _, context, _, _mcp) =
            parse_chat_args(&args(&["model.safetensors", "--max-tokens", "256"])).unwrap();
        assert!(context.chat_options_specified);

        let (operation, options) = parse_context_args(&args(&[
            "delete",
            "work",
            "--context-dir",
            "/tmp/cheetah-contexts",
        ]))
        .unwrap();
        assert_eq!(operation, ContextCommand::Delete("work".into()));
        assert_eq!(
            options.directory,
            Some(PathBuf::from("/tmp/cheetah-contexts"))
        );
        assert_eq!(
            parse_context_args(&args(&["list"])).unwrap().0,
            ContextCommand::List
        );
    }

    #[test]
    fn parses_mcp_flags_for_chat() {
        let (path, _, _, _, mcp) = parse_chat_args(&args(&[
            "model.safetensors",
            "--mcp-server",
            "fs=npx -y @mcp/fs /tmp",
            "--mcp-config",
            "/tmp/mcp.json",
        ]))
        .unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert_eq!(mcp.servers, vec!["fs=npx -y @mcp/fs /tmp".to_owned()]);
        assert_eq!(mcp.config, Some(PathBuf::from("/tmp/mcp.json")));
    }

    #[test]
    fn rejects_repeated_mcp_config() {
        let error = parse_chat_args(&args(&[
            "model.safetensors",
            "--mcp-config",
            "/tmp/a.json",
            "--mcp-config",
            "/tmp/b.json",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("--mcp-config"));
    }

    #[test]
    fn parses_mcp_flags_for_serve() {
        let (serve, _, mcp) = parse_serve_args(&args(&[
            "model.safetensors",
            "--mcp-server",
            "tools=./server.sh --flag",
        ]))
        .unwrap();
        assert_eq!(serve.model, "model.safetensors");
        assert_eq!(mcp.servers, vec!["tools=./server.sh --flag".to_owned()]);
        assert!(mcp.config.is_none());
    }

    #[test]
    fn parses_serve_defaults() {
        let (serve, options, _mcp) = parse_serve_args(&args(&["qwen2.5-0.5b"])).unwrap();
        assert_eq!(serve.model, "qwen2.5-0.5b");
        assert_eq!(serve.model_dir, PathBuf::from("."));
        assert_eq!(serve.host, "127.0.0.1");
        assert_eq!(serve.port, 8080);
        assert_eq!(serve.max_concurrent_generations, 1);
        assert!(!serve.generate_key);
        assert_eq!(serve.api_key, None);
        assert_eq!(options, ChatOptions::default());
    }

    #[test]
    fn parses_serve_options_and_generation_overrides() {
        let (serve, options, _mcp) = parse_serve_args(&args(&[
            "--model",
            "qwen2.5-0.5b",
            "--model-dir",
            "./models",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--max-concurrent-generations",
            "3",
            "--api-key",
            "cheetah-secret",
            "--temperature",
            "0.7",
            "--max-tokens",
            "512",
            "--ctx-size",
            "16384",
        ]))
        .unwrap();
        assert_eq!(serve.model_dir, PathBuf::from("./models"));
        assert_eq!(serve.host, "0.0.0.0");
        assert_eq!(serve.port, 9000);
        assert_eq!(serve.max_concurrent_generations, 3);
        assert_eq!(serve.api_key.as_deref(), Some("cheetah-secret"));
        assert!(!serve.generate_key);
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.max_tokens, 512);
        assert_eq!(options.ctx_size, 16384);
    }

    #[test]
    fn accepts_ephemeral_port_and_generate_key() {
        let (serve, _, _mcp) = parse_serve_args(&args(&[
            "--model",
            "m.safetensors",
            "--port",
            "0",
            "--generate-key",
        ]))
        .unwrap();
        assert_eq!(serve.port, 0);
        assert!(serve.generate_key);
    }

    #[test]
    fn serve_rejects_mutually_exclusive_keys() {
        let error = parse_serve_args(&args(&[
            "--model",
            "m.safetensors",
            "--generate-key",
            "--api-key",
            "cheetah-secret",
        ]))
        .unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
        assert!(error.to_string().contains("cannot be used together"));
    }

    #[test]
    fn serve_rejects_missing_model_and_invalid_port() {
        assert!(matches!(
            parse_serve_args(&args(&["--port", "8080"])).unwrap_err(),
            RunError::Usage(_)
        ));
        let error =
            parse_serve_args(&args(&["--model", "m.safetensors", "--port", "70000"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
        assert!(error.to_string().contains("0 and 65535"));
    }

    #[test]
    fn serve_rejects_invalid_or_repeated_generation_limit() {
        for values in [
            args(&["m.safetensors", "--max-concurrent-generations", "0"]),
            args(&[
                "m.safetensors",
                "--max-concurrent-generations",
                "2",
                "--max-concurrent-generations",
                "3",
            ]),
        ] {
            assert!(matches!(
                parse_serve_args(&values).unwrap_err(),
                RunError::Usage(_)
            ));
        }
    }

    #[test]
    fn parses_positional_serve_model_and_options() {
        let (serve, options, _mcp) = parse_serve_args(&args(&[
            "m.safetensors",
            "--model-dir",
            "./models",
            "--port",
            "9000",
        ]))
        .unwrap();
        assert_eq!(serve.model, "m.safetensors");
        assert_eq!(serve.model_dir, PathBuf::from("./models"));
        assert_eq!(serve.port, 9000);
        assert_eq!(options, ChatOptions::default());
    }

    #[test]
    fn serve_rejects_multiple_model_arguments() {
        for values in [
            args(&["m.safetensors", "other.safetensors"]),
            args(&["m.safetensors", "--model", "other.safetensors"]),
            args(&["--model", "m.safetensors", "other.safetensors"]),
        ] {
            let error = parse_serve_args(&values).unwrap_err();
            assert!(matches!(error, RunError::Usage(_)));
            assert!(error.to_string().contains("model"));
        }
    }

    #[test]
    fn parses_inspect_output_options() {
        let options = parse_inspect_args(&args(&[
            "model.safetensors",
            "--all",
            "--json",
            "-o",
            "report.json",
        ]))
        .unwrap();
        assert_eq!(options.path, PathBuf::from("model.safetensors"));
        assert!(options.show_all);
        assert!(options.json);
        assert_eq!(options.output, Some(PathBuf::from("report.json")));
    }

    #[test]
    fn inspect_rejects_missing_or_repeated_output_path() {
        for values in [
            args(&["model.safetensors", "--output"]),
            args(&["model.safetensors", "-o"]),
            args(&["model.safetensors", "-o", "one.txt", "--output", "two.txt"]),
        ] {
            assert!(matches!(
                parse_inspect_args(&values).unwrap_err(),
                RunError::Usage(_)
            ));
        }
    }
}
