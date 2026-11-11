//! Command-line argument parsing and runtime options.

use crate::ant::AntSettings;
use crate::options::ChatOptions;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextCliOptions {
    pub name: Option<String>,
    pub directory: Option<PathBuf>,
    pub chat_options_specified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommand {
    List,
    Delete(String),
}

/// Cheetah serve settings for model selection, listener, and authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeOptions {
    /// Model name, model directory, or Safetensors weight path.
    pub model: String,
    /// 解析模型名称时搜索的目录。
    pub model_dir: PathBuf,
    pub host: String,
    /// 0 表示由系统分配空闲端口。
    pub port: u16,
    /// 同时执行模型推理的生成请求上限。
    pub max_concurrent_generations: usize,
    pub generate_key: bool,
    pub api_key: Option<String>,
}

impl Default for ServeOptions {
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

/// 运行期错误:参数/用法错误以退出码 2 报告,其余运行时错误以退出码 1 报告。
#[derive(Debug)]
pub enum RunError {
    Usage(String),
    Runtime(String),
}

impl RunError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Usage(_) => ExitCode::from(2),
            Self::Runtime(_) => ExitCode::FAILURE,
        }
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(message) | Self::Runtime(message) => formatter.write_str(message),
        }
    }
}

pub fn parse_chat_args(
    args: &[String],
) -> Result<(PathBuf, ChatOptions, ContextCliOptions, AntCliOptions), RunError> {
    let mut path = None;
    let mut options = ChatOptions::default();
    let mut context = ContextCliOptions::default();
    let mut ant = AntCliOptions::default();
    let mut index = 0;
    // Track explicit inference flags separately so restored archives can distinguish defaults from overrides.
    // 单独记录用户显式传入的推理参数，以便恢复归档时区分默认值与命令行覆盖值。
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
            "--ant-setup" => {
                index += 1;
                ant.interactive = Some(parse_bool_option(args, index, "--ant-setup")?);
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
                ant.resident_units.push(parse_text_option(args, index, "--ant-resident")?);
            }
            "--ant-eviction-policy" => {
                index += 1;
                ant.eviction_policy = Some(parse_text_option(args, index, "--ant-eviction-policy")?);
            }
            "--ant-prefetch-stages" => {
                index += 1;
                ant.prefetch_stages = Some(parse_nonnegative_option(args, index, "--ant-prefetch-stages")? as i64);
            }
            "--ant-allow-vram-swap" => {
                index += 1;
                ant.allow_vram_swap = Some(parse_bool_option(args, index, "--ant-allow-vram-swap")?);
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
    Ok((path, options, context, ant))
}

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

pub fn parse_serve_args(args: &[String]) -> Result<(ServeOptions, ChatOptions), RunError> {
    let mut serve = ServeOptions::default();
    let mut options = ChatOptions::default();
    let mut max_concurrent_generations_specified = false;
    let mut index = 0;
    // Serve accepts both a positional model and the legacy --model flag, but never both.
    // 服务命令同时兼容位置模型参数和旧式 --model 标志，但不允许二者同时出现。
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
    Ok((serve, options))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectOptions {
    pub path: PathBuf,
    pub json: bool,
    pub show_all: bool,
    pub output: Option<PathBuf>,
}

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

fn parse_text_option(args: &[String], index: usize, option: &str) -> Result<String, RunError> {
    args.get(index).cloned().ok_or_else(|| {
        RunError::usage(format!(
            "{option} requires a value, e.g. --system \"You are a concise assistant.\"."
        ))
    })
}

/// 解析布尔开关值,接受 true/false/1/0/yes/no/on/off。
fn is_boolean_literal(value: &str) -> bool {
    [
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
        "on",
        "off",
    ]
    .iter()
    .any(|candidate| candidate.eq_ignore_ascii_case(value))
}

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

/// Ant 内存优化的命令行覆盖项;`None` 表示未指定，交由用户配置/默认值处理。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AntCliOptions {
    pub enabled: Option<bool>,
    pub vram_pool_bytes: Option<u64>,
    pub ram_budget_bytes: Option<u64>,
    pub disk_dir: Option<String>,
    pub resident_units: Vec<String>,
    pub eviction_policy: Option<String>,
    pub prefetch_stages: Option<i64>,
    pub allow_vram_swap: Option<bool>,
    /// 交互式设置Ant加载策略;`None` 表示未指定。
    pub interactive: Option<bool>,
}

impl AntCliOptions {
    /// 合并为完整的 AntSettings(未指定项回落到默认值)。
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

/// 解析 Ant 内存优化参数:`--ant <bool>` 以及预算/驻留类覆盖项。
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
            "--ant-setup" => {
                index += 1;
                options.interactive = Some(parse_bool_option(args, index, option)?);
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
                options.resident_units.push(parse_text_option(args, index, option)?);
            }
            "--ant-eviction-policy" => {
                index += 1;
                options.eviction_policy = Some(parse_text_option(args, index, option)?);
            }
            "--ant-prefetch-stages" => {
                index += 1;
                options.prefetch_stages = Some(parse_nonnegative_option(args, index, option)? as i64);
            }
            "--ant-allow-vram-swap" => {
                index += 1;
                options.allow_vram_swap = Some(parse_bool_option(args, index, option)?);
            }
            _ => {
                return Err(RunError::usage(format!(
                    "Unknown Ant option '{option}'. Run `cheetah help` for usage."
                )))
            }
        }
        index += 1;
    }
    Ok(options)
}

/// 解析下界为 `min` 的浮点数(拒绝 NaN)。
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

/// 解析介于 `min` 与 `max` 之间(含边界)的浮点数。
fn parse_bounded_float_option(
    args: &[String],
    index: usize,
    option: &str,
    min: f64,
    max: f64,
) -> Result<f64, RunError> {
    // Range::contains also rejects NaN, which cannot be meaningfully ordered or used for sampling.
    // 范围检查也会拒绝 NaN，因为它既无法可靠比较，也不适合作为采样参数。
    let value = parse_float_raw(args, index, option)?;
    if !(min..=max).contains(&value) {
        return Err(RunError::usage(format!(
            "{option} must be between {min} and {max}; received '{value}'."
        )));
    }
    Ok(value)
}

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
        let (path, options, context, _ant) = parse_chat_args(&values).unwrap();
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
        let (_, options, _, _) = parse_chat_args(&values).unwrap();
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
        let ant = parse_ant_args(&args(&["--ant", "true", "--ant-vram", "1024", "--ant-ram", "2048"]))
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
        let (path, _options, _context, ant) = parse_chat_args(&values).unwrap();
        assert_eq!(path, PathBuf::from("model.safetensors"));
        assert_eq!(ant.enabled, Some(false));
        assert_eq!(ant.vram_pool_bytes, Some(512));
        assert_eq!(ant.resident_units, vec!["model.embed_tokens.weight".to_string()]);
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
        let (path, _options, _context, ant) = parse_chat_args(&values).unwrap();
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
    fn reports_usage_errors_with_usage_variant() {
        let error = parse_chat_args(&args(&["--bogus"])).unwrap_err();
        assert!(matches!(error, RunError::Usage(_)));
    }

    #[test]
    fn parses_context_options_and_management_commands() {
        let (_, _, context, _) = parse_chat_args(&args(&[
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
        let (_, _, context, _) =
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
    fn parses_serve_defaults() {
        let (serve, options) = parse_serve_args(&args(&["qwen2.5-0.5b"])).unwrap();
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
        let (serve, options) = parse_serve_args(&args(&[
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
        let (serve, _) = parse_serve_args(&args(&[
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
        let (serve, options) = parse_serve_args(&args(&[
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
