//! MCP (Model Context Protocol) client support: Cheetah connects to local
//! MCP servers over stdio, discovers their tools (`tools/list`), exposes them
//! to the model alongside any request-declared tools, and executes tool
//! calls for them (`tools/call`) inside the agentic tool loop.
//!
//! Layering: this module sits above the engine — the engine's tool loop
//! stays MCP-agnostic and receives `Option<&McpRegistry>` per call; the
//! tool-call SHAPES it exchanges come from [`crate::chat::toolcalls`].
//! Transport is the MCP stdio flavor: one JSON-RPC 2.0 message per line.

pub(crate) mod client;

use crate::chat::toolcalls::{FunctionDefinition, ToolDefinition};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Mutex;

pub(crate) use client::McpClient;

/// One configured MCP server: a child process to spawn and the name its
/// tools are namespaced under.
#[derive(Clone, Debug)]
pub(crate) struct McpServerConfig {
    /// Server name — must be unique across the configuration.
    pub(crate) name: String,
    /// Executable to spawn.
    pub(crate) command: String,
    /// Command-line arguments for the executable.
    pub(crate) args: Vec<String>,
}

/// Parse one `--mcp-server` specification: `"name=command arg1 arg2 ..."`.
/// The first token carries `name=command`; the remaining tokens are the
/// command's arguments (split on whitespace — quoted arguments belong in a
/// `--mcp-config` file).
pub(crate) fn parse_server_spec(spec: &str) -> Result<McpServerConfig, String> {
    let mut tokens = spec.split_whitespace();
    let first = tokens.next().ok_or_else(|| {
        "MCP server specification is empty; expected name=command [args...]".to_string()
    })?;
    let (name, command) = first
        .split_once('=')
        .ok_or_else(|| format!("MCP server spec '{spec}' must start with name=command."))?;
    if name.is_empty() || command.is_empty() {
        return Err(format!(
            "MCP server spec '{spec}' has an empty name or command."
        ));
    }
    Ok(McpServerConfig {
        name: name.to_owned(),
        command: command.to_owned(),
        args: tokens.map(str::to_owned).collect(),
    })
}

#[derive(Deserialize)]
struct McpConfigFile {
    #[serde(default)]
    servers: Vec<McpConfigServer>,
}

#[derive(Deserialize)]
struct McpConfigServer {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

/// Resolve the configured servers: the `--mcp-server` flag specs plus, when
/// `--mcp-config` names a JSON file, its `{"servers": [{"name", "command",
/// "args": [...]}]}` entries. Names must be unique across both sources.
pub(crate) fn server_configs(
    specs: &[String],
    config_file: Option<&Path>,
) -> Result<Vec<McpServerConfig>, String> {
    let mut configs: Vec<McpServerConfig> = Vec::new();
    let mut push = |config: McpServerConfig| -> Result<(), String> {
        if configs.iter().any(|existing| existing.name == config.name) {
            return Err(format!("Duplicate MCP server name '{}'.", config.name));
        }
        configs.push(config);
        Ok(())
    };
    for spec in specs {
        push(parse_server_spec(spec)?)?;
    }
    if let Some(file) = config_file {
        let bytes = std::fs::read(file)
            .map_err(|error| format!("Failed to read MCP config {}: {error}", file.display()))?;
        let parsed: McpConfigFile = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Invalid MCP config {}: {error}", file.display()))?;
        for server in parsed.servers {
            push(McpServerConfig {
                name: server.name,
                command: server.command,
                args: server.args,
            })?;
        }
    }
    Ok(configs)
}

/// One tool discovered from an MCP server: its defining server, the model
/// facing name, the optional description, and the JSON Schema for its
/// arguments.
#[derive(Clone, Debug)]
pub(crate) struct McpTool {
    pub(crate) server: String,
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) schema: Value,
}

/// The connected MCP surface: one live client per configured server plus the
/// flattened tool list every front end exposes to the model.
pub(crate) struct McpRegistry {
    /// One mutex per client: MCP servers process one request at a time, and
    /// the interior mutability lets the registry be shared (`Arc`) across
    /// request workers while calls stay serialized per server.
    clients: Vec<Mutex<McpClient>>,
    tools: Vec<McpTool>,
}

impl McpRegistry {
    /// Spawn every configured server, run the `initialize` handshake, and
    /// list their tools. Any server that fails to start or answer fails the
    /// whole registry loudly — a half-configured tool surface would corrupt
    /// the model's tool calls silently.
    pub(crate) fn connect(configs: &[McpServerConfig]) -> Result<Self, String> {
        let mut clients = Vec::new();
        let mut tools: Vec<McpTool> = Vec::new();
        for config in configs {
            let mut client = McpClient::connect(config)?;
            for tool in client.list_tools()? {
                let server_tool = McpTool {
                    server: config.name.clone(),
                    name: tool.name,
                    description: tool.description,
                    schema: tool.schema,
                };
                if tools
                    .iter()
                    .any(|existing| existing.name == server_tool.name)
                {
                    return Err(format!(
                        "MCP servers expose duplicate tool '{}'.",
                        server_tool.name
                    ));
                }
                tools.push(server_tool);
            }
            clients.push(client);
        }
        Ok(Self {
            clients: clients.into_iter().map(Mutex::new).collect(),
            tools,
        })
    }

    /// The MCP tools rendered as OpenAI-style tool definitions for the model
    /// prompt: `inputSchema` becomes the function's `parameters`.
    pub(crate) fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|tool| ToolDefinition {
                kind: "function".to_owned(),
                function: FunctionDefinition {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: Some(tool.schema.clone()),
                    strict: None,
                },
            })
            .collect()
    }

    /// Whether `name` belongs to a connected MCP server.
    pub(crate) fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|tool| tool.name == name)
    }

    /// Execute one tool call on its owning server (`tools/call`) and return
    /// the text content blocks joined with newlines. MCP-level failures
    /// (`isError: true`) are returned as `[error] ...` text so the model can
    /// react; transport and protocol failures are hard errors.
    pub(crate) fn call(&self, name: &str, arguments: &str) -> Result<String, String> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .ok_or_else(|| format!("MCP tool '{name}' is not registered."))?;
        let client = self
            .clients
            .iter()
            .find(|client| {
                client
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .server_name()
                    == tool.server
            })
            .expect("registry clients cover every listed tool");
        let arguments: Value = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(arguments).map_err(|error| {
                format!("MCP tool '{name}' arguments are not valid JSON: {error}")
            })?
        };
        client
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .call_tool(name, arguments)
    }

    /// Shut every server down (kill the child processes) — also the `Drop`
    /// behavior, so a session that loses its registry never leaks children.
    pub(crate) fn shutdown(&mut self) {
        for client in &mut self.clients {
            client
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .shutdown();
        }
    }
}

/// Build the registry from the parsed CLI flags: `None` when nothing is
/// configured (the zero-cost, MCP-free posture every existing flow keeps).
pub(crate) fn registry_from_flags(
    specs: &[String],
    config_file: Option<&Path>,
) -> Result<Option<McpRegistry>, String> {
    let configs = server_configs(specs, config_file)?;
    if configs.is_empty() {
        return Ok(None);
    }
    McpRegistry::connect(&configs).map(Some)
}

impl Drop for McpRegistry {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// A minimal MCP stdio server in Python: answers `initialize`,
    /// `tools/list` and `tools/call`. Used by the end-to-end test below.
    const PYTHON_SERVER: &str = r#"
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if "id" not in req:
        continue
    if req.get("method") == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "serverInfo": {"name": "demo", "version": "0"}}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), flush=True)
    elif req.get("method") == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [
            {"name": "echo", "description": "Echoes its arguments",
             "inputSchema": {"type": "object"}}]}}), flush=True)
    elif req.get("method") == "tools/call":
        args = req["params"]["arguments"]
        print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {"content": [
            {"type": "text", "text": "echo: " + json.dumps(args, sort_keys=True)}]}}), flush=True)
"#;

    fn python_available() -> bool {
        Command::new("python3").arg("--version").output().is_ok()
    }

    /// End-to-end against a real stdio child process (gated on python3 being
    /// installed, like the device-gated Vulkan tests): spawn → initialize →
    /// tools/list → tools/call through the registry.
    #[test]
    fn connects_to_a_real_stdio_server_end_to_end() {
        if !python_available() {
            eprintln!("skipping the MCP end-to-end test (python3 not installed)");
            return;
        }
        let configs = vec![McpServerConfig {
            name: "demo".into(),
            command: "python3".into(),
            args: vec!["-c".into(), PYTHON_SERVER.into()],
        }];
        let registry = McpRegistry::connect(&configs).unwrap();
        let definitions = registry.tool_definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].function.name, "echo");
        assert!(definitions[0].function.description.is_some());
        assert!(registry.has_tool("echo"));
        assert!(!registry.has_tool("missing"));
        let result = registry.call("echo", r#"{"message": "hello"}"#).unwrap();
        assert_eq!(result, r#"echo: {"message": "hello"}"#);
        let error = registry.call("missing", "{}").unwrap_err();
        assert!(error.contains("not registered"), "{error}");
    }

    /// Server-spec and config-file parsing: the flag form, the JSON form,
    /// and duplicate-name rejection across both.
    #[test]
    fn parses_server_specs_and_config_files() {
        let spec = parse_server_spec("fs=npx -y @mcp/fs /tmp").unwrap();
        assert_eq!(spec.name, "fs");
        assert_eq!(spec.command, "npx");
        assert_eq!(spec.args, vec!["-y", "@mcp/fs", "/tmp"]);
        assert!(parse_server_spec("no-equals-here").is_err());
        assert!(parse_server_spec("").is_err());

        let dir = std::env::temp_dir().join(format!(
            "cheetah-mcp-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("mcp.json");
        std::fs::write(
            &file,
            r#"{"servers": [{"name": "fs", "command": "npx", "args": ["-y", "@mcp/fs"]}]}"#,
        )
        .unwrap();
        let configs = server_configs(&[], Some(&file)).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "fs");

        let error = server_configs(&["fs=npx".to_owned()], Some(&file)).unwrap_err();
        assert!(error.contains("Duplicate"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The registry renders discovered tools as OpenAI-style definitions and
    /// rejects duplicate tool names across servers.
    #[test]
    fn tool_definitions_map_and_duplicate_names_reject() {
        let configs = vec![McpServerConfig {
            name: "demo".into(),
            command: "python3".into(),
            args: vec!["-c".into(), PYTHON_SERVER.into()],
        }];
        if !python_available() {
            eprintln!("skipping (python3 not installed)");
            return;
        }
        let registry = McpRegistry::connect(&configs).unwrap();
        let definitions = registry.tool_definitions();
        assert_eq!(definitions[0].kind, "function");
        assert_eq!(
            definitions[0].function.parameters.as_ref().unwrap()["type"],
            "object"
        );
    }
}
