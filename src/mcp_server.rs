//! Cheetah as an MCP server: exposes the loaded model to MCP agents over
//! stdio (newline-delimited JSON-RPC 2.0). Any MCP client — Claude Desktop,
//! Cursor, or a scripted peer — spawns `cheetah mcp <model-path>` and calls
//! the `chat_generate` tool to run generation through the model's chat
//! template.
//!
//! Protocol channel discipline: **stdout carries protocol frames only.**
//! The engine is loaded through `Engine::load` directly (the CLI wrapper's
//! banners and spinner go elsewhere or are skipped), and every diagnostic
//! here uses stderr.

use crate::chat::{ChatMessage, Engine, TextSink};
use crate::options::ChatOptions;
use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;
use std::io::{BufRead, StdoutLock, Write};
use std::path::Path;

/// Tools exposed to MCP agents.
const CHAT_GENERATE_TOOL: &str = "chat_generate";
const MODEL_INFO_TOOL: &str = "model_info";

/// Per-call state: the loaded engine plus the session's generation defaults.
pub(crate) struct McpServerState {
    engine: Engine,
    options: ChatOptions,
}

/// Run the MCP stdio server: load the engine, then serve JSON-RPC requests
/// from stdin until the stream closes (the agent exiting = shutdown).
/// Responses are written to stdout, one per line, flushed immediately.
pub(crate) fn run(
    path: &Path,
    options: &ChatOptions,
    ant: Option<&crate::ant::AntSettings>,
) -> Result<(), String> {
    let engine = Engine::load(path, options.threads, options.ctx_size, ant)?;
    let mut state = McpServerState {
        engine,
        options: options.clone(),
    };

    let stdin = std::io::stdin();
    let mut stdout = stdout_lock();
    for line in stdin.lock().lines() {
        let line = line.map_err(|error| format!("Failed to read MCP request: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(message) => message,
            Err(error) => {
                // Malformed JSON: answer with a parse error (id null) and
                // keep serving — a chatty client must not kill the session.
                let response = json!({
                    "jsonrpc": "2.0", "id": Value::Null,
                    "error": {"code": -32700, "message": format!("Parse error: {error}")},
                });
                write_response(&mut stdout, &response)?;
                continue;
            }
        };
        if let Some(response) = handle_request(&message, &mut state) {
            write_response(&mut stdout, &response)?;
        }
    }
    Ok(())
}

fn write_response(stdout: &mut StdoutLock<'static>, response: &Value) -> Result<(), String> {
    let line = serde_json::to_string(response)
        .map_err(|error| format!("Failed to serialize MCP response: {error}"))?;
    writeln!(stdout, "{line}")
        .and_then(|()| stdout.flush())
        .map_err(|error| format!("Failed to write MCP response: {error}"))
}

/// Protocol dispatch: returns the JSON-RPC response for a request, or `None`
/// for notifications (no id → nothing to answer). The tool-invoking arm
/// delegates to [`McpServerState::invoke_tool`].
/// The tool-invoking surface of the server state — split out so protocol
/// tests can stub generation without loading a model.
trait McpServerStateLike {
    fn invoke_tool(&mut self, params: &Value) -> Result<Value, Value>;
}

fn handle_request<S: McpServerStateLike>(message: &Value, state: &mut S) -> Option<Value> {
    let id = message.get("id").cloned()?;
    let method = message.get("method").and_then(Value::as_str)?;
    let params = message.get("params").cloned().unwrap_or(Value::Null);

    let result = match method {
        "initialize" => Ok(initialize_result()),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tools_list_result()),
        "tools/call" => state.invoke_tool(&params),
        other => Err(json!({
            "code": -32601,
            "message": format!("Method not found: {other}"),
        })),
    };

    Some(match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err(error) => json!({"jsonrpc": "2.0", "id": id, "error": error}),
    })
}

/// The MCP `initialize` result: protocol version, tool capability, and the
/// server identity (name + crate version).
fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "cheetah", "version": env!("CARGO_PKG_VERSION")},
    })
}

/// The server's tool catalog: one tool per entry, with JSON Schema for the
/// arguments the agent may pass.
fn tools_list_result() -> Value {
    json!({"tools": [
        {
            "name": CHAT_GENERATE_TOOL,
            "description": "Generate a response with the loaded local model. Pass the conversation as messages; the model's chat template is applied automatically. Returns the assistant text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The conversation so far, oldest first.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "description": "system / user / assistant"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "max_tokens": {"type": "integer", "description": "Generation budget for this call (default: server-configured)."},
                    "temperature": {"type": "number", "description": "Sampling temperature; 0 = greedy."},
                },
                "required": ["messages"],
            },
        },
        {
            "name": MODEL_INFO_TOOL,
            "description": "Returns the loaded model's id, architecture, and device.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]})
}

impl McpServerState {
    /// Render the conversation through the model's chat template, generate,
    /// and return the assistant text.
    fn run_generation(
        &mut self,
        messages: Vec<ChatMessage>,
        options: &ChatOptions,
    ) -> Result<String, String> {
        let (tokens, _) = self
            .engine
            .prepare(&mut messages.clone(), None, options.max_tokens)?;
        let mut context = self.engine.new_context()?;
        let mut sink = DiscardSink;
        let generation = self.engine.generate(
            &mut context,
            &tokens,
            options,
            &[],
            &mut sink,
            &HashSet::new(),
        )?;
        Ok(generation.text)
    }
}

impl McpServerStateLike for McpServerState {
    /// Tool-call dispatch: parse the arguments, execute, and return the MCP
    /// result. Per the MCP contract, execution failures surface as
    /// `isError: true` results (the agent sees them); only argument/
    /// routing problems are JSON-RPC errors.
    fn invoke_tool(&mut self, params: &Value) -> Result<Value, Value> {
        let name = params.get("name").and_then(Value::as_str).ok_or_else(
            || json!({"code": -32602, "message": "tools/call is missing a tool name."}),
        )?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            CHAT_GENERATE_TOOL => {
                let messages = parse_tool_messages(&arguments)
                    .map_err(|message| json!({"code": -32602, "message": message}))?;
                let mut options = self.options.clone();
                if let Some(max_tokens) = arguments.get("max_tokens").and_then(Value::as_u64) {
                    options.max_tokens = usize::try_from(max_tokens).map_err(
                        |_| json!({"code": -32602, "message": "max_tokens does not fit usize."}),
                    )?;
                }
                if let Some(temperature) = arguments.get("temperature").and_then(Value::as_f64) {
                    if !(0.0..=2.0).contains(&temperature) {
                        return Err(
                            json!({"code": -32602, "message": "temperature must be between 0 and 2."}),
                        );
                    }
                    options.temperature = temperature;
                }
                let text = self
                    .run_generation(messages, &options)
                    .map_err(|message| json!({"code": -32602, "message": message}))?;
                Ok(json!({
                    "content": [{"type": "text", "text": text}],
                    "isError": false,
                }))
            }
            MODEL_INFO_TOOL => {
                let info = json!({
                    "model_id": self.engine.model_id(),
                    "architecture": self.engine.model_architecture(),
                    "device": self.engine.device_name(),
                });
                Ok(json!({
                    "content": [{"type": "text", "text": info.to_string()}],
                    "isError": false,
                }))
            }
            other => Err(json!({
                "code": -32602,
                "message": format!("Unknown tool: {other}"),
            })),
        }
    }
}

/// Discards streamed deltas — the MCP tool result carries `generation.text`
/// (captured because [`TextSink::capture_text`] defaults to true).
struct DiscardSink;

impl TextSink for DiscardSink {
    fn push(&mut self, _text: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Convert the tool arguments' `messages` array into engine chat messages.
fn parse_tool_messages(arguments: &Value) -> Result<Vec<ChatMessage>, String> {
    let entries = arguments
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "chat_generate requires a messages array.".to_string())?;
    if entries.is_empty() {
        return Err("messages must contain at least one message.".to_string());
    }
    entries
        .iter()
        .map(|entry| {
            let role = entry
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| "Each message requires a string role.".to_string())?;
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| "Each message requires a string content.".to_string())?;
            Ok(ChatMessage::new(role, content))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool invoker stub for protocol tests: records the invoked tool and
    /// returns a canned result.
    struct StubInvoker {
        result: Value,
        invoked: std::cell::RefCell<Vec<String>>,
    }

    impl StubInvoker {
        fn ok(text: &str) -> Self {
            Self {
                result: json!({"content": [{"type": "text", "text": text}], "isError": false}),
                invoked: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    impl McpServerStateLike for StubInvoker {
        fn invoke_tool(&mut self, params: &Value) -> Result<Value, Value> {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            self.invoked.borrow_mut().push(name.clone());
            // Mirror the real dispatch: unknown tools are protocol errors.
            if name != CHAT_GENERATE_TOOL && name != MODEL_INFO_TOOL {
                return Err(json!({"code": -32602, "message": format!("Unknown tool: {name}")}));
            }
            Ok(self.result.clone())
        }
    }

    fn request(id: i64, method: &str, params: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
    }

    /// The initialize result carries the protocol version, the tools
    /// capability, and the cheetah server identity.
    #[test]
    fn initialize_returns_protocol_version_and_capability() {
        let mut stub = StubInvoker::ok("");
        let response = handle_request(&request(1, "initialize", json!({})), &mut stub).unwrap();
        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["protocolVersion"], "2024-11-05");
        assert!(response["result"]["capabilities"]["tools"].is_object());
        assert_eq!(response["result"]["serverInfo"]["name"], "cheetah");
        assert!(stub.invoked.borrow().is_empty());
    }

    /// tools/list catalogs both tools with their JSON Schema.
    #[test]
    fn tools_list_catalogs_chat_generate_and_model_info() {
        let mut stub = StubInvoker::ok("");
        let response = handle_request(&request(2, "tools/list", json!({})), &mut stub).unwrap();
        let tools = response["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, [CHAT_GENERATE_TOOL, MODEL_INFO_TOOL]);
        let schema = &tools[0]["inputSchema"];
        assert_eq!(schema["properties"]["messages"]["type"], "array");
        assert!(stub.invoked.borrow().is_empty());
    }

    /// tools/call routes to the invoker with the parsed arguments.
    #[test]
    fn tools_call_routes_to_invoker() {
        let mut stub = StubInvoker::ok("12:00");
        let params = json!({"name": CHAT_GENERATE_TOOL, "arguments": {
            "messages": [{"role": "user", "content": "time?"}],
        }});
        let response = handle_request(&request(3, "tools/call", params), &mut stub).unwrap();
        assert_eq!(response["result"]["content"][0]["text"], "12:00");
        assert_eq!(response["result"]["isError"], false);
        assert_eq!(*stub.invoked.borrow(), vec![CHAT_GENERATE_TOOL.to_owned()]);
    }

    /// An unknown tool is a protocol error (-32602), not an execution result.
    #[test]
    fn unknown_tool_is_a_protocol_error() {
        let mut stub = StubInvoker::ok("");
        let params = json!({"name": "no_such_tool", "arguments": {}});
        let response = handle_request(&request(4, "tools/call", params), &mut stub).unwrap();
        assert_eq!(response["error"]["code"], -32602);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no_such_tool")
        );
    }

    /// Unknown methods are JSON-RPC errors; notifications (no id) return
    /// nothing; ping answers with an empty result.
    #[test]
    fn protocol_level_cases() {
        let mut stub = StubInvoker::ok("");
        let unknown = handle_request(&request(5, "resources/list", json!({})), &mut stub).unwrap();
        assert_eq!(unknown["error"]["code"], -32601);

        let notification = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        assert!(handle_request(&notification, &mut stub).is_none());

        let ping = handle_request(&request(6, "ping", json!({})), &mut stub).unwrap();
        assert_eq!(ping["result"], json!({}));
    }

    /// chat_generate argument validation: a missing/empty messages array is
    /// rejected before any generation is attempted.
    #[test]
    fn chat_generate_rejects_invalid_messages() {
        assert!(
            parse_tool_messages(&json!({}))
                .err()
                .unwrap()
                .contains("messages array")
        );
        assert!(
            parse_tool_messages(&json!({"messages": []}))
                .err()
                .unwrap()
                .contains("at least one message")
        );
        assert!(
            parse_tool_messages(&json!({"messages": [
                {"role": "user"}
            ]}))
            .err()
            .unwrap()
            .contains("string content")
        );
        let ok = parse_tool_messages(&json!({"messages": [
            {"role": "user", "content": "hi"}
        ]}))
        .unwrap();
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].role, "user");
    }
}

/// Hold the process-wide stdout lock for the whole session (the MCP loop is
/// the only writer; a scoped lock would fight the `stdin.lock()` lifetime).
fn stdout_lock() -> StdoutLock<'static> {
    std::io::stdout().lock()
}
