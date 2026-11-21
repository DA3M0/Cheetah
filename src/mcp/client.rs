//! The MCP stdio client: one child process per server, newline-delimited
//! JSON-RPC 2.0 framing, id-correlated request/response matching with a
//! deadline, and the three protocol operations Cheetah uses (`initialize`,
//! `tools/list`, `tools/call`).
//!
//! The transport is a trait so tests can drive the protocol against a
//! scripted peer instead of a real child process.

use super::McpServerConfig;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// Per-request deadline; a server that cannot answer inside it fails the
/// operation (and the registry build, when it happens during startup).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// One direction of the newline-delimited transport.
pub(crate) trait LineTransport: Send {
    /// Write one JSON line (newline + flush included).
    fn send_line(&mut self, line: &str) -> Result<(), String>;
    /// Downcast support (tests inspect the scripted transport's writes).
    #[cfg(test)]
    fn as_any(&mut self) -> &mut dyn std::any::Any;
    /// Read one line, waiting up to `timeout`; `None` = the peer closed its
    /// stdout (EOF).
    fn recv_line(&mut self, timeout: Duration) -> Result<Option<String>, String>;
}

/// The real transport: the child's piped stdin plus a reader thread pumping
/// its stdout into a channel (a blocking read on the child's stdout must
/// never block the engine thread past the request deadline).
pub(crate) struct ChildTransport {
    stdin: ChildStdin,
    lines: Receiver<String>,
    child: Child,
}

impl ChildTransport {
    /// Spawn the MCP server with piped stdio (stderr inherits so server
    /// diagnostics stay visible).
    pub(crate) fn spawn(config: &McpServerConfig) -> Result<Self, String> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                format!(
                    "Failed to spawn MCP server '{}' ({}): {error}",
                    config.name, config.command
                )
            })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "MCP server stdout is not piped.".to_string())?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "MCP server stdin is not piped.".to_string())?;
        let (sender, receiver) = mpsc::channel::<String>();
        std::thread::Builder::new()
            .name(format!("mcp-{}-reader", config.name))
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    if sender.send(line.unwrap_or_default()).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("Failed to start the MCP reader thread: {error}"))?;
        Ok(Self {
            stdin,
            lines: receiver,
            child,
        })
    }
}

impl LineTransport for ChildTransport {
    #[cfg(test)]
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn send_line(&mut self, line: &str) -> Result<(), String> {
        if self
            .child
            .try_wait()
            .map_err(|error| format!("Failed to poll the MCP server: {error}"))?
            .is_some()
        {
            return Err("MCP server process exited unexpectedly.".into());
        }
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("Failed to write to the MCP server: {error}"))
    }

    fn recv_line(&mut self, timeout: Duration) -> Result<Option<String>, String> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Ok(Some(line)),
            Err(RecvTimeoutError::Timeout) => Err("MCP server did not answer in time.".into()),
            Err(RecvTimeoutError::Disconnected) => Ok(None),
        }
    }
}

impl Drop for ChildTransport {
    fn drop(&mut self) {
        // Kill the child; dropping stdin closes the server's stdio transport,
        // which is the MCP shutdown signal — the kill only backstops servers
        // that ignore it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One live MCP server connection.
pub(crate) struct McpClient {
    server_name: String,
    transport: Box<dyn LineTransport + Send>,
    next_id: u64,
}

/// One tool from `tools/list`.
pub(crate) struct ListedTool {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) schema: Value,
}

impl McpClient {
    /// Spawn the server and run the `initialize` handshake.
    pub(crate) fn connect(config: &McpServerConfig) -> Result<Self, String> {
        let mut client = Self {
            server_name: config.name.clone(),
            transport: Box::new(ChildTransport::spawn(config)?),
            next_id: 0,
        };
        client.initialize()?;
        Ok(client)
    }

    /// The server name this client serves (registry routing key).
    pub(crate) fn server_name(&self) -> &str {
        &self.server_name
    }

    fn send(&mut self, value: &Value) -> Result<(), String> {
        let line = serde_json::to_string(value)
            .map_err(|error| format!("Failed to serialize an MCP message: {error}"))?;
        self.transport.send_line(&line)
    }

    /// Send a JSON-RPC request and wait for the response with the matching
    /// id. Notifications and server→client requests seen on the way are
    /// ignored / answered with a protocol error, keeping the stream clean.
    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            let elapsed = Instant::now();
            if elapsed >= deadline {
                return Err(format!(
                    "MCP server '{}' timed out answering '{method}'.",
                    self.server_name
                ));
            }
            let remaining = deadline - elapsed;
            let Some(line) = self.transport.recv_line(remaining)? else {
                return Err(format!(
                    "MCP server '{}' closed its output while answering '{method}'.",
                    self.server_name
                ));
            };
            if line.trim().is_empty() {
                continue;
            }
            let message: Value = serde_json::from_str(&line).map_err(|error| {
                format!(
                    "MCP server '{}' sent invalid JSON: {error}",
                    self.server_name
                )
            })?;
            if message.get("method").is_some() {
                // A notification (no id) is information only; a request (with
                // id) must be answered or well-behaved servers stall.
                if message.get("id").is_some() {
                    let request_id = message.get("id").cloned().unwrap_or(Value::Null);
                    self.send(&json!({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found."},
                    }))?;
                }
                continue;
            }
            if message.get("id") != Some(&json!(id)) {
                continue; // a stale response; keep waiting for ours
            }
            if let Some(error) = message.get("error") {
                return Err(format!(
                    "MCP server '{}' rejected '{method}': {error}",
                    self.server_name
                ));
            }
            return message.get("result").cloned().ok_or_else(|| {
                format!(
                    "MCP server '{}' sent a response without a result.",
                    self.server_name
                )
            });
        }
    }

    /// The `initialize` handshake plus the `notifications/initialized`
    /// acknowledgment — the MCP session setup every interaction requires.
    fn initialize(&mut self) -> Result<(), String> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cheetah", "version": env!("CARGO_PKG_VERSION")},
            }),
        )?;
        self.send(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }))
    }

    /// List the tools the server serves (following `nextCursor` pagination).
    pub(crate) fn list_tools(&mut self) -> Result<Vec<ListedTool>, String> {
        let mut listed = Vec::new();
        let mut cursor: Option<Value> = None;
        loop {
            let params = match &cursor {
                Some(cursor) => json!({"cursor": cursor}),
                None => json!({}),
            };
            let result = self.request("tools/list", params)?;
            for tool in result
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| "MCP tools/list response is missing its tools list.".to_string())?
            {
                let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "MCP tool entry is missing its name.".to_string())?;
                listed.push(ListedTool {
                    name: name.to_owned(),
                    description: tool
                        .get("description")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    schema: tool.get("inputSchema").cloned().unwrap_or(json!({})),
                });
            }
            cursor = result.get("nextCursor").cloned().filter(|c| !c.is_null());
            if cursor.is_none() {
                return Ok(listed);
            }
        }
    }

    /// Execute one tool (`tools/call`) and return the text content blocks
    /// joined with newlines. An `isError: true` result surfaces as
    /// `[error] ...` text — the model sees it and can react — while
    /// transport and protocol failures are hard errors.
    pub(crate) fn call_tool(&mut self, name: &str, arguments: Value) -> Result<String, String> {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments}))?;
        let mut texts = Vec::new();
        for block in result
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| "MCP tools/call response is missing its content.".to_string())?
        {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    texts.push(text.to_owned());
                }
            }
        }
        let mut text = texts.join("\n");
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            text = format!("[error] {text}");
        }
        Ok(text)
    }

    /// Kill the server process (the transport's `Drop` backstops it).
    pub(crate) fn shutdown(&mut self) {
        self.transport.send_line("").ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Scripted transport: writes are recorded, reads pop the script.
    struct ScriptedTransport {
        script: RefCell<Vec<String>>,
        sent: RefCell<Vec<String>>,
    }

    impl ScriptedTransport {
        fn new(script: &[&str]) -> Self {
            Self {
                script: RefCell::new(script.iter().map(|l| l.to_string()).collect()),
                sent: RefCell::new(Vec::new()),
            }
        }
    }

    impl LineTransport for ScriptedTransport {
        fn as_any(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn send_line(&mut self, line: &str) -> Result<(), String> {
            self.sent.borrow_mut().push(line.to_owned());
            Ok(())
        }

        fn recv_line(&mut self, _timeout: Duration) -> Result<Option<String>, String> {
            Ok(Some(
                self.script.borrow_mut().pop().expect("script exhausted"),
            ))
        }
    }

    fn client_with(script: &[&str]) -> McpClient {
        McpClient {
            server_name: "test".into(),
            transport: Box::new(ScriptedTransport::new(script)),
            next_id: 0,
        }
    }

    /// The initialize handshake sends a well-formed JSON-RPC request with the
    /// protocol version and follows it with the initialized notification.
    #[test]
    fn initialize_sends_handshake_and_notification() {
        let response = serde_json::json!({
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
        });
        let mut client = client_with(&[&response.to_string()]);
        client.initialize().unwrap();
        let sent = client
            .transport
            .as_any()
            .downcast_ref::<ScriptedTransport>()
            .unwrap()
            .sent
            .borrow();
        assert_eq!(sent.len(), 2);
        let request: Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(request["jsonrpc"], "2.0");
        assert_eq!(request["id"], 0);
        assert_eq!(request["method"], "initialize");
        assert_eq!(request["params"]["protocolVersion"], "2024-11-05");
        assert_eq!(request["params"]["clientInfo"]["name"], "cheetah");
        let notification: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(notification["method"], "notifications/initialized");
        assert!(notification.get("id").is_none());
    }

    /// `tools/list` extracts names/descriptions/schemas and follows the
    /// `nextCursor` pagination.
    #[test]
    fn list_tools_extracts_and_paginates() {
        let page1 = serde_json::json!({
            "jsonrpc": "2.0", "id": 0,
            "result": {"tools": [
                {"name": "read_file", "description": "Reads a file",
                 "inputSchema": {"type": "object"}},
            ], "nextCursor": "page2"},
        });
        let page2 = serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"tools": [{"name": "write_file"}]},
        });
        let mut client = client_with(&[&page2.to_string(), &page1.to_string()]);
        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description.as_deref(), Some("Reads a file"));
        assert_eq!(tools[0].schema["type"], "object");
        assert_eq!(tools[1].name, "write_file");
        assert!(tools[1].description.is_none());
        let sent = client
            .transport
            .as_any()
            .downcast_ref::<ScriptedTransport>()
            .unwrap()
            .sent
            .borrow();
        let second: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(second["params"]["cursor"], "page2");
    }

    /// `tools/call` joins the text content blocks and surfaces
    /// `isError: true` as `[error]` text for the model.
    #[test]
    fn call_tool_joins_text_and_flags_errors() {
        let ok = serde_json::json!({
            "jsonrpc": "2.0", "id": 0,
            "result": {"content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ]},
        });
        let failing = serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"isError": true, "content": [{"type": "text", "text": "boom"}]},
        });
        let mut client = client_with(&[&failing.to_string(), &ok.to_string()]);
        assert_eq!(
            client.call_tool("t", serde_json::json!({"a": 1})).unwrap(),
            "line one\nline two"
        );
        assert_eq!(
            client.call_tool("t", serde_json::json!({})).unwrap(),
            "[error] boom"
        );
    }

    /// Server→client requests are answered with a method-not-found error so
    /// well-behaved servers never stall waiting for a response.
    #[test]
    fn server_requests_are_answered_with_protocol_error() {
        let server_request = serde_json::json!({
            "jsonrpc": "2.0", "id": 99, "method": "sampling/createMessage",
        });
        let response = serde_json::json!({"jsonrpc": "2.0", "id": 0, "result": {}});
        let mut client = client_with(&[&response.to_string(), &server_request.to_string()]);
        client.request("tools/list", serde_json::json!({})).unwrap();
        let sent = client
            .transport
            .as_any()
            .downcast_ref::<ScriptedTransport>()
            .unwrap()
            .sent
            .borrow();
        let error_reply: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(error_reply["id"], 99);
        assert_eq!(error_reply["error"]["code"], -32601);
    }

    /// JSON-RPC errors from the server surface as request failures.
    #[test]
    fn server_error_rejects_the_request() {
        let response = serde_json::json!({
            "jsonrpc": "2.0", "id": 0,
            "error": {"code": -32000, "message": "not initialized"},
        });
        let mut client = client_with(&[&response.to_string()]);
        let error = client
            .request("tools/list", serde_json::json!({}))
            .unwrap_err();
        assert!(error.contains("not initialized"), "{error}");
    }
}
