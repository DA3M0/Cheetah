# MCP — Model Context Protocol — developer guide

Cheetah supports MCP in **both directions**:

1. **Cheetah as an MCP server** (`cheetah mcp <model-path>`): exposes the
   loaded model to MCP agents over stdio. Any MCP client — Claude Desktop,
   Cursor, or a scripted peer — spawns Cheetah and calls the
   `chat_generate` tool to run generation through the model's chat template.
2. **Cheetah as an MCP client** (`--mcp-server` / `--mcp-config` on chat and
   serve): connects to local MCP servers, lets the model call their tools,
   and executes server-side inside an agentic loop.

Both directions speak the MCP stdio transport: newline-delimited JSON-RPC
2.0 over the process's stdin/stdout.

---

## 1. Cheetah as an MCP server (for agents)

```sh
cheetah mcp <model-path> [--max-tokens N] [--temperature F]
```

Runs until the agent closes stdin (`Ctrl-C` also stops it). The loaded model
is exposed as two tools:

| Tool | Input | Returns |
|---|---|---|
| `chat_generate` | `{messages: [{role, content}], max_tokens?, temperature?}` | The assistant text, generated through the model's chat template |
| `model_info` | `{}` | `{model_id, architecture, device}` as JSON text |

### Agent configuration (Claude Desktop / any MCP client)

```json
{
  "mcpServers": {
    "cheetah": {
      "command": "/path/to/cheetah",
      "args": ["mcp", "/path/to/model-prepared"]
    }
  }
}
```

### Server behavior

- **Protocol**: JSON-RPC 2.0 over newline-delimited stdio, protocol version
  `2024-11-05`; capabilities advertise `tools`. Malformed JSON lines answer
  with a parse error and keep serving; unknown methods answer `-32601`.
- **`tools/call` execution**: the messages render through the model's chat
  template (the agent controls `role`/`content`; the server applies the
  template and context limits). `max_tokens` and `temperature` override the
  server defaults per call. Execution failures surface as `isError: true`
  results — the agent sees them and can react.
- **stdout discipline**: the protocol channel carries JSON frames only; all
  diagnostics go to stderr.

## 2. Cheetah as an MCP client (tool calling)

```sh
# Chat REPL with one MCP server (repeatable flag)
cheetah chat ./qwen-0.5b-prepared   --mcp-server "fs=npx -y @modelcontextprotocol/server-filesystem /tmp"

# Same, via a JSON config file
cheetah serve ./qwen-0.5b-prepared --mcp-config mcp.json

# mcp.json
{
  "servers": [
    {"name": "fs",     "command": "npx",     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    {"name": "time",   "command": "python3", "args": ["time_server.py"]}
  ]
}
```

- `--mcp-server "name=command [args...]"` — repeatable. The first token
  carries `name=command`; the rest are the command's arguments (split on
  whitespace — arguments containing spaces belong in a `--mcp-config` file).
- `--mcp-config <path>` — JSON config file (once).

Servers connect **before the model loads**; a server that fails to spawn,
answer the handshake, or list tools fails startup loudly — a half-configured
tool surface would silently corrupt tool calls.

```sh
# Chat REPL with one MCP server (repeatable flag)
cheetah chat ./qwen-0.5b-prepared \
  --mcp-server "fs=npx -y @modelcontextprotocol/server-filesystem /tmp"

# Same, via a JSON config file
cheetah serve ./qwen-0.5b-prepared --mcp-config mcp.json

# mcp.json
{
  "servers": [
    {"name": "fs",     "command": "npx",     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    {"name": "time",   "command": "python3", "args": ["time_server.py"]}
  ]
}
```

- `--mcp-server "name=command [args...]"` — repeatable. The first token
  carries `name=command`; the rest are the command's arguments (split on
  whitespace — arguments containing spaces belong in a `--mcp-config` file).
- `--mcp-config <path>` — JSON config file (once).

Servers connect **before the model loads**; a server that fails to spawn,
answer the handshake, or list tools fails startup loudly — a half-configured
tool surface would silently corrupt tool calls.

## 3. How a request flows

```text
request ──► tools = request-declared tools ∪ MCP tools (MCP names lose ties)
        ──► Engine::generate_tool_chat          (the agentic loop, ≤ 8 rounds)
              │  round r: tool instruction + conversation ──► generate
              │  parse the output into tool calls (chat::toolcalls shapes)
              │
              ├─ every call targets a connected MCP server?
              │     yes ──► tools/call on the owning server (per-client mutex)
              │             results join the conversation as role:"tool"
              │             ──► next round
              │     no  ──► return the calls to the caller
              │             (OpenAI clients execute them and send back
              │              role:"tool" messages — the client-driven
              │              contract is unchanged)
              └─ plain answer ──► final response
```

- **MCP-owned calls execute server-side.** A streaming request never sees
  the tool-call JSON: the engine's `ToolStreamSink` holds back everything
  from the first `{` / code fence and the tool-loop round decision drops it.
  The client only sees the final text deltas (or `tool_calls` frames for
  client-owned calls).
- **Client-declared calls keep the OpenAI contract.** If a round returns any
  call that is not an MCP tool, the calls return to the HTTP client as
  `tool_calls` — the client executes them and sends results back as
  `role:"tool"` messages, exactly like the no-MCP flow.
- **Streaming rounds are real.** With axum the SSE body flushes per frame;
  text deltas reach the client live between tool rounds, and a heartbeat
  frame flows after 4s of silence.

## 4. Architecture

| Piece | Where | Role |
|---|---|---|
| Shapes + parsing | `src/chat/toolcalls.rs` | The tool-call JSON shapes and the tolerant model-output parser, shared by the API, the REPL, and the MCP layer (HTTP-free) |
| Client | `src/mcp/client.rs` | JSON-RPC 2.0 over stdio: `initialize` handshake, `tools/list` (paginated), `tools/call`; id-correlated requests, 30s deadline, reader thread; server→client requests answered with a protocol error |
| Registry | `src/mcp.rs` | Server configs → connected clients + flattened tool list; `tool_definitions()` renders OpenAI-style definitions; `call()` routes to the owning server (per-client mutex, `Arc`-shared across request workers); `Drop` kills children |
| Agentic loop | `Engine::generate_tool_chat` | The round driver (MCP-agnostic — it routes calls to `Option<&McpRegistry>`) |
| MCP **server** | `src/mcp_server.rs` | `cheetah mcp <model-path>`: exposes `chat_generate` / `model_info` tools to agents over stdio; loads the engine via `Engine::load` (protocol-only stdout) |
| Streaming hold-back | `ToolStreamSink` (engine.rs) | Live text deltas with tool-call JSON held back until the round decision |

The full function-level reference lives in
[CODING_STANDARDS.md](CODING_STANDARDS.md) §10.19–10.20.

## 5. Testing

- `cargo test mcp` — protocol tests against a scripted transport (handshake
  shape, pagination, content joining, `isError`, protocol-error replies,
  server-error rejection), config/spec parsing, and an **end-to-end test
  against a real stdio child process** (gated on `python3` being installed).
- `cargo test tool_stream` — the hold-back wrapper: braces/fences hold,
  plain text streams through, flush/drop at the round decision.
- `cargo test mcp_server` — the server's protocol dispatch (initialize
  shape, tools catalog, routing, notifications, unknown methods/tools,
  argument validation) against a stubbed invoker.
- Live verification: `cheetah serve ... --mcp-server "demo=python3
  /tmp/echo_mcp_server.py"` + a streaming request — heartbeats and content
  deltas arrive incrementally, and `[mcp] calling tool '...'` lines appear on
  stderr per execution. Server side: pipe JSON-RPC lines into
  `cheetah mcp <model>` (or connect any real MCP client) — see the
  end-to-end patterns in this document.

## 6. Limitations

- **stdio only** — no HTTP/SSE MCP transport yet.
- **Tools, not resources** — only `tools/list` / `tools/call` are consumed;
  MCP resources and prompts are ignored.
- **No sampling** — server→client `sampling/createMessage` requests are
  answered with a method-not-found error.
- **Round cap** — the agentic loop stops after 8 tool rounds and returns the
  last calls to the caller.
- **Duplicate tool names** across servers fail the registry build; an MCP
  tool name colliding with a request-declared tool is shadowed by the
  request's declaration.
- **Server mode** (`cheetah mcp`): one model per server process; sequential
  request processing (stdio agents issue one call at a time); Ant settings
  supported via the shared chat flags.
