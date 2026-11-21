# MCP tool calling — developer guide

Cheetah can connect to local **Model Context Protocol (MCP)** servers and let
the model call their tools. Servers are spawned as child processes over the
MCP stdio transport (newline-delimited JSON-RPC 2.0); their tools are
discovered at startup (`tools/list`), exposed to the model alongside any
request-declared tools, and executed server-side (`tools/call`) inside an
agentic loop.

Both front ends support it: `cheetah chat` (REPL) and `cheetah serve`
(OpenAI-compatible API).

---

## 1. Quick start

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

## 2. How a request flows

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

## 3. Architecture

| Piece | Where | Role |
|---|---|---|
| Shapes + parsing | `src/chat/toolcalls.rs` | The tool-call JSON shapes and the tolerant model-output parser, shared by the API, the REPL, and the MCP layer (HTTP-free) |
| Client | `src/mcp/client.rs` | JSON-RPC 2.0 over stdio: `initialize` handshake, `tools/list` (paginated), `tools/call`; id-correlated requests, 30s deadline, reader thread; server→client requests answered with a protocol error |
| Registry | `src/mcp.rs` | Server configs → connected clients + flattened tool list; `tool_definitions()` renders OpenAI-style definitions; `call()` routes to the owning server (per-client mutex, `Arc`-shared across request workers); `Drop` kills children |
| Agentic loop | `Engine::generate_tool_chat` | The round driver (MCP-agnostic — it routes calls to `Option<&McpRegistry>`) |
| Streaming hold-back | `ToolStreamSink` (engine.rs) | Live text deltas with tool-call JSON held back until the round decision |

The full function-level reference lives in
[CODING_STANDARDS.md](CODING_STANDARDS.md) §10.19–10.20.

## 4. Testing

- `cargo test mcp` — protocol tests against a scripted transport (handshake
  shape, pagination, content joining, `isError`, protocol-error replies,
  server-error rejection), config/spec parsing, and an **end-to-end test
  against a real stdio child process** (gated on `python3` being installed).
- `cargo test tool_stream` — the hold-back wrapper: braces/fences hold,
  plain text streams through, flush/drop at the round decision.
- Live verification: `cheetah serve ... --mcp-server "demo=python3
  /tmp/echo_mcp_server.py"` + a streaming request — heartbeats and content
  deltas arrive incrementally, and `[mcp] calling tool '...'` lines appear on
  stderr per execution.

## 5. Limitations

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
