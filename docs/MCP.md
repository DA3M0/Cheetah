# MCP — Model Context Protocol — developer guide

Cheetah supports MCP in **both directions**:

1. **Cheetah as an MCP server** — built into `cheetah serve`: every serve
   instance exposes the loaded model at `POST /mcp` (JSON-RPC 2.0) with no
   extra configuration. Any MCP agent connects to `http://host:port/mcp`
   and calls the `chat_generate` tool to run generation through the model's
   chat template — the same server that answers the OpenAI API.
2. **Cheetah as an MCP client** (`--mcp-server` / `--mcp-config` on chat and
   serve): connects to local MCP servers, lets the model call their tools,
   and executes server-side inside an agentic loop.

The server direction speaks MCP over HTTP (one JSON-RPC message per POST);
the client direction spawns MCP servers over the stdio transport
(newline-delimited JSON-RPC 2.0).

---

## 1. Cheetah as an MCP server (for agents)

```sh
cheetah serve <model-path>   # OpenAI API + MCP protocol, nothing to configure
```

The loaded model is exposed as two tools:

| Tool | Input | Returns |
|---|---|---|
| `chat_generate` | `{messages: [{role, content}], max_tokens?, temperature?}` | The assistant text, generated through the model's chat template |
| `model_info` | `{}` | `{model_id, architecture, device}` as JSON text |

### Agent configuration (any HTTP MCP client)

Point the agent at the serve URL:

```json
{
  "mcpServers": {
    "cheetah": {
      "type": "streamable-http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### Endpoint behavior

- **Transport**: `POST /mcp` with a JSON-RPC 2.0 message body; the response
  is the JSON-RPC reply (`application/json`). One message per request —
  the server is stateless, so agents need no session management. Requests
  answer 200; notifications answer 202 with no body; `GET /mcp` answers
  405 (there is no SSE listen channel — agents must use an HTTP MCP client
  that accepts plain-JSON replies, e.g. a `streamable-http`-type
  connection; SSE-only 2024-11-05 clients cannot interop).
- **Authentication**: the same Bearer contract as the OpenAI endpoints —
  when `--api-key`/`--generate-key` is set, agents send
  `Authorization: Bearer <key>` (loopback servers without a key accept
  unauthenticated calls).
- **Queue**: an MCP `tools/call` that generates holds a slot in the same
  FIFO generation queue as the OpenAI endpoints — one model, one fair line.
- **Protocol**: version `2024-11-05`; capabilities advertise `tools`.
  Malformed JSON answers the JSON-RPC parse error (`-32700`, id null) and
  keeps serving; unknown methods answer `-32601`.
- **`tools/call` execution**: the messages render through the model's chat
  template with no system-prompt injection (the agent owns `role`/`content`;
  the server applies the template and context limits). `max_tokens`
  overrides the server budget per call; `temperature` must be within
  `0..=2` (out-of-range values answer `-32602`). Execution failures
  surface as `isError: true` results — the agent sees them and can react.
- **Performance**: serve warms the model up before binding, and generation
  runs through the KV prefix cache — a `chat_generate` call whose messages
  extend a recently prefilled conversation skips re-prefilling the shared
  history. Measured on qwen-0.5b (~1.4k-token prompt): repeat/multi-turn
  calls answer in ~0.2–0.5 s where a cold prompt takes ~75 s
  (docs/CODEBASE.md §2.2 has the full numbers). The cache is engine-level
  (LRU, ≤ 4 entries), so agents get the reuse without any session protocol.
- **Nested tool calling**: when serve runs with `--mcp-server` /
  `--mcp-config`, those servers' tools join the agentic loop — a model tool
  call inside `chat_generate` executes against the owning MCP server, the
  result feeds back, and the loop continues to a plain answer (capped at 8
  rounds). A malformed or hallucinated call that matches no connected tool
  surfaces as plain text.
- **Diagnostics**: tool executions log `[mcp] calling tool '...'` lines on
  stderr per round.

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

## 3. How a request flows

```text
OpenAI request ──► tools = request-declared tools ∪ MCP tools (MCP names lose ties)
MCP tools/call ──► chat_generate: messages + the configured MCP tools
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
              │              contract is unchanged; the MCP endpoint
              │              renders them as text — every tool is MCP-owned)
              └─ plain answer ──► final response
```

- **MCP-owned calls execute server-side.** A streaming OpenAI request never
  sees the tool-call JSON: the engine's `ToolStreamSink` holds back
  everything from the first `{` / code fence and the tool-loop round
  decision drops it. The client only sees the final text deltas (or
  `tool_calls` frames for client-owned calls).
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
| MCP **endpoint** | `src/serve/mcp.rs` | Serve's MCP surface: `POST /mcp` dispatch (`handle_request` is HTTP-free and generic over the invoker), the `chat_generate` / `model_info` catalog, per-request state borrowing the shared engine + generation ticket |
| Streaming hold-back | `ToolStreamSink` (engine.rs) | Live text deltas with tool-call JSON held back until the round decision |

The full function-level reference lives in
[CODING_STANDARDS.md](CODING_STANDARDS.md) §10.19–10.20b.

## 5. Testing

- `cargo test mcp` — client protocol tests against a scripted transport
  (handshake shape, pagination, content joining, `isError`, protocol-error
  replies, server-error rejection), config/spec parsing, and an
  **end-to-end test against a real stdio child process** (gated on
  `python3` being installed).
- `cargo test serve::mcp` — the MCP endpoint's protocol dispatch
  (initialize shape, tools catalog, routing, notifications, unknown
  methods/tools, argument validation) against a stubbed invoker.
- `cargo test tool_stream` — the hold-back wrapper: braces/fences hold,
  plain text streams through, flush/drop at the round decision.
- Live verification: `cheetah serve ... --mcp-server "demo=python3
  /tmp/echo_mcp_server.py"` + a streaming request — heartbeats and content
  deltas arrive incrementally, and `[mcp] calling tool '...'` lines appear on
  stderr per execution. Server side: `POST /mcp` with an initialize /
  tools/call line — see the patterns in §1.

## 6. Limitations

- **Client direction is stdio only** — no HTTP/SSE MCP client transport
  yet. The server direction is HTTP only (no stdio spawn mode; agents
  connect to the serve URL; `GET /mcp` is 405 — no SSE listen channel).
- **Stateless server** — no MCP sessions (`Mcp-Session-Id` is neither
  issued nor required); one JSON-RPC message per POST. Engine-level KV
  prefix reuse still applies across calls automatically (see §1
  Performance) — reuse is keyed on the token prefix, not on a session.
- **Tools, not resources** — only `tools/list` / `tools/call` are consumed;
  MCP resources and prompts are ignored.
- **No sampling** — server→client `sampling/createMessage` requests are
  answered with a method-not-found error.
- **Round cap** — the agentic loop stops after 8 tool rounds and returns the
  last calls to the caller.
- **Duplicate tool names** across servers fail the registry build; an MCP
  tool name colliding with a request-declared tool is shadowed by the
  request's declaration.
- **Server direction** (`POST /mcp`): one model per serve instance; the MCP
  tool calls share the OpenAI endpoints' FIFO generation queue; nested MCP
  servers share the 8-round agentic cap; Ant settings apply via the shared
  serve flags.
