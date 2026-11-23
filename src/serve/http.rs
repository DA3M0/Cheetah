//! HTTP routing, authentication, and streaming for `cheetah serve` (axum).

use crate::chat::{Engine, Generation, TextSink, ToolChoice};
use crate::color::{self, Color};
use crate::options::ChatOptions;
use crate::serve::openai::{self, *};
use crate::serve::queue::{QueueGuard, QueueTicket};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::Sleep;

const SSE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);
/// Request-body cap (parity with the former unlimited `read_to_end`).
const BODY_LIMIT: usize = usize::MAX;

pub(crate) struct ServerState {
    pub(crate) engine: Arc<Engine>,
    pub(crate) default_options: ChatOptions,
    pub(crate) model_id: String,
    pub(crate) api_key: Option<String>,
    pub(crate) generation_queue: Arc<super::queue::FifoQueue>,
    pub(crate) created: u64,
    /// Connected MCP servers (`None` when none are configured): their tools
    /// join every chat request and their calls execute in the agentic loop.
    pub(crate) mcp: Option<Arc<crate::mcp::McpRegistry>>,
}

/// Whether an incoming request is a generation endpoint (POST
/// `/v1/chat/completions`, `/v1/completions`, or `/mcp` — an MCP
/// `tools/call` may generate — ignoring query strings and trailing slashes)
/// — decides whether a queue ticket is taken.
pub(crate) fn is_generation_request(method: &Method, path: &str) -> bool {
    method == Method::POST && matches!(path, "/v1/chat/completions" | "/v1/completions" | "/mcp")
}

/// The request dispatcher: OPTIONS → CORS preflight, `/health` free of auth
/// so probes can check liveness, then the Bearer check and the route table
/// (models / chat / completions). Generation requests take a queue ticket at
/// arrival, preserving the FIFO-at-arrival ordering.
/// The axum app: every request (any method, any path) flows through
/// [`handle_any`], which owns the OPTIONS/auth/route contract verbatim.
pub(crate) fn router(state: Arc<ServerState>) -> axum::Router {
    axum::Router::new()
        .fallback(any_handler)
        .layer(axum::extract::DefaultBodyLimit::max(usize::MAX))
        .with_state(state)
}

async fn any_handler(State(state): State<Arc<ServerState>>, request: Request) -> Response {
    handle_any(state, request).await
}

pub(crate) async fn handle_any(state: Arc<ServerState>, request: Request) -> Response {
    let method = request.method().clone();
    let raw = request.uri().path().to_owned();
    let path = raw.strip_suffix('/').unwrap_or(&raw);
    let path = if path.is_empty() { "/" } else { path };

    if method == Method::OPTIONS {
        return options_response();
    }

    // /health is auth-free so probes and load balancers can check liveness.
    // /health 不要求鉴权，便于健康探针和负载均衡器检查服务状态。
    if method == Method::GET && path == "/health" {
        return json_response(StatusCode::OK, health_body());
    }

    // Generation requests take a ticket at arrival (before auth), preserving
    // the FIFO-at-arrival ordering the queue contract promises.
    let generation_ticket = if is_generation_request(&method, path) {
        Some(state.generation_queue.enqueue())
    } else {
        None
    };

    if !authorized(request.headers(), state.api_key.as_deref()) {
        return error_response(
            401,
            "Missing or invalid API key; provide a Bearer token in the Authorization header.",
        );
    }

    match route(&method, path) {
        Ok(RouteEnum::Health) => json_response(StatusCode::OK, health_body()),
        Ok(RouteEnum::ListModels) => {
            let body = model_list_string(&state);
            json_response(StatusCode::OK, body)
        }
        Ok(RouteEnum::GetModel(id)) => {
            if id == state.model_id {
                let body = single_model_string(&state);
                json_response(StatusCode::OK, body)
            } else {
                error_response(
                    404,
                    &format!(
                        "Model '{id}' does not exist; the loaded model is '{}'.",
                        state.model_id
                    ),
                )
            }
        }
        Ok(RouteEnum::ChatCompletions) => {
            let ticket = generation_ticket.expect("generation request receives a queue ticket");
            handle_chat_completions(state, request, ticket).await
        }
        Ok(RouteEnum::Completions) => {
            let ticket = generation_ticket.expect("generation request receives a queue ticket");
            handle_completions(state, request, ticket).await
        }
        Ok(RouteEnum::Mcp) => {
            let ticket = generation_ticket.expect("generation request receives a queue ticket");
            handle_mcp(state, request, ticket).await
        }
        Err(status) => error_response(
            status.as_u16(),
            if status == StatusCode::METHOD_NOT_ALLOWED {
                "Method not allowed."
            } else {
                "Endpoint not found."
            },
        ),
    }
}

// ---- routing ----

#[derive(Debug, PartialEq, Eq)]
enum RouteEnum {
    Health,
    ListModels,
    GetModel(String),
    ChatCompletions,
    Completions,
    Mcp,
}

/// Route table: a known path with a wrong method is 405, anything unknown
/// is 404.
fn route(method: &Method, path: &str) -> Result<RouteEnum, StatusCode> {
    if method == Method::GET && path == "/health" {
        return Ok(RouteEnum::Health);
    }
    if method == Method::GET && path == "/v1/models" {
        return Ok(RouteEnum::ListModels);
    }
    if method == Method::GET
        && let Some(id) = path.strip_prefix("/v1/models/")
        && !id.is_empty()
        && !id.contains('/')
    {
        return Ok(RouteEnum::GetModel(id.to_string()));
    }
    if method == Method::POST && path == "/v1/chat/completions" {
        return Ok(RouteEnum::ChatCompletions);
    }
    if method == Method::POST && path == "/v1/completions" {
        return Ok(RouteEnum::Completions);
    }
    if method == Method::POST && path == "/mcp" {
        return Ok(RouteEnum::Mcp);
    }

    let known = path == "/health"
        || path == "/v1/models"
        || path.starts_with("/v1/models/")
        || path == "/v1/chat/completions"
        || path == "/v1/completions"
        || path == "/mcp";
    if known {
        Err(StatusCode::METHOD_NOT_ALLOWED)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

// ---- auth ----

/// Bearer-token check against the expected key; no expected key means the
/// request is allowed (loopback-only posture enforced upstream).
fn authorized(headers: &axum::http::HeaderMap, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let Some(token) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(token.as_bytes(), expected.as_bytes())
}

/// Length-checked constant-time byte comparison so token validation does
/// not leak the first differing position.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // Compare every byte when lengths match to avoid leaking the first differing position.
    // 长度相同时遍历所有字节，避免通过首个差异位置泄露令牌内容。
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---- response helpers ----

/// The CORS header set (origin, methods, headers).
fn cors_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        (
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        ),
    ]
}

/// The 204 CORS preflight response with CORS + Allow headers.
fn options_response() -> Response {
    let mut response = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header("Allow", "GET, POST, OPTIONS");
    for (name, value) in cors_headers() {
        response = response.header(name, value);
    }
    response.body(Body::empty()).expect("static response")
}

/// CORS headers plus `application/json`.
fn json_response(status: StatusCode, body: String) -> Response {
    let mut response = Response::builder()
        .status(status)
        .header("Content-Type", "application/json");
    for (name, value) in cors_headers() {
        response = response.header(name, value);
    }
    response.body(Body::from(body)).expect("json response")
}

/// Sends an OpenAI-shaped error envelope (`error_body`) for a status code.
fn error_response(status: u16, message: &str) -> Response {
    json_response(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        error_body(status, message),
    )
}

/// A bare status response with CORS headers and no body (202 notifications).
fn empty_response(status: StatusCode) -> Response {
    let mut response = Response::builder().status(status);
    for (name, value) in cors_headers() {
        response = response.header(name, value);
    }
    response.body(Body::empty()).expect("empty response")
}

// ---- models endpoints ----

/// The static `{"status":"ok"}` liveness body.
fn health_body() -> String {
    r#"{"status":"ok"}"#.to_string()
}

/// Serializes the OpenAI `/v1/models` list (the single loaded model).
fn model_list_string(state: &ServerState) -> String {
    let list = ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model",
            created: state.created,
            owned_by: "cheetah",
        }],
    };
    serde_json::to_string(&list).unwrap_or_default()
}

/// Serializes the `/v1/models/{id}` entry for the loaded model.
fn single_model_string(state: &ServerState) -> String {
    let entry = ModelEntry {
        id: state.model_id.clone(),
        object: "model",
        created: state.created,
        owned_by: "cheetah",
    };
    serde_json::to_string(&entry).unwrap_or_default()
}

// ---- chat completions ----

/// The chat endpoint: parse → model match → tool-choice validation →
/// message building → option resolution; tool-enabled requests route
/// through the tool prompt path (the agentic MCP loop executes server-side
/// tools), then either stream (SSE with disconnect tracking) or respond
/// non-streaming. Every validation failure maps to a status + error body
/// before any generation starts.
async fn handle_chat_completions(
    state: Arc<ServerState>,
    request: Request,
    generation_ticket: QueueTicket,
) -> Response {
    let arrived = Instant::now();
    let body = match axum::body::to_bytes(request.into_body(), BODY_LIMIT).await {
        Ok(body) => body,
        Err(error) => {
            return error_response(400, &format!("Failed to read request body: {error}"));
        }
    };
    let api_request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            return error_response(400, &format!("Request body is not valid JSON: {error}"));
        }
    };
    if let Err(error) = ensure_model_matches(&api_request.model, &state.model_id) {
        return error_response(error.status, &error.message);
    }
    let mut tool_choice = match resolve_tool_choice(&api_request.tools, &api_request.tool_choice) {
        Ok(choice) => choice,
        Err(error) => return error_response(error.status, &error.message),
    };
    // MCP tools join every request once servers are configured: the merged
    // tool list feeds the prompt, and MCP-named calls execute in the agentic
    // loop (client-declared tools keep their client-driven contract).
    let mut tools: Vec<ToolDefinition> = api_request.tools.clone().unwrap_or_default();
    let declared: HashSet<String> = tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect();
    let mcp_definitions = state
        .mcp
        .as_ref()
        .map(|mcp| mcp.tool_definitions())
        .unwrap_or_default();
    tools.extend(
        mcp_definitions
            .into_iter()
            .filter(|tool| !declared.contains(&tool.function.name)),
    );
    if state.mcp.is_some() && tool_choice.is_none() && api_request.tool_choice.is_none() {
        tool_choice = Some(ResolvedToolChoice::Auto);
    }

    let prepared = match build_chat_messages(&api_request) {
        Ok(p) => p,
        Err(error) => return error_response(error.status, &error.message),
    };
    let options = match resolve_chat_options(&state.default_options, &api_request) {
        Ok(o) => o,
        Err(error) => return error_response(error.status, &error.message),
    };
    let resident_overrides: HashSet<String> = api_request
        .resident_units
        .as_ref()
        .map(|units| units.iter().cloned().collect())
        .unwrap_or_default();
    let system = prepared
        .system
        .clone()
        .or_else(|| state.default_options.system.clone());
    let stop = stop_strings(&api_request.stop);
    let model = if api_request.model.is_empty() {
        state.model_id.clone()
    } else {
        api_request.model.clone()
    };
    let completion_id = completion_id();
    let created = openai::now_unix();

    // Tool-enabled requests use the same chat prompt path and are parsed from model output.
    if let Some(tool_choice) = tool_choice.filter(|_| !tools.is_empty()) {
        let (messages_json, tools_json) = match build_tool_prompt_inputs(
            &api_request,
            &tools,
            tool_choice.clone(),
            state.default_options.system.as_deref(),
        ) {
            Ok(inputs) => inputs,
            Err(error) => return error_response(400, &error),
        };
        let engine_choice = match &tool_choice {
            ResolvedToolChoice::Auto => ToolChoice::Auto,
            ResolvedToolChoice::None => ToolChoice::None,
            ResolvedToolChoice::Required | ResolvedToolChoice::Named(_) => ToolChoice::Required,
        };
        let gen_state = Arc::clone(&state);
        let requested_tools = tools.clone();
        let generation_input = ToolChatInput {
            messages_json,
            tools_json,
            tools: requested_tools,
            choice: tool_choice.clone(),
            engine_choice,
            options,
            stop,
            resident_overrides,
        };
        if api_request.stream {
            let heartbeat_frame =
                match sse_frame(&completion_id, created, &model, None, Some(""), None, None) {
                    Ok(frame) => frame,
                    Err(error) => return error_response(500, &error),
                };
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
            let response = sse_response(rx, heartbeat_frame, None);
            // The SSE sink goes INTO the loop: text deltas stream live (with
            // tool-call hold-back), so streaming tool requests get
            // incremental frames per round instead of one post-hoc blob.
            tokio::task::spawn_blocking(move || {
                let guard = generation_ticket.acquire();
                let mut sink = SseSink::new(tx, completion_id, created, model);
                let result = generate_tool_chat(&gen_state, guard, generation_input, &mut sink);
                match result {
                    Ok((generation, ToolResponse::Text(_))) => {
                        // Content deltas already streamed round by round.
                        if let Err(error) = sink
                            .send_chunk(None, None, None, Some(generation.finish_reason.as_str()))
                            .and_then(|()| sink.send_done())
                        {
                            eprintln!(
                                "{}",
                                color::stderr(
                                    &format!("Failed to send tool-mode SSE response: {error}"),
                                    Color::Red
                                )
                            );
                        }
                    }
                    Ok((_, ToolResponse::Calls(calls))) => {
                        // Client-declared calls: the OpenAI client executes
                        // them and sends the results back.
                        if let Err(error) = sink
                            .send_tool_calls(&calls)
                            .and_then(|()| sink.send_chunk(None, None, None, Some("tool_calls")))
                            .and_then(|()| sink.send_done())
                        {
                            eprintln!(
                                "{}",
                                color::stderr(
                                    &format!("Failed to send tool-call SSE response: {error}"),
                                    Color::Red
                                )
                            );
                        }
                    }
                    Err(error) => {
                        let _ = sink.send_chunk(None, None, None, Some("server_error"));
                        let _ = sink.send_done();
                        eprintln!(
                            "{}",
                            color::stderr(
                                &format!("Tool-call generation failed: {error}"),
                                Color::Red
                            )
                        );
                    }
                }
            });
            return response;
        }
        let generation = tokio::task::spawn_blocking(move || {
            let guard = generation_ticket.acquire();
            let mut sink = NullSink;
            generate_tool_chat(&gen_state, guard, generation_input, &mut sink)
        })
        .await
        .expect("tool-mode generation task panicked");
        return match generation {
            Ok((generation, result)) => {
                let (content, calls, finish_reason) = match result {
                    ToolResponse::Text(content) => (
                        Some(content),
                        None,
                        generation.finish_reason.as_str().to_string(),
                    ),
                    ToolResponse::Calls(calls) => (None, Some(calls), "tool_calls".to_string()),
                };
                let body = serde_json::to_string(&ChatCompletionResponse {
                    id: completion_id,
                    object: "chat.completion",
                    created,
                    model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChoiceMessage {
                            role: "assistant",
                            content,
                            tool_calls: calls,
                        },
                        finish_reason,
                    }],
                    usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
                })
                .unwrap_or_default();
                json_response(StatusCode::OK, body)
            }
            Err(error) => error_response(prompt_error_status(&error), &error),
        };
    }

    let mut messages = prepared.messages;
    let tokenize_started = Instant::now();
    let tokens = match state
        .engine
        .prepare(&mut messages, system.as_deref(), options.max_tokens)
    {
        Ok((tokens, _)) => tokens,
        Err(error) => return error_response(prompt_error_status(&error), &error),
    };
    let tokenize_duration = tokenize_started.elapsed();

    if api_request.stream {
        let heartbeat_frame =
            match sse_frame(&completion_id, created, &model, None, Some(""), None, None) {
                Ok(frame) => frame,
                Err(error) => return error_response(500, &error),
            };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
        let disconnected = Arc::new(AtomicBool::new(false));
        let response = sse_response(rx, heartbeat_frame, Some(Arc::clone(&disconnected)));
        let gen_state = Arc::clone(&state);
        let queue_started = Instant::now();
        let request_total_ms = arrived.elapsed().as_millis();
        tokio::task::spawn_blocking(move || {
            let guard = generation_ticket.acquire();
            let queue_wait_ms = queue_started.elapsed().as_millis();
            let acquired_at = Instant::now();
            let mut sink =
                SseSink::new_with_disconnect(tx, completion_id, created, model, disconnected);
            let result = generate_chat(
                &gen_state,
                guard,
                tokens,
                &options,
                &stop,
                &mut sink,
                &resident_overrides,
            );
            match result {
                Ok(generation) => {
                    let ttft_ms = sink
                        .first_push_at
                        .map(|first_push| first_push.duration_since(acquired_at).as_millis());
                    tracing::info!(
                        queue_wait_ms,
                        tokenize_ms = tokenize_duration.as_millis(),
                        prefill_ms = generation.prefill_duration.as_millis(),
                        ttft_ms = ttft_ms.unwrap_or(0),
                        prompt_tokens = generation.prompt_tokens,
                        generated_tokens = generation.generated_tokens,
                        request_total_ms,
                        "chat completion (stream) timing"
                    );
                    if let Err(err) = sink
                        .send_chunk(None, None, None, Some(generation.finish_reason.as_str()))
                        .and_then(|()| sink.send_done())
                    {
                        eprintln!(
                            "{}",
                            color::stderr(
                                &format!("Failed to send chat SSE response: {err}"),
                                Color::Red
                            )
                        );
                    }
                }
                Err(err) => {
                    let _ = sink.send_chunk(None, None, None, Some("server_error"));
                    let _ = sink.send_done();
                    eprintln!(
                        "{}",
                        color::stderr(&format!("Generation failed: {err}"), Color::Red)
                    );
                }
            }
        });
        response
    } else {
        let queue_started = Instant::now();
        let generation = tokio::task::spawn_blocking(move || {
            let guard = generation_ticket.acquire();
            let queue_wait_ms = queue_started.elapsed().as_millis();
            let mut sink = NullSink;
            let result = generate_chat(
                &state,
                guard,
                tokens,
                &options,
                &stop,
                &mut sink,
                &resident_overrides,
            );
            if let Ok(generation) = &result {
                tracing::info!(
                    queue_wait_ms,
                    tokenize_ms = tokenize_duration.as_millis(),
                    prefill_ms = generation.prefill_duration.as_millis(),
                    prompt_tokens = generation.prompt_tokens,
                    generated_tokens = generation.generated_tokens,
                    "chat completion timing"
                );
            }
            result
        })
        .await
        .expect("generation task panicked");
        match generation {
            Ok(generation) => {
                let (content, calls, finish_reason) =
                    chat_completion_parts(&generation.text, generation.finish_reason.as_str());
                let body = serde_json::to_string(&ChatCompletionResponse {
                    id: completion_id,
                    object: "chat.completion",
                    created,
                    model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChoiceMessage {
                            role: "assistant",
                            content,
                            tool_calls: calls,
                        },
                        finish_reason,
                    }],
                    usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
                })
                .unwrap_or_default();
                json_response(StatusCode::OK, body)
            }
            Err(err) => error_response(500, &err),
        }
    }
}

/// Splits a finished generation into content / tool-calls / finish-reason —
/// tool JSON detected in the text turns the response into a tool_calls
/// message.
fn chat_completion_parts(
    text: &str,
    finish_reason: &str,
) -> (Option<String>, Option<Vec<ToolCall>>, String) {
    match parse_generated_tool_calls(text) {
        Some(calls) => (None, Some(calls), "tool_calls".to_string()),
        None => (Some(text.to_string()), None, finish_reason.to_string()),
    }
}

/// Serializes the messages (injecting the default system prompt when the
/// request carries none) and the tools — filtered to the named function
/// when `tool_choice` names one — for the template's tool parser.
fn build_tool_prompt_inputs(
    request: &ChatCompletionRequest,
    tools: &[ToolDefinition],
    choice: ResolvedToolChoice,
    default_system: Option<&str>,
) -> Result<(String, String), String> {
    let mut messages = request.messages.clone();
    if !messages
        .iter()
        .any(|message| matches!(message.role.as_str(), "system" | "developer"))
        && let Some(system) = default_system
    {
        messages.insert(
            0,
            RequestMessage {
                role: "system".to_string(),
                content: Some(Content::Text(system.to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        );
    }
    let tools: Vec<_> = match choice {
        ResolvedToolChoice::Named(ref name) => tools
            .iter()
            .filter(|tool| tool.function.name == *name)
            .cloned()
            .collect(),
        _ => tools.to_vec(),
    };
    let messages_json = serde_json::to_string(&messages)
        .map_err(|error| format!("Failed to serialize messages: {error}"))?;
    let tools_json = serde_json::to_string(&tools)
        .map_err(|error| format!("Failed to serialize tool definitions: {error}"))?;
    Ok((messages_json, tools_json))
}

/// Everything one tool-mode request needs: the serialized conversation +
/// merged tool list, the resolved tool choice (for parsing) and its engine
/// instruction form, and the generation options.
struct ToolChatInput {
    messages_json: String,
    tools_json: String,
    tools: Vec<ToolDefinition>,
    choice: ResolvedToolChoice,
    engine_choice: ToolChoice,
    options: ChatOptions,
    stop: Vec<String>,
    resident_overrides: HashSet<String>,
}

/// Runs one tool-mode request through the engine's agentic loop (shared with
/// the chat REPL): MCP tool calls execute in place, client-declared calls
/// return to the client. The caller supplies the sink — the loop's
/// `ToolStreamSink` holds back tool-call JSON so streaming callers never see
/// it as content.
fn generate_tool_chat(
    state: &ServerState,
    _guard: QueueGuard,
    input: ToolChatInput,
    sink: &mut dyn TextSink,
) -> Result<(Generation, ToolResponse), String> {
    state.engine.generate_tool_chat(
        &input.messages_json,
        &input.tools_json,
        &input.tools,
        input.engine_choice,
        &input.choice,
        input.options.max_tokens,
        &input.options,
        &input.stop,
        sink,
        &input.resident_overrides,
        state.mcp.as_deref(),
    )
}

// ---- completions (legacy) ----

/// The legacy `/v1/completions` endpoint (non-streaming only; `echo` is
/// unsupported): prompt → single user message → generate → text completion
/// response.
async fn handle_completions(
    state: Arc<ServerState>,
    request: Request,
    generation_ticket: QueueTicket,
) -> Response {
    let body = match axum::body::to_bytes(request.into_body(), BODY_LIMIT).await {
        Ok(body) => body,
        Err(error) => {
            return error_response(400, &format!("Failed to read request body: {error}"));
        }
    };
    let api_request: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(error) => {
            return error_response(400, &format!("Request body is not valid JSON: {error}"));
        }
    };
    if let Err(error) = ensure_model_matches(&api_request.model, &state.model_id) {
        return error_response(error.status, &error.message);
    }
    if api_request.stream {
        return error_response(
            400,
            "Streaming for /v1/completions is not supported yet; use /v1/chat/completions.",
        );
    }

    let message = match build_completion_message(&api_request) {
        Ok(m) => m,
        Err(error) => return error_response(error.status, &error.message),
    };
    let options = match resolve_completion_options(&state.default_options, &api_request) {
        Ok(o) => o,
        Err(error) => return error_response(error.status, &error.message),
    };
    let resident_overrides: HashSet<String> = api_request
        .resident_units
        .as_ref()
        .map(|units| units.iter().cloned().collect())
        .unwrap_or_default();
    let stop = stop_strings(&api_request.stop);
    let model = if api_request.model.is_empty() {
        state.model_id.clone()
    } else {
        api_request.model.clone()
    };
    let system = state.default_options.system.clone();
    let completion_id = text_completion_id();
    let created = openai::now_unix();

    let generation =
        tokio::task::spawn_blocking(move || -> Result<(Generation, ChatOptions), String> {
            let guard = generation_ticket.acquire();
            let mut messages = vec![message];
            let (tokens, _) =
                state
                    .engine
                    .prepare(&mut messages, system.as_deref(), options.max_tokens)?;
            let mut sink = NullSink;
            let generation = generate_chat(
                &state,
                guard,
                tokens,
                &options,
                &stop,
                &mut sink,
                &resident_overrides,
            )?;
            Ok((generation, options))
        })
        .await
        .expect("completions generation task panicked");

    let (generation, _options) = match generation {
        Ok(pair) => pair,
        Err(err) => return error_response(prompt_error_status(&err), &err),
    };

    let body = serde_json::to_string(&TextCompletionResponse {
        id: completion_id,
        object: "text_completion",
        created,
        model,
        choices: vec![TextChoice {
            index: 0,
            text: generation.text,
            finish_reason: generation.finish_reason.as_str().to_string(),
        }],
        usage: Usage::new(generation.prompt_tokens, generation.generated_tokens),
    })
    .unwrap_or_default();
    json_response(StatusCode::OK, body)
}

// ---- MCP endpoint ----

/// `POST /mcp`: the MCP protocol on the same server as the OpenAI API — no
/// flags needed. One JSON-RPC 2.0 message per request, dispatched from the
/// same auth/queue contract (the Bearer check happened upstream; the
/// request's ticket is acquired only when a tool call generates). Requests
/// answer 200 with the JSON-RPC response, notifications answer 202 with no
/// body, and malformed JSON answers the JSON-RPC parse error (id null).
async fn handle_mcp(
    state: Arc<ServerState>,
    request: Request,
    generation_ticket: QueueTicket,
) -> Response {
    let body = match axum::body::to_bytes(request.into_body(), BODY_LIMIT).await {
        Ok(body) => body,
        Err(error) => {
            return error_response(400, &format!("Failed to read request body: {error}"));
        }
    };
    let message = match serde_json::from_slice::<Value>(&body) {
        Ok(message) => message,
        Err(error) => {
            return json_response(
                StatusCode::OK,
                super::mcp::parse_error_response(&error).to_string(),
            );
        }
    };
    let response = tokio::task::spawn_blocking(move || {
        let mut mcp_state = super::mcp::McpServeState::new(&state, generation_ticket);
        super::mcp::handle_request(&message, &mut mcp_state)
    })
    .await
    .expect("MCP dispatch task panicked");
    match response {
        Some(response) => json_response(StatusCode::OK, response.to_string()),
        None => empty_response(StatusCode::ACCEPTED),
    }
}

// ---- generation ----

/// Shared generation core under the queue guard: new context plus
/// `Engine::generate`.
fn generate_chat(
    state: &ServerState,
    _guard: QueueGuard,
    tokens: Vec<u32>,
    options: &ChatOptions,
    stop: &[String],
    sink: &mut dyn TextSink,
    resident_overrides: &HashSet<String>,
) -> Result<Generation, String> {
    let mut context = state.engine.new_context()?;
    state.engine.generate(
        &mut context,
        &tokens,
        options,
        stop,
        sink,
        resident_overrides,
    )
}

/// Maps context-overflow prompt errors to 400 (client error); every other
/// prompt failure is a 500.
fn prompt_error_status(error: &str) -> u16 {
    if error.starts_with("Input plus reserved generation tokens exceeds the model context limit") {
        400
    } else {
        500
    }
}

// ---- sinks ----

struct NullSink;

impl TextSink for NullSink {
    /// NullSink discards streamed text — non-streaming requests still stream
    /// internally and simply do not keep the text.
    fn push(&mut self, _text: &str) -> Result<(), String> {
        Ok(())
    }
}

/// 流式 SSE 输出目标:把每段文本封装成 OpenAI 格式的 `data:` 帧。
/// The channel is an unbounded tokio channel fed from the blocking
/// generation task; the response body drains it (hyper flushes per frame).
struct SseSink {
    tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
    id: String,
    created: u64,
    model: String,
    first: bool,
    /// Timestamp of the first content delta pushed to the client — the raw
    /// material for the TTFT timing log.
    first_push_at: Option<Instant>,
    disconnected: Option<Arc<AtomicBool>>,
}

impl SseSink {
    /// Builds an SSE sink without disconnect tracking (tool responses).
    fn new(
        tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
        id: String,
        created: u64,
        model: String,
    ) -> Self {
        Self::new_inner(tx, id, created, model, None)
    }

    /// Builds an SSE sink that shares the client-disconnect flag with the
    /// response body stream, so a dropped connection stops the generation.
    fn new_with_disconnect(
        tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
        id: String,
        created: u64,
        model: String,
        disconnected: Arc<AtomicBool>,
    ) -> Self {
        Self::new_inner(tx, id, created, model, Some(disconnected))
    }

    /// Shared construction behind the two sink constructors.
    fn new_inner(
        tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
        id: String,
        created: u64,
        model: String,
        disconnected: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            tx,
            id,
            created,
            model,
            first: true,
            first_push_at: None,
            disconnected,
        }
    }

    /// Emits one OpenAI `chat.completion.chunk` frame; a closed channel maps
    /// to a "Client disconnected" error.
    fn send_chunk(
        &self,
        role: Option<&'static str>,
        content: Option<&str>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish_reason: Option<&str>,
    ) -> Result<(), String> {
        self.tx
            .send(sse_frame(
                &self.id,
                self.created,
                &self.model,
                role,
                content,
                tool_calls,
                finish_reason,
            )?)
            .map_err(|_| "Client disconnected".to_string())
    }

    /// Emits the tool-call delta frame — indexed calls with id / name /
    /// arguments — carrying the assistant role on the first emission.
    fn send_tool_calls(&mut self, calls: &[ToolCall]) -> Result<(), String> {
        let role = self.first.then_some("assistant");
        self.first = false;
        let calls = calls
            .iter()
            .enumerate()
            .map(|(index, call)| ToolCallDelta {
                index,
                id: Some(call.id.clone()),
                kind: Some(call.kind),
                function: ToolCallFunctionDelta {
                    name: Some(call.function.name.clone()),
                    arguments: Some(call.function.arguments.clone()),
                },
            })
            .collect();
        self.send_chunk(role, None, Some(calls), None)
    }

    /// Emits the `data: [DONE]` SSE terminator.
    fn send_done(&self) -> Result<(), String> {
        self.tx
            .send(Bytes::from_static(b"data: [DONE]\n\n"))
            .map_err(|_| "Client disconnected".to_string())
    }
}

#[derive(Serialize)]
struct SseChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [SseChunkChoice<'a>; 1],
}

#[derive(Serialize)]
struct SseChunkChoice<'a> {
    index: usize,
    delta: SseChunkDelta<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'a str>,
}

#[derive(Serialize)]
struct SseChunkDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Serializes one SSE frame; `None` fields are skipped per the OpenAI
/// delta shape.
fn sse_frame(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<&str>,
    tool_calls: Option<Vec<ToolCallDelta>>,
    finish_reason: Option<&str>,
) -> Result<Bytes, String> {
    let chunk = SseChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [SseChunkChoice {
            index: 0,
            delta: SseChunkDelta {
                role,
                content,
                tool_calls,
            },
            finish_reason,
        }],
    };
    let mut frame = Vec::with_capacity(192 + content.map_or(0, str::len));
    frame.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut frame, &chunk)
        .map_err(|error| format!("Failed to serialize SSE frame: {error}"))?;
    frame.extend_from_slice(b"\n\n");
    Ok(Bytes::from(frame))
}

impl TextSink for SseSink {
    /// TextSink impl for SSE: streams one content delta (empty chunks ignored)
    /// and aborts the generation once the client disconnected.
    fn push(&mut self, text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        if self
            .disconnected
            .as_ref()
            .is_some_and(|disconnected| disconnected.load(Ordering::Relaxed))
        {
            return Err("Client disconnected".to_string());
        }
        if self.first_push_at.is_none() {
            self.first_push_at = Some(Instant::now());
        }
        let role = self.first.then_some("assistant");
        self.first = false;
        self.send_chunk(role, Some(text), None, None)
    }

    /// Always false — streamed text is never retained server-side.
    fn capture_text(&self) -> bool {
        false
    }

    /// True once the client disconnected from the SSE stream.
    fn interrupted(&mut self) -> bool {
        self.disconnected
            .as_ref()
            .is_some_and(|disconnected| disconnected.load(Ordering::Relaxed))
    }
}

// ---- SSE body ----

/// Builds the `text/event-stream` response over the frame channel, with
/// CORS + no-cache headers and the disconnect flag wired to the sink. hyper
/// flushes the body as the stream yields, so frames reach the client live.
fn sse_response(
    receiver: UnboundedReceiver<Bytes>,
    heartbeat_frame: Bytes,
    disconnected: Option<Arc<AtomicBool>>,
) -> Response {
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache");
    for (name, value) in cors_headers() {
        response = response.header(name, value);
    }
    let body = SseBody {
        receiver,
        heartbeat_frame,
        heartbeat: Box::pin(tokio::time::sleep(SSE_HEARTBEAT_INTERVAL)),
        disconnected,
    };
    response.body(Body::from_stream(body)).expect("sse body")
}

/// The SSE response body: drains the frame channel, emits the heartbeat
/// frame when the stream goes idle past [`SSE_HEARTBEAT_INTERVAL`], and marks
/// the client disconnected on drop (the generation sink polls that flag and
/// stops).
struct SseBody {
    receiver: UnboundedReceiver<Bytes>,
    heartbeat_frame: Bytes,
    heartbeat: std::pin::Pin<Box<Sleep>>,
    disconnected: Option<Arc<AtomicBool>>,
}

impl SseBody {
    /// Resets the idle timer after every emitted frame.
    fn reset_heartbeat(&mut self) {
        self.heartbeat
            .as_mut()
            .reset(tokio::time::Instant::now() + SSE_HEARTBEAT_INTERVAL);
    }
}

impl futures_core::Stream for SseBody {
    type Item = Result<Bytes, std::convert::Infallible>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Idle past the interval → emit the heartbeat frame and re-arm.
        if self.heartbeat.as_mut().poll(cx).is_ready() {
            self.reset_heartbeat();
            return std::task::Poll::Ready(Some(Ok(self.heartbeat_frame.clone())));
        }
        match self.receiver.poll_recv(cx) {
            std::task::Poll::Ready(Some(chunk)) => {
                self.reset_heartbeat();
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            // The generation task finished (sender dropped): end the stream.
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for SseBody {
    /// Marks the client disconnected — the generation sink observes the flag
    /// on its next delta and stops.
    fn drop(&mut self) {
        if let Some(disconnected) = &self.disconnected {
            disconnected.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_compares_bearer_tokens() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn prompt_context_overflow_maps_to_bad_request() {
        assert_eq!(
            prompt_error_status(
                "Input plus reserved generation tokens exceeds the model context limit of 64"
            ),
            400
        );
        assert_eq!(prompt_error_status("Failed to render prompt"), 500);
    }

    #[test]
    fn routes_known_paths() {
        assert!(matches!(
            route(&Method::GET, "/health"),
            Ok(RouteEnum::Health)
        ));
        assert!(matches!(
            route(&Method::GET, "/v1/models"),
            Ok(RouteEnum::ListModels)
        ));
        assert!(matches!(
            route(&Method::GET, "/v1/models/qwen2"),
            Ok(RouteEnum::GetModel(id)) if id == "qwen2"
        ));
        assert!(matches!(
            route(&Method::POST, "/v1/chat/completions"),
            Ok(RouteEnum::ChatCompletions)
        ));
        assert!(matches!(
            route(&Method::POST, "/v1/completions"),
            Ok(RouteEnum::Completions)
        ));
    }

    #[test]
    fn routes_return_405_wrong_method_and_404_unknown() {
        assert_eq!(
            route(&Method::POST, "/v1/models"),
            Err(StatusCode::METHOD_NOT_ALLOWED)
        );
        assert_eq!(
            route(&Method::GET, "/v1/chat/completions"),
            Err(StatusCode::METHOD_NOT_ALLOWED)
        );
        assert_eq!(
            route(&Method::GET, "/v1/unknown"),
            Err(StatusCode::NOT_FOUND)
        );
    }

    #[test]
    fn sse_tool_call_delta_uses_openai_fields() {
        let frame = sse_frame(
            "chatcmpl-1",
            1,
            "qwen",
            Some("assistant"),
            None,
            Some(vec![ToolCallDelta {
                index: 0,
                id: Some("call_1".to_string()),
                kind: Some("function"),
                function: ToolCallFunctionDelta {
                    name: Some("weather".to_string()),
                    arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
                },
            }]),
            None,
        )
        .unwrap();
        let payload = std::str::from_utf8(&frame[6..frame.len() - 2]).unwrap();
        let chunk: serde_json::Value = serde_json::from_str(payload).unwrap();
        let delta = &chunk["choices"][0]["delta"];
        assert_eq!(delta["role"], "assistant");
        assert_eq!(delta["tool_calls"][0]["index"], 0);
        assert_eq!(delta["tool_calls"][0]["id"], "call_1");
        assert_eq!(delta["tool_calls"][0]["type"], "function");
        assert_eq!(
            delta["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Tokyo"}"#
        );
    }

    /// The SSE sink emits one frame per push with the OpenAI delta fields,
    /// ignoring empty chunks and carrying the assistant role once.
    #[tokio::test]
    async fn chat_stream_emits_each_generated_chunk_without_copying_text() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
        let mut sink = SseSink::new(tx, "chatcmpl-test".to_string(), 1, "test".to_string());
        assert!(!sink.capture_text());
        sink.push("plain ").unwrap();
        sink.push("response").unwrap();
        sink.send_chunk(None, None, None, Some("stop")).unwrap();
        sink.send_done().unwrap();

        let content_frame = rx.try_recv().unwrap();
        let content_json = std::str::from_utf8(&content_frame[6..content_frame.len() - 2]).unwrap();
        let content_chunk: serde_json::Value = serde_json::from_str(content_json).unwrap();
        assert_eq!(content_chunk["choices"][0]["delta"]["content"], "plain ");

        let next_content_frame = rx.try_recv().unwrap();
        let next_content_json =
            std::str::from_utf8(&next_content_frame[6..next_content_frame.len() - 2]).unwrap();
        let next_content_chunk: serde_json::Value =
            serde_json::from_str(next_content_json).unwrap();
        assert_eq!(
            next_content_chunk["choices"][0]["delta"]["content"],
            "response"
        );

        let finish_frame = rx.try_recv().unwrap();
        let finish_json = std::str::from_utf8(&finish_frame[6..finish_frame.len() - 2]).unwrap();
        let finish_chunk: serde_json::Value = serde_json::from_str(finish_json).unwrap();
        assert_eq!(finish_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(rx.try_recv().unwrap().as_ref(), b"data: [DONE]\n\n");
    }

    #[test]
    fn non_stream_chat_converts_tool_json_and_falls_back_to_text() {
        let (content, calls, finish_reason) = chat_completion_parts(
            r#"{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}"#,
            "stop",
        );
        assert!(content.is_none());
        assert_eq!(finish_reason, "tool_calls");
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Tokyo"}"#);

        let (content, calls, finish_reason) = chat_completion_parts("plain response", "length");
        assert_eq!(content.as_deref(), Some("plain response"));
        assert!(calls.is_none());
        assert_eq!(finish_reason, "length");
    }

    #[test]
    fn sse_sink_ignores_empty_generated_chunks() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
        let mut sink = SseSink {
            tx,
            id: "test".to_string(),
            created: 0,
            model: "test".to_string(),
            first: true,
            first_push_at: None,
            disconnected: None,
        };

        sink.push("").unwrap();
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        sink.push("token").unwrap();
        let chunk: serde_json::Value = serde_json::from_slice(
            &rx.try_recv()
                .unwrap()
                .strip_prefix(b"data: ")
                .unwrap()
                .strip_suffix(b"\n\n")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "token");
    }

    /// The SSE body emits queued frames, EOFs after the sender drops, and
    /// emits the heartbeat frame when the stream goes idle.
    #[tokio::test]
    async fn sse_body_streams_frames_eof_and_heartbeat() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
        let heartbeat = sse_frame("test", 1, "test", None, Some(""), None, None).unwrap();
        let body = SseBody {
            receiver: rx,
            heartbeat_frame: heartbeat.clone(),
            heartbeat: Box::pin(tokio::time::sleep(Duration::from_millis(10))),
            disconnected: None,
        };
        use futures_core::Stream;
        let mut pinned = std::pin::pin!(body);

        tx.send(Bytes::from_static(b"data: token\n\n")).unwrap();
        match pinned
            .as_mut()
            .poll_next(&mut std::task::Context::from_waker(
                futures_core::task::Waker::noop(),
            )) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                assert_eq!(bytes.as_ref(), b"data: token\n\n")
            }
            other => panic!("expected the token frame, got {other:?}"),
        }
        drop(tx);
        // EOF: the sender dropped, so the stream ends.
        match pinned
            .as_mut()
            .poll_next(&mut std::task::Context::from_waker(
                futures_core::task::Waker::noop(),
            )) {
            std::task::Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        let _ = heartbeat;
    }

    /// Dropping the SSE body marks the client disconnected — the generation
    /// sink observes the flag and stops.
    #[tokio::test]
    async fn dropping_sse_body_marks_disconnected() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
        let disconnected = Arc::new(AtomicBool::new(false));
        let body = SseBody {
            receiver: rx,
            heartbeat_frame: Bytes::from_static(b"heartbeat"),
            heartbeat: Box::pin(tokio::time::sleep(Duration::from_millis(10))),
            disconnected: Some(Arc::clone(&disconnected)),
        };
        let mut sink = SseSink::new_with_disconnect(
            tx,
            "test".into(),
            0,
            "test".into(),
            Arc::clone(&disconnected),
        );
        assert!(sink.push("token").is_ok());
        drop(body);
        assert!(disconnected.load(Ordering::Relaxed));
        assert!(sink.interrupted());
        assert_eq!(sink.push("token"), Err("Client disconnected".to_string()));
    }
}
