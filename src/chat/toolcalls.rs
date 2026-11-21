//! Tool-call shape parsing shared by every front end (the OpenAI function
//! calling JSON shapes are the de-facto tool protocol for both the HTTP API
//! and the local REPL). Pure serde_json + std: no HTTP types live here, so
//! the engine layer can run tool loops without reaching into `serve`.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::hash::{BuildHasher, Hasher};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ToolCallFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ToolDefinition {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

#[derive(Debug)]
pub(crate) enum ToolResponse {
    Text(String),
    Calls(Vec<ToolCall>),
}

#[derive(Debug, Deserialize)]
struct ParsedToolCall {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<Value>,
    #[serde(default)]
    function: Option<ParsedToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct ParsedToolCallFunction {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ParsedAssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ParsedToolCall>>,
}

/// Tool-output resolution ladder: template-parsed `tool_calls` → compatible
/// raw JSON shapes → free-text JSON extraction validated against the tools
/// and choice → plain-text fallback.
pub(crate) fn parse_tool_response(
    parsed: &str,
    raw_text: &str,
    tools: &[ToolDefinition],
    choice: &ResolvedToolChoice,
) -> ToolResponse {
    // Prefer the model template's parsed form, then accept compatible raw JSON shapes before falling back to text.
    // 优先使用模型模板解析结果，再兼容原始 JSON 工具调用格式；均无法识别时保留为普通文本。
    let parsed_message = serde_json::from_str::<ParsedAssistantMessage>(parsed).ok();
    let nested_message = parsed_message
        .as_ref()
        .and_then(|message| message.content.as_deref())
        .and_then(|content| serde_json::from_str::<ParsedAssistantMessage>(content).ok());
    let candidates = parsed_message
        .as_ref()
        .and_then(|message| message.tool_calls.as_deref())
        .or_else(|| {
            nested_message
                .as_ref()
                .and_then(|message| message.tool_calls.as_deref())
        });

    if let Some(calls) = candidates
        && !calls.is_empty()
        && let Some(calls) = convert_tool_calls(calls, tools, choice)
    {
        return ToolResponse::Calls(calls);
    }

    if let Ok(message) = serde_json::from_str::<ParsedAssistantMessage>(raw_text)
        && let Some(calls) = message.tool_calls
        && !calls.is_empty()
        && let Some(calls) = convert_tool_calls(&calls, tools, choice)
    {
        return ToolResponse::Calls(calls);
    }

    if let Some(calls) = parse_generated_tool_calls(raw_text) {
        let valid_calls: Vec<_> = calls
            .into_iter()
            .filter(|call| {
                !matches!(choice, ResolvedToolChoice::None)
                    && tools
                        .iter()
                        .any(|tool| tool.function.name == call.function.name)
            })
            .collect();
        if !valid_calls.is_empty() {
            return ToolResponse::Calls(valid_calls);
        }
    }

    ToolResponse::Text(
        parsed_message
            .as_ref()
            .and_then(|message| message.content.clone())
            .or_else(|| nested_message.and_then(|message| message.content))
            .unwrap_or_else(|| raw_text.to_string()),
    )
}

/// Parses tool calls from free model text (compact, OpenAI, or flat shapes)
/// after fence/brace extraction; invalid shapes yield None so the text
/// fallback takes over.
pub(crate) fn parse_generated_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let json = extract_generated_tool_call_json(text)?;
    let calls: Vec<ParsedToolCall> = serde_json::from_str::<ParsedAssistantMessage>(json)
        .ok()
        .and_then(|message| message.tool_calls)
        .filter(|calls| !calls.is_empty())
        .or_else(|| {
            serde_json::from_str::<ParsedToolCall>(json)
                .ok()
                .map(|call| vec![call])
        })?;
    calls
        .iter()
        .map(|call| {
            if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                return None;
            }
            let name = call
                .function
                .as_ref()
                .map(|function| &function.name)
                .or(call.name.as_ref())?;
            if name.is_empty() {
                return None;
            }
            let arguments = call
                .function
                .as_ref()
                .map(|function| &function.arguments)
                .or(call.arguments.as_ref())?;
            Some(ToolCall {
                id: tool_call_id(),
                kind: "function",
                function: ToolCallFunction {
                    name: name.clone(),
                    arguments: normalize_tool_arguments(arguments)?,
                },
            })
        })
        .collect()
}

/// Strips code fences and slices the outermost JSON object out of the
/// model text.
fn extract_generated_tool_call_json(text: &str) -> Option<&str> {
    let mut text = text.trim();
    if text.starts_with("```") {
        if let Some(body) = text.strip_prefix("```json") {
            text = body;
        } else if let Some(body) = text.strip_prefix("```") {
            text = body;
        }
        text = text.trim();
    }

    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (start <= end).then(|| &text[start..=end])
}

/// Converts parsed calls to the response shape, filtering by tool existence
/// and a named `tool_choice`; an empty result falls back to text.
fn convert_tool_calls(
    calls: &[ParsedToolCall],
    tools: &[ToolDefinition],
    choice: &ResolvedToolChoice,
) -> Option<Vec<ToolCall>> {
    calls
        .iter()
        .map(|call| {
            let name = call
                .function
                .as_ref()
                .map(|function| &function.name)
                .or(call.name.as_ref())?;
            if matches!(choice, ResolvedToolChoice::None)
                || call.kind.as_deref().is_some_and(|kind| kind != "function")
                || !tools.iter().any(|tool| tool.function.name == *name)
                || matches!(choice, ResolvedToolChoice::Named(selected) if selected != name)
            {
                return None;
            }
            let arguments = call
                .function
                .as_ref()
                .map(|function| &function.arguments)
                .or(call.arguments.as_ref())?;
            Some(ToolCall {
                id: tool_call_id(),
                kind: "function",
                function: ToolCallFunction {
                    name: name.clone(),
                    arguments: normalize_tool_arguments(arguments)?,
                },
            })
        })
        .collect()
}

/// Accepts arguments as a JSON object or a JSON-encoded object string; both
/// validate and normalize to object text.
pub(crate) fn normalize_tool_arguments(arguments: &Value) -> Option<String> {
    // OpenAI accepts arguments as an object or JSON-encoded object string; normalize both for the response.
    // OpenAI 参数既可能是对象，也可能是 JSON 字符串；这里统一校验并序列化为对象文本。
    let arguments = match arguments {
        Value::String(value) => serde_json::from_str(value).ok()?,
        Value::Object(_) => arguments.clone(),
        _ => return None,
    };
    arguments.is_object().then(|| arguments.to_string())
}

/// Generates `call_<hex>` ids for outgoing tool calls.
pub(crate) fn tool_call_id() -> String {
    format!("call_{}", random_hex(12))
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: ToolCallFunction,
}

/// 由操作系统熵播种的随机十六进制字符串(适合生成本地 API Key / 请求 ID)。
pub(crate) fn random_hex(bytes: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes * 2);
    for _ in 0..bytes {
        let value = random_u64() as u8;
        out.push(HEX[usize::from(value >> 4)] as char);
        out.push(HEX[usize::from(value & 0x0f)] as char);
    }
    out
}

/// Entropy source built from `RandomState` (per-process randomized hasher
/// seed) — enough for local keys and ids, not a CSPRNG contract.
fn random_u64() -> u64 {
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

/// Build the next tool-loop round's message list: the assistant's tool calls
/// (serialized into its content — the same history convention the request
/// parser applies) followed by one `role: "tool"` result message per call.
pub(crate) fn next_tool_round(
    messages_json: &str,
    calls: &[ToolCall],
    results: &[(String, String)],
) -> Result<String, String> {
    let mut messages: Vec<Value> = serde_json::from_str(messages_json)
        .map_err(|error| format!("Tool loop messages are not valid JSON: {error}"))?;
    let calls_json = serde_json::to_string(calls)
        .map_err(|error| format!("Failed to serialize tool calls: {error}"))?;
    messages.push(json!({"role": "assistant", "content": calls_json}));
    for (id, text) in results {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": text,
        }));
    }
    serde_json::to_string(&messages)
        .map_err(|error| format!("Failed to serialize tool-loop messages: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The next round carries the assistant's serialized tool calls plus one
    /// `role: "tool"` result per call, preserving the earlier history.
    #[test]
    fn next_tool_round_appends_calls_and_results() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            kind: "function",
            function: ToolCallFunction {
                name: "weather".into(),
                arguments: r#"{"city":"Oslo"}"#.into(),
            },
        }];
        let messages = r#"[{"role":"user","content":"weather?"}]"#;
        let next = next_tool_round(messages, &calls, &[("call_1".into(), "sunny".into())]).unwrap();
        let parsed: Vec<Value> = serde_json::from_str(&next).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[1]["role"], "assistant");
        assert!(parsed[1]["content"].as_str().unwrap().contains("weather"));
        assert!(parsed[1].get("tool_call_id").is_none());
        assert_eq!(parsed[2]["role"], "tool");
        assert_eq!(parsed[2]["tool_call_id"], "call_1");
        assert_eq!(parsed[2]["content"], "sunny");
    }

    /// Non-JSON input history is a hard error, never silently reset.
    #[test]
    fn next_tool_round_rejects_invalid_history() {
        let calls: Vec<ToolCall> = Vec::new();
        assert!(next_tool_round("not json", &calls, &[]).is_err());
    }
}
