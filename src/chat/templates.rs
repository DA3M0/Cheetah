use crate::chat::ChatMessage;

pub(super) fn qwen(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut rendered = String::new();
    if let Some(system) = system {
        rendered.push_str("<|im_start|>system\n");
        rendered.push_str(system);
        rendered.push_str("<|im_end|>\n");
    }
    for message in messages {
        rendered.push_str("<|im_start|>");
        rendered.push_str(&message.role);
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<|im_end|>\n");
    }
    rendered.push_str("<|im_start|>assistant\n");
    rendered
}

pub(super) fn gemma(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut rendered = String::new();
    for (index, message) in messages.iter().enumerate() {
        rendered.push_str("<start_of_turn>");
        rendered.push_str(if message.role == "assistant" {
            "model"
        } else {
            "user"
        });
        rendered.push('\n');
        if index == 0
            && message.role != "assistant"
            && let Some(system) = system
        {
            rendered.push_str(system);
            rendered.push_str("\n\n");
        }
        rendered.push_str(&message.content);
        rendered.push_str("<end_of_turn>\n");
    }
    rendered.push_str("<start_of_turn>model\n");
    rendered
}

pub(super) fn llama2(messages: &[ChatMessage], system: Option<&str>) -> String {
    instruction(messages, system)
}

pub(super) fn mistral(messages: &[ChatMessage], system: Option<&str>) -> String {
    instruction(messages, system)
}

pub(super) fn mixtral(messages: &[ChatMessage], system: Option<&str>) -> String {
    instruction(messages, system)
}

fn instruction(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut rendered = String::new();
    let mut system_message = system.map(str::to_owned);
    let mut first_user = true;
    for message in messages {
        match message.role.as_str() {
            "system" => system_message = Some(message.content.clone()),
            "assistant" => {
                rendered.push(' ');
                rendered.push_str(&message.content);
                rendered.push_str(" </s>");
            }
            _ => {
                rendered.push_str("<s>[INST] ");
                if first_user && let Some(system) = system_message.as_deref() {
                    rendered.push_str("<<SYS>>\n");
                    rendered.push_str(system);
                    rendered.push_str("\n<</SYS>>\n\n");
                }
                rendered.push_str(&message.content);
                rendered.push_str(" [/INST]");
                first_user = false;
            }
        }
    }
    rendered
}

pub(super) fn llama3(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut rendered = String::from("<|begin_of_text|>");
    if let Some(system) = system {
        rendered.push_str("<|start_header_id|>system<|end_header_id|>\n");
        rendered.push_str(system);
        rendered.push_str("<|eot_id|>");
    }
    for message in messages {
        rendered.push_str("<|start_header_id|>");
        rendered.push_str(&message.role);
        rendered.push_str("<|end_header_id|>\n");
        rendered.push_str(&message.content);
        rendered.push_str("<|eot_id|>");
    }
    rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n");
    rendered
}

pub(super) fn phi(messages: &[ChatMessage], system: Option<&str>) -> String {
    let mut rendered = String::new();
    if let Some(system) = system {
        rendered.push_str("<|system|>\n");
        rendered.push_str(system);
        rendered.push_str("<|end|>\n");
    }
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" => "assistant",
            "system" => "system",
            _ => "user",
        };
        rendered.push_str("<|");
        rendered.push_str(role);
        rendered.push_str("|>\n");
        rendered.push_str(&message.content);
        rendered.push_str("<|end|>\n");
    }
    rendered.push_str("<|assistant|>\n");
    rendered
}
