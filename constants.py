# Phi-3.5-mini-instruct 对话模板（需根据实际格式调整）
phi_template = {
    "system_format": "<|system|>\n{content}</s>",
    "user_format": "<|user|>\n{content}</s>",
    "assistant_format": "<|assistant|>\n{content}</s>",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|observation|>\n{content}</s>",
    "system": "You are an AI assistant.",
}

model2template = {
    "microsoft/Phi-3.5-mini-instruct": phi_template
}

model2size = {
    "microsoft/Phi-3.5-mini-instruct": 4_000_000_000  # 假设 4B 参数
}

model2base_model = {
    "microsoft/Phi-3.5-mini-instruct": "phi3"
}
