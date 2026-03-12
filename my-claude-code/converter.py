"""Convert Anthropic message format to OpenAI format for NIM."""

import json
from typing import Any


def _get_attr(block: Any, attr: str, default: Any = None) -> Any:
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def _get_type(block: Any) -> str | None:
    return _get_attr(block, "type")


def convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic messages to OpenAI format."""
    result = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            result.append({"role": msg.role, "content": content})
        elif isinstance(content, list):
            if msg.role == "assistant":
                result.extend(_convert_assistant(content))
            elif msg.role == "user":
                result.extend(_convert_user(content))
        else:
            result.append({"role": msg.role, "content": str(content)})
    return result


def _convert_assistant(content: list[Any]) -> list[dict[str, Any]]:
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        bt = _get_type(block)
        if bt == "text":
            content_parts.append(_get_attr(block, "text", ""))
        elif bt == "thinking":
            thinking = _get_attr(block, "thinking", "")
            content_parts.append(f"<think>\n{thinking}\n</think>")
        elif bt == "tool_use":
            tool_input = _get_attr(block, "input", {})
            tool_calls.append({
                "id": _get_attr(block, "id"),
                "type": "function",
                "function": {
                    "name": _get_attr(block, "name"),
                    "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                },
            })

    content_str = "\n\n".join(content_parts)
    if not content_str and not tool_calls:
        content_str = " "

    msg: dict[str, Any] = {"role": "assistant", "content": content_str}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _convert_user(content: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    text_parts: list[str] = []

    def flush():
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
            text_parts.clear()

    for block in content:
        bt = _get_type(block)
        if bt == "text":
            text_parts.append(_get_attr(block, "text", ""))
        elif bt == "tool_result":
            flush()
            tool_content = _get_attr(block, "content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in tool_content
                )
            result.append({
                "role": "tool",
                "tool_call_id": _get_attr(block, "tool_use_id"),
                "content": str(tool_content) if tool_content else "",
            })

    flush()
    return result


def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic tools to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def convert_system(system: Any) -> dict[str, str] | None:
    """Convert Anthropic system prompt to OpenAI format."""
    if isinstance(system, str):
        return {"role": "system", "content": system}
    elif isinstance(system, list):
        parts = [_get_attr(b, "text", "") for b in system if _get_type(b) == "text"]
        if parts:
            return {"role": "system", "content": "\n\n".join(parts).strip()}
    return None


def build_nim_request(request_data: Any) -> dict[str, Any]:
    """Build complete OpenAI-format request body from Anthropic request for NIM."""
    messages = convert_messages(request_data.messages)

    system = getattr(request_data, "system", None)
    if system:
        sys_msg = convert_system(system)
        if sys_msg:
            messages.insert(0, sys_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = getattr(request_data, "max_tokens", None) or 81920
    body["max_tokens"] = min(max_tokens, 81920)

    temp = getattr(request_data, "temperature", None)
    if temp is not None:
        body["temperature"] = temp
    else:
        body["temperature"] = 1.0

    top_p = getattr(request_data, "top_p", None)
    if top_p is not None:
        body["top_p"] = top_p

    stop = getattr(request_data, "stop_sequences", None)
    if stop:
        body["stop"] = stop

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = tool_choice

    body["parallel_tool_calls"] = True

    extra_body: dict[str, Any] = {}
    req_extra = getattr(request_data, "extra_body", None)
    if req_extra:
        extra_body.update(req_extra)

    extra_body.setdefault("thinking", {"type": "enabled"})
    extra_body.setdefault("reasoning_split", True)
    extra_body.setdefault("chat_template_kwargs", {
        "thinking": True,
        "enable_thinking": True,
        "reasoning_split": True,
        "clear_thinking": False,
    })
    extra_body.setdefault("reasoning_effort", "high")
    extra_body.setdefault("include_reasoning", True)

    body["extra_body"] = extra_body
    return body


def extract_text_from_content(content: Any) -> str:
    """Extract text from message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(b, "text", "") for b in content if hasattr(b, "text"))
    return ""
