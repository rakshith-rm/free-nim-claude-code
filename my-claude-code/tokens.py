"""Token counting for Anthropic API requests."""

import json

import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")


def _attr(block, key, default=None):
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key, default)
    return default


def count_tokens(messages: list, system=None, tools=None) -> int:
    total = 0

    if system:
        if isinstance(system, str):
            total += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for b in system:
                t = _attr(b, "text", "")
                if t:
                    total += len(ENCODER.encode(str(t)))
        total += 4

    for msg in messages:
        if isinstance(msg.content, str):
            total += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                bt = _attr(block, "type")
                if bt == "text":
                    total += len(ENCODER.encode(str(_attr(block, "text", ""))))
                elif bt == "thinking":
                    total += len(ENCODER.encode(str(_attr(block, "thinking", ""))))
                elif bt == "tool_use":
                    total += len(ENCODER.encode(str(_attr(block, "name", ""))))
                    total += len(ENCODER.encode(json.dumps(_attr(block, "input", {}))))
                    total += len(ENCODER.encode(str(_attr(block, "id", ""))))
                    total += 15
                elif bt == "tool_result":
                    c = _attr(block, "content", "")
                    total += len(ENCODER.encode(str(c) if isinstance(c, str) else json.dumps(c)))
                    total += len(ENCODER.encode(str(_attr(block, "tool_use_id", ""))))
                    total += 8
                elif bt == "image":
                    total += 765

    if tools:
        for tool in tools:
            total += len(ENCODER.encode(tool.name + (tool.description or "") + json.dumps(tool.input_schema)))

    total += len(messages) * 4
    if tools:
        total += len(tools) * 5

    return max(1, total)
