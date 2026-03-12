"""SSE event builder, think-tag parser, and heuristic tool parser.

Converts OpenAI streaming chunks into Anthropic SSE format that Claude Code CLI expects.
"""

import json
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


# =============================================================================
# Stop reason mapping
# =============================================================================

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    return STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"


# =============================================================================
# Think tag parser
# =============================================================================

class ContentType(Enum):
    TEXT = "text"
    THINKING = "thinking"


@dataclass
class ContentChunk:
    type: ContentType
    content: str


class ThinkTagParser:
    """Streaming parser for <think>...</think> tags."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self._buf = ""
        self._inside = False

    def feed(self, content: str) -> Iterator[ContentChunk]:
        self._buf += content
        while self._buf:
            prev = len(self._buf)
            chunk = self._parse_inside() if self._inside else self._parse_outside()
            if chunk:
                yield chunk
            elif len(self._buf) == prev:
                break

    def _parse_outside(self) -> ContentChunk | None:
        idx = self._buf.find(self.OPEN)
        orphan = self._buf.find(self.CLOSE)

        if orphan != -1 and (idx == -1 or orphan < idx):
            pre = self._buf[:orphan]
            self._buf = self._buf[orphan + 8:]
            return ContentChunk(ContentType.TEXT, pre) if pre else None

        if idx == -1:
            last = self._buf.rfind("<")
            if last != -1:
                tail = self._buf[last:]
                if (len(tail) < 7 and self.OPEN.startswith(tail)) or (len(tail) < 8 and self.CLOSE.startswith(tail)):
                    emit = self._buf[:last]
                    self._buf = self._buf[last:]
                    return ContentChunk(ContentType.TEXT, emit) if emit else None
            emit = self._buf
            self._buf = ""
            return ContentChunk(ContentType.TEXT, emit) if emit else None

        pre = self._buf[:idx]
        self._buf = self._buf[idx + 7:]
        self._inside = True
        return ContentChunk(ContentType.TEXT, pre) if pre else None

    def _parse_inside(self) -> ContentChunk | None:
        idx = self._buf.find(self.CLOSE)
        if idx == -1:
            last = self._buf.rfind("<")
            if last != -1 and len(self._buf) - last < 8:
                tail = self._buf[last:]
                if self.CLOSE.startswith(tail):
                    emit = self._buf[:last]
                    self._buf = self._buf[last:]
                    return ContentChunk(ContentType.THINKING, emit) if emit else None
            emit = self._buf
            self._buf = ""
            return ContentChunk(ContentType.THINKING, emit) if emit else None

        thinking = self._buf[:idx]
        self._buf = self._buf[idx + 8:]
        self._inside = False
        return ContentChunk(ContentType.THINKING, thinking) if thinking else None

    def flush(self) -> ContentChunk | None:
        if self._buf:
            ct = ContentType.THINKING if self._inside else ContentType.TEXT
            c = self._buf
            self._buf = ""
            return ContentChunk(ct, c)
        return None


# =============================================================================
# Heuristic tool parser (fallback for models that emit tool calls as text)
# =============================================================================

_CONTROL_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_FUNC_RE = re.compile(r"●\s*<function=([^>]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL)


class HeuristicToolParser:
    def __init__(self):
        self._buf = ""
        self._state = "text"
        self._tool_id = None
        self._func_name = None
        self._params: dict[str, str] = {}

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        self._buf += text
        self._buf = _CONTROL_RE.sub("", self._buf)
        tools: list[dict[str, Any]] = []
        parts: list[str] = []

        while True:
            if self._state == "text":
                if "●" in self._buf:
                    i = self._buf.find("●")
                    parts.append(self._buf[:i])
                    self._buf = self._buf[i:]
                    self._state = "match_func"
                else:
                    parts.append(self._buf)
                    self._buf = ""
                    break

            if self._state == "match_func":
                m = _FUNC_RE.search(self._buf)
                if m:
                    self._func_name = m.group(1).strip()
                    self._tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._params = {}
                    self._buf = self._buf[m.end():]
                    self._state = "parse_params"
                elif len(self._buf) > 100:
                    parts.append(self._buf[0])
                    self._buf = self._buf[1:]
                    self._state = "text"
                else:
                    break

            if self._state == "parse_params":
                done = False
                while True:
                    pm = _PARAM_RE.search(self._buf)
                    if pm and "</parameter>" in pm.group(0):
                        pre = self._buf[:pm.start()]
                        if pre:
                            parts.append(pre)
                        self._params[pm.group(1).strip()] = pm.group(2).strip()
                        self._buf = self._buf[pm.end():]
                    else:
                        break

                if "●" in self._buf:
                    i = self._buf.find("●")
                    if i > 0:
                        parts.append(self._buf[:i])
                        self._buf = self._buf[i:]
                    done = True
                elif self._buf and not self._buf.strip().startswith("<") and "<parameter=" not in self._buf:
                    parts.append(self._buf)
                    self._buf = ""
                    done = True

                if done:
                    tools.append({
                        "type": "tool_use",
                        "id": self._tool_id,
                        "name": self._func_name,
                        "input": self._params,
                    })
                    self._state = "text"
                else:
                    break

        return "".join(parts), tools

    def flush(self) -> list[dict[str, Any]]:
        self._buf = _CONTROL_RE.sub("", self._buf)
        if self._state == "parse_params":
            for m in re.finditer(r"<parameter=([^>]+)>(.*)$", self._buf, re.DOTALL):
                self._params[m.group(1).strip()] = m.group(2).strip()
            self._state = "text"
            self._buf = ""
            return [{"type": "tool_use", "id": self._tool_id, "name": self._func_name, "input": self._params}]
        return []


# =============================================================================
# Content block state tracking
# =============================================================================

@dataclass
class ToolCallState:
    block_index: int
    tool_id: str
    name: str
    contents: list[str] = field(default_factory=list)
    started: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False


@dataclass
class BlockManager:
    next_index: int = 0
    thinking_index: int = -1
    text_index: int = -1
    thinking_started: bool = False
    text_started: bool = False
    tool_states: dict[int, ToolCallState] = field(default_factory=dict)

    def alloc(self) -> int:
        idx = self.next_index
        self.next_index += 1
        return idx

    def register_tool_name(self, index: int, name: str) -> None:
        if index not in self.tool_states:
            self.tool_states[index] = ToolCallState(block_index=-1, tool_id="", name=name)
            return
        s = self.tool_states[index]
        if not s.name or name.startswith(s.name):
            s.name = name
        elif not s.name.startswith(name):
            s.name = s.name + name

    def buffer_task_args(self, index: int, args: str) -> dict | None:
        s = self.tool_states.get(index)
        if s is None or s.task_args_emitted:
            return None
        s.task_arg_buffer += args
        try:
            parsed = json.loads(s.task_arg_buffer)
        except Exception:
            return None
        if parsed.get("run_in_background") is not False:
            parsed["run_in_background"] = False
        s.task_args_emitted = True
        s.task_arg_buffer = ""
        return parsed

    def flush_task_buffers(self) -> list[tuple[int, str]]:
        results = []
        for ti, s in list(self.tool_states.items()):
            if not s.task_arg_buffer or s.task_args_emitted:
                continue
            out = "{}"
            try:
                parsed = json.loads(s.task_arg_buffer)
                if parsed.get("run_in_background") is not False:
                    parsed["run_in_background"] = False
                out = json.dumps(parsed)
            except Exception:
                pass
            s.task_args_emitted = True
            s.task_arg_buffer = ""
            results.append((ti, out))
        return results


# =============================================================================
# SSE Builder - generates Anthropic SSE events
# =============================================================================

class SSEBuilder:
    def __init__(self, message_id: str, model: str, input_tokens: int = 0):
        self.message_id = message_id
        self.model = model
        self.input_tokens = input_tokens
        self.blocks = BlockManager()
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []

    def _event(self, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # --- Message lifecycle ---

    def message_start(self) -> str:
        return self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 1},
            },
        })

    def message_delta(self, stop_reason: str, output_tokens: int) -> str:
        return self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": self.input_tokens, "output_tokens": output_tokens},
        })

    def message_stop(self) -> str:
        return self._event("message_stop", {"type": "message_stop"})

    # --- Block lifecycle ---

    def block_start(self, index: int, block_type: str, **kw) -> str:
        cb: dict[str, Any] = {"type": block_type}
        if block_type == "thinking":
            cb["thinking"] = kw.get("thinking", "")
        elif block_type == "text":
            cb["text"] = kw.get("text", "")
        elif block_type == "tool_use":
            cb["id"] = kw.get("id", "")
            cb["name"] = kw.get("name", "")
            cb["input"] = kw.get("input", {})
        return self._event("content_block_start", {"type": "content_block_start", "index": index, "content_block": cb})

    def block_delta(self, index: int, delta_type: str, content: str) -> str:
        delta: dict[str, Any] = {"type": delta_type}
        if delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content
        return self._event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

    def block_stop(self, index: int) -> str:
        return self._event("content_block_stop", {"type": "content_block_stop", "index": index})

    # --- High-level helpers ---

    def start_thinking(self) -> str:
        self.blocks.thinking_index = self.blocks.alloc()
        self.blocks.thinking_started = True
        return self.block_start(self.blocks.thinking_index, "thinking")

    def thinking_delta(self, content: str) -> str:
        self._reasoning_parts.append(content)
        return self.block_delta(self.blocks.thinking_index, "thinking_delta", content)

    def stop_thinking(self) -> str:
        self.blocks.thinking_started = False
        return self.block_stop(self.blocks.thinking_index)

    def start_text(self) -> str:
        self.blocks.text_index = self.blocks.alloc()
        self.blocks.text_started = True
        return self.block_start(self.blocks.text_index, "text")

    def text_delta(self, content: str) -> str:
        self._text_parts.append(content)
        return self.block_delta(self.blocks.text_index, "text_delta", content)

    def stop_text(self) -> str:
        self.blocks.text_started = False
        return self.block_stop(self.blocks.text_index)

    def start_tool(self, tool_index: int, tool_id: str, name: str) -> str:
        bi = self.blocks.alloc()
        if tool_index in self.blocks.tool_states:
            s = self.blocks.tool_states[tool_index]
            s.block_index = bi
            s.tool_id = tool_id
            s.started = True
        else:
            self.blocks.tool_states[tool_index] = ToolCallState(block_index=bi, tool_id=tool_id, name=name, started=True)
        return self.block_start(bi, "tool_use", id=tool_id, name=name)

    def tool_delta(self, tool_index: int, partial_json: str) -> str:
        s = self.blocks.tool_states[tool_index]
        s.contents.append(partial_json)
        return self.block_delta(s.block_index, "input_json_delta", partial_json)

    # --- State transitions ---

    def ensure_thinking(self) -> Iterator[str]:
        if self.blocks.text_started:
            yield self.stop_text()
        if not self.blocks.thinking_started:
            yield self.start_thinking()

    def ensure_text(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if not self.blocks.text_started:
            yield self.start_text()

    def close_content(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if self.blocks.text_started:
            yield self.stop_text()

    def close_all(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking()
        if self.blocks.text_started:
            yield self.stop_text()
        for ti, s in list(self.blocks.tool_states.items()):
            if s.started:
                yield self.block_stop(s.block_index)

    def emit_error(self, msg: str) -> Iterator[str]:
        ei = self.blocks.alloc()
        yield self.block_start(ei, "text")
        yield self.block_delta(ei, "text_delta", msg)
        yield self.block_stop(ei)

    # --- Token estimation ---

    def estimate_output_tokens(self) -> int:
        text = "".join(self._text_parts)
        reasoning = "".join(self._reasoning_parts)
        if ENCODER:
            tt = len(ENCODER.encode(text))
            rt = len(ENCODER.encode(reasoning))
            tool_t = sum(
                len(ENCODER.encode(s.name)) + len(ENCODER.encode("".join(s.contents))) + 15
                for s in self.blocks.tool_states.values()
            )
            blocks = (1 if reasoning else 0) + (1 if text else 0) + sum(1 for s in self.blocks.tool_states.values() if s.started)
            return tt + rt + tool_t + blocks * 4
        return len(text) // 4 + len(reasoning) // 4 + sum(1 for s in self.blocks.tool_states.values() if s.started) * 50
