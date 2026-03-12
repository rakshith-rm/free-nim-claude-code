"""Pydantic models for Anthropic-compatible API requests and responses."""

from typing import Any, Literal

from pydantic import BaseModel


# --- Content blocks ---

class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[Any] | dict[str, Any]


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


# --- Messages ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: (
        str
        | list[
            ContentBlockText
            | ContentBlockImage
            | ContentBlockToolUse
            | ContentBlockToolResult
            | ContentBlockThinking
        ]
    )


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


# --- Request ---

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    extra_body: dict[str, Any] | None = None


class TokenCountRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None


# --- Responses ---

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlockText | ContentBlockToolUse | ContentBlockThinking | dict[str, Any]]
    type: Literal["message"] = "message"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: Usage


class TokenCountResponse(BaseModel):
    input_tokens: int
