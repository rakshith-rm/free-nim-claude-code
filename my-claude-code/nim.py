"""NVIDIA NIM streaming provider.

Sends requests to NIM via the OpenAI SDK and streams back Anthropic SSE events.
"""

import asyncio
import json
import random
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import openai
from loguru import logger
from openai import AsyncOpenAI

from converter import build_nim_request
from sse import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


# =============================================================================
# Rate limiter (singleton)
# =============================================================================

class RateLimiter:
    _instance = None

    def __init__(self, rate_limit: int = 40, rate_window: float = 60.0, max_concurrency: int = 5):
        if hasattr(self, "_ok"):
            return
        self._rate_limit = rate_limit
        self._rate_window = rate_window
        self._times: deque[float] = deque()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ok = True

    @classmethod
    def get(cls, **kw) -> "RateLimiter":
        if cls._instance is None:
            cls._instance = cls(**kw)
        return cls._instance

    async def wait(self) -> None:
        now = time.monotonic()
        if now < self._blocked_until:
            await asyncio.sleep(self._blocked_until - now)
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._rate_window
                while self._times and self._times[0] <= cutoff:
                    self._times.popleft()
                if len(self._times) < self._rate_limit:
                    self._times.append(now)
                    return
                wait = max(0.0, (self._times[0] + self._rate_window) - now)
            await asyncio.sleep(wait) if wait > 0 else await asyncio.sleep(0)

    def block(self, seconds: float = 60.0) -> None:
        self._blocked_until = time.monotonic() + seconds

    @asynccontextmanager
    async def slot(self):
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

    async def execute(self, fn, *args, max_retries: int = 3, **kwargs) -> Any:
        last_exc = None
        for attempt in range(1 + max_retries):
            await self.wait()
            try:
                return await fn(*args, **kwargs)
            except openai.RateLimitError as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                delay = min(2.0 * (2 ** attempt), 60.0) + random.uniform(0, 1.0)
                logger.warning(f"Rate limited, retry {attempt + 1}/{max_retries + 1} in {delay:.1f}s")
                self.block(delay)
                await asyncio.sleep(delay)
        raise last_exc


# =============================================================================
# Error mapping
# =============================================================================

def _user_error_msg(e: Exception, timeout_s: float = 300.0) -> str:
    msg = str(e).strip()
    if msg:
        return msg
    if isinstance(e, httpx.ReadTimeout):
        return f"NIM request timed out after {timeout_s:g}s."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to NIM."
    return "NIM request failed."


# =============================================================================
# NIM Provider
# =============================================================================

class NimProvider:
    def __init__(self, api_key: str, timeout: float = 300.0):
        self._api_key = api_key
        self._timeout = timeout
        self._limiter = RateLimiter.get()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=NIM_BASE_URL,
            max_retries=0,
            timeout=httpx.Timeout(timeout, connect=2.0, read=timeout, write=10.0),
        )

    async def cleanup(self) -> None:
        if self._client:
            await self._client.aclose()

    def _process_tool_call(self, tc: dict, sse: SSEBuilder) -> Iterator[str]:
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_states)

        fn = tc.get("function", {})
        name = fn.get("name")
        if name is not None:
            sse.blocks.register_tool_name(tc_index, name)

        state = sse.blocks.tool_states.get(tc_index)
        if state is None or not state.started:
            n = state.name if state else ""
            if n or tc.get("id"):
                yield sse.start_tool(tc_index, tc.get("id") or f"tool_{uuid.uuid4()}", n)

        args = fn.get("arguments", "")
        if args:
            state = sse.blocks.tool_states.get(tc_index)
            if state is None or not state.started:
                yield sse.start_tool(tc_index, tc.get("id") or f"tool_{uuid.uuid4()}", (state.name if state else "") or "tool_call")
                state = sse.blocks.tool_states.get(tc_index)

            if state and state.name == "Task":
                parsed = sse.blocks.buffer_task_args(tc_index, args)
                if parsed is not None:
                    yield sse.tool_delta(tc_index, json.dumps(parsed))
                return

            yield sse.tool_delta(tc_index, args)

    async def stream(self, request: Any, input_tokens: int = 0, request_id: str | None = None) -> AsyncIterator[str]:
        msg_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(msg_id, request.model, input_tokens)
        body = build_nim_request(request)

        logger.info("NIM_STREAM: model={} msgs={} tools={}", body.get("model"), len(body.get("messages", [])), len(body.get("tools", [])))

        yield sse.message_start()

        think_parser = ThinkTagParser()
        tool_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None
        error_occurred = False

        async with self._limiter.slot():
            try:
                stream = await self._limiter.execute(self._client.chat.completions.create, **body, stream=True)
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Reasoning content (OpenAI extended format)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        for ev in sse.ensure_thinking():
                            yield ev
                        yield sse.thinking_delta(reasoning)

                    # Text content (may contain <think> tags)
                    if delta.content:
                        for part in think_parser.feed(delta.content):
                            if part.type == ContentType.THINKING:
                                for ev in sse.ensure_thinking():
                                    yield ev
                                yield sse.thinking_delta(part.content)
                            else:
                                filtered, detected = tool_parser.feed(part.content)
                                if filtered:
                                    for ev in sse.ensure_text():
                                        yield ev
                                    yield sse.text_delta(filtered)
                                for tu in detected:
                                    for ev in sse.close_content():
                                        yield ev
                                    bi = sse.blocks.alloc()
                                    if tu.get("name") == "Task" and isinstance(tu.get("input"), dict):
                                        tu["input"]["run_in_background"] = False
                                    yield sse.block_start(bi, "tool_use", id=tu["id"], name=tu["name"])
                                    yield sse.block_delta(bi, "input_json_delta", json.dumps(tu["input"]))
                                    yield sse.block_stop(bi)

                    # Native tool calls
                    if delta.tool_calls:
                        for ev in sse.close_content():
                            yield ev
                        for tc in delta.tool_calls:
                            tc_info = {
                                "index": tc.index,
                                "id": tc.id,
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for ev in self._process_tool_call(tc_info, sse):
                                yield ev

            except Exception as e:
                logger.error("NIM_ERROR: {}: {}", type(e).__name__, e)
                error_occurred = True
                error_msg = _user_error_msg(e, self._timeout)
                if request_id:
                    error_msg += f" (request_id={request_id})"
                for ev in sse.close_content():
                    yield ev
                for ev in sse.emit_error(error_msg):
                    yield ev

        # Flush remaining
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                for ev in sse.ensure_thinking():
                    yield ev
                yield sse.thinking_delta(remaining.content)
            else:
                for ev in sse.ensure_text():
                    yield ev
                yield sse.text_delta(remaining.content)

        for tu in tool_parser.flush():
            for ev in sse.close_content():
                yield ev
            bi = sse.blocks.alloc()
            yield sse.block_start(bi, "tool_use", id=tu["id"], name=tu["name"])
            if tu.get("name") == "Task" and isinstance(tu.get("input"), dict):
                tu["input"]["run_in_background"] = False
            yield sse.block_delta(bi, "input_json_delta", json.dumps(tu["input"]))
            yield sse.block_stop(bi)

        if not error_occurred and sse.blocks.text_index == -1 and not sse.blocks.tool_states:
            for ev in sse.ensure_text():
                yield ev
            yield sse.text_delta(" ")

        for ti, out in sse.blocks.flush_task_buffers():
            yield sse.tool_delta(ti, out)

        for ev in sse.close_all():
            yield ev

        output_tokens = (
            usage_info.completion_tokens
            if usage_info and hasattr(usage_info, "completion_tokens")
            else sse.estimate_output_tokens()
        )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
