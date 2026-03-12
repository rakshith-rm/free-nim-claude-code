"""FastAPI application - the proxy between Claude Code CLI and NVIDIA NIM."""

import json
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse
from nim import NimProvider
from tokens import count_tokens

# ---------------------------------------------------------------------------
# Settings (loaded from .env)
# ---------------------------------------------------------------------------

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    nvidia_nim_api_key: str = ""
    nim_model: str = "meta/llama-3.3-70b-instruct"
    host: str = "0.0.0.0"
    port: int = 8082
    http_read_timeout: float = 300.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# ---------------------------------------------------------------------------
# Provider singleton
# ---------------------------------------------------------------------------

_provider: NimProvider | None = None


def get_provider() -> NimProvider:
    global _provider
    if _provider is None:
        if not settings.nvidia_nim_api_key.strip():
            raise HTTPException(status_code=503, detail="NVIDIA_NIM_API_KEY is not set in .env")
        _provider = NimProvider(api_key=settings.nvidia_nim_api_key, timeout=settings.http_read_timeout)
    return _provider


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting NIM Claude Code Proxy on port {}", settings.port)
    yield
    logger.info("Shutting down...")
    if _provider:
        await _provider.cleanup()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="My Claude Code (NIM)", version="1.0.0", lifespan=lifespan)


@app.post("/v1/messages")
async def create_message(request_data: MessagesRequest):
    if not request_data.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Map model name to configured NIM model
    request_data.model = settings.nim_model

    provider = get_provider()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    input_tokens = count_tokens(request_data.messages, request_data.system, request_data.tools)

    logger.info("REQUEST {}: model={} msgs={} tools={}",
                request_id, request_data.model,
                len(request_data.messages),
                len(request_data.tools or []))

    return StreamingResponse(
        provider.stream(request_data, input_tokens=input_tokens, request_id=request_id),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request_data: TokenCountRequest):
    request_data.model = settings.nim_model
    tokens = count_tokens(request_data.messages, request_data.system, request_data.tools)
    return TokenCountResponse(input_tokens=tokens)


@app.get("/")
async def root():
    return {"status": "ok", "provider": "nvidia_nim", "model": settings.nim_model}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def error_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc!s}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"type": "error", "error": {"type": "api_error", "message": str(exc) or "An unexpected error occurred."}},
    )
