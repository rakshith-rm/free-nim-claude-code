# my-claude-code

Use the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) for **free** by routing requests through [NVIDIA NIM](https://build.nvidia.com/) instead of paying for Anthropic's API.

This project is a local proxy server that sits between Claude Code and NVIDIA NIM. It translates Anthropic's API format into OpenAI-compatible format on the fly.

## How request routing works

```
┌──────────────┐       Anthropic API        ┌─────────────────┐      OpenAI API       ┌──────────────┐
│  Claude Code │  ───────────────────────►   │  This Proxy     │  ──────────────────►   │  NVIDIA NIM  │
│  CLI         │  POST /v1/messages          │  (localhost:8082)│  chat.completions     │  (free tier) │
│              │  ◄───────────────────────   │                 │  ◄──────────────────   │              │
│              │   Anthropic SSE stream      │  converter.py   │   OpenAI SSE stream   │  LLama 405B  │
└──────────────┘                             │  sse.py         │                       └──────────────┘
                                             └─────────────────┘
```

1. **Claude Code** sends requests to `http://localhost:8082/v1/messages` (Anthropic format)
2. **converter.py** translates messages, tools, and system prompts from Anthropic format to OpenAI format
3. **nim.py** forwards the converted request to NVIDIA NIM via the OpenAI SDK, with rate limiting and retries
4. **sse.py** converts the streaming OpenAI response back into Anthropic SSE events that Claude Code expects
5. Includes a **ThinkTagParser** (handles `<think>` tags) and **HeuristicToolParser** (catches tool calls emitted as plain text by weaker models)

## Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org/) (for Claude Code CLI)

## Step 1: Get an NVIDIA NIM API key (free)

1. Go to [build.nvidia.com](https://build.nvidia.com/)
2. Sign up or log in
3. Navigate to any model (e.g. LLama 3.1 405B Instruct) and click **Get API Key**
4. Copy the key — it starts with `nvapi-...`

## Step 2: Configure `.env`

Create a `.env` file in this directory:

```env
# NVIDIA NIM API Key (get one at https://build.nvidia.com/settings/api-keys)
NVIDIA_NIM_API_KEY="nvapi-YOUR_KEY_HERE"

# Model to use (any model available on NIM)
NIM_MODEL=meta/llama-3.1-405b-instruct

# Server (optional)
HOST=0.0.0.0
PORT=8082
```

### Popular available models

| Model | `.env` value |
|-------|-------------|
| LLama 3.1 405B (recommended) | `meta/llama-3.1-405b-instruct` |
| LLama 3.3 70B | `meta/llama-3.3-70b-instruct` |

NIM provides 100s of models, you can configure as per your needs

## Step 3: Install Python dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# .\venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -e .

# If that fails (old pip), install directly:
pip install "fastapi[standard]" uvicorn httpx pydantic pydantic-settings python-dotenv tiktoken openai loguru
```

## Step 4: Start the proxy server

```bash
python server.py
```

You should see:

```
Starting NIM Claude Code Proxy on port 8082
INFO:     Uvicorn running on http://0.0.0.0:8082
```

**Keep this terminal open.**

## Step 5: Install Claude Code CLI

In a **new terminal**:

```bash
npm install -g @anthropic-ai/claude-code
```

## Step 6: Launch Claude Code

In that same new terminal, set the environment variables and start Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=sk-dummy
claude
```

On Windows PowerShell:

```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8082"
$env:ANTHROPIC_API_KEY = "sk-dummy"
claude
```

### Environment variables reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_BASE_URL` | Yes | Must be `http://localhost:8082` to route through the proxy |
| `ANTHROPIC_API_KEY` | Yes | Any non-empty string (e.g. `sk-dummy`). Claude Code requires it but the proxy ignores it |
| `NVIDIA_NIM_API_KEY` | Yes (in `.env`) | Your NVIDIA NIM API key |
| `NIM_MODEL` | No (in `.env`) | Model to use. Defaults to `meta/llama-3.3-70b-instruct` |

### Claude Code setup prompts

When Claude Code starts for the first time:

1. **Login method** — choose **Anthropic Console account** (option 2)
2. **Detected custom API key** — choose **Yes** (option 1)

## Project structure

```
my-claude-code/
├── app.py           # FastAPI app, routes, settings
├── server.py        # Uvicorn entry point
├── converter.py     # Anthropic → OpenAI format conversion
├── nim.py           # NVIDIA NIM provider with rate limiting and streaming
├── sse.py           # OpenAI → Anthropic SSE event conversion
├── models.py        # Pydantic request/response models
├── tokens.py        # Token counting utilities
├── .env             # Your API key and model config
└── pyproject.toml   # Python project metadata and dependencies
```
