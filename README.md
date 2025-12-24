# ChatGPT App Tester

Lightweight local tester for ChatGPT apps with MCP servers. Iterate on your ChatGPT apps locally without using ChatGPT developer mode.

## Features

- **Chat UI** with message history
- **OpenAI-powered tool selection** using GPT-4o-mini
- **Widget rendering** with `window.openai` API mock
- **Light/dark theme toggle**
- **Single-file implementation** (~600 lines)

## Installation

```bash
# Clone the repo
git clone https://github.com/haldar/chatgpt-app-tester.git
cd chatgpt-app-tester

# Install dependencies with uv
uv sync
```

## Usage

### Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here

# Run against an MCP server
uv run python chatgpt_app_tester.py http://localhost:8000/mcp

# Open the UI in your browser
open http://localhost:3000
```

### CLI Options

```bash
# Specify a custom port
uv run python chatgpt_app_tester.py --port 3001 http://localhost:8000/mcp

# Use a config file
uv run python chatgpt_app_tester.py --config servers.json
```

### Config File Format

```json
{
  "servers": [
    {"name": "My App", "url": "http://localhost:8000/mcp"}
  ],
  "port": 3000
}
```

## How It Works

```
User types message
       │
       ▼
POST /api/chat → OpenAI API (with MCP tools as functions)
       │
       ▼
OpenAI returns tool_call → MCP Server (tools/call)
       │
       ▼
MCP returns structuredContent + _meta with widget template
       │
       ▼
Fetch widget HTML via resources/read
       │
       ▼
Inject window.openai mock → Render in iframe
```

## window.openai API

The tester mocks the ChatGPT `window.openai` API for widgets:

| Property/Method | Supported |
|-----------------|-----------|
| `toolOutput` | Yes |
| `toolResponseMetadata` | Yes |
| `theme` | Yes |
| `widgetState` | Yes |
| `setWidgetState()` | Yes |
| `callTool()` | Yes |
| `sendFollowUpMessage()` | Yes |
| `requestDisplayMode()` | Logged only |
| `openExternal()` | Yes |

## Example: Testing time-left app

```bash
# Terminal 1: Start the MCP server
cd ~/repos/gh/time-left-chatgpt-app
uv run python server/main.py

# Terminal 2: Start the tester
cd ~/repos/gh/chatgpt-app-tester
export OPENAI_API_KEY=your-key
uv run python chatgpt_app_tester.py http://localhost:8000/mcp

# Open http://localhost:3000 and type "How much time is left?"
```

## Dependencies

- `httpx` - Async HTTP client for MCP calls
- `starlette` - Lightweight web framework
- `uvicorn` - ASGI server
- `openai` - OpenAI Python client

## Limitations

- Single MCP server only (multi-server support not yet implemented)
- Display mode switching (fullscreen/pip) is logged but not rendered
- No file upload support
- In-memory message history (not persisted)

## License

MIT
