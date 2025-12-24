#!/usr/bin/env python3
# ABOUTME: Lightweight local tester for ChatGPT apps with MCP servers.
# ABOUTME: Provides a chat UI with OpenAI-powered tool calling and widget rendering.

"""
ChatGPT App Tester - Local development tool for ChatGPT apps

Usage:
    python chatgpt_app_tester.py http://localhost:8000/mcp
    python chatgpt_app_tester.py --config servers.json
    python chatgpt_app_tester.py --port 3001 http://localhost:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
from starlette.requests import Request


# =============================================================================
# MCP HTTP Client
# =============================================================================


@dataclass
class MCPClient:
    """Simple MCP client that communicates via HTTP JSON-RPC."""

    base_url: str
    _client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=30.0))
    _request_id: int = 0

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request to the MCP server."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        url = self.base_url if self.base_url.endswith("/mcp") else f"{self.base_url.rstrip('/')}/mcp"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        # Parse response - may be SSE format or plain JSON
        text = resp.text
        if text.startswith("event:"):
            # SSE format: parse the data line
            for line in text.split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    break
            else:
                raise Exception("No data in SSE response")
        else:
            data = resp.json()

        if "error" in data:
            raise Exception(f"MCP Error: {data['error']}")
        return data.get("result", {})

    async def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        result = await self._rpc("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Call a tool on the MCP server."""
        return await self._rpc("tools/call", {"name": name, "arguments": arguments or {}})

    async def list_resources(self) -> list[dict]:
        """List available resources from the MCP server."""
        result = await self._rpc("resources/list")
        return result.get("resources", [])

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI and return its content."""
        result = await self._rpc("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# OpenAI Tool Integration
# =============================================================================


def mcp_tools_to_openai_format(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tools to OpenAI function calling format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


async def chat_with_tools(
    openai_client: AsyncOpenAI,
    mcp_client: MCPClient,
    messages: list[dict],
    mcp_tools: list[dict],
) -> dict:
    """
    Send a chat request to OpenAI with MCP tools.
    Returns the final response and any tool calls/results.
    """
    openai_tools = mcp_tools_to_openai_format(mcp_tools)

    # First call to OpenAI
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=openai_tools if openai_tools else None,
    )

    choice = response.choices[0]
    assistant_message = choice.message

    result = {
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": [],
        "tool_results": [],
        "widget": None,
    }

    # If there are tool calls, execute them
    if assistant_message.tool_calls:
        # Add assistant message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_message.tool_calls
            ],
        })

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            # Call the MCP tool
            try:
                tool_result = await mcp_client.call_tool(tool_name, tool_args)
                result["tool_calls"].append({"name": tool_name, "arguments": tool_args})
                result["tool_results"].append(tool_result)

                # Check for widget template
                meta = tool_result.get("_meta", {})
                if "openai/outputTemplate" in meta:
                    widget_uri = meta["openai/outputTemplate"]
                    widget_html = await mcp_client.read_resource(widget_uri)
                    result["widget"] = {
                        "uri": widget_uri,
                        "html": widget_html,
                        "structuredContent": tool_result.get("structuredContent", {}),
                        "_meta": meta,
                    }

                # Get text content for tool result
                content_items = tool_result.get("content", [])
                tool_text = ""
                for item in content_items:
                    if item.get("type") == "text":
                        tool_text += item.get("text", "")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_text or json.dumps(tool_result.get("structuredContent", {})),
                })

            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error calling tool: {e}",
                })

        # Get final response from OpenAI
        final_response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        result["content"] = final_response.choices[0].message.content

    return result


# =============================================================================
# Web Server
# =============================================================================


# Global state
app_state: dict[str, Any] = {
    "mcp_client": None,
    "openai_client": None,
    "mcp_tools": [],
    "messages": [],
    "widget_states": {},
    "theme": "light",
}


async def index(request: Request) -> HTMLResponse:
    """Serve the main chat UI."""
    return HTMLResponse(CHAT_UI_HTML)


async def api_tools(request: Request) -> JSONResponse:
    """Return list of available MCP tools."""
    return JSONResponse({"tools": app_state["mcp_tools"]})


async def api_chat(request: Request) -> JSONResponse:
    """Handle chat messages with LLM tool calling."""
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return JSONResponse({"error": "No message provided"}, status_code=400)

    # Add user message to history
    app_state["messages"].append({"role": "user", "content": user_message})

    try:
        # Chat with OpenAI and MCP tools
        result = await chat_with_tools(
            app_state["openai_client"],
            app_state["mcp_client"],
            app_state["messages"].copy(),
            app_state["mcp_tools"],
        )

        # Add assistant response to history
        app_state["messages"].append({"role": "assistant", "content": result["content"]})

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_resource(request: Request) -> JSONResponse:
    """Fetch a resource by URI."""
    uri = request.query_params.get("uri", "")
    if not uri:
        return JSONResponse({"error": "No URI provided"}, status_code=400)

    try:
        content = await app_state["mcp_client"].read_resource(uri)
        return JSONResponse({"content": content})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_call_tool(request: Request) -> JSONResponse:
    """Direct tool call (for widget callTool)."""
    data = await request.json()
    tool_name = data.get("name", "")
    tool_args = data.get("arguments", {})

    if not tool_name:
        return JSONResponse({"error": "No tool name provided"}, status_code=400)

    try:
        result = await app_state["mcp_client"].call_tool(tool_name, tool_args)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_widget_state(request: Request) -> JSONResponse:
    """Get or set widget state."""
    if request.method == "GET":
        widget_id = request.query_params.get("id", "")
        return JSONResponse({"state": app_state["widget_states"].get(widget_id)})
    else:
        data = await request.json()
        widget_id = data.get("id", "")
        state = data.get("state")
        app_state["widget_states"][widget_id] = state
        return JSONResponse({"ok": True})


async def api_theme(request: Request) -> JSONResponse:
    """Get or set theme."""
    if request.method == "GET":
        return JSONResponse({"theme": app_state["theme"]})
    else:
        data = await request.json()
        app_state["theme"] = data.get("theme", "light")
        return JSONResponse({"theme": app_state["theme"]})


routes = [
    Route("/", index),
    Route("/api/tools", api_tools),
    Route("/api/chat", api_chat, methods=["POST"]),
    Route("/api/resource", api_resource),
    Route("/api/call_tool", api_call_tool, methods=["POST"]),
    Route("/api/widget_state", api_widget_state, methods=["GET", "POST"]),
    Route("/api/theme", api_theme, methods=["GET", "POST"]),
]

app = Starlette(routes=routes)


# =============================================================================
# Embedded Chat UI (HTML/CSS/JS)
# =============================================================================


CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChatGPT App Tester</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #ffffff;
      --bg-secondary: #f7f7f8;
      --text: #1a1a1a;
      --text-secondary: #666;
      --border: #e5e5e5;
      --primary: #10a37f;
      --user-bg: #f7f7f8;
      --assistant-bg: #ffffff;
    }

    body.dark {
      --bg: #212121;
      --bg-secondary: #2f2f2f;
      --text: #ececec;
      --text-secondary: #999;
      --border: #444;
      --user-bg: #2f2f2f;
      --assistant-bg: #212121;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-secondary);
    }

    header h1 {
      font-size: 1rem;
      font-weight: 600;
    }

    .theme-toggle {
      background: none;
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      color: var(--text);
      font-size: 0.875rem;
    }

    .chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      max-width: 800px;
      width: 100%;
      margin: 0 auto;
      padding: 16px 20px;
      border-radius: 12px;
    }

    .message.user {
      background: var(--user-bg);
    }

    .message.assistant {
      background: var(--assistant-bg);
      border: 1px solid var(--border);
    }

    .message-role {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }

    .message-content {
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .tool-call {
      background: var(--bg-secondary);
      padding: 8px 12px;
      border-radius: 6px;
      margin: 8px 0;
      font-family: monospace;
      font-size: 0.875rem;
      color: var(--primary);
    }

    .widget-container {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .widget-container iframe {
      width: 100%;
      border: none;
      min-height: 200px;
    }

    .input-container {
      padding: 16px 20px;
      border-top: 1px solid var(--border);
      background: var(--bg-secondary);
    }

    .input-wrapper {
      max-width: 800px;
      margin: 0 auto;
      display: flex;
      gap: 12px;
    }

    #chat-input {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 1rem;
      background: var(--bg);
      color: var(--text);
      outline: none;
    }

    #chat-input:focus {
      border-color: var(--primary);
    }

    #send-btn {
      padding: 12px 24px;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
    }

    #send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .loading {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-secondary);
    }

    .loading::after {
      content: '';
      width: 16px;
      height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--primary);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .tools-panel {
      padding: 8px 20px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      font-size: 0.875rem;
      color: var(--text-secondary);
    }

    .tool-badge {
      display: inline-block;
      background: var(--bg);
      border: 1px solid var(--border);
      padding: 4px 8px;
      border-radius: 4px;
      margin: 4px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <header>
    <h1>ChatGPT App Tester</h1>
    <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
  </header>

  <div class="tools-panel" id="tools-panel">
    Loading tools...
  </div>

  <div class="chat-container" id="chat-container"></div>

  <div class="input-container">
    <div class="input-wrapper">
      <input type="text" id="chat-input" placeholder="Type a message..." onkeydown="handleKeyDown(event)">
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    let currentTheme = 'light';
    let widgetStates = {};
    let widgetCounter = 0;

    // Initialize
    async function init() {
      await loadTools();
      document.getElementById('chat-input').focus();
    }

    async function loadTools() {
      try {
        const resp = await fetch('/api/tools');
        const data = await resp.json();
        const toolsPanel = document.getElementById('tools-panel');
        if (data.tools && data.tools.length > 0) {
          toolsPanel.innerHTML = 'Available tools: ' +
            data.tools.map(t => `<span class="tool-badge">${t.name}</span>`).join('');
        } else {
          toolsPanel.innerHTML = 'No tools available';
        }
      } catch (e) {
        document.getElementById('tools-panel').innerHTML = 'Error loading tools: ' + e.message;
      }
    }

    function toggleTheme() {
      currentTheme = currentTheme === 'light' ? 'dark' : 'light';
      document.body.classList.toggle('dark', currentTheme === 'dark');
      // Update all widget iframes
      document.querySelectorAll('.widget-container iframe').forEach(iframe => {
        if (iframe.contentWindow) {
          iframe.contentWindow.postMessage({ type: 'theme-change', theme: currentTheme }, '*');
        }
      });
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    async function sendMessage() {
      const input = document.getElementById('chat-input');
      const message = input.value.trim();
      if (!message) return;

      input.value = '';
      const sendBtn = document.getElementById('send-btn');
      sendBtn.disabled = true;

      // Add user message to UI
      addMessage('user', message);

      // Add loading indicator
      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'message assistant';
      loadingDiv.innerHTML = '<div class="loading">Thinking</div>';
      document.getElementById('chat-container').appendChild(loadingDiv);

      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message }),
        });

        const data = await resp.json();

        // Remove loading indicator
        loadingDiv.remove();

        if (data.error) {
          addMessage('assistant', `Error: ${data.error}`);
        } else {
          // Show tool calls if any
          let content = '';
          if (data.tool_calls && data.tool_calls.length > 0) {
            for (const tc of data.tool_calls) {
              content += `<div class="tool-call">Called: ${tc.name}(${JSON.stringify(tc.arguments)})</div>`;
            }
          }

          // Add text content
          if (data.content) {
            content += `<div class="message-content">${escapeHtml(data.content)}</div>`;
          }

          // Add widget if present
          if (data.widget) {
            content += renderWidget(data.widget);
          }

          addMessageHtml('assistant', content);
        }
      } catch (e) {
        loadingDiv.remove();
        addMessage('assistant', `Error: ${e.message}`);
      }

      sendBtn.disabled = false;
      input.focus();
    }

    function addMessage(role, content) {
      addMessageHtml(role, `<div class="message-content">${escapeHtml(content)}</div>`);
    }

    function addMessageHtml(role, html) {
      const container = document.getElementById('chat-container');
      const div = document.createElement('div');
      div.className = `message ${role}`;
      div.innerHTML = `<div class="message-role">${role}</div>${html}`;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }

    function renderWidget(widget) {
      const widgetId = `widget-${++widgetCounter}`;

      // Build window.openai mock
      const mockOpenAI = {
        toolOutput: widget.structuredContent || {},
        toolResponseMetadata: widget._meta || {},
        theme: currentTheme,
        widgetState: widgetStates[widgetId] || null,
      };

      // Create injection script
      const injectionScript = `
        <script>
          window.openai = ${JSON.stringify(mockOpenAI)};
          window.openai.setWidgetState = function(state) {
            window.parent.postMessage({ type: 'widget-state', widgetId: '${widgetId}', state: state }, '*');
          };
          window.openai.callTool = async function(name, args) {
            const resp = await fetch('/api/call_tool', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: name, arguments: args || {} })
            });
            return resp.json();
          };
          window.openai.sendFollowUpMessage = function(req) {
            window.parent.postMessage({ type: 'follow-up', prompt: req.prompt }, '*');
          };
          window.openai.requestDisplayMode = function(req) {
            console.log('requestDisplayMode:', req.mode);
          };
          window.openai.openExternal = function(req) {
            window.open(req.href, '_blank');
          };
          window.dispatchEvent(new Event('openai:set-globals'));

          // Listen for theme changes from parent
          window.addEventListener('message', function(e) {
            if (e.data && e.data.type === 'theme-change') {
              window.openai.theme = e.data.theme;
              window.dispatchEvent(new Event('openai:set-globals'));
            }
          });
        <\\/script>
      `;

      // Inject mock into widget HTML
      let html = widget.html || '';
      html = html.replace('</head>', injectionScript + '</head>');

      // Escape for srcdoc
      const srcdoc = html.replace(/"/g, '&quot;');

      return `
        <div class="widget-container">
          <iframe srcdoc="${srcdoc}" sandbox="allow-scripts allow-same-origin"></iframe>
        </div>
      `;
    }

    function escapeHtml(text) {
      if (!text) return '';
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    // Handle messages from widget iframes
    window.addEventListener('message', function(e) {
      if (e.data && e.data.type === 'widget-state') {
        widgetStates[e.data.widgetId] = e.data.state;
      } else if (e.data && e.data.type === 'follow-up') {
        document.getElementById('chat-input').value = e.data.prompt;
        sendMessage();
      }
    });

    init();
  </script>
</body>
</html>
"""


# =============================================================================
# CLI Entry Point
# =============================================================================


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


async def initialize(mcp_url: str):
    """Initialize MCP client and load tools."""
    app_state["mcp_client"] = MCPClient(mcp_url)
    app_state["openai_client"] = AsyncOpenAI()

    # Load tools
    try:
        app_state["mcp_tools"] = await app_state["mcp_client"].list_tools()
        print(f"Loaded {len(app_state['mcp_tools'])} tools from MCP server")
        for tool in app_state["mcp_tools"]:
            print(f"  - {tool.get('name')}: {tool.get('description', '')[:60]}...")
    except Exception as e:
        print(f"Warning: Could not load tools: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ChatGPT App Tester")
    parser.add_argument("url", nargs="?", help="MCP server URL (e.g., http://localhost:8000/mcp)")
    parser.add_argument("--config", "-c", help="Path to config JSON file")
    parser.add_argument("--port", "-p", type=int, default=3000, help="Port to run on (default: 3000)")
    args = parser.parse_args()

    # Determine MCP URL
    mcp_url = None
    port = args.port

    if args.config:
        config = load_config(args.config)
        servers = config.get("servers", [])
        if servers:
            mcp_url = servers[0].get("url")
        port = config.get("port", port)
    elif args.url:
        mcp_url = args.url
    else:
        parser.print_help()
        print("\nError: Please provide an MCP server URL or config file")
        return

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-key-here")
        return

    print(f"ChatGPT App Tester")
    print(f"  MCP Server: {mcp_url}")
    print(f"  Web UI: http://localhost:{port}")
    print()

    # Initialize and run
    async def startup():
        await initialize(mcp_url)

    @app.on_event("startup")
    async def on_startup():
        await startup()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
