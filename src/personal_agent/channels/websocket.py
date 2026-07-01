"""WebSocketChannel — WebSocket server channel for browser-based UI access."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from personal_agent.channels.base import Channel, ChannelMessage, SessionKey
from personal_agent.server.router import MessageRouter

logger = logging.getLogger(__name__)

WS_CHANNEL = "websocket"

# JSON-RPC style message types
# Client → Server:
#   {"type": "task", "text": "..."}
#   {"type": "session_create", "name": "..."}
#   {"type": "session_switch", "id": "..."}
#   {"type": "session_list"}
#   {"type": "session_info"}
#
# Server → Client:
#   {"type": "thought", "text": "..."}
#   {"type": "tool_call", "name": "...", "arguments": {...}}
#   {"type": "tool_result", "name": "...", "output": "..."}
#   {"type": "answer", "text": "..."}
#   {"type": "status", "token_usage": {...}, "elapsed_ms": ...}
#   {"type": "error", "text": "..."}
#   {"type": "session_info", ...}
#   {"type": "session_list", "sessions": [...]}


class WebSocketChannel(Channel):
    """WebSocket server channel for browser-based agent UI.

    Starts a WebSocket server that accepts connections from browser clients.
    Each connection is mapped to a session, and agent execution callbacks
    are streamed back to the client in real time.

    Usage:
        ws = WebSocketChannel(settings, router, host="localhost", port=8765)
        server.add_channel(ws)
        await server.start()  # WebSocket server starts alongside other channels
    """

    def __init__(
        self,
        settings: Any,
        router: MessageRouter,
        host: str = "localhost",
        port: int = 8765,
    ):
        super().__init__(WS_CHANNEL)
        self._settings = settings
        self._router = router
        self._host = host
        self._port = port
        self._server: Any = None
        self._connections: dict[int, ServerConnection] = {}
        self._conn_agents: dict[int, Any] = {}
        self._conn_sessions: dict[int, Any] = {}
        self._conn_counter = itertools.count()
        self._agent_lock = asyncio.Lock()
        self._conn_locks: dict[int, asyncio.Lock] = {}
        self._conn_locks_lock = asyncio.Lock()

    # ── Channel interface ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server."""
        logger.info("WebSocket server starting on ws://%s:%d", self._host, self._port)
        self._server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        print(f"  {_C_GREEN}WebSocket{_C_RESET} server: {_C_BOLD}ws://{self._host}:{self._port}{_C_RESET}")
        # Keep running until stopped
        await self._server.wait_closed()

    async def stop(self) -> None:
        """Stop the WebSocket server and close all connections.

        Idempotent: a second call (e.g. from signal handler + AgentServer
        teardown) is a no-op.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Close all agent connections
        for conn_id, agent in list(self._conn_agents.items()):
            try:
                await agent.close()
            except BaseException as e:
                logger.warning("Error closing agent for conn %d: %s", conn_id, e)
        self._conn_agents.clear()
        self._conn_sessions.clear()
        self._connections.clear()

    # ── Connection handling ──────────────────────────────────────────────────

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket connection (one per browser tab)."""
        # itertools.count.__next__ is atomic under the GIL, avoiding the
        # read-modify-write race of `self._conn_counter += 1` where two
        # connections landing in the same tick could share an id.
        conn_id = next(self._conn_counter)
        self._connections[conn_id] = websocket

        remote = websocket.remote_address
        logger.info("WebSocket connection #%d from %s", conn_id, remote)

        try:
            await self._conn_loop(conn_id, websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection #%d closed", conn_id)
        except Exception as e:
            logger.exception("WebSocket connection #%d error: %s", conn_id, e)
        finally:
            # Clean up
            agent = self._conn_agents.pop(conn_id, None)
            if agent:
                try:
                    await agent.close()
                except BaseException:
                    pass
            self._conn_sessions.pop(conn_id, None)
            self._connections.pop(conn_id, None)
            self._conn_locks.pop(conn_id, None)

    async def _conn_loop(self, conn_id: int, websocket: ServerConnection) -> None:
        """Main message loop for a single connection."""
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                await self._send(websocket, {"type": "error", "text": "Invalid JSON"})
                continue

            msg_type = data.get("type", "")

            if msg_type == "task":
                await self._handle_task(conn_id, websocket, data)
            elif msg_type == "session_create":
                await self._handle_session_create(conn_id, websocket, data)
            elif msg_type == "session_switch":
                await self._handle_session_switch(conn_id, websocket, data)
            elif msg_type == "session_list":
                await self._handle_session_list(websocket)
            elif msg_type == "session_info":
                await self._handle_session_info(conn_id, websocket)
            elif msg_type == "ping":
                await self._send(websocket, {"type": "pong"})
            else:
                await self._send(websocket, {"type": "error", "text": f"Unknown message type: {msg_type}"})

    # ── Task processing ──────────────────────────────────────────────────────

    async def _handle_task(self, conn_id: int, websocket: ServerConnection, data: dict) -> None:
        """Process a task from the WebSocket client."""
        text = data.get("text", "").strip()
        if not text:
            await self._send(websocket, {"type": "error", "text": "Empty task"})
            return

        # Serialize task execution per connection to prevent concurrent state corruption
        async with self._conn_locks_lock:
            if conn_id not in self._conn_locks:
                self._conn_locks[conn_id] = asyncio.Lock()
        conn_lock = self._conn_locks[conn_id]

        async with conn_lock:
            # Determine user_id for this connection
            session = self._conn_sessions.get(conn_id)
            user_id = session.user_id if session else "web-user"
            conv_id = session.conversation_id if session else f"conn-{conn_id}"

            # Create channel message and resolve session (before creating agent so
            # _get_or_create_agent sees the correct user_id for memory isolation)
            msg = ChannelMessage(
                channel=WS_CHANNEL,
                user_id=user_id,
                conversation_id=conv_id,
                text=text,
            )
            resolved = self._router.resolve(msg)
            self._conn_sessions[conn_id] = resolved

            # Get or create agent for this connection
            agent = await self._get_or_create_agent(conn_id)

            # Restore session memory into agent
            async with resolved.memory_lock:
                agent.short_term = resolved.short_term
                agent.working = resolved.working

            # Wire up callbacks to stream to WebSocket
            from personal_agent.types import AgentCallbacks

            agent._callbacks = AgentCallbacks(
                on_step_start=lambda step, total: self._send(
                    websocket, {"type": "step_start", "step": step, "total": total},
                ),
                on_thought=lambda thought: self._send(
                    websocket, {"type": "thought", "text": thought},
                ),
                on_tool_call=lambda name, args: self._send(
                    websocket, {"type": "tool_call", "name": name, "arguments": args},
                ),
                on_tool_result=lambda name, output, error: self._send(
                    websocket, {
                        "type": "tool_result",
                        "name": name,
                        "output": str(output)[:5000] if output else None,
                        "error": error,
                    },
                ),
                on_answer=lambda answer: self._send(
                    websocket, {"type": "answer", "text": answer},
                ),
                on_text_delta=lambda text: self._send(
                    websocket, {"type": "text_delta", "text": text},
                ),
                on_tool_call_stream=lambda name, args: self._send(
                    websocket, {"type": "tool_call_stream", "name": name, "arguments": args},
                ),
            )
            agent._streaming_enabled = True

            start = time.time()
            try:
                result = await agent.run(text)
                elapsed_ms = (time.time() - start) * 1000

                # Send final status
                await self._send(websocket, {
                    "type": "status",
                    "token_usage": result.token_usage,
                    "elapsed_ms": elapsed_ms,
                    "steps": len(result.steps),
                })
            except Exception as e:
                logger.exception("Task execution failed for conn %d", conn_id)
                await self._send(websocket, {
                    "type": "error",
                    "text": "An internal error occurred while processing your request.",
                })
            finally:
                # Persist session state even on error
                async with resolved.memory_lock:
                    resolved.short_term = agent.short_term
                    resolved.working = agent.working
                self._router.session_manager.save_session(resolved)

    # ── Agent management ─────────────────────────────────────────────────────

    async def _get_or_create_agent(self, conn_id: int) -> Any:
        """Get or create an agent for a WebSocket connection."""
        agent = self._conn_agents.get(conn_id)
        if agent is not None:
            return agent

        async with self._agent_lock:
            # Double-check: another task may have created the agent while we waited
            if conn_id in self._conn_agents:
                return self._conn_agents[conn_id]

            from personal_agent.factory import create_agent

            session = self._conn_sessions.get(conn_id)
            user_id = session.user_id if session else "web-user"
            agent = await create_agent(self._settings, user_id=user_id)
            self._conn_agents[conn_id] = agent
            return agent

    # ── Session management ──────────────────────────────────────────────────

    async def _handle_session_create(self, conn_id: int, websocket: ServerConnection, data: dict) -> None:
        """Create a new session."""
        name = data.get("name", f"web-{conn_id}")
        session_mgr = self._router.session_manager

        # Save old session state before replacing it
        old_session = self._conn_sessions.get(conn_id)
        if old_session and conn_id in self._conn_agents:
            async with old_session.memory_lock:
                old_session.short_term = self._conn_agents[conn_id].short_term
                old_session.working = self._conn_agents[conn_id].working
            session_mgr.save_session(old_session)

        session = session_mgr.create(name)
        self._conn_sessions[conn_id] = session
        # Reset agent for new session (under lock to prevent races with _get_or_create_agent)
        async with self._agent_lock:
            if conn_id in self._conn_agents:
                try:
                    await self._conn_agents[conn_id].close()
                except BaseException:
                    pass
                del self._conn_agents[conn_id]
        await self._send(websocket, {
            "type": "session_info",
            "id": session.id,
            "name": session.name,
            "channel": session.channel,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "messages": len(session.short_term),
        })

    async def _handle_session_switch(self, conn_id: int, websocket: ServerConnection, data: dict) -> None:
        """Switch to a different session."""
        session_id = data.get("id", "")
        session_mgr = self._router.session_manager
        # Persist the current connection's session state before switching
        current = self._conn_sessions.get(conn_id)
        if current:
            if conn_id in self._conn_agents:
                async with current.memory_lock:
                    current.short_term = self._conn_agents[conn_id].short_term
                    current.working = self._conn_agents[conn_id].working
            session_mgr.save_session(current)

        target = session_mgr.switch(session_id)
        if target is None:
            await self._send(websocket, {"type": "error", "text": f"Session not found: {session_id}"})
            return

        self._conn_sessions[conn_id] = target
        if conn_id in self._conn_agents:
            async with target.memory_lock:
                self._conn_agents[conn_id].short_term = target.short_term
                self._conn_agents[conn_id].working = target.working
        await self._send(websocket, {
            "type": "session_info",
            "id": target.id,
            "name": target.name,
            "messages": len(target.short_term),
        })

    async def _handle_session_list(self, websocket: ServerConnection) -> None:
        """List all sessions."""
        session_mgr = self._router.session_manager
        sessions = []
        for s in session_mgr.list_sessions():
            sessions.append({
                "id": s.id,
                "name": s.name,
                "channel": s.channel,
                "user_id": s.user_id,
                "messages": len(s.short_term),
            })
        await self._send(websocket, {"type": "session_list", "sessions": sessions})

    async def _handle_session_info(self, conn_id: int, websocket: ServerConnection) -> None:
        """Send current session info."""
        session = self._conn_sessions.get(conn_id)
        if session is None:
            await self._send(websocket, {"type": "error", "text": "No active session"})
            return
        await self._send(websocket, {
            "type": "session_info",
            "id": session.id,
            "name": session.name,
            "channel": session.channel,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "messages": len(session.short_term),
        })

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send(self, websocket: ServerConnection, data: dict) -> None:
        """Send a JSON message to a WebSocket client, ignoring connection errors."""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False, default=str))
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.debug("Failed to send WebSocket message: %s", e)


# ANSI color for startup message
_C_GREEN = "\033[32m"
_C_BOLD = "\033[1m"
_C_RESET = "\033[0m"