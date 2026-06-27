"""FeishuChannel — Feishu (Lark) IM bot channel for the personal agent."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from aiohttp import web

from personal_agent.channels.base import Channel, ChannelMessage
from personal_agent.server.router import MessageRouter

logger = logging.getLogger(__name__)

FEISHU_CHANNEL = "feishu"
FEISHU_API_HOST = "https://open.feishu.cn"
TOKEN_REFRESH_MARGIN = 300  # Refresh token 5 minutes before expiry


class FeishuAPIClient:
    """Minimal Feishu Open API client for bot messaging.

    Handles tenant access token acquisition and message sending.
    """

    def __init__(self, app_id: str, app_secret: str):
        import httpx

        self._app_id = app_id
        self._app_secret = app_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._lock = asyncio.Lock()
        self._http = httpx.AsyncClient(timeout=30.0)

    async def _ensure_token(self) -> str:
        """Get or refresh the tenant access token."""
        async with self._lock:
            if self._token and time.time() < self._token_expires_at - TOKEN_REFRESH_MARGIN:
                return self._token

            resp = await self._http.post(
                f"{FEISHU_API_HOST}/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self._app_id,
                    "app_secret": self._app_secret,
                },
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Feishu token error: {data.get('msg', 'unknown')}")

            self._token = data["tenant_access_token"]
            self._token_expires_at = time.time() + data.get("expire", 7200)
            logger.info("Feishu tenant access token refreshed")
            return self._token  # type: ignore[return-value]

    async def reply_text(self, message_id: str, content: str) -> dict:
        """Reply to a message with plain text."""
        token = await self._ensure_token()

        resp = await self._http.post(
            f"{FEISHU_API_HOST}/open-apis/im/v1/messages/{message_id}/reply",
            json={
                "content": json.dumps({"text": content}),
                "msg_type": "text",
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        return resp.json()

    async def send_text(self, chat_id: str, content: str) -> dict:
        """Send a text message to a chat."""
        token = await self._ensure_token()

        resp = await self._http.post(
            f"{FEISHU_API_HOST}/open-apis/im/v1/messages",
            json={
                "receive_id": chat_id,
                "content": json.dumps({"text": content}),
                "msg_type": "text",
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        return resp.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()


class FeishuChannel(Channel):
    """Feishu (Lark) IM bot channel.

    Receives messages via Feishu webhook, processes them through the agent,
    and sends replies back via the Feishu API.

    Setup:
      1. Create a bot app at https://open.feishu.cn/app
      2. Enable "Bot" capability, get App ID and App secret
      3. Configure the webhook URL to point to this server
      4. Set env vars: PA_FEISHU__APP_ID, PA_FEISHU__APP_secret, PA_FEISHU__VERIFICATION_TOKEN

    Usage:
        feishu = FeishuChannel(settings, router)
        server.add_channel(feishu)
    """

    # Max text length for Feishu messages (API limit)
    MAX_REPLY_LENGTH = 15000

    def __init__(
        self,
        settings: Any,
        router: MessageRouter,
        app_id: str = "",
        app_secret: str = "",
        verification_token: str = "",
        webhook_port: int = 8080,
        webhook_path: str = "/feishu/webhook",
        encrypt_key: str = "",
    ):
        super().__init__(FEISHU_CHANNEL)
        self._settings = settings
        self._router = router
        self._app_id = app_id or settings.feishu.app_id
        self._app_secret = app_secret or settings.feishu.app_secret
        self._verification_token = verification_token or settings.feishu.verification_token
        self._webhook_port = webhook_port or settings.feishu.webhook_port
        self._webhook_path = webhook_path or settings.feishu.webhook_path
        self._encrypt_key = encrypt_key or settings.feishu.encrypt_key

        self._api: FeishuAPIClient | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._conn_agents: dict[str, Any] = {}
        self._conn_sessions: dict[str, Any] = {}
        self._pending_tasks: set[asyncio.Task] = set()
        self._agent_lock = asyncio.Lock()
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._stop_event = asyncio.Event()

    # ── Channel interface ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Feishu webhook HTTP server."""
        if not self._app_id or not self._app_secret:
            logger.warning(
                "FeishuChannel started but app_id or app_secret not configured. "
                "Set PA_FEISHU__APP_ID and PA_FEISHU__APP_secret env vars."
            )
            return

        self._api = FeishuAPIClient(self._app_id, self._app_secret)

        self._app = web.Application()
        self._app.router.add_get(self._webhook_path, self._handle_verification)
        self._app.router.add_post(self._webhook_path, self._handle_event)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._webhook_port)
        await site.start()

        logger.info(
            "Feishu webhook listening on http://0.0.0.0:%d%s",
            self._webhook_port, self._webhook_path,
        )
        print(f"  {_C_GREEN}Feishu{_C_RESET} webhook: {_C_BOLD}http://0.0.0.0:{self._webhook_port}{self._webhook_path}{_C_RESET}")

        # Keep running until stopped
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Stop the Feishu webhook server."""
        self._stop_event.set()
        if self._runner:
            await self._runner.cleanup()
        # Cancel pending tasks first, then await them, THEN close agents
        # to avoid closing agents while tasks are still executing agent.run()
        for task in list(self._pending_tasks):
            if not task.done():
                task.cancel()
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()
        for agent in self._conn_agents.values():
            try:
                await agent.close()
            except Exception:
                pass
        self._conn_agents.clear()
        self._conn_sessions.clear()
        self._user_locks.clear()
        if self._api:
            try:
                await self._api.close()
            except Exception:
                pass

    # ── Webhook handlers ─────────────────────────────────────────────────────

    async def _handle_verification(self, request: web.Request) -> web.Response:
        """Handle Feishu URL verification (GET request with challenge)."""
        challenge = request.query.get("challenge", "")
        token = request.query.get("token", "")
        if not self._verification_token:
            logger.warning("Feishu verification token not configured")
            return web.json_response({"code": 1, "msg": "Verification token not configured"}, status=403)
        if token != self._verification_token:
            logger.warning("Feishu verification token mismatch")
            return web.json_response({"code": 1, "msg": "Invalid token"}, status=403)
        logger.info("Feishu URL verification OK")
        return web.json_response({"challenge": challenge})

    async def _handle_event(self, request: web.Request) -> web.Response:
        """Handle Feishu event callback (POST request with event data)."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"code": 1, "msg": "Invalid JSON"}, status=400)

        # Feishu event format: {"schema": "2.0", "header": {...}, "event": {...}}
        event_type = body.get("header", {}).get("event_type", "")
        event = body.get("event", {})

        if event_type == "im.message.receive_v1":
            # Process in background, respond immediately to Feishu
            task = asyncio.create_task(self._process_message(event))
            self._pending_tasks.add(task)
            task.add_done_callback(
                lambda t: (
                    logger.error("Feishu message processing failed: %s", t.exception())
                    if t.exception() else None
                )
            )
            task.add_done_callback(self._pending_tasks.discard)
            return web.json_response({"code": 0})

        # Challenge event (old format, POST with challenge)
        if body.get("type") == "url_verification":
            challenge = body.get("challenge", "")
            token = body.get("token", "")
            if not self._verification_token:
                logger.warning("Feishu verification token not configured")
                return web.json_response({"code": 1, "msg": "Verification token not configured"}, status=403)
            if token != self._verification_token:
                return web.json_response({"code": 1, "msg": "Invalid token"}, status=403)
            return web.json_response({"challenge": challenge})

        # Unknown event type — acknowledge
        return web.json_response({"code": 0})

    # ── Message processing ───────────────────────────────────────────────────

    async def _process_message(self, event: dict) -> None:
        """Process an incoming Feishu message event."""
        message = event.get("message", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})
        user_id = sender_id.get("open_id", "")

        chat_id = message.get("chat_id", "")
        message_id = message.get("message_id", "")
        msg_type = message.get("msg_type", "text")

        # Only process text messages
        if msg_type != "text":
            logger.debug("Ignoring non-text message type: %s", msg_type)
            return

        # Parse text content (Feishu sends it as JSON string)
        content_str = message.get("content", "{}")
        try:
            content_data = json.loads(content_str)
            text = content_data.get("text", "")
        except json.JSONDecodeError:
            text = content_str

        text = text.strip()
        if not text:
            return

        logger.info(
            "Feishu message from %s in chat %s: %s",
            user_id, chat_id, text[:100],
        )

        # Serialize message processing per user to prevent concurrent state corruption
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        user_lock = self._user_locks[user_id]

        async with user_lock:
            # Create channel message and route to session (under lock to avoid races)
            msg = ChannelMessage(
                channel=FEISHU_CHANNEL,
                user_id=user_id,
                conversation_id=chat_id,
                text=text,
            )
            session = self._router.resolve(msg)
            self._conn_sessions[user_id] = session

            # Get or create agent for this user
            agent = await self._get_or_create_agent(user_id)
            async with session.memory_lock:
                agent.short_term = session.short_term
                agent.working = session.working

            await self._run_agent(agent, session, text, message_id)

    # ── Agent management ─────────────────────────────────────────────────────

    async def _get_or_create_agent(self, user_id: str) -> Any:
        """Get or create an agent for a Feishu user."""
        if user_id in self._conn_agents:
            return self._conn_agents[user_id]

        async with self._agent_lock:
            # Double-check: another task may have created the agent while we waited
            if user_id in self._conn_agents:
                return self._conn_agents[user_id]

            from personal_agent.factory import create_agent

            agent = await create_agent(self._settings, user_id=user_id)
            self._conn_agents[user_id] = agent
            return agent

    async def _run_agent(self, agent: Any, session: Any, text: str, message_id: str) -> None:
        """Run the agent and send the reply. Called under per-user lock."""
        try:
            result = await agent.run(text)
            reply = result.answer[:self.MAX_REPLY_LENGTH]

            # Send reply
            if self._api:
                resp = await self._api.reply_text(message_id, reply)
                code = resp.get("code", -1)
                if code != 0:
                    logger.error("Feishu reply failed: %s", resp.get("msg", "unknown"))
                else:
                    logger.info("Feishu reply sent (msg_id=%s)", message_id)
        except Exception as e:
            logger.exception("Feishu message processing failed: %s", e)
            if self._api:
                try:
                    await self._api.reply_text(message_id, f"Error: {str(e)[:500]}")
                except Exception:
                    pass
        finally:
            # Persist session state even on error (partial progress may be useful)
            async with session.memory_lock:
                session.short_term = agent.short_term
                session.working = agent.working
            self._router.session_manager.save_session(session)


# ANSI color for startup message
_C_GREEN = "\033[32m"
_C_BOLD = "\033[1m"
_C_RESET = "\033[0m"
