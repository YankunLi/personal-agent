"""FeishuChannel — Feishu (Lark) IM bot channel for the personal agent."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
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
AGENT_TTL = 1800  # Evict idle per-user agents after 30 minutes


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
            # A gateway 5xx/4xx returns an HTML error page, not JSON —
            # resp.json() would raise JSONDecodeError with no HTTP context.
            resp.raise_for_status()
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
        resp.raise_for_status()
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
        resp.raise_for_status()
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
        self._conn_agent_times: dict[str, float] = {}
        self._conn_sessions: dict[str, Any] = {}
        self._pending_tasks: set[asyncio.Task] = set()
        self._agent_lock = asyncio.Lock()
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._user_locks_lock = asyncio.Lock()
        # Per-user count of in-flight agent.run() calls. Eviction skips users
        # with active runs so an agent is never closed mid-run.
        self._active_runs: dict[str, int] = {}
        self._stop_event = asyncio.Event()

    # ── Channel interface ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the Feishu webhook HTTP server."""
        self._stop_event.clear()

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
        self._conn_agent_times.clear()
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
        raw_body = await request.read()

        # Verify request signature when encrypt_key is configured.
        # Feishu computes X-Lark-Signature as HMAC-SHA256 over
        # timestamp + nonce + raw_body using the SHA256 digest of encrypt_key
        # as the HMAC key. Without this check any party able to POST can
        # forge events.
        if self._encrypt_key:
            sig = request.headers.get("X-Lark-Signature", "")
            ts = request.headers.get("X-Lark-Request-Timestamp", "")
            nonce = request.headers.get("X-Lark-Request-Nonce", "")
            key = hashlib.sha256(self._encrypt_key.encode("utf-8")).digest()
            expected = hmac.new(
                key,
                (ts + nonce + raw_body.decode("utf-8", errors="replace")).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(sig, expected):
                logger.warning("Feishu signature verification failed")
                return web.json_response({"code": 1, "msg": "Invalid signature"}, status=403)
            # Reject replay: timestamp must be within 5 minutes.
            try:
                ts_int = int(ts)
            except ValueError:
                return web.json_response({"code": 1, "msg": "Invalid timestamp"}, status=400)
            if abs(time.time() - ts_int) > 300:
                logger.warning("Feishu event timestamp out of window: %s", ts)
                return web.json_response({"code": 1, "msg": "Stale timestamp"}, status=403)

        try:
            body = json.loads(raw_body)
        except Exception:
            return web.json_response({"code": 1, "msg": "Invalid JSON"}, status=400)

        # Encrypted event payload — Feishu sends {"encrypt": "<base64>"} when
        # encrypt_key is set. We fail closed (reject) rather than silently
        # dropping, since processing an undecryptable body would only ever
        # yield None header/event lookups.
        if isinstance(body, dict) and body.get("encrypt"):
            if not self._encrypt_key:
                logger.warning("Received encrypted Feishu event but no encrypt_key configured")
                return web.json_response({"code": 1, "msg": "Encryption not configured"}, status=500)
            logger.warning("Feishu event encryption decryption not implemented; rejecting encrypted event")
            return web.json_response({"code": 1, "msg": "Encrypted events not supported"}, status=501)

        # Feishu event format: {"schema": "2.0", "header": {...}, "event": {...}}
        header = body.get("header") if isinstance(body, dict) else None
        event_type = header.get("event_type", "") if header else ""
        event = body.get("event", {}) if isinstance(body, dict) else {}

        # Verify token for event callbacks
        if not self._verification_token:
            logger.warning("Feishu verification token not configured, rejecting event")
            return web.json_response({"code": 1, "msg": "Verification token not configured"}, status=403)
        event_token = header.get("token", "") if header else body.get("token", "")
        if event_token != self._verification_token:
                logger.warning("Feishu event rejected: invalid token")
                return web.json_response({"code": 1, "msg": "Invalid token"}, status=403)

        if event_type == "im.message.receive_v1":
            # Don't process messages if the server is shutting down.
            # Return 5xx so Feishu retries delivery after restart.
            if self._stop_event.is_set():
                return web.json_response({"code": 1, "msg": "Server shutting down"}, status=503)
            # Process in background, respond immediately to Feishu
            task = asyncio.create_task(self._process_message(event))
            self._pending_tasks.add(task)
            task.add_done_callback(
                lambda t: (
                    logger.error("Feishu message processing failed: %s", t.exception())
                    if not t.cancelled() and t.exception() else None
                )
            )
            task.add_done_callback(self._pending_tasks.discard)
            # Re-check: if stop() beat us between the initial check and task
            # creation, cancel the task to prevent use-after-close on resources.
            if self._stop_event.is_set():
                task.cancel()
                return web.json_response({"code": 1, "msg": "Server shutting down"}, status=503)
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

        # Feishu fires im.message.receive_v1 for the bot's own replies too.
        # Without this filter the bot would process its own messages and loop.
        if sender.get("sender_type") != "user":
            logger.debug("Ignoring non-user message (sender_type=%s)", sender.get("sender_type"))
            return

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
        async with self._user_locks_lock:
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

            # Get or create agent for this user (increments active-run count
            # under _agent_lock so eviction cannot close it mid-run).
            agent = await self._get_or_create_agent(user_id)
            try:
                async with session.memory_lock:
                    agent.short_term = session.short_term
                    agent.working = session.working

                await self._run_agent(agent, session, text, message_id)
            finally:
                async with self._agent_lock:
                    remaining = self._active_runs.get(user_id, 0) - 1
                    if remaining > 0:
                        self._active_runs[user_id] = remaining
                    else:
                        self._active_runs.pop(user_id, None)

    # ── Agent management ─────────────────────────────────────────────────────

    async def _get_or_create_agent(self, user_id: str) -> Any:
        """Get or create an agent for a Feishu user.

        The active-run counter is incremented under ``_agent_lock`` so that
        ``_evict_idle_agents`` (which holds the same lock) cannot close this
        agent between lookup and run. The caller decrements in a ``finally``.
        """
        async with self._agent_lock:
            agent = self._conn_agents.get(user_id)
            if agent is not None:
                self._conn_agent_times[user_id] = time.time()
                self._active_runs[user_id] = self._active_runs.get(user_id, 0) + 1
                return agent

            # Evict idle agents to prevent unbounded memory growth
            await self._evict_idle_agents()

            from personal_agent.factory import create_agent

            agent = await create_agent(self._settings, user_id=user_id)
            self._conn_agents[user_id] = agent
            self._conn_agent_times[user_id] = time.time()
            self._active_runs[user_id] = self._active_runs.get(user_id, 0) + 1
            return agent

    async def _evict_idle_agents(self) -> None:
        """Evict agents that have been idle longer than AGENT_TTL.

        Skips any user with an in-flight run so an agent is never closed while
        ``agent.run()`` is executing.
        """
        now = time.time()
        stale = [
            uid for uid, last_used in self._conn_agent_times.items()
            if now - last_used > AGENT_TTL and self._active_runs.get(uid, 0) == 0
        ]
        for uid in stale:
            agent = self._conn_agents.pop(uid, None)
            self._conn_agent_times.pop(uid, None)
            self._conn_sessions.pop(uid, None)
            # Do NOT pop _user_locks[uid]: a task may already be waiting on
            # the old lock object (between releasing user_lock and incrementing
            # _active_runs inside _get_or_create_agent). Removing it lets the
            # next task create a different lock and run concurrently with the
            # in-flight one, corrupting agent/session state. Lock objects are
            # small; keep them to preserve per-user serialization.
            self._active_runs.pop(uid, None)
            if agent:
                try:
                    await agent.close()
                except Exception:
                    pass
        if stale:
            logger.info("Evicted %d idle Feishu agent(s)", len(stale))

    async def _run_agent(self, agent: Any, session: Any, text: str, message_id: str) -> None:
        """Run the agent and send the reply. Called under per-user lock."""
        # Refresh the timestamp to prevent eviction during long-running agent.run()
        self._conn_agent_times[session.user_id] = time.time()
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
                    await self._api.reply_text(
                        message_id,
                        "Sorry, an internal error occurred while processing your request. Please try again.",
                    )
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
