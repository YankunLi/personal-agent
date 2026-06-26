"""AgentServer — main process that coordinates channels, sessions, and agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from personal_agent.channels.base import Channel
from personal_agent.server.router import MessageRouter

if TYPE_CHECKING:
    from personal_agent.config import Settings

logger = logging.getLogger(__name__)


class AgentServer:
    """Main server process that coordinates channels and agent sessions.

    The AgentServer is the top-level orchestrator. It:
    1. Holds shared resources (session manager, message router)
    2. Manages channel lifecycle (start/stop)
    3. Provides a unified entry point for all agent communication

    Usage:
        server = AgentServer(settings)
        server.add_channel(CLIChannel(settings, ...))
        await server.start()   # Blocks until stopped
    """

    def __init__(self, settings: Settings):
        from personal_agent.session import SessionManager

        self.settings = settings
        self.session_manager = SessionManager()
        self.router = MessageRouter(self.session_manager)
        self._channels: list[Channel] = []
        self._running = False

    @property
    def channels(self) -> list[Channel]:
        return self._channels

    def add_channel(self, channel: Channel) -> None:
        """Register a channel to be started with the server."""
        self._channels.append(channel)

    async def start(self) -> None:
        """Start all registered channels.

        Loads existing sessions from disk, then starts each channel.
        This method blocks until all channels stop.
        """
        self._running = True
        self.session_manager.load_all()
        logger.info("AgentServer starting with %d channel(s)", len(self._channels))

        # Start all channels concurrently
        import asyncio
        tasks = []
        for channel in self._channels:
            logger.info("Starting channel: %s", channel.name)
            tasks.append(asyncio.create_task(channel.start()))

        # Wait for all channels to complete
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop all channels and persist sessions."""
        if not self._running:
            return
        self._running = False

        logger.info("AgentServer stopping %d channel(s)", len(self._channels))
        for channel in self._channels:
            try:
                await channel.stop()
            except Exception as e:
                logger.warning("Error stopping channel '%s': %s", channel.name, e)

        self.session_manager.save_current()
        logger.info("AgentServer stopped")
