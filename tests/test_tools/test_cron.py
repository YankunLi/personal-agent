"""Tests for Cron tools and CronScheduler."""

from __future__ import annotations

import pytest

from personal_agent.cron_scheduler import CronScheduler, _cron_matches, _next_cron_match
from personal_agent.tools.builtin.cron import (
    create_cron_create_tool,
    create_cron_delete_tool,
    create_cron_list_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def scheduler():
    return CronScheduler()


@pytest.fixture
def executor(scheduler):
    registry = ToolRegistry()
    registry.register(create_cron_create_tool(scheduler=scheduler))
    registry.register(create_cron_delete_tool(scheduler=scheduler))
    registry.register(create_cron_list_tool(scheduler=scheduler))
    return ToolExecutor(registry=registry)


class TestCronParsing:
    """Test cron expression parsing."""

    def test_every_minute(self):
        from datetime import datetime
        assert _cron_matches("* * * * *", datetime(2026, 1, 1, 12, 0))

    def test_specific_hour(self):
        from datetime import datetime
        assert _cron_matches("0 9 * * *", datetime(2026, 1, 1, 9, 0))
        assert not _cron_matches("0 9 * * *", datetime(2026, 1, 1, 10, 0))

    def test_weekdays_only(self):
        from datetime import datetime
        # Monday = 0, so 1-5 = Tue-Sat... wait, Python's weekday(): Monday=0, Sunday=6
        # "1-5" = Monday-Friday
        mon = datetime(2026, 6, 1, 9, 0)  # Monday
        assert _cron_matches("0 9 * * 1-5", mon)

    def test_comma_separated(self):
        from datetime import datetime
        # 9am on the 1st and 15th
        assert _cron_matches("0 9 1,15 * *", datetime(2026, 1, 1, 9, 0))
        assert _cron_matches("0 9 1,15 * *", datetime(2026, 1, 15, 9, 0))
        assert not _cron_matches("0 9 1,15 * *", datetime(2026, 1, 2, 9, 0))

    def test_step_values(self):
        from datetime import datetime
        # Every 5 minutes
        assert _cron_matches("*/5 * * * *", datetime(2026, 1, 1, 12, 0))
        assert _cron_matches("*/5 * * * *", datetime(2026, 1, 1, 12, 5))
        assert not _cron_matches("*/5 * * * *", datetime(2026, 1, 1, 12, 1))

    def test_next_cron_match(self):
        from datetime import datetime
        # Next 9am
        next_match = _next_cron_match("0 9 * * *", datetime(2026, 1, 1, 5, 0))
        assert next_match is not None
        assert next_match.hour == 9
        assert next_match.minute == 0

    def test_invalid_cron(self):
        assert _next_cron_match("invalid cron expression") is None


class TestCronCreate:
    """Test CronCreate tool."""

    @pytest.mark.asyncio
    async def test_create_job(self, executor, scheduler):
        tc = ToolCall(
            id="1", name="cron_create",
            arguments={
                "cron": "0 9 * * *",
                "prompt": "Remind me to check email",
                "recurring": True,
                "durable": False,
            },
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Cron job created" in result.output
        assert len(await scheduler.list_jobs()) == 1

    @pytest.mark.asyncio
    async def test_create_invalid_cron(self, executor, scheduler):
        tc = ToolCall(
            id="1", name="cron_create",
            arguments={
                "cron": "invalid",
                "prompt": "test",
            },
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Error" in result.output


class TestCronDelete:
    """Test CronDelete tool."""

    @pytest.mark.asyncio
    async def test_delete_job(self, executor, scheduler):
        job_id = await scheduler.add_job("0 9 * * *", "test prompt")
        tc = ToolCall(
            id="1", name="cron_delete",
            arguments={"id": job_id},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "deleted" in result.output
        assert len(await scheduler.list_jobs()) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, executor):
        tc = ToolCall(
            id="1", name="cron_delete",
            arguments={"id": "nonexistent"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "not found" in result.output


class TestCronList:
    """Test CronList tool."""

    @pytest.mark.asyncio
    async def test_list_empty(self, executor):
        tc = ToolCall(
            id="1", name="cron_list",
            arguments={},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "No scheduled cron jobs" in result.output

    @pytest.mark.asyncio
    async def test_list_jobs(self, executor, scheduler):
        await scheduler.add_job("0 9 * * *", "Morning check", recurring=True)
        await scheduler.add_job("30 14 * * *", "Afternoon reminder", recurring=False)

        tc = ToolCall(
            id="1", name="cron_list",
            arguments={},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Morning check" in result.output
        assert "Afternoon reminder" in result.output
        assert "recurring" in result.output
        assert "one-shot" in result.output


class TestCronSchedulerMaxJobs:
    """Test max jobs limit."""

    @pytest.mark.asyncio
    async def test_max_jobs(self, scheduler):
        for i in range(scheduler.MAX_JOBS):
            await scheduler.add_job(f"0 {i % 24} * * *", f"job {i}")
        with pytest.raises(ValueError, match="Maximum"):
            await scheduler.add_job("0 0 * * *", "one too many")