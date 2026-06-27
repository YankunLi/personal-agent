"""Tests for task tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from personal_agent.tools.builtin.task import (
    create_task_create_tool,
    create_task_get_tool,
    create_task_list_tool,
    create_task_stop_tool,
    create_task_update_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def session_id():
    """Unique session ID for test isolation."""
    import uuid
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def executor(session_id):
    """Executor with all task tools registered."""
    registry = ToolRegistry()
    registry.register(create_task_create_tool(session_id=session_id))
    registry.register(create_task_get_tool(session_id=session_id))
    registry.register(create_task_list_tool(session_id=session_id))
    registry.register(create_task_update_tool(session_id=session_id))
    registry.register(create_task_stop_tool(session_id=session_id))
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_create_and_get_task(executor):
    """Create a task then retrieve it."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={
            "subject": "Fix login bug",
            "description": "The login endpoint returns 500 when given invalid credentials.",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    data = json.loads(result.output)
    task_id = data["task"]["id"]
    assert task_id == "1"
    assert data["task"]["subject"] == "Fix login bug"

    # Get the task
    tc2 = ToolCall(
        id="2", name="task_get",
        arguments={"taskId": task_id},
    )
    result2 = await executor.execute(tc2)
    assert result2.error is None
    data2 = json.loads(result2.output)
    assert data2["task"]["id"] == task_id
    assert data2["task"]["subject"] == "Fix login bug"
    assert data2["task"]["description"] == "The login endpoint returns 500 when given invalid credentials."
    assert data2["task"]["status"] == "pending"
    assert data2["task"]["blocks"] == []
    assert data2["task"]["blockedBy"] == []


@pytest.mark.asyncio
async def test_create_task_with_active_form(executor):
    """Create task with activeForm."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={
            "subject": "Run tests",
            "description": "Run all unit tests",
            "activeForm": "Running tests",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    data = json.loads(result.output)
    task_id = data["task"]["id"]

    tc2 = ToolCall(id="2", name="task_get", arguments={"taskId": task_id})
    result2 = await executor.execute(tc2)
    data2 = json.loads(result2.output)
    assert data2["task"]["activeForm"] == "Running tests"


@pytest.mark.asyncio
async def test_auto_increment_ids(executor):
    """Task IDs should auto-increment."""
    for i in range(3):
        tc = ToolCall(
            id=str(i), name="task_create",
            arguments={"subject": f"Task {i+1}", "description": f"Desc {i+1}"},
        )
        result = await executor.execute(tc)
        data = json.loads(result.output)
        assert data["task"]["id"] == str(i + 1)


@pytest.mark.asyncio
async def test_list_tasks(executor):
    """List all tasks."""
    for i in range(3):
        tc = ToolCall(
            id=str(i), name="task_create",
            arguments={"subject": f"Task {i+1}", "description": f"Desc {i+1}"},
        )
        await executor.execute(tc)

    tc = ToolCall(id="list", name="task_list", arguments={})
    result = await executor.execute(tc)
    assert result.error is None
    assert "Task 1" in result.output
    assert "Task 2" in result.output
    assert "Task 3" in result.output
    assert "3 tasks total" in result.output


@pytest.mark.asyncio
async def test_list_empty(executor):
    """Listing with no tasks should show message."""
    tc = ToolCall(id="1", name="task_list", arguments={})
    result = await executor.execute(tc)
    assert result.error is None
    assert "No tasks found" in result.output


@pytest.mark.asyncio
async def test_update_status(executor):
    """Update task status."""
    # Create
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Test", "description": "Test desc"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    # Update to in_progress
    tc2 = ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "status": "in_progress"},
    )
    result2 = await executor.execute(tc2)
    data2 = json.loads(result2.output)
    assert data2["success"] is True
    assert "status" in data2["updatedFields"]

    # Verify
    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["task"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_update_subject(executor):
    """Update task subject."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Original", "description": "Desc"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "subject": "Updated"},
    )
    result2 = await executor.execute(tc2)
    data2 = json.loads(result2.output)
    assert data2["success"] is True

    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["task"]["subject"] == "Updated"


@pytest.mark.asyncio
async def test_update_nonexistent(executor):
    """Update non-existent task should return error."""
    tc = ToolCall(
        id="1", name="task_update",
        arguments={"taskId": "999", "status": "completed"},
    )
    result = await executor.execute(tc)
    data = json.loads(result.output)
    assert data["success"] is False
    assert "not found" in data["error"].lower()


@pytest.mark.asyncio
async def test_delete_task(executor):
    """Delete a task via status='deleted'."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "To delete", "description": "Will be deleted"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "status": "deleted"},
    )
    result2 = await executor.execute(tc2)
    data2 = json.loads(result2.output)
    assert data2["success"] is True

    # Verify gone
    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    assert "not found" in result3.output.lower()


@pytest.mark.asyncio
async def test_task_dependencies(executor):
    """Set up and verify task dependencies."""
    # Create two tasks
    tc1 = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Task A", "description": "First task"},
    )
    result1 = await executor.execute(tc1)
    id_a = json.loads(result1.output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_create",
        arguments={"subject": "Task B", "description": "Second task"},
    )
    result2 = await executor.execute(tc2)
    id_b = json.loads(result2.output)["task"]["id"]

    # Task A blocks Task B
    tc3 = ToolCall(
        id="3", name="task_update",
        arguments={"taskId": id_a, "addBlocks": [id_b]},
    )
    result3 = await executor.execute(tc3)
    assert json.loads(result3.output)["success"] is True

    # Verify dependency
    tc4 = ToolCall(id="4", name="task_get", arguments={"taskId": id_a})
    result4 = await executor.execute(tc4)
    data4 = json.loads(result4.output)
    assert id_b in data4["task"]["blocks"]

    tc5 = ToolCall(id="5", name="task_get", arguments={"taskId": id_b})
    result5 = await executor.execute(tc5)
    data5 = json.loads(result5.output)
    assert id_a in data5["task"]["blockedBy"]


@pytest.mark.asyncio
async def test_resolve_dependencies_on_complete(executor):
    """Completing a task should unblock dependents."""
    # Create two tasks with dependency
    tc1 = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Task A", "description": "Blocker"},
    )
    id_a = json.loads((await executor.execute(tc1)).output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_create",
        arguments={"subject": "Task B", "description": "Blocked"},
    )
    id_b = json.loads((await executor.execute(tc2)).output)["task"]["id"]

    # A blocks B
    await executor.execute(ToolCall(
        id="3", name="task_update",
        arguments={"taskId": id_a, "addBlocks": [id_b]},
    ))

    # Complete A
    await executor.execute(ToolCall(
        id="4", name="task_update",
        arguments={"taskId": id_a, "status": "completed"},
    ))

    # B should no longer be blocked by A
    result = await executor.execute(ToolCall(
        id="5", name="task_get", arguments={"taskId": id_b},
    ))
    data = json.loads(result.output)
    assert id_a not in data["task"]["blockedBy"]


@pytest.mark.asyncio
async def test_list_filters_completed_blockers(executor):
    """TaskList should filter out completed blockers."""
    # Create two tasks
    tc1 = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Task A", "description": "Will be completed"},
    )
    id_a = json.loads((await executor.execute(tc1)).output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_create",
        arguments={"subject": "Task B", "description": "Was blocked"},
    )
    id_b = json.loads((await executor.execute(tc2)).output)["task"]["id"]

    # A blocks B
    await executor.execute(ToolCall(
        id="3", name="task_update",
        arguments={"taskId": id_a, "addBlocks": [id_b]},
    ))

    # Complete A
    await executor.execute(ToolCall(
        id="4", name="task_update",
        arguments={"taskId": id_a, "status": "completed"},
    ))

    # List should not show A as blocking B
    result = await executor.execute(ToolCall(id="5", name="task_list", arguments={}))
    assert "blocked by" not in result.output


@pytest.mark.asyncio
async def test_stop_task(executor):
    """Stop a running task."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Long task", "description": "Running"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    # Set to in_progress
    await executor.execute(ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "status": "in_progress"},
    ))

    # Stop it
    tc3 = ToolCall(
        id="3", name="task_stop",
        arguments={"task_id": task_id},
    )
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["success"] is True

    # Verify completed
    tc4 = ToolCall(id="4", name="task_get", arguments={"taskId": task_id})
    result4 = await executor.execute(tc4)
    data4 = json.loads(result4.output)
    assert data4["task"]["status"] == "completed"


@pytest.mark.asyncio
async def test_stop_task_not_running(executor):
    """Stopping a non-running task should fail."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Pending task", "description": "Not started"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    tc2 = ToolCall(
        id="2", name="task_stop",
        arguments={"task_id": task_id},
    )
    result2 = await executor.execute(tc2)
    data2 = json.loads(result2.output)
    assert data2["success"] is False
    assert "not running" in data2["error"].lower()


@pytest.mark.asyncio
async def test_stop_task_with_shell_id(executor):
    """Stop using deprecated shell_id parameter."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Shell task", "description": "Running"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    await executor.execute(ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "status": "in_progress"},
    ))

    tc3 = ToolCall(
        id="3", name="task_stop",
        arguments={"shell_id": task_id},
    )
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["success"] is True


@pytest.mark.asyncio
async def test_stop_task_no_id(executor):
    """Stop without any ID should fail."""
    tc = ToolCall(id="1", name="task_stop", arguments={})
    result = await executor.execute(tc)
    data = json.loads(result.output)
    assert data["success"] is False


@pytest.mark.asyncio
async def test_update_metadata_merge(executor):
    """Metadata should be merged, not replaced."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={
            "subject": "Meta test",
            "description": "Testing metadata",
            "metadata": {"key1": "value1"},
        },
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    # Merge new key
    await executor.execute(ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "metadata": {"key2": "value2"}},
    ))

    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["task"]["metadata"]["key1"] == "value1"
    assert data3["task"]["metadata"]["key2"] == "value2"


@pytest.mark.asyncio
async def test_update_metadata_delete_key(executor):
    """Setting metadata key to null should delete it."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={
            "subject": "Meta delete",
            "description": "Testing null metadata",
            "metadata": {"keep": "val", "remove": "val"},
        },
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    await executor.execute(ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "metadata": {"remove": None}},
    ))

    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert "keep" in data3["task"]["metadata"]
    assert "remove" not in data3["task"]["metadata"]


@pytest.mark.asyncio
async def test_get_nonexistent(executor):
    """Get non-existent task should return not found."""
    tc = ToolCall(id="1", name="task_get", arguments={"taskId": "999"})
    result = await executor.execute(tc)
    assert "not found" in result.output.lower()


@pytest.mark.asyncio
async def test_list_filters_internal_tasks(executor):
    """Tasks with _internal metadata should be filtered from list."""
    # Create a normal task
    tc1 = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Visible", "description": "Should appear"},
    )
    await executor.execute(tc1)

    # Create an internal task
    tc2 = ToolCall(
        id="2", name="task_create",
        arguments={
            "subject": "Hidden",
            "description": "Should not appear",
            "metadata": {"_internal": True},
        },
    )
    await executor.execute(tc2)

    tc3 = ToolCall(id="3", name="task_list", arguments={})
    result = await executor.execute(tc3)
    assert "Visible" in result.output
    assert "Hidden" not in result.output
    assert "1 tasks total" in result.output


@pytest.mark.asyncio
async def test_update_owner(executor):
    """Update task owner."""
    tc = ToolCall(
        id="1", name="task_create",
        arguments={"subject": "Owner test", "description": "Testing owner"},
    )
    result = await executor.execute(tc)
    task_id = json.loads(result.output)["task"]["id"]

    await executor.execute(ToolCall(
        id="2", name="task_update",
        arguments={"taskId": task_id, "owner": "agent-1"},
    ))

    tc3 = ToolCall(id="3", name="task_get", arguments={"taskId": task_id})
    result3 = await executor.execute(tc3)
    data3 = json.loads(result3.output)
    assert data3["task"]["owner"] == "agent-1"