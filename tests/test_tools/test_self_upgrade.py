"""Regression tests: update_instruction long-term delete by key.

The set path stores with metadata['key'] and a content-hash name, so
forget(key) never matched and the tool reported \"Key ... removed\" even
though nothing was deleted. The delete path must actually remove the memory
and report honestly when it is not found.
"""

import pytest

from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.long_term import LongTermMemory
from personal_agent.tools.builtin.self_upgrade import create_self_upgrade_tool


@pytest.mark.asyncio
async def test_set_then_delete_long_term_by_key(temp_memory_dir):
    store = FileMemoryStore(storage_dir=temp_memory_dir)
    ltm = LongTermMemory(store)
    tool = create_self_upgrade_tool(long_term_memory=ltm)

    result = await tool.execute(
        instruction="Always run lint before committing.",
        memory_type="long_term",
        action="set",
        key="rules",
    )
    assert "Stored in long-term memory" in result
    assert await ltm.count() == 1

    result = await tool.execute(
        instruction="Always run lint before committing.",
        memory_type="long_term",
        action="delete",
        key="rules",
    )
    assert "removed from long-term memory" in result
    assert await ltm.count() == 0


@pytest.mark.asyncio
async def test_delete_missing_long_term_reports_honestly(temp_memory_dir):
    store = FileMemoryStore(storage_dir=temp_memory_dir)
    ltm = LongTermMemory(store)
    tool = create_self_upgrade_tool(long_term_memory=ltm)

    result = await tool.execute(
        instruction="irrelevant",
        memory_type="long_term",
        action="delete",
        key="does-not-exist",
    )
    assert "No long-term memory found for key 'does-not-exist'" in result
    assert await ltm.count() == 0
