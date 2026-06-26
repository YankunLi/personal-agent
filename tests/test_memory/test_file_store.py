"""Tests for FileMemoryStore — markdown file-based memory CRUD."""

import pytest

from personal_agent.memory.file_store import (
    FileMemoryStore,
    _parse_frontmatter,
    _format_frontmatter,
    _slugify,
)


class TestFrontmatterParsing:
    def test_parse_complete_frontmatter(self):
        text = """---
name: user role
description: The user's role and background
type: user
---

The user is a senior backend engineer."""
        meta, body = _parse_frontmatter(text)
        assert meta == {
            "name": "user role",
            "description": "The user's role and background",
            "type": "user",
        }
        assert body == "The user is a senior backend engineer."

    def test_parse_no_frontmatter(self):
        text = "Just plain text, no frontmatter."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_parse_partial_frontmatter(self):
        text = """---
name: test
---

Body text."""
        meta, body = _parse_frontmatter(text)
        assert meta == {"name": "test"}
        assert body == "Body text."

    def test_format_frontmatter(self):
        meta = {
            "name": "Test Memory",
            "description": "A test",
            "type": "user",
        }
        result = _format_frontmatter(meta)
        assert "name: Test Memory" in result
        assert "description: A test" in result
        assert "type: user" in result
        assert result.startswith("---")
        assert result.endswith("---")

    def test_format_frontmatter_extra_keys_ignored(self):
        """Only name, description, type are included in frontmatter."""
        meta = {"name": "test", "type": "user", "extra": "ignored"}
        result = _format_frontmatter(meta)
        assert "extra" not in result


class TestSlugify:
    def test_simple_name(self):
        assert _slugify("User Role") == "user_role"

    def test_special_characters(self):
        assert _slugify("What's my role?") == "whats_my_role"

    def test_multiple_spaces_and_dashes(self):
        assert _slugify("  foo---bar  ") == "foo_bar"

    def test_empty_name(self):
        assert _slugify("") == "memory"


class TestFileMemoryStoreCRUD:
    @pytest.mark.asyncio
    async def test_add_and_get(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        path = await store.add("User Role", "The user is an engineer.", memory_type="user")
        assert path.exists()
        assert path.suffix == ".md"

        result = await store.get("User Role")
        assert result is not None
        meta, body = result
        assert meta["name"] == "User Role"
        assert meta["type"] == "user"
        assert "engineer" in body

    @pytest.mark.asyncio
    async def test_add_updates_existing(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("Test", "Original content", memory_type="user")
        await store.add("Test", "Updated content", memory_type="user")

        result = await store.get("Test")
        assert result is not None
        _, body = result
        assert body == "Updated content"

    @pytest.mark.asyncio
    async def test_get_missing(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        result = await store.get("Nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("To Delete", "Content", memory_type="user")
        assert await store.get("To Delete") is not None

        deleted = await store.delete("To Delete")
        assert deleted is True
        assert await store.get("To Delete") is None

    @pytest.mark.asyncio
    async def test_delete_missing(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        deleted = await store.delete("Nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_invalid_memory_type(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        with pytest.raises(ValueError, match="Invalid memory type"):
            await store.add("Test", "Content", memory_type="invalid_type")

    @pytest.mark.asyncio
    async def test_get_by_type(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Mem", "User content", memory_type="user")
        await store.add("Feedback Mem", "Feedback content", memory_type="feedback")
        await store.add("Project Mem", "Project content", memory_type="project")

        user_mems = await store.get_by_type("user")
        assert len(user_mems) == 1
        assert user_mems[0]["name"] == "User Mem"

        feedback_mems = await store.get_by_type("feedback")
        assert len(feedback_mems) == 1
        assert feedback_mems[0]["name"] == "Feedback Mem"

    @pytest.mark.asyncio
    async def test_count(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        assert store.count() == 0
        await store.add("First", "Content", memory_type="user")
        assert store.count() == 1
        await store.add("Second", "Content", memory_type="feedback")
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("First", "Content", memory_type="user")
        await store.add("Second", "Content", memory_type="feedback")
        assert store.count() == 2

        await store.clear()
        assert store.count() == 0
        assert store.list_all() == []


class TestMemoryIndex:
    @pytest.mark.asyncio
    async def test_index_created_on_add(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Role", "The user is an engineer.", memory_type="user")

        index_text = store.load_index_text()
        assert "User Role" in index_text
        assert "engineer" not in index_text  # Index only has description, not body

    @pytest.mark.asyncio
    async def test_index_updated_on_delete(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("A", "Content A", memory_type="user")
        await store.add("B", "Content B", memory_type="user")

        assert "A" in store.load_index_text()
        await store.delete("A")
        assert "A" not in store.load_index_text()
        assert "B" in store.load_index_text()

    @pytest.mark.asyncio
    async def test_list_all(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("First", "Content 1", memory_type="user", description="First memory")
        await store.add("Second", "Content 2", memory_type="feedback", description="Second memory")

        entries = store.list_all()
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert names == {"First", "Second"}

    @pytest.mark.asyncio
    async def test_build_index_regenerates(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("A", "Content A", memory_type="user")
        await store.add("B", "Content B", memory_type="user")

        # Corrupt index by deleting it
        store.index_path.unlink()
        assert not store.index_path.exists()

        await store.build_index()
        assert store.index_path.exists()
        entries = store.list_all()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_repair_index_removes_stale(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("A", "Content A", memory_type="user")

        # Manually add a stale entry to the index
        entries = store.list_all()
        entries.append({"name": "Ghost", "filename": "user_ghost.md", "description": "Stale"})
        store._write_index_locked(entries)

        assert "Ghost" in store.load_index_text()

        removed = await store.repair_index()
        assert removed == 1
        assert "Ghost" not in store.load_index_text()

    @pytest.mark.asyncio
    async def test_empty_index_auto_created(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        text = store.load_index_text()
        assert "No memories stored yet" in text

    @pytest.mark.asyncio
    async def test_memory_types_in_filename(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Mem", "Content", memory_type="user")
        await store.add("Feedback Mem", "Content", memory_type="feedback")
        await store.add("Project Mem", "Content", memory_type="project")
        await store.add("Reference Mem", "Content", memory_type="reference")

        files = list(temp_memory_dir.glob("*.md"))
        filenames = {f.name for f in files if f.name != "MEMORY.md"}
        assert "user_user_mem.md" in filenames
        assert "feedback_feedback_mem.md" in filenames
        assert "project_project_mem.md" in filenames
        assert "reference_reference_mem.md" in filenames