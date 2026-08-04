"""Regression tests: reviewer bug scoping must work on Windows paths.

``str(Path)`` yields backslashes on Windows, so a module prefix built from
``module_dir.relative_to(repo_root)`` became ``src\\personal_agent/`` — a
hybrid that never matched the reviewer's ``/``-separated bug locations. As a
result, prev_bugs/applied_fixes context was silently dropped on Windows and
already-fixed bugs were re-reported every round.
"""

from personal_agent.orchestrator.reviewer import _bugs_for_module, _norm_path
from personal_agent.orchestrator.state import Bug


def _bug(location: str) -> Bug:
    return Bug(location=location, severity="major", description="d")


class TestNormPath:
    def test_backslashes_normalized(self):
        assert _norm_path(r"src\personal_agent\foo.py") == "src/personal_agent/foo.py"

    def test_forward_slashes_unchanged(self):
        assert _norm_path("src/personal_agent/foo.py") == "src/personal_agent/foo.py"


class TestBugsForModule:
    def test_windows_style_module_prefix_matches_forward_slash_locations(self):
        # Module prefix from Path.relative_to() on win32 (backslashes).
        bugs = [
            _bug("src/personal_agent/core/agent.py:10"),
            _bug("src/personal_agent/other.py:5"),
            _bug("src/personal_agent/core/agent.py:30"),
        ]
        matched = _bugs_for_module(bugs, r"src\personal_agent\core")
        assert [b.location for b in matched] == [
            "src/personal_agent/core/agent.py:10",
            "src/personal_agent/core/agent.py:30",
        ]

    def test_backslash_locations_match_windows_prefix(self):
        # LLM may copy file headers verbatim, producing backslash locations.
        bugs = [
            _bug(r"src\personal_agent\core\agent.py:10"),
            _bug(r"src\personal_agent\other.py:5"),
        ]
        matched = _bugs_for_module(bugs, r"src\personal_agent\core")
        assert [b.location for b in matched] == [r"src\personal_agent\core\agent.py:10"]

    def test_exact_module_match_included(self):
        bugs = [_bug("src/personal_agent/core")]
        assert _bugs_for_module(bugs, "src/personal_agent/core") == bugs

    def test_other_modules_excluded(self):
        bugs = [_bug("src/other_module/foo.py:1")]
        assert _bugs_for_module(bugs, "src/personal_agent/core") == []
