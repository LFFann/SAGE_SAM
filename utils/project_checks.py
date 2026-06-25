"""Project independence checks shared by tests and CLI tools."""

from __future__ import annotations

from pathlib import Path


FORBIDDEN_SNIPPETS = (
    ".." + "/a3-sam",
    "sys" + ".path." + "append",
    "sys" + ".path." + "insert",
    "PYTHON" + "PATH",
)


def scan_for_forbidden_runtime_links(root: str | Path) -> list[tuple[str, str]]:
    root = Path(root)
    findings = []
    for path in root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        if path.name == "project_checks.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                findings.append((str(path.relative_to(root)), snippet))
    return findings


def find_symlinks(root: str | Path) -> list[str]:
    root = Path(root)
    return [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
