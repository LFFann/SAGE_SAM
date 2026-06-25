"""Tiny JSONL logger used by smoke and small runs."""

from __future__ import annotations

import json
from pathlib import Path


class JSONLLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **payload) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
