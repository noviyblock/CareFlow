"""Content-addressed SQLite cache for LLM calls.

API models are not bit-reproducible even at temperature 0, so every generation
is keyed by (model, messages, sampling params) and stored. Re-running an
experiment in `read_only` mode replays exactly the same responses and fails
loudly on a miss instead of silently calling the API.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class CacheMiss(KeyError):
    pass


class LLMCache:
    def __init__(self, path: str | Path, mode: str = "read_write") -> None:
        if mode not in {"read_write", "read_only", "off"}:
            raise ValueError(f"Unknown cache mode {mode!r}")
        self.mode = mode
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path / "llm_cache.sqlite")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            " key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL,"
            " created_at REAL NOT NULL)"
        )

    @staticmethod
    def make_key(request: dict[str, Any]) -> str:
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if self.mode == "off":
            return None
        row = self._db.execute(
            "SELECT response FROM calls WHERE key = ?", (self.make_key(request),)
        ).fetchone()
        if row is None and self.mode == "read_only":
            raise CacheMiss(self.make_key(request))
        return json.loads(row[0]) if row else None

    def put(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        if self.mode != "read_write":
            return
        self._db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?)",
            (
                self.make_key(request),
                json.dumps(request, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                time.time(),
            ),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
