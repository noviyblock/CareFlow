"""LLM-cache maintenance for multi-machine runs (Mac <-> Colab).

  snapshot SRC DST   consistent copy of a live cache (sqlite backup API; safe while a run is writing)
  merge SRC DST      add SRC's calls to DST without overwriting existing keys; prints how many were new
  stats PATH         number of cached calls per model

SQLite must not be opened in place on Google Drive (network FS breaks WAL locking): keep the working cache on local
disk, snapshot it to Drive, and restore with `snapshot` the other way round.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

SCHEMA = ("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL,"
          " created_at REAL NOT NULL)")


def _db(path: str) -> Path:
    p = Path(path)
    return p / "llm_cache.sqlite" if p.is_dir() or not p.suffix else p


def snapshot(src: str, dst: str) -> None:
    d = _db(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_db(src)) as a, sqlite3.connect(d) as b:
        a.backup(b)
    print(f"snapshot {_db(src)} -> {d}")


def merge(src: str, dst: str) -> None:
    d = _db(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(d) as b:
        b.execute(SCHEMA)
        before = b.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        b.execute("ATTACH DATABASE ? AS s", (str(_db(src)),))
        b.execute("INSERT OR IGNORE INTO calls SELECT * FROM s.calls")
        b.commit()
        after = b.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    print(f"merged {after - before} new calls into {d} (now {after})")


def stats(path: str) -> None:
    with sqlite3.connect(_db(path)) as a:
        c = Counter(json.loads(r)["model"] for (r,) in a.execute("SELECT request FROM calls"))
    for m, n in c.most_common():
        print(f"{n:8d}  {m}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"snapshot": snapshot, "merge": merge, "stats": stats}[cmd](*args)
