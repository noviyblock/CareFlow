"""Download pinned source files and verify their sha256 before anything reads them."""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cfg: dict[str, Any]) -> Path:
    """Download every file in `cfg.source.files` into `cfg.raw_dir`; skip files whose hash already matches."""
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    src = cfg["source"]
    for name, expected in src["files"].items():
        dst = raw / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and sha256(dst) == expected):
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(f"{src['base_url']}/{name}", tmp)
            tmp.rename(dst)
        actual = sha256(dst)
        if actual != expected:
            raise ValueError(f"sha256 mismatch for {dst}: expected {expected}, got {actual}")
    if archive := src.get("extract"):
        with tarfile.open(raw / archive) as tar:
            tar.extractall(raw, filter="data")
    return raw
