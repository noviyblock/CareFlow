"""Dataset registry, JSONL I/O and corpus statistics."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from careflow.data import empathetic_dialogues, esconv
from careflow.data.schema import Dialogue

LOADERS: dict[str, Callable[[str | Path], list[Dialogue]]] = {
    "esconv": esconv.load,
    "empathetic_dialogues": empathetic_dialogues.load,
}


def write_jsonl(dialogues: list[Dialogue], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Dialogue]] = {}
    for d in dialogues:
        by_split.setdefault(d.split, []).append(d)
    for split, items in by_split.items():
        with (out / f"{split}.jsonl").open("w") as f:
            for d in sorted(items, key=lambda x: x.dialogue_id):
                f.write(d.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Dialogue]:
    with Path(path).open() as f:
        return [Dialogue.model_validate_json(line) for line in f if line.strip()]


def stats(dialogues: list[Dialogue]) -> dict[str, Any]:
    turns = [len(d.turns) for d in dialogues]
    out: dict[str, Any] = {
        "n_dialogues": len(dialogues),
        "n_turns": sum(turns),
        "turns_per_dialogue_mean": round(mean(turns), 2),
        "turns_per_dialogue_max": max(turns),
        "splits": dict(Counter(d.split for d in dialogues)),
        "emotions_top10": dict(Counter(d.emotion for d in dialogues).most_common(10)),
    }
    strategies = Counter(t.strategy for d in dialogues for t in d.turns if t.strategy)
    if strategies:
        out["strategies"] = dict(strategies.most_common())
    deltas = [
        d.initial_intensity - d.final_intensity
        for d in dialogues
        if d.initial_intensity is not None and d.final_intensity is not None
    ]
    if deltas:
        out["intensity_drop_mean"] = round(mean(deltas), 3)
        out["share_deescalated"] = round(sum(x > 0 for x in deltas) / len(deltas), 3)
    return out


def write_stats(dialogues: list[Dialogue], path: str | Path) -> dict[str, Any]:
    s = stats(dialogues)
    Path(path).write_text(json.dumps(s, indent=2, ensure_ascii=False))
    return s
