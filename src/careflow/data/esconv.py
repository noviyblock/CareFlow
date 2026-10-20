"""ESConv -> common schema.

The thu-coai release ships the full corpus (ESConv.json) and the official
70/15/15 split as preprocessed JSONL (train/valid/test.txt, text normalized
with `' '.join(x.split())`). We read annotations from ESConv.json and recover
split membership by matching the normalized turn sequence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from careflow.data.schema import Dialogue, Turn

SPLITS = ("train", "valid", "test")
_ROLE = {"seeker": "seeker", "supporter": "supporter"}


def _norm(text: str) -> str:
    return " ".join(text.strip().split())


def _fingerprint(texts: list[str]) -> str:
    return hashlib.sha256("\x1f".join(texts).encode()).hexdigest()


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _official_split(raw: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in SPLITS:
        with (raw / f"{split}.txt").open() as f:
            for line in f:
                fp = _fingerprint([u["text"] for u in json.loads(line)["dialog"]])
                if membership.setdefault(fp, split) != split:
                    raise ValueError(f"Dialogue appears in two official splits: {fp}")
    return membership


def load(raw_dir: str | Path) -> list[Dialogue]:
    raw = Path(raw_dir)
    membership = _official_split(raw)
    corpus = json.loads((raw / "ESConv.json").read_text())

    dialogues = []
    for i, d in enumerate(corpus):
        texts = [_norm(u["content"]) for u in d["dialog"]]
        split = membership.get(_fingerprint(texts))
        if split is None:
            raise ValueError(f"ESConv dialogue {i} not found in any official split")
        turns = [
            Turn(
                idx=j,
                role=_ROLE[u["speaker"]],
                text=text,
                strategy=u["annotation"].get("strategy"),
                feedback=_int_or_none(u["annotation"].get("feedback")),
            )
            for j, (u, text) in enumerate(zip(d["dialog"], texts, strict=True))
        ]
        seeker_survey = d.get("survey_score", {}).get("seeker", {})
        dialogues.append(
            Dialogue(
                dialogue_id=f"esconv-{i:04d}",
                source="esconv",
                split=split,
                turns=turns,
                situation=_norm(d.get("situation", "")) or None,
                emotion=d.get("emotion_type"),
                problem_type=d.get("problem_type"),
                initial_intensity=_int_or_none(seeker_survey.get("initial_emotion_intensity")),
                final_intensity=_int_or_none(seeker_survey.get("final_emotion_intensity")),
                meta={
                    "experience_type": d.get("experience_type"),
                    "seeker_empathy": _int_or_none(seeker_survey.get("empathy")),
                    "seeker_relevance": _int_or_none(seeker_survey.get("relevance")),
                },
            )
        )
    return dialogues
