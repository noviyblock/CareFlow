"""EmpatheticDialogues -> common schema.

The CSVs are not RFC-4180: commas inside text are escaped as `_comma_`, and
in valid/test about half the rows carry a 9th, header-less column (candidate
responses). Like ParlAI we split on ',' and keep the first 8 fields.
The speaker who wrote the situation prompt (first utterance) is the seeker.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from careflow.data.schema import Dialogue, Turn

SPLITS = ("train", "valid", "test")
_FIELDS = ("conv_id", "utterance_idx", "context", "prompt", "speaker_idx", "utterance", "selfeval", "tags")


def _clean(text: str) -> str:
    return " ".join(text.replace("_comma_", ",").split())


def _rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    header = tuple(lines[0].split(","))
    if header != _FIELDS:
        raise ValueError(f"Unexpected ED header in {path}: {header}")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < len(_FIELDS):
            raise ValueError(f"Malformed ED row in {path}: {line[:80]!r}")
        rows.append(dict(zip(_FIELDS, parts[: len(_FIELDS)], strict=True)))
    return rows


def load(raw_dir: str | Path) -> list[Dialogue]:
    base = Path(raw_dir) / "empatheticdialogues"
    dialogues = []
    for split in SPLITS:
        convs: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _rows(base / f"{split}.csv"):
            convs[row["conv_id"]].append(row)
        for conv_id, rows in convs.items():
            rows.sort(key=lambda r: int(r["utterance_idx"]))
            seeker_speaker = rows[0]["speaker_idx"]
            turns = [
                Turn(
                    idx=j,
                    role="seeker" if r["speaker_idx"] == seeker_speaker else "supporter",
                    text=_clean(r["utterance"]),
                )
                for j, r in enumerate(rows)
            ]
            dialogues.append(
                Dialogue(
                    dialogue_id=f"ed-{conv_id}",
                    source="empathetic_dialogues",
                    split=split,
                    turns=turns,
                    situation=_clean(rows[0]["prompt"]),
                    emotion=rows[0]["context"],
                    meta={"selfeval": rows[0]["selfeval"] or None},
                )
            )
    return dialogues
