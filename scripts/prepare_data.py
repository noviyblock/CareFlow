"""DVC stage entry point for datasets.

    python scripts/prepare_data.py download --dataset esconv
    python scripts/prepare_data.py prepare  --dataset esconv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from careflow.data.download import fetch
from careflow.data.prepare import LOADERS, write_jsonl, write_stats
from careflow.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "prepare"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)[args.dataset]

    if args.stage == "download":
        fetch(cfg)
        return

    if args.dataset not in LOADERS:
        raise SystemExit(f"No prepare step for {args.dataset!r}; available: {sorted(LOADERS)}")
    dialogues = LOADERS[args.dataset](cfg["raw_dir"])
    write_jsonl(dialogues, cfg["processed_dir"])
    s = write_stats(dialogues, Path(cfg["processed_dir"]) / "stats.json")
    print(f"{args.dataset}: {s['n_dialogues']} dialogues, splits {s['splits']}")


if __name__ == "__main__":
    main()
