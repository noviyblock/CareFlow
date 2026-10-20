"""Single entry point for all experiments.

    python scripts/run_experiment.py --config configs/experiments/smoke.yaml [key.sub=value ...]
"""

from __future__ import annotations

import argparse

from careflow.tasks import TASKS
from careflow.utils.config import apply_overrides, load_config
from careflow.utils.seed import resolve_device, set_seed
from careflow.utils.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. seed=1")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    task = cfg["experiment"]["task"]
    if task not in TASKS:
        raise SystemExit(f"Unknown task {task!r}; registered: {sorted(TASKS)}")

    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = resolve_device(cfg.get("compute", {}).get("device", "cpu"))
    with start_run(cfg, device):
        TASKS[task](cfg, device)


if __name__ == "__main__":
    main()
