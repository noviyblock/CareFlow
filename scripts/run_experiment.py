"""Single entry point for all experiments.

    python scripts/run_experiment.py --config configs/experiments/smoke.yaml [--overlay extra.yaml] [key.sub=value ...]

--overlay deep-merges another YAML on top of the experiment config (e.g. configs/colab.yaml switches any experiment
to the Colab vLLM backend); dotted overrides are applied last.
"""

from __future__ import annotations

import argparse

from careflow.tasks import TASKS
from careflow.utils.config import apply_overrides, deep_merge, load_config
from careflow.utils.seed import resolve_device, set_seed
from careflow.utils.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overlay", action="append", default=[], help="YAML deep-merged on top (repeatable)")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. seed=1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for extra in args.overlay:
        cfg = deep_merge(cfg, load_config(extra))
    cfg = apply_overrides(cfg, args.overrides)
    task = cfg["experiment"]["task"]
    if task not in TASKS:
        raise SystemExit(f"Unknown task {task!r}; registered: {sorted(TASKS)}")

    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = resolve_device(cfg.get("compute", {}).get("device", "cpu"))
    with start_run(cfg, device):
        TASKS[task](cfg, device)


if __name__ == "__main__":
    main()
