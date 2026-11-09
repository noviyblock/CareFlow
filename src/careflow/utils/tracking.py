"""MLflow run setup: every run logs its full resolved config, git commit, DVC lock and environment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import yaml

from careflow.utils.config import flatten

# MLflow caps param values at 6000 chars; long values stay in the config artifact.
_MAX_PARAM_LEN = 500
_AUTH_ENV = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _environment(device: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
    }


def _check_remote_auth(uri: str) -> None:
    """Fail before training starts, not at the first log call, if remote credentials are missing."""
    if not uri.startswith("http"):
        return
    missing = [v for v in _AUTH_ENV if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"MLflow tracking URI {uri} needs env {', '.join(missing)} "
            "(DagsHub username and access token). "
            "For offline runs pass mlflow.tracking_uri=sqlite:///mlflow.db"
        )


@contextmanager
def start_run(cfg: dict[str, Any], device: str) -> Iterator[mlflow.ActiveRun]:
    _check_remote_auth(cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=cfg["experiment"]["name"]) as run:
        mlflow.set_tags(
            {"git_commit": _git_commit(), "task": cfg["experiment"]["task"], "device": device}
        )
        params = {
            k: (str(v)[:_MAX_PARAM_LEN]) for k, v in flatten(cfg).items()
        }
        mlflow.log_params(params)
        mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config.resolved.yaml")
        if cfg["mlflow"].get("log_env", True):
            mlflow.log_text(json.dumps(_environment(device), indent=2), "environment.json")
        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock")
        # pre-registration: freeze hypotheses/analysis code with the run, before any result exists
        for f in cfg.get("prereg_files", []):
            mlflow.log_artifact(f, "prereg")
            mlflow.set_tag(f"prereg.sha256.{Path(f).name}", hashlib.sha256(Path(f).read_bytes()).hexdigest())
        yield run
