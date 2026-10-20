import random

import numpy as np
import pytest

from careflow.cache import CacheMiss, LLMCache
from careflow.utils.config import apply_overrides, flatten, load_config
from careflow.utils.seed import set_seed


def test_inherit_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\nmlflow:\n  experiment: a\n  log_env: true\n")
    (tmp_path / "exp.yaml").write_text("inherit: [base.yaml]\nmlflow:\n  experiment: b\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg == {"seed": 1, "mlflow": {"experiment": "b", "log_env": True}}
    cfg = apply_overrides(cfg, ["seed=7", "mlflow.log_env=false", "new.key=0.5"])
    assert cfg["seed"] == 7 and cfg["mlflow"]["log_env"] is False and cfg["new"]["key"] == 0.5
    assert flatten(cfg)["mlflow.experiment"] == "b"


def test_cyclic_inherit(tmp_path):
    (tmp_path / "a.yaml").write_text("inherit: [b.yaml]\n")
    (tmp_path / "b.yaml").write_text("inherit: [a.yaml]\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_seed_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    assert (random.random(), np.random.rand()) == first


def test_cache_modes(tmp_path):
    req = {"model": "m", "messages": [{"role": "user", "content": "привет"}], "temperature": 0}
    rw = LLMCache(tmp_path, "read_write")
    assert rw.get(req) is None
    rw.put(req, {"content": "ok"})
    assert rw.get(dict(reversed(list(req.items())))) == {"content": "ok"}  # key order-insensitive
    rw.close()

    ro = LLMCache(tmp_path, "read_only")
    assert ro.get(req) == {"content": "ok"}
    with pytest.raises(CacheMiss):
        ro.get({**req, "temperature": 1})
    ro.close()
