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


def test_cache_off_from_yaml_boolean(tmp_path):
    c = LLMCache(tmp_path, False)  # `mode: off` in YAML arrives as False
    assert c.mode == "off" and c.get({"x": 1}) is None
    c.close()


def test_cache_tool_snapshot_and_merge(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("cache_tool", "scripts/cache_tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    a, b = LLMCache(tmp_path / "a"), LLMCache(tmp_path / "b")
    a.put({"model": "m", "x": 1}, {"text": "one"})
    b.put({"model": "m", "x": 2}, {"text": "two"})
    b.put({"model": "m", "x": 1}, {"text": "kept"})  # existing key in DST must not be overwritten
    a.close(), b.close()
    tool.merge(str(tmp_path / "a"), str(tmp_path / "b"))
    tool.snapshot(str(tmp_path / "b"), str(tmp_path / "c"))
    c = LLMCache(tmp_path / "c", "read_only")
    assert c.get({"model": "m", "x": 2}) == {"text": "two"} and c.get({"model": "m", "x": 1}) == {"text": "kept"}
    c.close()


def test_openai_compat_backend_request_and_parse(monkeypatch):
    import sys
    import types

    sent = {}

    class _Completions:
        def create(self, **kw):
            sent.update(kw)
            msg = types.SimpleNamespace(content="  hello  ")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                         usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=2))

    class _Client:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Completions())

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    from careflow.llm.openai_backend import OpenAICompatChat

    m = OpenAICompatChat("Qwen/Qwen3.5-9B", "c2022362357", "http://localhost:8001/v1", "agent",
                         chat_template_kwargs={"enable_thinking": False})
    g = m.chat([{"role": "user", "content": "hi"}], 16, 0.0, 5)
    assert g.text == "hello" and g.prompt_tokens == 7 and g.completion_tokens == 2 and g.finish_reason == "stop"
    assert sent["model"] == "agent" and sent["seed"] == 5 and sent["max_tokens"] == 16
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert m.name == "Qwen/Qwen3.5-9B@c2022362357"
