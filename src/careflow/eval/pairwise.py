"""System-vs-reference pairwise evaluation with the HEART judge, both presentation orders per pair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from careflow.data.contexts import Context
from careflow.eval import judge as J


@dataclass
class PairResult:
    """Per context: share of the two orders the system won (0, 0.5, 1); None if a judgment failed."""

    overall: list[float | None] = field(default_factory=list)
    categories: dict[str, list[float]] = field(default_factory=lambda: {c: [] for c in J.CATEGORIES})
    missing_categories: int = 0
    failures: int = 0
    records: list[dict] = field(default_factory=list)


def _judge_context(c: Context, system: str, reference: str, ask: Callable[[str], str]) -> dict:
    texts = {"human": c.human, **c.responses}
    out = {"records": [], "sys_wins": [], "cat_wins": {k: [] for k in J.CATEGORIES}, "ok": True,
           "failures": 0, "missing": 0}
    for first, second in ((system, reference), (reference, system)):
        raw = ask(J.build_prompt(c.history, c.speakers, texts[first], texts[second], c.emotion))
        try:
            v = J.parse(raw)
        except J.JudgeParseError as e:
            out["failures"] += 1
            out["ok"] = False
            out["records"].append({"ctx": c.dialogue_id, "first": first, "second": second, "error": str(e), "raw": raw})
            continue
        name = {J.ID_A: first, J.ID_B: second}
        out["sys_wins"].append(name[v["overall_eq"].winner] == system)
        for k in J.CATEGORIES:
            if v[k] is None:
                out["missing"] += 1
            else:
                out["cat_wins"][k].append(name[v[k].winner] == system)
        out["records"].append({"ctx": c.dialogue_id, "first": first, "second": second,
                               "winner": name[v["overall_eq"].winner]})
    return out


def judge_vs_reference(
    contexts: Sequence[Context], system: str, reference: str, ask: Callable[[str], str], progress_every: int = 20,
    workers: int = 1,
) -> PairResult:
    """`ask(prompt) -> judge text`. `reference` is 'human' or another system name. `workers` > 1 judges contexts
    concurrently (for a batching server); results are aggregated in context order, so output is identical."""
    from concurrent.futures import ThreadPoolExecutor

    res = PairResult()

    def run(item):
        i, c = item
        if progress_every and i and i % progress_every == 0:
            print(f"[judge] {system} vs {reference}: {i}/{len(contexts)} contexts", flush=True)
        return _judge_context(c, system, reference, ask)

    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            outs = list(ex.map(run, enumerate(contexts)))
    else:
        outs = [run(x) for x in enumerate(contexts)]
    for o in outs:
        res.records += o["records"]
        res.failures += o["failures"]
        res.missing_categories += o["missing"]
        res.overall.append(float(np.mean(o["sys_wins"])) if o["ok"] else None)
        if o["ok"]:
            for k in J.CATEGORIES:
                if o["cat_wins"][k]:
                    res.categories[k].append(float(np.mean(o["cat_wins"][k])))
    return res


def win_rate(values: Sequence[float | None], n_boot: int, seed: int) -> dict[str, float]:
    v = np.array([x for x in values if x is not None], float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0,
                "order_consistent": float("nan")}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return {"mean": float(v.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": float(len(v)),
            "order_consistent": float(np.mean(v != 0.5))}
