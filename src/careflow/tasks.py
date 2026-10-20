"""Task registry: `experiment.task` in a YAML config selects the function to run."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow

from careflow.cache import LLMCache

TaskFn = Callable[[dict[str, Any], str], None]
TASKS: dict[str, TaskFn] = {}


def register(name: str) -> Callable[[TaskFn], TaskFn]:
    def deco(fn: TaskFn) -> TaskFn:
        if name in TASKS:
            raise ValueError(f"Task {name!r} already registered")
        TASKS[name] = fn
        return fn

    return deco


@register("smoke")
def smoke(cfg: dict[str, Any], device: str) -> None:
    """Infrastructure check only; produces no research results."""
    import numpy as np

    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    request = {"model": "smoke", "messages": [{"role": "user", "content": "ping"}]}
    if cache.get(request) is None:
        cache.put(request, {"content": "pong"})
    cache.close()
    mlflow.log_metric("smoke.rng_draw", float(np.random.rand()))


@register("esconv_dynamics")
def esconv_dynamics(cfg: dict[str, Any], device: str) -> None:
    """Seeker emotion trajectories in ESConv + validation of the proxy against self-reports."""
    import json

    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, spearmanr

    from careflow.data.prepare import read_jsonl
    from careflow.eval.dynamics import summarize, trajectory, transition_matrix
    from careflow.eval.emotion import EmotionClassifier

    dcfg, mcfg, fcfg = cfg["data"], cfg["emotion_model"], cfg["flags"]
    dialogues = [
        d for s in dcfg["splits"] for d in read_jsonl(Path(dcfg["dataset_dir"]) / f"{s}.jsonl")
    ]
    seeker = [(d.dialogue_id, t.idx, t.text) for d in dialogues for t in d.turns if t.role == "seeker"]

    clf = EmotionClassifier(mcfg["id"], mcfg["revision"], device, mcfg["max_length"])
    probs = clf.predict_proba([x[2] for x in seeker], mcfg["batch_size"]).numpy()
    unknown = set(clf.labels) - set(cfg["polarity"])
    if unknown:
        raise ValueError(f"Classifier labels without polarity mapping: {sorted(unknown)}")

    turns = pd.DataFrame(seeker, columns=["dialogue_id", "turn_idx", "text"])
    turns["emotion"] = [clf.labels[i] for i in probs.argmax(1)]
    turns["confidence"] = probs.max(1)
    turns["polarity"] = turns["emotion"].map(cfg["polarity"])

    by_dlg = {k: g.sort_values("turn_idx") for k, g in turns.groupby("dialogue_id")}
    trajs = {k: trajectory(g["polarity"].tolist(), fcfg["fast_k"], fcfg["stuck_k"]) for k, g in by_dlg.items()}
    low = {
        d.dialogue_id
        for d in dialogues
        if any(t.feedback is not None and t.feedback <= dcfg["low_feedback_max"] for t in d.turns)
    }

    metrics: dict[str, float] = {f"all.{k}": v for k, v in summarize(list(trajs.values())).items()}
    for name, ids in (("low_feedback", low), ("other", set(trajs) - low)):
        metrics.update({f"{name}.{k}": v for k, v in summarize([trajs[i] for i in ids]).items()})

    # Proxy validity: classifier-based polarity change vs self-reported intensity drop.
    pairs = [
        (trajs[d.dialogue_id].end - trajs[d.dialogue_id].start, d.initial_intensity - d.final_intensity)
        for d in dialogues
        if d.final_intensity is not None and d.initial_intensity is not None
    ]
    rho, p = spearmanr(*zip(*pairs, strict=True))
    metrics.update({"proxy.spearman_rho": float(rho), "proxy.spearman_p": float(p), "proxy.n": float(len(pairs))})

    # Do low-rated dialogues stay negative longer? (two-sided Mann-Whitney U)
    neg_low = [trajs[i].longest_neg_run for i in low]
    neg_other = [trajs[i].longest_neg_run for i in set(trajs) - low]
    u = mannwhitneyu(neg_low, neg_other, alternative="two-sided")
    metrics.update({"contrast.neg_run_mwu_p": float(u.pvalue)})

    metrics["turns.n_seeker"] = float(len(turns))
    metrics["turns.mean_confidence"] = float(turns["confidence"].mean())
    for lab, share in turns["emotion"].value_counts(normalize=True).items():
        metrics[f"turns.share.{lab}"] = float(share)
    mlflow.log_metrics(metrics)

    tm = transition_matrix([g["emotion"].tolist() for g in by_dlg.values()], clf.labels)
    mlflow.log_text(
        json.dumps({"labels": clf.labels, "matrix": np.round(tm, 4).tolist()}, indent=2),
        "transition_matrix.json",
    )
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    turns.drop(columns="text").to_csv(out / "seeker_emotions.csv", index=False)
    mlflow.log_artifact(str(out / "seeker_emotions.csv"))


@register("judge_validation")
def judge_validation(cfg: dict[str, Any], device: str) -> None:
    """Local judge vs HEART's published ranking + position bias + parse failures."""
    import json
    import time

    import numpy as np

    from careflow.eval import bradley_terry
    from careflow.eval import judge as J
    from careflow.eval.judge_validation import (
        HUMAN,
        bootstrap_tau,
        outcomes,
        players_for_row,
        position_consistency,
        rank_agreement,
        sample_pairs,
    )
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    rows = [r for r in json.loads(Path(cfg["data"]["completions"]).read_text()) if "prev_speakers" in r]
    ref = cfg["reference_elo"]
    players = [players_for_row(r, ref, cfg["sample"]["include_human"]) for r in rows]
    s = cfg["sample"]
    pairs = sample_pairs([list(p) for p in players], s["n_contexts"], s["pairs_per_context"], s["swap_fraction"], cfg["seed"])

    mcfg = load_config(cfg["models_config"])
    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    pool = ModelPool(mcfg["models"], mcfg["pool"]["max_loaded"], cache)
    judge = pool.get(cfg["judge_role"])
    mlflow.set_tag("judge_model", judge.name)

    gen_cfg = cfg["generation"]
    winners: list[str | None] = []
    records, latencies, tokens, failures = [], [], [], 0
    t0 = time.perf_counter()
    for k, p in enumerate(pairs):
        row, texts = rows[p.context], players[p.context]
        prompt = J.build_prompt(row["dialogue_history"], row["prev_speakers"], texts[p.a], texts[p.b], row.get("emotion_type"))
        gen = judge.chat(J.messages(prompt), gen_cfg["max_tokens"], gen_cfg["temperature"], seed=cfg["seed"])
        if not gen.cached:
            latencies.append(gen.latency_s)
        tokens.append(gen.completion_tokens)
        try:
            v = J.parse(gen.text)
            win = p.a if v["overall_eq"].winner == J.ID_A else p.b
            cats = {c: (p.a if x.winner == J.ID_A else p.b, x.strength) for c, x in v.items()}
            err = None
        except J.JudgeParseError as e:
            win, cats, err = None, None, str(e)
            failures += 1
        winners.append(win)
        records.append({"pair": k, "context": p.context, "a": p.a, "b": p.b, "swapped_of": p.swapped_of,
                        "winner": win, "categories": cats, "error": err, "raw": gen.text,
                        "completion_tokens": gen.completion_tokens, "finish_reason": gen.finish_reason})
        if (k + 1) % 10 == 0:
            print(f"[judge_validation] {k + 1}/{len(pairs)} pairs, {failures} parse failures, "
                  f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
    pool.close()
    cache.close()

    res = outcomes(pairs, winners)
    scores = bradley_terry.fit(res)
    ref_elo = {m: v["elo"] for m, v in ref.items()}
    agree = rank_agreement(scores, ref_elo)
    lo, hi = bootstrap_tau(pairs, winners, ref_elo, cfg["bootstrap"]["n"], cfg["seed"])
    human_games = [(a, b, w) for a, b, w in res if HUMAN in (a, b)]
    human_win = np.mean([w if a == HUMAN else 1 - w for a, b, w in human_games]) if human_games else float("nan")

    metrics = {
        **{f"agreement.{k}": v for k, v in agree.items()},
        "agreement.kendall_tau_ci_low": lo,
        "agreement.kendall_tau_ci_high": hi,
        "judge.parse_failure_rate": failures / len(pairs),
        "judge.position_consistency": position_consistency(pairs, winners),
        "judge.first_position_win_rate": float(np.mean([w for _, _, w in res])),
        "judge.latency_s_mean": float(np.mean(latencies)) if latencies else float("nan"),
        "judge.completion_tokens_mean": float(np.mean(tokens)),
        "judge.truncated_rate": float(np.mean([r["finish_reason"] == "length" for r in records])),
        "human.win_rate": float(human_win),
        "sample.n_pairs": float(len(pairs)),
        "sample.n_contexts": float(len({p.context for p in pairs})),
    }
    metrics.update({f"bt.{m}": v for m, v in scores.items()})
    mlflow.log_metrics(metrics)
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "judgments.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mlflow.log_artifact(str(out / "judgments.jsonl"))
    for k in ("agreement.kendall_tau", "agreement.kendall_tau_ci_low", "agreement.kendall_tau_ci_high",
              "judge.parse_failure_rate", "judge.position_consistency", "human.win_rate"):
        print(f"{k}: {metrics[k]:.3f}")
