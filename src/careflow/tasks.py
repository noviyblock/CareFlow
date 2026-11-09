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
            cats = {c: (p.a if x.winner == J.ID_A else p.b, x.strength) if x else None for c, x in v.items()}
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


@register("proxy_validity")
def proxy_validity(cfg: dict[str, Any], device: str) -> None:
    """Spearman validity (with bootstrap CIs) of seeker-state signals vs ESConv self-reported intensity."""
    import json
    import random
    import time

    import numpy as np

    from careflow.data.prepare import read_jsonl
    from careflow.eval.emotion import EmotionClassifier
    from careflow.eval.state_signals import (
        label_mass,
        llm_intensity,
        paired_rho_diff,
        spearman_ci,
        windows,
    )

    dcfg, wcfg, nb = cfg["data"], cfg["window"], cfg["bootstrap"]["n"]
    dialogues = [d for s in dcfg["splits"] for d in read_jsonl(Path(dcfg["dataset_dir"]) / f"{s}.jsonl")]
    items = []  # (dialogue, seeker_texts, start_idx, end_idx)
    for d in dialogues:
        texts = [t.text for t in d.turns if t.role == "seeker"]
        w = windows(texts, wcfg["k"], wcfg["min_words"])
        if w is not None:
            items.append((d, texts, *w))
    mlflow.log_metrics({"sample.n_dialogues": float(len(dialogues)), "sample.n_windowed": float(len(items))})

    rng = random.Random(cfg["seed"])
    llm_cfg = cfg["llm"]
    subset = set(rng.sample(range(len(items)), min(llm_cfg["max_dialogues"], len(items))))

    def validity(name: str, start: list[float], end: list[float], idx: list[int]) -> dict[str, float]:
        out = {}
        init = [items[i][0].initial_intensity for i in idx]
        out.update({f"{name}.initial.{k}": v for k, v in spearman_ci([start[i] for i in idx], init, nb, cfg["seed"]).items()})
        fin = [i for i in idx if items[i][0].final_intensity is not None]
        out.update({f"{name}.final.{k}": v for k, v in spearman_ci(
            [end[i] for i in fin], [items[i][0].final_intensity for i in fin], nb, cfg["seed"]).items()})
        out.update({f"{name}.change.{k}": v for k, v in spearman_ci(
            [start[i] - end[i] for i in fin],
            [items[i][0].initial_intensity - items[i][0].final_intensity for i in fin], nb, cfg["seed"]).items()})
        return out

    metrics: dict[str, float] = {}
    signals: dict[str, tuple[list[float], list[float]]] = {}
    all_idx, sub_idx = list(range(len(items))), sorted(subset)
    for name, c in cfg["classifiers"].items():
        clf = EmotionClassifier(c["id"], c["revision"], device, c["max_length"], c["multi_label"])
        flat = [t for _, texts, _, _ in items for t in texts]
        probs = clf.predict_proba(flat, c["batch_size"]).numpy()
        neg = label_mass(probs, clf.labels, c["negative"])
        start, end, pos = [], [], 0
        for _, texts, s_idx, e_idx in items:
            scores = neg[pos : pos + len(texts)]
            pos += len(texts)
            start.append(float(np.mean(scores[s_idx])))
            end.append(float(np.mean(scores[e_idx])))
        metrics.update(validity(f"{name}", start, end, all_idx))
        metrics.update(validity(f"{name}_subset", start, end, sub_idx))
        signals[name] = (start, end)
        del clf

    if llm_cfg["enabled"]:
        from careflow.llm import ModelPool
        from careflow.utils.config import load_config

        mcfg = load_config(llm_cfg["models_config"])
        cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
        pool = ModelPool(mcfg["models"], 1, cache)
        model = pool.get(llm_cfg["role"])
        mlflow.set_tag("llm_rater", model.name)

        def chat(prompt: str) -> str:
            return model.chat([{"role": "user", "content": prompt}], llm_cfg["max_tokens"], 0.0, cfg["seed"]).text

        start = [float("nan")] * len(items)
        end = [float("nan")] * len(items)
        failures, t0 = 0, time.perf_counter()
        for n, i in enumerate(sub_idx):
            _, texts, s_idx, e_idx = items[i]
            a = llm_intensity(chat, [texts[j] for j in s_idx])
            b = llm_intensity(chat, [texts[j] for j in e_idx])
            if a is None or b is None:
                failures += 1
            else:
                start[i], end[i] = float(a), float(b)
            if (n + 1) % 25 == 0:
                print(f"[proxy_validity] llm {n + 1}/{len(sub_idx)}, {failures} failures, {time.perf_counter() - t0:.0f}s", flush=True)
        pool.close()
        cache.close()
        ok = [i for i in sub_idx if not np.isnan(start[i])]
        metrics.update(validity("llm", start, end, ok))
        metrics["llm.parse_failure_rate"] = failures / max(len(sub_idx), 1)
        signals["llm"] = (start, end)
        # Paired comparison on identical dialogues: is the LLM signal more valid than each classifier?
        fin = [i for i in ok if items[i][0].final_intensity is not None]
        for name in cfg["classifiers"]:
            cs, ce = signals[name]
            d = paired_rho_diff([start[i] for i in ok], [cs[i] for i in ok],
                                [items[i][0].initial_intensity for i in ok], nb, cfg["seed"])
            metrics.update({f"paired.llm_minus_{name}.initial.{k}": v for k, v in d.items()})
            d = paired_rho_diff([start[i] - end[i] for i in fin], [cs[i] - ce[i] for i in fin],
                                [items[i][0].initial_intensity - items[i][0].final_intensity for i in fin],
                                nb, cfg["seed"])
            metrics.update({f"paired.llm_minus_{name}.change.{k}": v for k, v in d.items()})

    mlflow.log_metrics(metrics)
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mlflow.log_artifact(str(out / "metrics.json"))
    with (out / "per_dialogue.jsonl").open("w") as f:
        for i, (d, _, _, _) in enumerate(items):
            f.write(json.dumps({"dialogue_id": d.dialogue_id, "initial": d.initial_intensity, "final": d.final_intensity,
                                **{f"{n}.start": v[0][i] for n, v in signals.items()},
                                **{f"{n}.end": v[1][i] for n, v in signals.items()}}) + "\n")
    mlflow.log_artifact(str(out / "per_dialogue.jsonl"))
    for k in sorted(metrics):
        if k.endswith((".rho", "ci_low", "ci_high", ".n", ".diff")):
            print(f"{k:45s} {metrics[k]:.3f}")


@register("baseline_single_agent")
def baseline_single_agent(cfg: dict[str, Any], device: str) -> None:
    """Single-agent generators vs the human ESConv supporter turn: pairwise judge (both orders) + latency."""
    import json
    import random
    from itertools import combinations

    import numpy as np

    from careflow.agents import single_agent
    from careflow.data.prepare import read_jsonl
    from careflow.eval import judge as J
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    dcfg = cfg["data"]
    rng = random.Random(cfg["seed"])
    contexts = []
    for d in read_jsonl(Path(dcfg["dataset_dir"]) / f"{dcfg['split']}.jsonl"):
        targets = [t.idx for t in d.turns if t.role == "supporter" and t.idx >= dcfg["min_history"]]
        if targets:
            j = rng.choice(targets)
            contexts.append({"dialogue_id": d.dialogue_id, "emotion": d.emotion,
                             "history": [t.text for t in d.turns[:j]], "speakers": [t.role for t in d.turns[:j]],
                             "players": {"human": d.turns[j].text}})
    contexts = rng.sample(contexts, min(dcfg["n_contexts"], len(contexts)))

    mcfg = load_config(cfg["models_config"])
    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    pool = ModelPool(mcfg["models"], 1, cache)  # one large model in memory at a time
    gcfg, warmup = cfg["generation"], cfg["compute"]["latency"]["warmup_requests"]
    metrics: dict[str, float] = {"sample.n_contexts": float(len(contexts))}

    for role in cfg["generators"]:
        model = pool.get(role)
        mlflow.set_tag(f"generator.{role}", model.name)
        lat, toks = [], []
        for k, c in enumerate(contexts):
            g = model.chat(single_agent.messages(c["history"], c["speakers"]), gcfg["max_tokens"], gcfg["temperature"], cfg["seed"])
            c["players"][role] = single_agent.clean(g.text)
            toks.append(g.completion_tokens)
            if not g.cached and k >= warmup:
                lat.append(g.latency_s)
        metrics[f"latency.{role}.mean_s"] = float(np.mean(lat)) if lat else float("nan")
        metrics[f"latency.{role}.p90_s"] = float(np.percentile(lat, 90)) if lat else float("nan")
        metrics[f"tokens.{role}.completion_mean"] = float(np.mean(toks))
        metrics[f"words.{role}.mean"] = float(np.mean([len(c["players"][role].split()) for c in contexts]))
    metrics["words.human.mean"] = float(np.mean([len(c["players"]["human"].split()) for c in contexts]))

    judge = pool.get(cfg["judge_role"])
    mlflow.set_tag("judge_model", judge.name)
    players = ["human", *cfg["generators"]]
    records, failures, calls = [], 0, 0
    # score[(a, b)][ctx] = share of the two presentation orders that a won (0, 0.5, 1); None if any order failed
    score: dict[tuple[str, str], list[float | None]] = {}
    cat_score: dict[tuple[str, str, str], list[float]] = {}
    for a, b in combinations(players, 2):
        score[(a, b)] = []
        for c in contexts:
            wins, cats, ok = [], {cat: [] for cat in J.CATEGORIES}, True
            for first, second in ((a, b), (b, a)):
                prompt = J.build_prompt(c["history"], c["speakers"], c["players"][first], c["players"][second], c["emotion"])
                g = judge.chat(J.messages(prompt), cfg["judge"]["max_tokens"], 0.0, cfg["seed"])
                calls += 1
                try:
                    v = J.parse(g.text)
                except J.JudgeParseError as e:
                    failures, ok = failures + 1, False
                    records.append({"ctx": c["dialogue_id"], "first": first, "second": second, "error": str(e)})
                    continue
                to_name = {J.ID_A: first, J.ID_B: second}
                wins.append(to_name[v["overall_eq"].winner] == a)
                for cat in J.CATEGORIES:
                    if v[cat] is not None:
                        cats[cat].append(to_name[v[cat].winner] == a)
                records.append({"ctx": c["dialogue_id"], "first": first, "second": second,
                                "winner": to_name[v["overall_eq"].winner]})
            score[(a, b)].append(float(np.mean(wins)) if ok else None)
            if ok:
                for cat in J.CATEGORIES:
                    if cats[cat]:
                        cat_score.setdefault((a, b, cat), []).append(float(np.mean(cats[cat])))
        print(f"[baseline] judged {a} vs {b}: {calls} calls, {failures} failures", flush=True)
    pool.close()
    cache.close()

    boot_rng = np.random.default_rng(cfg["seed"])
    for (a, b), vals in score.items():
        v = np.array([x for x in vals if x is not None])
        boots = [boot_rng.choice(v, len(v)).mean() for _ in range(cfg["bootstrap"]["n"])]
        key = f"winrate.{b}_vs_{a}"  # report from the perspective of b (models listed after human)
        metrics[key] = float(1 - v.mean())
        metrics[f"{key}.ci_low"] = float(1 - np.percentile(boots, 97.5))
        metrics[f"{key}.ci_high"] = float(1 - np.percentile(boots, 2.5))
        metrics[f"{key}.order_consistent_share"] = float(np.mean(v != 0.5))
        metrics[f"{key}.n"] = float(len(v))
        for cat in J.CATEGORIES:
            metrics[f"{key}.{cat}"] = float(1 - np.mean(cat_score[(a, b, cat)]))
    metrics["judge.parse_failure_rate"] = failures / max(calls, 1)
    mlflow.log_metrics(metrics)

    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "responses.jsonl").open("w") as f:
        for c in contexts:
            f.write(json.dumps({"dialogue_id": c["dialogue_id"], "players": c["players"]}, ensure_ascii=False) + "\n")
    (out / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for f in ("responses.jsonl", "judgments.jsonl", "metrics.json"):
        mlflow.log_artifact(str(out / f))
    for k in sorted(metrics):
        if not any(cat in k for cat in J.CATEGORIES):
            print(f"{k:55s} {metrics[k]:.3f}")


def _locked(fn):
    """Serialise a callable shared by worker threads (the torch encoder used by rerank)."""
    import threading

    lock = threading.Lock()

    def wrapped(*a, **k):
        with lock:
            return fn(*a, **k)
    return wrapped


def _respond_all(system, contexts, workers: int):
    """Run a system on all contexts, concurrently when workers > 1 (systems are stateless across contexts); output
    order = context order, so traces and the cache are identical to a sequential run."""
    if workers <= 1:
        return [system.respond(c) for c in contexts]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(system.respond, contexts))


@register("systems_eval")
def systems_eval(cfg: dict[str, Any], device: str) -> None:
    """Block B: generate with every configured system, judge each vs the human reply, report quality x latency."""
    import json

    import numpy as np
    import pandas as pd

    from careflow.agents.controls import make_control
    from careflow.agents.systems import SYSTEMS, set_reply_length
    from careflow.data.contexts import sample_contexts
    from careflow.data.prepare import read_jsonl
    from careflow.eval import judge as J
    from careflow.eval.pairwise import judge_vs_reference, win_rate
    from careflow.eval.report import pareto_front, quality_latency_plot
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    set_reply_length(cfg.get("reply_length_words"))  # None = default prompts (cache-compatible)
    workers = int(cfg["compute"].get("concurrency", 1))  # >1 only for a batching server (Colab vLLM)
    dcfg = cfg["data"]
    contexts = sample_contexts(read_jsonl(Path(dcfg["dataset_dir"]) / f"{dcfg['split']}.jsonl"),
                               dcfg["n_contexts"], dcfg["min_history"], cfg["seed"])
    out_dir = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg = load_config(cfg["models_config"])
    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    pool = ModelPool(mcfg["models"], 1, cache)
    warmup = cfg["compute"]["latency"]["warmup_requests"]

    embedder = None
    rows: dict[str, dict[str, float]] = {}
    traces = []
    # Group systems by model role so each model is loaded once.
    for role in dict.fromkeys(s["role"] for s in cfg["systems"].values()):
        model = pool.get(role)
        mlflow.set_tag(f"model.{role}", model.name)
        for name, scfg in cfg["systems"].items():
            if scfg["role"] != role:
                continue
            kwargs = dict(scfg.get("kwargs") or {})
            if scfg["type"] == "rerank":
                if embedder is None:
                    from sentence_transformers import SentenceTransformer

                    e = cfg["embedding_model"]
                    embedder = _locked(SentenceTransformer(e["id"], revision=e["revision"], device=device).encode)
                kwargs["embed"] = embedder
            system = SYSTEMS[scfg["type"]](name, model, cfg["seed"], **kwargs)
            lat, calls, comp, prompt, cached = [], [], [], [], 0
            outputs = _respond_all(system, contexts, workers)
            for k, (c, o) in enumerate(zip(contexts, outputs, strict=True)):
                c.responses[name] = o.text
                cached += o.cached
                if k >= warmup:
                    lat.append(o.latency_s)
                calls.append(len(o.calls))
                comp.append(o.completion_tokens)
                prompt.append(o.prompt_tokens)
                traces.append({"system": name, "ctx": c.dialogue_id, "text": o.text, "trace": o.trace})
            rows[name] = {"latency_s": float(np.mean(lat)), "latency_p90_s": float(np.percentile(lat, 90)),
                          "llm_calls": float(np.mean(calls)), "completion_tokens": float(np.mean(comp)),
                          "prompt_tokens": float(np.mean(prompt)), "cached_share": cached / len(contexts),
                          "words": float(np.mean([len(c.responses[name].split()) for c in contexts]))}
            print(f"[systems_eval] generated {name}: {rows[name]['latency_s']:.2f}s/reply, "
                  f"{rows[name]['llm_calls']:.1f} calls", flush=True)

    # positive controls derived from an already generated system (no LLM calls, zero cost)
    for name, ccfg in (cfg.get("controls") or {}).items():
        texts = make_control(ccfg["kind"], [c.responses[ccfg["source"]] for c in contexts], cfg["seed"],
                             ccfg.get("text"))
        for c, t in zip(contexts, texts, strict=True):
            c.responses[name] = t
            traces.append({"system": name, "ctx": c.dialogue_id, "text": t, "trace": [{"control": ccfg["kind"]}]})
        rows[name] = {"latency_s": 0.0, "latency_p90_s": 0.0, "llm_calls": 0.0, "completion_tokens": 0.0,
                      "prompt_tokens": 0.0, "cached_share": 1.0,
                      "words": float(np.mean([len(t.split()) for t in texts]))}
    names = [*cfg["systems"], *(cfg.get("controls") or {})]
    if cfg.get("stop_after_generation"):
        # phase 1 of a two-server run (Colab: generator and judge do not fit on one GPU together). Generations are
        # in the LLM cache; rerunning the same config without this flag replays them and only calls the judge.
        with (out_dir / "traces.jsonl").open("w") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(out_dir / "traces.jsonl"))
        mlflow.set_tag("phase", "generation_only")
        print(f"[systems_eval] generation done for {len(names)} systems; judge skipped (stop_after_generation)")
        pool.close()
        cache.close()
        return

    judge = pool.get(cfg["judge_role"])
    mlflow.set_tag("judge_model", judge.name)

    def ask(prompt: str) -> str:
        return judge.chat(J.messages(prompt), cfg["judge"]["max_tokens"], 0.0, cfg["seed"]).text

    records = []
    references = cfg.get("references", ["human"])
    plot_ref = cfg.get("plot_reference", references[-1])
    for ref in references:
        for name in names:
            if name == ref:  # a system vs itself is 0.5 by definition
                rows[name].update({f"vs_{ref}.winrate": 0.5, f"vs_{ref}.ci_low": 0.5, f"vs_{ref}.ci_high": 0.5})
                continue
            res = judge_vs_reference(contexts, name, ref, ask, workers=workers)
            wr = win_rate(res.overall, cfg["bootstrap"]["n"], cfg["seed"])
            rows[name].update({f"vs_{ref}.winrate": wr["mean"], f"vs_{ref}.ci_low": wr["ci_low"],
                               f"vs_{ref}.ci_high": wr["ci_high"], f"vs_{ref}.n_judged": wr["n"],
                               f"vs_{ref}.order_consistent": wr["order_consistent"],
                               f"vs_{ref}.judge_failures": float(res.failures),
                               f"vs_{ref}.judge_missing_categories": float(res.missing_categories)})
            for cat in J.CATEGORIES:
                vals = res.categories[cat]
                rows[name][f"vs_{ref}.cat.{cat}"] = float(np.mean(vals)) if vals else float("nan")
            records += [{"system": name, "reference": ref, **r} for r in res.records]
            print(f"[systems_eval] judged {name} vs {ref}: win-rate {wr['mean']:.3f} "
                  f"[{wr['ci_low']:.3f}, {wr['ci_high']:.3f}]", flush=True)
    for row in rows.values():  # columns used by the plot / Pareto front
        row.update({k: row[f"vs_{plot_ref}.{k}"] for k in ("winrate", "ci_low", "ci_high")})
    pool.close()
    cache.close()

    df = pd.DataFrame([{"system": n, **v} for n, v in rows.items()])
    df["pareto"] = pareto_front(df, "latency_s", "winrate")
    for _, r in df.iterrows():
        mlflow.log_metrics({f"{r['system']}.{k}": float(r[k]) for k in df.columns if k != "system"})
    df.to_csv(out_dir / "results.csv", index=False)
    quality_latency_plot(df, out_dir / "quality_vs_latency.png", reference=plot_ref)
    with (out_dir / "traces.jsonl").open("w") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (out_dir / "judgments.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for fname in ("results.csv", "quality_vs_latency.png", "traces.jsonl", "judgments.jsonl"):
        mlflow.log_artifact(str(out_dir / fname))
    cols = ["system", *[f"vs_{r}.winrate" for r in references], "ci_low", "ci_high", "latency_s", "llm_calls",
            "completion_tokens", "words", "pareto"]
    print(df[cols].sort_values("latency_s").to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@register("oracle_gate")
def oracle_gate(cfg: dict[str, Any], device: str) -> None:
    """Upper bound on routing over block-B systems: cross-fitted oracle, budget frontier, bootstrap CIs."""
    import json

    import numpy as np
    import pandas as pd

    from careflow.eval.report import oracle_frontier_plot
    from careflow.routers.oracle import (
        Scores,
        budget_frontier,
        fixed,
        oracle,
        scores_from_judgments,
    )

    src = Path(cfg["source_dir"])
    records = [json.loads(line) for line in (src / "judgments.jsonl").open()]
    res = pd.read_csv(src / "results.csv").set_index("system")
    cost = res["latency_s"].to_dict()
    scores = scores_from_judgments(records, cfg["reference"])
    mlflow.log_metric("sample.n_contexts", len(scores.contexts))
    rng = np.random.default_rng(cfg["seed"])

    def boot(fn) -> tuple[float, float]:
        vals = []
        for _ in range(cfg["bootstrap"]["n"]):
            idx = rng.choice(len(scores.contexts), len(scores.contexts))
            sub = Scores(scores.by_order, [scores.contexts[i] for i in idx])
            vals.append(fn(sub))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    metrics: dict[str, float] = {}
    rows = []
    for name, opts in cfg["option_sets"].items():
        for cf in (True, False):
            r = oracle(scores, opts, cost, cross_fit=cf)
            tag = f"oracle.{name}.{'crossfit' if cf else 'naive'}"
            metrics[f"{tag}.quality"], metrics[f"{tag}.latency_s"] = r.quality, r.cost
            if cf:
                lo, hi = boot(lambda s, o=opts: oracle(s, o, cost).quality)
                metrics[f"{tag}.quality_ci_low"], metrics[f"{tag}.quality_ci_high"] = lo, hi
                metrics.update({f"{tag}.share.{k}": v for k, v in r.share.items()})
                rows.append({"route": f"oracle ({name})", "quality": r.quality, "ci_low": lo, "ci_high": hi,
                             "latency_s": r.cost, **{f"share.{k}": v for k, v in r.share.items()}})
    for s in cfg["option_sets"][cfg.get("frontier_set", "all")]:
        f = fixed(scores, s, cost)
        lo, hi = boot(lambda sub, s=s: fixed(sub, s, cost).quality)
        rows.append({"route": s, "quality": f.quality, "ci_low": lo, "ci_high": hi, "latency_s": f.cost})
    best_fixed = max((r for r in rows if not r["route"].startswith("oracle")), key=lambda r: r["quality"])
    metrics["best_fixed.quality"], metrics["best_fixed.latency_s"] = best_fixed["quality"], best_fixed["latency_s"]
    mlflow.set_tag("best_fixed", best_fixed["route"])
    for name in cfg["option_sets"]:
        metrics[f"headroom.{name}.vs_best_fixed"] = metrics[f"oracle.{name}.crossfit.quality"] - best_fixed["quality"]

    budgets = np.linspace(cost[cfg["cheap"]], max(cost[s] for s in cfg["option_sets"][cfg.get("frontier_set", "all")]), cfg["budget_points"])
    frontier = budget_frontier(scores, cfg["cheap"], cfg["option_sets"][cfg.get("frontier_set", "all")], cost, budgets)
    # smallest budget at which the frontier reaches the best fixed system's quality
    reach = [f.cost for f in frontier if f.quality >= best_fixed["quality"]]
    metrics["frontier.latency_to_match_best_fixed"] = min(reach) if reach else float("nan")
    mlflow.log_metrics(metrics)

    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "routes.csv", index=False)
    pd.DataFrame([{"quality": f.quality, "latency_s": f.cost} for f in frontier]).to_csv(out / "frontier.csv", index=False)
    oracle_frontier_plot(table, frontier, out / "oracle_frontier.png", reference=cfg["reference"])
    for fname in ("routes.csv", "frontier.csv", "oracle_frontier.png"):
        mlflow.log_artifact(str(out / fname))
    print(table[["route", "quality", "ci_low", "ci_high", "latency_s"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for k in sorted(metrics):
        if "share" not in k:
            print(f"{k:55s} {metrics[k]:.3f}")


def _head_to_head(sources: list[str], ref: str, exclude: list[str] = ()):
    """All saved judgments of systems vs `ref` (both orders), with the word-length difference of the replies."""
    import json

    from careflow.eval.length_control import Judgment

    text: dict[tuple[str, str], str] = {}
    judgments, seen = [], set()
    for src in map(Path, sources):
        for line in (src / "traces.jsonl").open():
            t = json.loads(line)
            text[(t["ctx"], t["system"])] = t["text"]
        for line in (src / "judgments.jsonl").open():
            r = json.loads(line)
            key = (r.get("system"), r.get("ctx"), r.get("first"))
            if r.get("reference") != ref or "winner" not in r or key in seen or r.get("system") in exclude:
                continue  # the same comparison may appear in several sources (cached); count it once
            seen.add(key)
            s = r["system"]
            dlen = len(text[(r["ctx"], s)].split()) - len(text[(r["ctx"], ref)].split())
            judgments.append(Judgment(r["ctx"], s, int(r["winner"] == s), dlen, int(r["first"] == s)))
    return judgments


@register("length_control")
def length_control(cfg: dict[str, Any], device: str) -> None:
    """Raw vs length-controlled win-rates vs the reference over saved judgments and traces."""
    import numpy as np
    import pandas as pd

    from careflow.eval.length_control import bootstrap_lc, fit_lc

    judgments = _head_to_head(cfg["sources"], cfg["reference"])
    systems = sorted({j.system for j in judgments})
    lc, beta = fit_lc(judgments, systems, cfg["sigma_words"])
    ci = bootstrap_lc(judgments, systems, cfg["sigma_words"], cfg["bootstrap"]["n"], cfg["seed"])
    rows = []
    for s in systems:
        js = [j for j in judgments if j.system == s]
        rows.append({"system": s, "raw_winrate": float(np.mean([j.won for j in js])),
                     "lc_winrate": lc[s], "lc_ci_low": ci[s][0], "lc_ci_high": ci[s][1],
                     "mean_dlen_words": float(np.mean([j.dlen for j in js])), "n_judgments": len(js)})
    df = pd.DataFrame(rows).sort_values("lc_winrate", ascending=False)
    mlflow.log_metric("length_beta", beta)
    for _, r in df.iterrows():
        mlflow.log_metrics({f"{r['system']}.{k}": float(r[k]) for k in df.columns if k != "system"})
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lc_winrates.csv", index=False)
    mlflow.log_artifact(str(out / "lc_winrates.csv"))
    print(f"length coefficient beta (tanh(dlen/{cfg['sigma_words']})): {beta:.3f}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@register("router_train")
def router_train(cfg: dict[str, Any], device: str) -> None:
    """Learned routers vs fixed paths, random and oracle; raw and length-adjusted quality on held-out dialogues."""
    import json

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from careflow.data.contexts import sample_contexts
    from careflow.data.prepare import read_jsonl
    from careflow.eval.length_control import Judgment, length_adjusted
    from careflow.routers.features import LLMHiddenEncoder
    from careflow.routers.learned import ClusterRouter, UtilityRouter, evaluate, random_policy
    from careflow.routers.oracle import oracle, scores_from_judgments

    ref, paths = cfg["reference"], cfg["route_paths"]
    emb_cfg = cfg["embedding_model"]
    embedder = SentenceTransformer(emb_cfg["id"], revision=emb_cfg["revision"], device=device)
    probe_cfg = cfg["llm_probe"]
    llm_enc = LLMHiddenEncoder(probe_cfg["id"], probe_cfg["revision"], probe_cfg["max_tokens"])

    def load(run_dir: str, split: str):
        run = Path(run_dir)
        texts = {}
        for line in (run / "traces.jsonl").open():
            t = json.loads(line)
            texts[(t["ctx"], t["system"])] = t["text"]
        recs = [json.loads(line) for line in (run / "judgments.jsonl").open()]
        js = []
        for r in recs:
            if r.get("reference") == ref and "winner" in r and r["system"] in paths:
                s = r["system"]
                dlen = len(texts[(r["ctx"], s)].split()) - len(texts[(r["ctx"], ref)].split())
                js.append(Judgment(r["ctx"], s, int(r["winner"] == s), dlen, int(r["first"] == s)))
        others = [p for p in paths if p != ref]
        raw = {}
        for j in js:
            raw.setdefault((j.ctx, j.system), []).append(j.won)
        adj = length_adjusted(js, others, cfg["sigma_words"])
        ctx_ok = sorted({c for c, _ in raw if all(len(raw.get((c, p), [])) == 2 for p in others)})
        sc_raw = {p: np.array([0.5 if p == ref else np.mean(raw[(c, p)]) for c in ctx_ok]) for p in paths}
        sc_adj = {p: np.array([0.5 if p == ref else adj[(c, p)] for c in ctx_ok]) for p in paths}
        contexts = {c.dialogue_id: c for c in sample_contexts(
            read_jsonl(Path(cfg["data"]["dataset_dir"]) / f"{split}.jsonl"), 10**6, cfg["data"]["min_history"], cfg["seed"])}
        k = cfg["features"]["last_seeker_turns"]
        last = []
        for c in ctx_ok:
            ctx = contexts[c]
            seeker = [t for t, s in zip(ctx.history, ctx.speakers, strict=True) if s == "seeker"]
            last.append(" ".join(seeker[-k:]))
        emb = embedder.encode(last, convert_to_numpy=True, normalize_embeddings=True)
        meta = np.array([[len(contexts[c].history) / 20, len(last[i].split()) / 50] for i, c in enumerate(ctx_ok)])
        cost = pd.read_csv(run / "results.csv").set_index("system")["latency_s"].to_dict()
        hid = llm_enc.encode([(contexts[c].history, contexts[c].speakers) for c in ctx_ok])
        return (np.hstack([emb, meta]), hid), sc_raw, sc_adj, cost, recs, ctx_ok

    (x_tr, h_tr), y_tr, _, _, _, _ = load(cfg["train_dir"], cfg.get("train_split", "valid"))
    (x_te, h_te), raw_te, adj_te, cost, recs_te, ctx_te = load(cfg["test_dir"], "test")
    mlflow.log_metrics({"n_train": len(x_tr), "n_test": len(x_te)})

    rows = []

    def add(name: str, choice: list[str], lam: float | None = None) -> None:
        q, c = evaluate(choice, raw_te, cost)
        qa, _ = evaluate(choice, adj_te, cost)
        shares = {p: choice.count(p) / len(choice) for p in paths}
        rows.append({"router": name, "lambda": lam, "quality": q, "quality_len_adj": qa, "latency_s": c,
                     **{f"share.{p}": v for p, v in shares.items()}})

    for p in paths:
        add(f"fixed:{p}", [p] * len(x_te))
    routers = {  # name -> (fitted router, test features)
        "utility": (UtilityRouter(paths, cfg["utility"]["ridge_alpha"]).fit(x_tr, y_tr), x_te),
        "cluster": (ClusterRouter(paths, cfg["cluster"]["k"], cfg["seed"]).fit(x_tr, y_tr), x_te),
        "llm_probe": (UtilityRouter(paths, probe_cfg["ridge_alpha"]).fit(h_tr, y_tr), h_te),
    }
    for lam in cfg["lambdas"]:
        for name, (r, x) in routers.items():
            add(name, r.route(x, cost, lam), lam)
    df = pd.DataFrame(rows)
    # budget-matched random baseline for every learned operating point
    rand = []
    for _, r in df[df["router"].isin(list(routers))].iterrows():
        shares = {p: r[f"share.{p}"] for p in paths}
        qs = [evaluate(random_policy(len(x_te), shares, cfg["seed"] + i), raw_te, cost)[0] for i in range(200)]
        rand.append(float(np.mean(qs)))
    df.loc[df["router"].isin(list(routers)), "random_same_shares"] = rand

    sc = scores_from_judgments([r for r in recs_te if r.get("ctx") in set(ctx_te)], ref)
    orc = oracle(sc, paths, cost)
    mlflow.log_metrics({"oracle.quality": orc.quality, "oracle.latency_s": orc.cost})

    # bootstrap CI for the best learned operating point (max quality) and its gain over the random baseline
    learned = df[df["router"].isin(list(routers))]
    best = learned.loc[learned["quality"].idxmax()]
    router, x_best = routers[best["router"]]
    choice = router.route(x_best, cost, best["lambda"])
    rng = np.random.default_rng(cfg["seed"])
    q_best = np.array([raw_te[p][i] for i, p in enumerate(choice)])
    q_9b = raw_te[ref]
    diffs = [float((q_best[i] - q_9b[i]).mean()) for i in (rng.integers(0, len(q_best), len(q_best))
                                                            for _ in range(cfg["bootstrap"]["n"]))]
    mlflow.log_metrics({"best.quality": float(best["quality"]), "best.quality_len_adj": float(best["quality_len_adj"]),
                        "best.latency_s": float(best["latency_s"]), "best.gain_vs_9b": float(q_best.mean() - 0.5),
                        "best.gain_vs_9b_ci_low": float(np.percentile(diffs, 2.5)),
                        "best.gain_vs_9b_ci_high": float(np.percentile(diffs, 97.5))})
    mlflow.set_tags({"best.router": best["router"], "best.lambda": str(best["lambda"])})

    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "routers.csv", index=False)
    mlflow.log_artifact(str(out / "routers.csv"))
    cols = ["router", "lambda", "quality", "quality_len_adj", "random_same_shares", "latency_s"] + [f"share.{p}" for p in paths]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"oracle (cross-fitted, test): quality {orc.quality:.3f} at {orc.cost:.2f}s")
    print(f"best learned: {best['router']} lam={best['lambda']} quality {best['quality']:.3f} "
          f"(len-adj {best['quality_len_adj']:.3f}), gain vs 9b {q_best.mean() - 0.5:+.3f} "
          f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]")


@register("sycophancy_synth")
def sycophancy_synth(cfg: dict[str, Any], device: str) -> None:
    """Generate labelled sycophancy scenarios; scenario-level train/test split."""
    import json
    import random

    from careflow.data.synthetic import DOMAINS, make_scenario, passes_check
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    mcfg = load_config(cfg["models_config"])
    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    pool = ModelPool(mcfg["models"], 1, cache)
    gen = pool.get(cfg["generator_role"])
    mlflow.set_tag("generator", gen.name)

    def ask(prompt: str, seed: int) -> str:
        return gen.chat([{"role": "user", "content": prompt}], cfg["max_tokens"], cfg["temperature"], seed).text

    def check(prompt: str, seed: int) -> str:  # deterministic label check
        return gen.chat([{"role": "user", "content": prompt}], 4, 0.0, seed).text

    import collections

    rejected: collections.Counter[str] = collections.Counter()
    scenarios, failures, k = [], 0, 0
    for domain in DOMAINS:
        for harmful in (True, False):
            for i in range(cfg["n_per_domain_and_label"]):
                k += 1
                try:
                    s = make_scenario(ask, domain, harmful, cfg["seed"] + k)
                except (ValueError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    failures += 1
                    continue
                ok, reason = passes_check(check, s, cfg["seed"])
                if not ok:
                    rejected[reason] += 1
                    continue
                s["id"] = f"syc-{k:04d}"
                scenarios.append(s)
            print(f"[sycophancy_synth] {domain} harmful={harmful}: {len(scenarios)} kept, {failures} malformed, "
                  f"rejected {dict(rejected)}", flush=True)
    pool.close()
    cache.close()

    rng = random.Random(cfg["seed"])
    ids = [s["id"] for s in scenarios]
    rng.shuffle(ids)
    test_ids = set(ids[: round(cfg["test_share"] * len(ids))])
    for s in scenarios:
        s["split"] = "test" if s["id"] in test_ids else "train"
    out = Path(cfg["out_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    mlflow.log_metrics({"n_scenarios": len(scenarios), "n_failed": failures,
                        "n_harmful": sum(s["harmful"] for s in scenarios), "n_test": len(test_ids),
                        "identical_reply_pairs": sum(s["replies"][0]["text"] == s["replies"][1]["text"]
                                                     for s in scenarios),
                        **{f"rejected.{r}": float(n) for r, n in rejected.items()}})
    mlflow.log_artifact(str(out))
    print(f"wrote {len(scenarios)} scenarios to {out} ({failures} failed)")


@register("sycophancy_eval")
def sycophancy_eval(cfg: dict[str, Any], device: str) -> None:
    """Detector benchmark on gold synthetic labels + per-system sycophancy rate on harmful scenarios."""
    import json

    import numpy as np
    import pandas as pd

    from careflow.agents.systems import SYSTEMS
    from careflow.data.contexts import Context
    from careflow.eval.sycophancy import DebertaDetector, LLMDetector, RuleDetector, binary_metrics
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    scen = [json.loads(line) for line in Path(cfg["data_file"]).read_text().splitlines() if line.strip()]
    def items(split: str):
        out = []
        for s in scen:
            if s["split"] == split:
                for r in s["replies"]:
                    out.append((s["turns"][-1]["text"], r["text"], r["endorses_harmful"], s["harmful"], r["kind"]))
        return out
    train, test = items("train"), items("test")
    mlflow.log_metrics({"n_train_replies": len(train), "n_test_replies": len(test)})

    mcfg = load_config(cfg["models_config"])
    cache = LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"])
    pool = ModelPool(mcfg["models"], 1, cache)
    llm_model = pool.get(cfg["llm_detector_role"])
    llm = LLMDetector(lambda p: llm_model.chat([{"role": "user", "content": p}], 4, 0.0, cfg["seed"]).text)
    rule = RuleDetector()
    d = cfg["deberta"]
    deb = DebertaDetector(d["id"], d["revision"], device, d["max_length"], cfg["seed"])
    losses = deb.fit([(a, b) for a, b, *_ in train], [y for _, _, y, *_ in train], d["epochs"], d["lr"],
                     d["batch_size"], cfg["seed"])
    mlflow.log_metric("deberta.final_train_loss", float(np.mean(losses[-5:])))

    y_test = [y for _, _, y, *_ in test]
    scores = {
        "rule": [rule.score(a, b) for a, b, *_ in test],
        "llm": [llm.score(a, b) for a, b, *_ in test],
        "deberta": list(deb.score_many([(a, b) for a, b, *_ in test])),
    }
    benign_agree = [i for i, t in enumerate(test) if not t[3] and t[4] == "agree"]  # hard negatives
    rows = []
    for name, s in scores.items():
        m = binary_metrics(s, y_test)
        m["fpr_benign_agree"] = float(np.mean([s[i] >= 0.5 for i in benign_agree])) if benign_agree else float("nan")
        rows.append({"detector": name, **m})
        mlflow.log_metrics({f"detector.{name}.{k}": v for k, v in m.items()})
    det = pd.DataFrame(rows)
    print(det.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # (b) system sycophancy rate on harmful test scenarios
    harmful = [s for s in scen if s["split"] == "test" and s["harmful"]]
    ctxs = [Context(s["id"], [t["text"] for t in s["turns"]], [t["role"] for t in s["turns"]], None, None, "")
            for s in harmful]
    embedder = None
    sys_rows = []
    for role in dict.fromkeys(v["role"] for v in cfg["systems"].values()):
        model = pool.get(role)
        for name, scfg in cfg["systems"].items():
            if scfg["role"] != role:
                continue
            kwargs = dict(scfg.get("kwargs") or {})
            if scfg["type"] == "rerank":
                if embedder is None:
                    from sentence_transformers import SentenceTransformer

                    e = cfg["embedding_model"]
                    embedder = SentenceTransformer(e["id"], revision=e["revision"], device=device).encode
                kwargs["embed"] = embedder
            system = SYSTEMS[scfg["type"]](name, model, cfg["seed"], **kwargs)
            for c in ctxs:
                c.responses[name] = system.respond(c).text
            print(f"[sycophancy_eval] generated {name}", flush=True)
    llm_model = pool.get(cfg["llm_detector_role"])
    # Synthetic-test accuracy need not transfer to real system replies (different generator, style), so each
    # reply also gets an independent rater: the judge model with the label-check prompt (full conversation).
    audit = None
    if cfg.get("audit_role"):
        from careflow.data.synthetic import CHECK_REPLY, conversation_text

        audit_model = pool.get(cfg["audit_role"])
        conv = {s["id"]: conversation_text(s["turns"]) for s in harmful}

        def audit(cid: str, reply: str) -> float:
            t = audit_model.chat([{"role": "user", "content": CHECK_REPLY.format(conv=conv[cid], reply=reply)}],
                                 4, 0.0, cfg["seed"]).text.strip().upper()
            return 1.0 if t.startswith("YES") else 0.0 if t.startswith("NO") else float("nan")
    rng = np.random.default_rng(cfg["seed"])
    per_reply = []
    for name in cfg["systems"]:
        pairs = [(c.history[-1], c.responses[name]) for c in ctxs]
        det_scores = {"llm": np.array([llm.score(a, b) for a, b in pairs]),
                      "deberta": deb.score_many(pairs), "rule": np.array([rule.score(a, b) for a, b in pairs])}
        if audit is not None:
            det_scores["audit"] = np.array([audit(c.dialogue_id, c.responses[name]) for c in ctxs])
        for i, c in enumerate(ctxs):
            per_reply.append({"id": c.dialogue_id, "system": name, "reply": c.responses[name],
                              **{k: float(v[i]) for k, v in det_scores.items()}})
        row = {"system": name}
        for dn, sc in det_scores.items():
            flags = (sc >= 0.5).astype(float)
            boots = [rng.choice(flags, len(flags)).mean() for _ in range(cfg["bootstrap"]["n"])]
            row[f"rate.{dn}"] = float(flags.mean())
            row[f"rate.{dn}.ci_low"], row[f"rate.{dn}.ci_high"] = map(float, np.percentile(boots, [2.5, 97.5]))
            mlflow.log_metrics({f"system.{name}.rate.{dn}": row[f"rate.{dn}"]})
        sys_rows.append(row)
    pool.close()
    cache.close()

    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    det.to_csv(out / "detectors.csv", index=False)
    sys_df = pd.DataFrame(sys_rows)
    sys_df.to_csv(out / "system_rates.csv", index=False)
    rep = pd.DataFrame(per_reply)
    rep.to_csv(out / "system_reply_scores.csv", index=False)
    # agreement between raters on real system replies (binary at 0.5)
    from sklearn.metrics import cohen_kappa_score

    raters = [k for k in ("rule", "llm", "deberta", "audit") if k in rep]
    kappa = {}
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            m = rep[[r1, r2]].dropna()
            kappa[f"kappa.{r1}.{r2}"] = float(cohen_kappa_score(m[r1] >= 0.5, m[r2] >= 0.5))
    mlflow.log_metrics(kappa)
    print("agreement on system replies:", {k: round(v, 3) for k, v in kappa.items()})
    # blind sheet for manual gold labels: stratified by system, shuffled, no system/detector columns
    k = cfg.get("label_sheet_per_system", 10)
    sheet = rep.groupby("system").sample(n=min(k, len(ctxs)), random_state=cfg["seed"])  # equal-size groups
    sheet = sheet.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    sheet["item"] = range(1, len(sheet) + 1)
    seeker = {c.dialogue_id: c.history[-1] for c in ctxs}
    pd.DataFrame({"item": sheet["item"], "seeker": sheet["id"].map(seeker), "reply": sheet["reply"],
                  "endorses_harmful_0_1": ""}).to_csv(out / "label_sheet.csv", index=False)
    sheet[["item", "id", "system"]].to_csv(out / "label_sheet_key.csv", index=False)
    with (out / "system_replies.jsonl").open("w") as f:
        for c in ctxs:
            f.write(json.dumps({"id": c.dialogue_id, "seeker": c.history[-1], "replies": c.responses},
                               ensure_ascii=False) + "\n")
    for fname in ("detectors.csv", "system_rates.csv", "system_replies.jsonl", "system_reply_scores.csv",
                  "label_sheet.csv", "label_sheet_key.csv"):
        mlflow.log_artifact(str(out / fname))
    print(sys_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@register("equivalence")
def equivalence(cfg: dict[str, Any], device: str) -> None:
    """TOST equivalence vs the reference (raw and length-controlled), Bonferroni-adjusted, plus power analysis:
    which "no difference" claims are supported, and how many contexts a confirmatory run needs."""
    import numpy as np
    import pandas as pd

    from careflow.eval.equivalence import (
        bootstrap_mean,
        min_detectable_effect,
        required_n,
        tost_p,
        verdict,
    )
    from careflow.eval.length_control import bootstrap_lc_draws, fit_lc

    judgments = _head_to_head(cfg["sources"], cfg["reference"], cfg.get("exclude", []))
    systems = sorted({j.system for j in judgments})
    n_boot, alpha = cfg["bootstrap"]["n"], cfg["alpha"]
    h1 = cfg.get("h1_systems")  # pre-registered family: Bonferroni over it only; other rows at plain alpha

    def alpha_for(s: str) -> float:
        if h1 is None:
            return alpha / len(systems)  # Bonferroni over all systems compared with the reference
        return alpha / len(h1) if s in h1 else alpha
    lc, _ = fit_lc(judgments, systems, cfg["sigma_words"])
    lc_draws = bootstrap_lc_draws(judgments, systems, cfg["sigma_words"], n_boot, cfg["seed"])

    def ci(d: np.ndarray, a: float) -> tuple[float, float]:
        return float(np.percentile(d, 100 * a / 2)), float(np.percentile(d, 100 * (1 - a / 2)))

    rows = []
    for s in systems:
        per_ctx: dict[str, list[int]] = {}
        for j in judgments:
            if j.system == s:
                per_ctx.setdefault(j.ctx, []).append(j.won)
        scores = np.array([np.mean(v) for v in per_ctx.values()])
        sd, n = float(scores.std(ddof=1)), len(scores)
        a_adj = alpha_for(s)
        for kind, point, draws in (("raw", float(scores.mean()), bootstrap_mean(scores, n_boot, cfg["seed"])),
                                   ("lc", lc[s], lc_draws[s])):
            row = {"system": s, "kind": kind, "n_contexts": n, "winrate": point, "sd_ctx": sd, "alpha": a_adj,
                   "mde_80": min_detectable_effect(sd, n, a_adj)}
            for delta in cfg["deltas"]:
                row[f"verdict@{delta}"] = verdict(ci(draws, a_adj), ci(draws, 2 * a_adj), delta)
                row[f"tost_p@{delta}"] = tost_p(draws, delta)
                row[f"n_needed@{delta}"] = required_n(sd, delta, a_adj, cfg["power"])
            rows.append(row)
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        mlflow.log_metrics({f"{r['system']}.{r['kind']}.{k}".replace("@", "_at_"): float(r[k])
                            for k in df.columns if k not in ("system", "kind") and not k.startswith("verdict")})
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "equivalence.csv", index=False)
    mlflow.log_artifact(str(out / "equivalence.csv"))
    print(f"alpha={alpha}; per-row adjusted alpha in column 'alpha'; power {cfg['power']}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@register("router_diagnostics")
def router_diagnostics(cfg: dict[str, Any], device: str) -> None:
    """Where the oracle headroom lives (stable vs judge-split contexts) and whether it is identifiable before
    generation (AUROC of a context-only classifier, trained on one split, tested on another). No LLM calls."""
    import json

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from careflow.data.contexts import sample_contexts
    from careflow.data.prepare import read_jsonl
    from careflow.routers.diagnostics import (
        context_classes,
        headroom_by_class,
        identifiability_auroc,
        order_consistency,
    )
    from careflow.routers.oracle import oracle, scores_from_judgments

    ref, paths = cfg["reference"], cfg["route_paths"]
    e = cfg["embedding_model"]
    embedder = SentenceTransformer(e["id"], revision=e["revision"], device=device)

    def load(run_dir: str, split: str):
        run = Path(run_dir)
        recs = [json.loads(line) for line in (run / "judgments.jsonl").open()]
        recs = [r for r in recs if r.get("system") in paths]
        scores = scores_from_judgments(recs, ref)
        cost = pd.read_csv(run / "results.csv").set_index("system")["latency_s"].to_dict()
        ctx = {c.dialogue_id: c for c in sample_contexts(
            read_jsonl(Path(cfg["data"]["dataset_dir"]) / f"{split}.jsonl"), 10**6, cfg["data"]["min_history"],
            cfg["seed"])}
        k = cfg["features"]["last_seeker_turns"]
        texts, meta = [], []
        for c in scores.contexts:
            seeker = [t for t, s in zip(ctx[c].history, ctx[c].speakers, strict=True) if s == "seeker"]
            texts.append(" ".join(seeker[-k:]))
            meta.append([len(ctx[c].history) / 20, len(texts[-1].split()) / 50])
        x = np.hstack([embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True), np.array(meta)])
        return scores, cost, x

    rows, metrics = [], {}
    loaded = {}
    for name, (run_dir, split) in cfg["splits"].items():
        scores, cost, x = load(run_dir, split)
        loaded[name] = (scores, x)
        total = oracle(scores, paths, cost).quality
        by_cls = headroom_by_class(scores, paths, ref, cost)
        cons = order_consistency(scores, paths, ref)
        for k, v in by_cls.items():
            rows.append({"split": name, "class": k, "share": v["share"], "oracle_gain": v["gain"],
                         "gain_share_of_total": v["gain"] / (total - 0.5) if total > 0.5 else float("nan")})
            metrics[f"{name}.{k}.share"], metrics[f"{name}.{k}.gain"] = v["share"], v["gain"]
        metrics[f"{name}.oracle"] = total
        metrics.update({f"{name}.order_consistency.{p}": v for p, v in cons.items()})
        print(f"[{name}] n={len(scores.contexts)} oracle={total:.3f}  order-consistency: "
              + ", ".join(f"{p} {v:.2f}" for p, v in cons.items()), flush=True)

    tr, te = cfg["identifiability"]["train"], cfg["identifiability"]["test"]
    (s_tr, x_tr), (s_te, x_te) = loaded[tr], loaded[te]
    targets = {"any_decisive": None, **{f"decisive_{p}": p for p in paths if p != ref}}
    for tname, p in targets.items():
        def label(s, p=p):
            if p is None:
                cls = context_classes(s, paths, ref)
                return [int(cls[c] == "decisive") for c in s.contexts]
            a, b = (s.by_order[o] for o in ("sys_first", "ref_first"))
            return [int(a[p][c] == 1 and b[p][c] == 1) for c in s.contexts]
        y_tr, y_te = label(s_tr), label(s_te)
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        r = identifiability_auroc(x_tr, y_tr, x_te, y_te, cfg["identifiability"]["C"], cfg["bootstrap"]["n"],
                                  cfg["seed"])
        metrics.update({f"identifiability.{tname}.{k}": v for k, v in r.items()})
        rows.append({"split": f"{tr}->{te}", "class": f"AUROC {tname}", "share": r["pos_rate_test"],
                     "oracle_gain": r["auroc"], "gain_share_of_total": float("nan"),
                     "ci": f"[{r['auroc_ci_low']:.2f}, {r['auroc_ci_high']:.2f}]"})
    mlflow.log_metrics(metrics)
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "diagnostics.csv", index=False)
    mlflow.log_artifact(str(out / "diagnostics.csv"))
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@register("export_contexts")
def export_contexts(cfg: dict[str, Any], device: str) -> None:
    """Shared context sample for the team: JSONL rows {ctx, history, speakers, human} so anyone can generate replies
    on exactly the contexts our systems were evaluated on."""
    import json

    from careflow.data.contexts import sample_contexts
    from careflow.data.prepare import read_jsonl

    d = cfg["data"]
    ctxs = sample_contexts(read_jsonl(Path(d["dataset_dir"]) / f"{d['split']}.jsonl"), d["n_contexts"],
                           d["min_history"], cfg["seed"])
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "contexts.jsonl").open("w") as f:
        for c in ctxs:
            f.write(json.dumps({"ctx": c.dialogue_id, "history": c.history, "speakers": c.speakers,
                                "emotion": c.emotion, "human": c.human}, ensure_ascii=False) + "\n")
    mlflow.log_artifact(str(out / "contexts.jsonl"))
    print(f"wrote {len(ctxs)} contexts to {out / 'contexts.jsonl'}")


@register("evaluate_replies")
def evaluate_replies(cfg: dict[str, Any], device: str) -> None:
    """One-command evaluation for any system (the team's entry point): replies of `system` and `reference` on the
    same contexts -> judge (HEART rubric, both orders), raw and length-controlled win-rate with CIs, TOST verdict.
    Input JSONL rows: {ctx, history, speakers, system, text} (+ optional emotion); see docs/evaluate.md."""
    import json

    import numpy as np

    from careflow.data.contexts import Context
    from careflow.eval import judge as J
    from careflow.eval.equivalence import bootstrap_mean, tost_p, verdict
    from careflow.eval.length_control import Judgment, bootstrap_lc_draws, fit_lc
    from careflow.eval.pairwise import judge_vs_reference
    from careflow.llm import ModelPool
    from careflow.utils.config import load_config

    sys_name, ref = cfg["system"], cfg["reference"]
    rows = [json.loads(line) for line in Path(cfg["replies_file"]).read_text().splitlines() if line.strip()]
    by_ctx: dict[str, Context] = {}
    for r in rows:
        if r["system"] not in (sys_name, ref):
            continue
        c = by_ctx.setdefault(r["ctx"], Context(r["ctx"], r["history"], r["speakers"], r.get("emotion"), None, ""))
        c.responses[r["system"]] = r["text"]
    contexts = [c for c in by_ctx.values() if sys_name in c.responses and ref in c.responses]
    if not contexts:
        raise ValueError(f"no context has replies of both {sys_name!r} and {ref!r}")
    mlflow.log_metric("n_contexts", len(contexts))

    pool = ModelPool(load_config(cfg["models_config"])["models"], 1,
                     LLMCache(Path(cfg["paths"]["cache"]), mode=cfg["cache"]["mode"]))
    judge = pool.get(cfg["judge_role"])
    res = judge_vs_reference(contexts, sys_name, ref,
                             lambda p: judge.chat(J.messages(p), cfg["judge"]["max_tokens"], 0.0, cfg["seed"]).text,
                             workers=int(cfg["compute"].get("concurrency", 1)))
    pool.close()

    words = {(c.dialogue_id, s): len(c.responses[s].split()) for c in contexts for s in (sys_name, ref)}
    js = [Judgment(r["ctx"], sys_name, int(r["winner"] == sys_name),
                   words[(r["ctx"], sys_name)] - words[(r["ctx"], ref)], int(r["first"] == sys_name))
          for r in res.records if "winner" in r]
    per_ctx = np.array([v for v in res.overall if v is not None])
    n_boot, alpha = cfg["bootstrap"]["n"], cfg["alpha"]
    raw_draws = bootstrap_mean(per_ctx, n_boot, cfg["seed"])
    lc, beta = fit_lc(js, [sys_name], cfg["sigma_words"])
    lc_draws = bootstrap_lc_draws(js, [sys_name], cfg["sigma_words"], n_boot, cfg["seed"])[sys_name]

    def ci(d, a):
        return float(np.percentile(d, 100 * a / 2)), float(np.percentile(d, 100 * (1 - a / 2)))

    report = {"system": sys_name, "reference": ref, "n_contexts": len(per_ctx),
              "judge_failures": res.failures, "order_consistency": float(np.mean(np.isin(per_ctx, (0.0, 1.0)))),
              "mean_words_system": float(np.mean([words[(c.dialogue_id, sys_name)] for c in contexts])),
              "mean_words_reference": float(np.mean([words[(c.dialogue_id, ref)] for c in contexts])),
              "length_beta": beta}
    for kind, point, d in (("raw", float(per_ctx.mean()), raw_draws), ("lc", lc[sys_name], lc_draws)):
        lo, hi = ci(d, alpha)
        report.update({f"{kind}.winrate": point, f"{kind}.ci_low": lo, f"{kind}.ci_high": hi})
        for delta in cfg["deltas"]:
            report[f"{kind}.verdict@{delta}"] = verdict((lo, hi), ci(d, 2 * alpha), delta)
            report[f"{kind}.tost_p@{delta}"] = tost_p(d, delta)
    for cat, vals in res.categories.items():
        report[f"cat.{cat}"] = float(np.mean(vals)) if vals else float("nan")
    mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in report.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = Path(cfg["paths"]["outputs"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out / "judgments.jsonl").open("w") as f:
        for r in res.records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for fname in ("report.json", "judgments.jsonl"):
        mlflow.log_artifact(str(out / fname))
    print(json.dumps(report, indent=2, ensure_ascii=False))
