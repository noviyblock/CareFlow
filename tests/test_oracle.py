from itertools import pairwise

from careflow.routers.oracle import budget_frontier, fixed, oracle, scores_from_judgments

COST = {"cheap": 1.0, "ref": 2.0, "big": 5.0}


def _rec(system, ctx, first_is_sys, won):
    first, second = (system, "ref") if first_is_sys else ("ref", system)
    return {"system": system, "reference": "ref", "ctx": ctx, "first": first, "second": second,
            "winner": system if won else "ref"}


def test_scores_and_fixed():
    recs = [_rec("cheap", "c1", True, True), _rec("cheap", "c1", False, False)]
    s = scores_from_judgments(recs, "ref")
    assert s.contexts == ["c1"] and s.by_order["sys_first"]["ref"]["c1"] == 0.5
    assert fixed(s, "cheap", COST).quality == 0.5


def test_crossfit_removes_winners_curse():
    # 'big' wins in exactly one order on every context: pure position noise, true quality 0.5
    recs = []
    for i in range(10):
        recs += [_rec("big", f"c{i}", True, True), _rec("big", f"c{i}", False, False)]
        recs += [_rec("cheap", f"c{i}", True, False), _rec("cheap", f"c{i}", False, False)]
    s = scores_from_judgments(recs, "ref")
    naive = oracle(s, ["cheap", "ref", "big"], COST, cross_fit=False)
    cf = oracle(s, ["cheap", "ref", "big"], COST, cross_fit=True)
    assert naive.quality > cf.quality and abs(cf.quality - 0.25) < 1e-9


def test_budget_frontier_monotone_cost():
    recs = []
    for i in range(6):
        win = i % 2 == 0
        recs += [_rec("big", f"c{i}", True, win), _rec("big", f"c{i}", False, win)]
        recs += [_rec("cheap", f"c{i}", True, False), _rec("cheap", f"c{i}", False, False)]
    s = scores_from_judgments(recs, "ref")
    fr = budget_frontier(s, "cheap", ["cheap", "ref", "big"], COST, [1.0, 3.0, 5.0])
    assert fr[0].cost == 1.0 and fr[0].quality == 0.0
    assert all(a.cost <= b.cost + 1e-9 for a, b in pairwise(fr))
    assert fr[-1].quality > fr[0].quality


def test_length_control_recovers_equal_quality():
    import numpy as np

    from careflow.eval.length_control import Judgment, fit_lc

    rng = np.random.default_rng(0)
    js = []
    for i in range(2000):  # system 'long' only wins because it is longer; true quality equal to reference
        dlen = rng.normal(4, 12)  # mostly longer, but spanning 0 so length and quality are separable
        p = 1 / (1 + np.exp(-(1.2 * np.tanh(dlen / 10))))
        js.append(Judgment(f"c{i}", "long", int(rng.random() < p), dlen, i % 2))
    raw = np.mean([j.won for j in js])
    lc, beta = fit_lc(js, ["long"], 10)
    assert raw > 0.55 and abs(lc["long"] - 0.5) < 0.05 and beta > 0.5


def test_learned_routers_pick_the_better_path_per_region():
    import numpy as np

    from careflow.routers.learned import ClusterRouter, UtilityRouter, evaluate

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-3, 0.3, (50, 2)), rng.normal(3, 0.3, (50, 2))])
    region = np.r_[np.zeros(50), np.ones(50)]
    scores = {"a": np.where(region == 0, 1.0, 0.0), "b": np.where(region == 1, 1.0, 0.0), "ref": np.full(100, 0.5)}
    cost = {"a": 1.0, "b": 1.0, "ref": 1.0}
    for r in (UtilityRouter(["a", "b", "ref"], 0.1).fit(x, scores), ClusterRouter(["a", "b", "ref"], 2, 0).fit(x, scores)):
        q, _ = evaluate(r.route(x, cost, 0.0), scores, cost)
        assert q > 0.95


def test_context_classes_and_headroom():
    from careflow.routers.diagnostics import context_classes, headroom_by_class
    from careflow.routers.oracle import Scores

    by_order = {"sys_first": {"a": {"c1": 1, "c2": 1, "c3": 0}, "r": {"c1": .5, "c2": .5, "c3": .5}},
                "ref_first": {"a": {"c1": 1, "c2": 0, "c3": 0}, "r": {"c1": .5, "c2": .5, "c3": .5}}}
    s = Scores(by_order, ["c1", "c2", "c3"])
    assert context_classes(s, ["a", "r"], "r") == {"c1": "decisive", "c2": "split", "c3": "none"}
    h = headroom_by_class(s, ["a", "r"], "r", {"a": 2.0, "r": 1.0})
    assert abs(h["decisive"]["gain"] - 0.5 / 3) < 1e-9  # c1: oracle picks a, wins in both orders
    assert h["split"]["gain"] < 0                         # c2: selected on the winning order, loses on the other
