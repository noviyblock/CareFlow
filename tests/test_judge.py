import pytest

from careflow.cache import LLMCache
from careflow.eval import bradley_terry
from careflow.eval import judge as J
from careflow.eval.judge_validation import (
    Pair,
    outcomes,
    players_for_row,
    position_consistency,
    rank_agreement,
    sample_pairs,
)
from careflow.llm import Generation, ModelPool


def _json(**over):
    vals = {c: "A0493++" for c in J.CATEGORIES}
    vals.update(over)
    body = ",".join(f'"{k}": "{v}"' for k, v in vals.items())
    return '```json\n{"chain_of_thought_reasoning": "x", ' + body + "}\n```"


def test_prompt_uses_given_speakers():
    p = J.build_prompt(["Hi, how can I help?", "I feel low"], ["supporter", "seeker"], "R1", "R2", "sadness")
    assert "Supporter: Hi, how can I help?\nSeeker: I feel low\nFinal Response: R1" in p
    assert "A0488 Final Turn: R2" in p and "addresses sadness support" in p


def test_parse_strict():
    v = J.parse(_json(overall_eq="A0488++++ (far stronger)"))
    assert v["overall_eq"] == J.Verdict("A0488", 4) and v["empathic_responsiveness"].strength == 2
    with pytest.raises(J.JudgeParseError, match="ambiguous"):
        J.parse(_json(overall_eq="A0493+ vs A0488++"))
    with pytest.raises(J.JudgeParseError, match="missing category"):
        J.parse('{"overall_eq": "A0493+"}')
    with pytest.raises(J.JudgeParseError):
        J.parse("Unable to judge A0493 versus A0488.")  # HEART's fallback would score this


def test_bradley_terry_orders_players():
    games = [("a", "b", 1)] * 8 + [("a", "b", 0)] * 2 + [("b", "c", 1)] * 8 + [("b", "c", 0)] * 2
    s = bradley_terry.fit(games)
    assert s["a"] > s["b"] > s["c"] and abs(sum(s.values()) / 3 - 1500) < 1e-6


def test_sampling_is_seeded_and_mirrors():
    avail = [["a", "b", "c"], ["a", "b"], ["a", "c", "d"]]
    p1 = sample_pairs(avail, 2, 2, 0.5, seed=1)
    assert p1 == sample_pairs(avail, 2, 2, 0.5, seed=1)
    mirrors = [p for p in p1 if p.swapped_of is not None]
    assert mirrors and all((m.a, m.b) == (p1[m.swapped_of].b, p1[m.swapped_of].a) for m in mirrors)


def test_outcomes_and_consistency():
    pairs = [Pair(0, "a", "b"), Pair(0, "b", "a", swapped_of=0)]
    assert outcomes(pairs, ["a", "a"]) == [("a", "b", 1), ("b", "a", 0)]
    assert position_consistency(pairs, ["a", "a"]) == 1.0
    assert position_consistency(pairs, ["a", "b"]) == 0.0  # always picks first slot
    assert rank_agreement({"x": 3, "y": 2, "z": 1}, {"x": 30, "y": 20, "z": 10})["kendall_tau"] == 1.0


def test_players_for_row_aliases():
    ref = {"g3": {"columns": ["gemini3pro", "gemini3"]}, "o3": {"columns": ["gpto3"]}}
    row = {"gemini3": "hi", "gemini3pro": " ", "human_completion_1": "me"}
    assert players_for_row(row, ref, include_human=True) == {"g3": "hi", "human": "me"}


class _Fake:
    def __init__(self, spec):
        self.name, self.calls, self.closed = spec["id"], 0, False

    def chat(self, messages, max_tokens, temperature=0.0, seed=0):
        self.calls += 1
        return Generation("ok", self.name, 3, 1, 0.1)

    def close(self):
        self.closed = True


def test_pool_lru_and_cache(tmp_path):
    specs = {r: {"id": r, "revision": "0"} for r in ("a", "b", "c")}
    built = []
    pool = ModelPool(specs, 2, LLMCache(tmp_path), builder=lambda s: built.append(_Fake(s)) or built[-1])
    msg = [{"role": "user", "content": "x"}]
    assert not pool.get("a").chat(msg, 5).cached
    assert pool.get("a").chat(msg, 5).cached and built[0].calls == 1
    pool.get("b")
    pool.get("c")  # evicts "a"
    assert built[0].closed and len(built) == 3
