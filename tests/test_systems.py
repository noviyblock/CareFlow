import numpy as np

from careflow.agents.systems import SYSTEMS, medoid
from careflow.data.contexts import Context, sample_contexts
from careflow.data.schema import Dialogue, Turn
from careflow.eval import judge as J
from careflow.eval.pairwise import judge_vs_reference, win_rate
from careflow.eval.report import pareto_front
from careflow.llm import Generation


class Scripted:
    """Fake chat model: replies by keyword in the prompt."""

    name = "fake"

    def chat(self, messages, max_tokens, temperature=0.0, seed=0):
        p = messages[-1]["content"]
        if "number only" in p:
            t = "2"
        elif "Answer 'OK'" in p or "Answer 'SAFE'" in p:
            t = "OK" if "Answer 'OK'" in p else "SAFE"
        elif "If it is fine" in p:
            t = "REVISE: That sounds really hard, I'm here with you."
        elif "Thought:" in p:
            t = "Thought: they are sad.\nResponse: I'm so sorry you're going through this."
        else:
            t = f"reply-{seed}"
        return Generation(t, self.name, 10, 5, 0.01)


CTX = Context("d1", ["Hi", "I lost my job"], ["supporter", "seeker"], "sadness", None, "Sorry to hear that.")


def test_all_systems_run():
    calls = {}
    for kind, cls in SYSTEMS.items():
        kw = {"embed": lambda xs: np.eye(len(xs)) + 0.1} if kind == "rerank" else {}
        out = cls(kind, Scripted(), 0, **kw).respond(CTX)
        assert out.text and out.calls, kind
        calls[kind] = len(out.calls)
    assert calls == {"single": 1, "react": 1, "pgc": 3, "debate": 5, "rerank": 4, "insideout": 7,
                     "roundtable": 5, "careflow_mas": 4}


def test_parsing_in_systems():
    assert SYSTEMS["react"]("r", Scripted(), 0).respond(CTX).text == "I'm so sorry you're going through this."
    assert SYSTEMS["pgc"]("p", Scripted(), 0).respond(CTX).text.startswith("That sounds really hard")
    assert SYSTEMS["debate"]("d", Scripted(), 0).respond(CTX).text == "reply-0"  # picked option 2 = revise_1


def test_medoid():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1.0]])
    assert medoid(emb) in (0, 1)


def test_sample_contexts_seeded():
    turns = [Turn(idx=i, role="seeker" if i % 2 == 0 else "supporter", text=f"t{i}") for i in range(10)]
    ds = [Dialogue(dialogue_id=f"d{k}", source="x", split="test", turns=turns) for k in range(5)]
    a = sample_contexts(ds, 3, 4, 7)
    assert [c.dialogue_id for c in a] == [c.dialogue_id for c in sample_contexts(ds, 3, 4, 7)]
    assert all(len(c.history) >= 4 and c.speakers[-1] == "seeker" for c in a)


def test_judge_vs_reference_both_orders():
    ctx = Context("d1", ["I feel low"], ["seeker"], None, None, "human reply", {"sys": "system reply"})

    def ask(prompt):  # always prefers whatever is shown first -> pure position bias -> 0.5
        return "{" + ",".join(f'"{c}": "A0493++"' for c in J.CATEGORIES) + "}"

    res = judge_vs_reference([ctx], "sys", "human", ask)
    assert res.overall == [0.5] and res.failures == 0
    assert win_rate(res.overall, 10, 0)["order_consistent"] == 0.0


def test_pareto_front():
    import pandas as pd

    df = pd.DataFrame({"latency_s": [1, 2, 3], "winrate": [0.4, 0.3, 0.6]})
    assert pareto_front(df, "latency_s", "winrate").tolist() == [True, False, True]


def test_pareto_ignores_nan():
    import pandas as pd

    df = pd.DataFrame({"latency_s": [1, 0.5, 3], "winrate": [0.4, float("nan"), 0.6]})
    assert pareto_front(df, "latency_s", "winrate").tolist() == [True, False, True]


def test_cascade_latency():
    from careflow.eval.latency import CascadeStep, cascade_summary

    steps = [CascadeStep(10, True), CascadeStep(20, True), CascadeStep(5, False, 100)]
    s = cascade_summary(steps)
    assert s["agl"] == 15 and s["arol"] == 5 and abs(s["small_share"] - 2 / 3) < 1e-9
    assert abs(s["mean_cost"] - (10 + 20 + 105) / 3) < 1e-9


def test_careflow_ablations_and_prompt_stability():
    class Rec(Scripted):
        def __init__(self):
            self.prompts = []

        def chat(self, messages, max_tokens, temperature=0.0, seed=0):
            self.prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, temperature, seed)

    full = Rec()
    SYSTEMS["careflow_mas"]("c", full, 0).respond(CTX)
    assert len(full.prompts) == 4 and "Answer in 5 short lines" in full.prompts[0] and "Risk:" in full.prompts[0]
    norisk = Rec()
    SYSTEMS["careflow_mas"]("c", norisk, 0, use_risk=False).respond(CTX)
    assert "Answer in 4 short lines" in norisk.prompts[0] and "Risk:" not in norisk.prompts[0]
    noctx = Rec()
    SYSTEMS["careflow_mas"]("c", noctx, 0, use_context=False).respond(CTX)
    assert len(noctx.prompts) == 3 and noctx.prompts[0].endswith("Supporter:")  # single-agent HEART prompt
    nocheck = Rec()
    SYSTEMS["careflow_mas"]("c", nocheck, 0, use_checks=False).respond(CTX)
    assert len(nocheck.prompts) == 2


def test_synthetic_label_check():
    from careflow.data.synthetic import passes_check

    s = {"harmful": True, "profile": {"belief": "I will stop my meds"},
         "turns": [{"role": "seeker", "text": "I will stop my meds, agree?"}],
         "replies": [{"kind": "agree", "text": "Yes, do it"}, {"kind": "balanced", "text": "Talk to your doctor first"}]}

    def ask(prompt, seed):
        if "risky" in prompt:
            return "YES"
        return "YES" if "Yes, do it" in prompt else "NO"

    assert passes_check(ask, s, 0) == (True, "ok")
    assert passes_check(lambda p, s_: "YES", s, 0) == (False, "balanced_reply")
    assert passes_check(lambda p, s_: "NO", s, 0) == (False, "belief_label")


def test_rule_detector_and_metrics():
    from careflow.eval.sycophancy import RuleDetector, binary_metrics

    r = RuleDetector()
    yes = r.score("I'll quit my meds, agree?", "You're absolutely right, go for it!")
    no = r.score("I'll quit my meds, agree?", "I hear you, but maybe talk to your doctor before stopping.")
    assert yes > 0.5 > no
    m = binary_metrics([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert m["auroc"] == 1.0 and m["f1"] == 1.0


def test_controls():
    from careflow.agents.controls import derangement, first_sentence, make_control

    assert first_sentence("I hear you. That sounds hard!") == "I hear you."
    p = derangement(10, 0)
    assert sorted(p) == list(range(10)) and all(i != j for i, j in enumerate(p))
    src = ["a.", "b.", "c."]
    assert set(make_control("shuffled", src, 1)) == set(src) and make_control("shuffled", src, 1) != src
    assert make_control("generic", src, 0, "I'm here.") == ["I'm here."] * 3


def test_reply_length_roundtrip():
    from careflow.agents import single_agent, systems

    before = (systems.REPLY_RULES, single_agent.messages(["hi"], ["seeker"]))
    systems.set_reply_length(30)
    assert "about 30 words" in systems.REPLY_RULES
    assert "about 30 words" in single_agent.messages(["hi"], ["seeker"])[0]["content"]
    systems.set_reply_length(None)
    assert (systems.REPLY_RULES, single_agent.messages(["hi"], ["seeker"])) == before
