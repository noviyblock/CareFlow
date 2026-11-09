"""Support systems compared in block B. All share the HEART reply constraints (1-3 sentences, <= 40 words).

Re-implementations from the papers' descriptions, not their released code:
- InsideOut (Mozikov et al., ECAI 2024, DOI 10.3233/FAIA241039): one agent per Ekman emotion + aggregator (star).
- Roundtable in the spirit of UniESC (Wang et al., IEEE TAC 2026, DOI 10.1109/TAFFC.2026.3697775):
  a Theory-of-Mind agent profiles the seeker, several supporter personas propose, a supervisor picks.
- Multi-agent debate (Du et al., 2023, "Improving Factuality and Reasoning in Language Models through
  Multiagent Debate", arXiv:2305.14325): propose -> see the other's answer -> revise.
- ReAct-style (Yao et al., 2023, arXiv:2210.03629): explicit Thought before the
  Response; there are no external tools in this task, so it is a single reason-then-act call.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from careflow.agents import single_agent
from careflow.agents.base import Chat, Recorder, SystemOutput
from careflow.data.contexts import Context
from careflow.eval.judge import transcript

ESCONV_STRATEGIES = (
    "Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
    "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others",
)
EKMAN = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")
REPLY_RULES = (
    "Write EXACTLY ONE Supporter reply: conversational, 1-3 sentences, at most 40 words, first person, "
    "no speaker labels, no quotation marks."
)
_DEFAULT_RULES = REPLY_RULES


def set_reply_length(words: int | None) -> None:
    """Length-matched design: every system's final reply targets the same length, so the judge's length bias
    cannot favour one architecture. None restores the default rules (and the cached prompts)."""
    global REPLY_RULES
    REPLY_RULES = _DEFAULT_RULES if words is None else _DEFAULT_RULES.replace(
        "at most 40 words", f"about {words} words (between {words - 3} and {words + 3})")
    single_agent.set_reply_length(words)


def _conv(ctx: Context) -> str:
    return transcript(ctx.history, ctx.speakers)


def _after(marker: str, text: str) -> str | None:
    m = re.search(rf"{marker}\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return single_agent.clean(m.group(1).strip().split("\n\n")[0]) if m else None


class SingleAgent:
    """`temperature`/`seed_offset` > 0 give independent samples of the same system (placebo paths for the
    oracle noise floor); the defaults reproduce the greedy baseline and its cached prompts exactly."""

    def __init__(self, name: str, model: Chat, seed: int, max_tokens: int = 120, temperature: float = 0.0,
                 seed_offset: int = 0) -> None:
        self.name, self.rec, self.max_tokens = name, Recorder(model, seed), max_tokens
        self.temperature, self.seed_offset = temperature, seed_offset

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        msgs = single_agent.messages(ctx.history, ctx.speakers)
        out.text = single_agent.clean(self.rec.ask(out, "generate", msgs[0]["content"], self.max_tokens,
                                                   temperature=self.temperature, seed_offset=self.seed_offset))
        return out


class ReActStyle:
    def __init__(self, name: str, model: Chat, seed: int) -> None:
        self.name, self.rec = name, Recorder(model, seed)

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        raw = self.rec.ask(out, "think_act", (
            "You are the Supporter in an emotional support conversation.\n"
            f"Conversation so far:\n{_conv(ctx)}\n\n"
            "First write 'Thought:' with 1-2 sentences on what the Seeker feels and needs right now and which "
            f"support strategy fits ({', '.join(ESCONV_STRATEGIES)}). Then write 'Response:' followed by the reply. "
            + REPLY_RULES), 200)
        out.text = _after("Response", raw) or single_agent.clean(raw)
        return out


class PlannerGeneratorCritic:
    def __init__(self, name: str, model: Chat, seed: int) -> None:
        self.name, self.rec = name, Recorder(model, seed)

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        plan = self.rec.ask(out, "plan", (
            f"Conversation between a Seeker and a Supporter:\n{_conv(ctx)}\n\n"
            "As a support planner, choose ONE strategy for the next Supporter turn from: "
            f"{', '.join(ESCONV_STRATEGIES)}. Answer as 'Strategy: <name>' and 'Plan: <one sentence>'."), 80)
        draft = single_agent.clean(self.rec.ask(out, "generate", (
            f"Conversation so far:\n{_conv(ctx)}\n\nPlanner's guidance:\n{plan}\n\n"
            f"You are the Supporter. Follow the guidance. {REPLY_RULES}"), 120))
        verdict = self.rec.ask(out, "critic", (
            f"Conversation so far:\n{_conv(ctx)}\n\nCandidate Supporter reply: {draft}\n\n"
            "Check it for empathy, appropriateness, intrusiveness (pushy advice, too many questions) and "
            "sycophancy (agreeing with harmful or false beliefs just to please). If it is fine, answer 'OK'. "
            f"Otherwise answer 'REVISE:' followed by an improved reply. {REPLY_RULES}"), 160)
        revised = _after("REVISE", verdict)
        out.text = revised if revised else draft
        out.trace.append({"step": "decision", "text": "revised" if revised else "kept"})
        return out


class Debate:
    def __init__(self, name: str, model: Chat, seed: int, n_agents: int = 2, temperature: float = 0.7) -> None:
        self.name, self.rec, self.n, self.t = name, Recorder(model, seed), n_agents, temperature

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        base = single_agent.messages(ctx.history, ctx.speakers)[0]["content"]
        props = [single_agent.clean(self.rec.ask(out, f"propose_{i}", base, 120, temperature=self.t, seed_offset=i))
                 for i in range(self.n)]
        revised = []
        for i in range(self.n):
            others = "\n".join(f"- {p}" for j, p in enumerate(props) if j != i)
            revised.append(single_agent.clean(self.rec.ask(out, f"revise_{i}", (
                f"Conversation so far:\n{_conv(ctx)}\n\nYour previous reply: {props[i]}\n"
                f"Other supporters replied:\n{others}\n\nConsidering their views, give your final reply. "
                + REPLY_RULES), 120)))
        options = "\n".join(f"{k + 1}. {r}" for k, r in enumerate(revised))
        pick = self.rec.ask(out, "aggregate", (
            f"Conversation so far:\n{_conv(ctx)}\n\nCandidate Supporter replies:\n{options}\n\n"
            "Which reply best supports the Seeker? Answer with the number only."), 4)
        m = re.search(r"\d", pick)
        k = int(m.group()) - 1 if m and 0 < int(m.group()) <= len(revised) else 0
        out.text = revised[k]
        return out


class EmbeddingRerank:
    """Sample N candidates and return the embedding medoid (highest mean cosine to the others)."""

    def __init__(self, name: str, model: Chat, seed: int, embed: Callable[[list[str]], np.ndarray],
                 n: int = 4, temperature: float = 0.8) -> None:
        self.name, self.rec, self.embed, self.n, self.t = name, Recorder(model, seed), embed, n, temperature

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        base = single_agent.messages(ctx.history, ctx.speakers)[0]["content"]
        cands = [single_agent.clean(self.rec.ask(out, f"sample_{i}", base, 120, temperature=self.t, seed_offset=i))
                 for i in range(self.n)]
        out.text = cands[medoid(self.embed(cands))]
        return out


def medoid(emb: np.ndarray) -> int:
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = e @ e.T
    np.fill_diagonal(sim, 0.0)
    return int(np.argmax(sim.sum(axis=1)))


class InsideOut:
    def __init__(self, name: str, model: Chat, seed: int, emotions: Sequence[str] = EKMAN) -> None:
        self.name, self.rec, self.emotions = name, Recorder(model, seed), emotions

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        views = []
        for emo in self.emotions:
            v = self.rec.ask(out, f"agent_{emo}", (
                f"Conversation so far:\n{_conv(ctx)}\n\nYou are the {emo.upper()} agent: you only look at the "
                f"conversation through {emo}. In 1-2 sentences, say how {emo} shows up in the Seeker (or that it "
                "does not) and what the next Supporter reply should take into account."), 80)
            views.append(f"[{emo}] {v.strip()}")
        out.text = single_agent.clean(self.rec.ask(out, "aggregate", (
            f"Conversation so far:\n{_conv(ctx)}\n\nPerspectives of emotion agents:\n" + "\n".join(views) +
            f"\n\nAs the Supporter, combine the relevant perspectives into one reply. {REPLY_RULES}"), 120))
        return out


PERSONAS = (
    "a warm, validating listener who focuses on feelings",
    "a practical helper who gently offers one concrete next step",
    "a reflective counselor who helps the Seeker make sense of their situation",
)


class Roundtable:
    def __init__(self, name: str, model: Chat, seed: int, personas: Sequence[str] = PERSONAS) -> None:
        self.name, self.rec, self.personas = name, Recorder(model, seed), personas

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        tom = self.rec.ask(out, "theory_of_mind", (
            f"Conversation so far:\n{_conv(ctx)}\n\nInfer the Seeker's mental state: what they feel, what they "
            "believe, what they want from the Supporter right now, and their apparent personality. 2-3 sentences."), 120)
        props = [single_agent.clean(self.rec.ask(out, f"persona_{i}", (
            f"Conversation so far:\n{_conv(ctx)}\n\nProfile of the Seeker: {tom}\n\nYou are {p}. {REPLY_RULES}"), 120))
            for i, p in enumerate(self.personas)]
        options = "\n".join(f"{k + 1}. {r}" for k, r in enumerate(props))
        pick = self.rec.ask(out, "supervisor", (
            f"Conversation so far:\n{_conv(ctx)}\n\nProfile of the Seeker: {tom}\n\nCandidate replies:\n{options}\n\n"
            "As the supervisor, which reply fits this Seeker best? Answer with the number only."), 4)
        m = re.search(r"\d", pick)
        k = int(m.group()) - 1 if m and 0 < int(m.group()) <= len(props) else 0
        out.text = props[k]
        return out


class CareFlowMAS:
    """Full pipeline: context analysis -> generation -> safety check -> empathy/sycophancy check.

    Ablation flags (defaults reproduce the full system and its cached prompts exactly):
    - use_context=False: no analysis agent; the generator uses the single-agent (HEART) prompt, so the
      draft equals single_agent's reply and only the checkers can change it.
    - use_risk=False: the analysis agent is not asked about risk.
    - use_checks=False: no safety / empathy agents.
    """

    def __init__(self, name: str, model: Chat, seed: int, use_context: bool = True, use_risk: bool = True,
                 use_checks: bool = True) -> None:
        self.name, self.rec = name, Recorder(model, seed)
        self.use_context, self.use_risk, self.use_checks = use_context, use_risk, use_checks

    def respond(self, ctx: Context) -> SystemOutput:
        out = SystemOutput("")
        if self.use_context:
            fields = ("Emotion: <main emotion>\nIntensity: <1-5>\nNeed: <what they need now>\n"
                      f"Strategy: <one of {', '.join(ESCONV_STRATEGIES)}>")
            n_lines = "5" if self.use_risk else "4"
            if self.use_risk:
                fields += "\nRisk: <none|self-harm|harm to others|other>"
            analysis = self.rec.ask(out, "context", (
                f"Conversation so far:\n{_conv(ctx)}\n\nAnalyse the Seeker. Answer in {n_lines} short lines:\n"
                + fields), 100)
            draft = single_agent.clean(self.rec.ask(out, "generate", (
                f"Conversation so far:\n{_conv(ctx)}\n\nAnalysis of the Seeker:\n{analysis}\n\n"
                f"You are the Supporter. Use the analysis. {REPLY_RULES}"), 120))
        else:
            msgs = single_agent.messages(ctx.history, ctx.speakers)
            draft = single_agent.clean(self.rec.ask(out, "generate", msgs[0]["content"], 120))
        if self.use_checks:
            safety = self.rec.ask(out, "safety", (
                f"Conversation so far:\n{_conv(ctx)}\n\nCandidate Supporter reply: {draft}\n\n"
                "Is the reply safe (no harmful advice, no dismissal of risk, appropriate referral if there is risk of "
                f"harm)? Answer 'SAFE', or 'UNSAFE:' followed by a safe reply. {REPLY_RULES}"), 160)
            draft = _after("UNSAFE", safety) or draft
            check = self.rec.ask(out, "empathy", (
                f"Conversation so far:\n{_conv(ctx)}\n\nCandidate Supporter reply: {draft}\n\n"
                "Is the reply empathic and attuned, without being intrusive or sycophantic (agreeing with harmful or "
                "false beliefs to please)? Answer 'OK', or 'REVISE:' followed by an improved reply. " + REPLY_RULES), 160)
            draft = _after("REVISE", check) or draft
        out.text = draft
        return out


SYSTEMS: dict[str, Any] = {
    "single": SingleAgent,
    "react": ReActStyle,
    "pgc": PlannerGeneratorCritic,
    "debate": Debate,
    "rerank": EmbeddingRerank,
    "insideout": InsideOut,
    "roundtable": Roundtable,
    "careflow_mas": CareFlowMAS,
}
