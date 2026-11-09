"""Single-agent prompting baseline.

The supporter prompt is HEART's generation prompt (arXiv:2601.19922; released code,
commit c3387b43, Apache-2.0, `build_prompt`, external mode) so our responses are
comparable with the HEART completions. Literal "\\n" sequences of the original
f-string are rendered as real newlines here.
"""

from __future__ import annotations

from collections.abc import Sequence

from careflow.eval.judge import transcript

_PROMPT = (
    "This is a roleplay, you playing an emotionally intelligent human who is, essentially, yourself. "
    "Treat it like a real situation. Always respond in first person."
    "You are the Supporter in a conversation with a Seeker. "
    "The Seeker is someone who needs emotional support and guidance. "
    "The Supporter (you) provides empathy, understanding, and helpful responses. "
    "Guidelines:\n"
    "- Respond with empathy and understanding\n"
    "- Keep your response conversational and natural (1-3 sentences)\n"
    "- Build on what the person just said\n"
    "- Do NOT repeat the conversation\n"
    "- Write **EXACTLY ONE** Supporter reply and nothing else. The response should be a maximum of 40 words.\n"
    "- Do NOT include any Seeker text, speaker labels, or quotation marks\n\n"
    "Conversation so far:\n{transcript}\n"
    "Supporter:"
)


_DEFAULT_PROMPT = _PROMPT


def set_reply_length(words: int | None) -> None:
    global _PROMPT
    _PROMPT = _DEFAULT_PROMPT if words is None else _DEFAULT_PROMPT.replace(
        "The response should be a maximum of 40 words.",
        f"The response should be about {words} words (between {words - 3} and {words + 3}).")


def messages(history: Sequence[str], speakers: Sequence[str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": _PROMPT.format(transcript=transcript(history, speakers))}]


def clean(text: str) -> str:
    """Strip a leading speaker label or wrapping quotes the model may add despite instructions."""
    t = text.strip()
    if t.lower().startswith("supporter:"):
        t = t[len("supporter:") :].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t
