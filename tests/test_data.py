import json

import pytest

from careflow.data import empathetic_dialogues, esconv
from careflow.data.download import fetch, sha256
from careflow.data.prepare import read_jsonl, stats, write_jsonl


def _esconv_raw(tmp_path):
    def dialog(text):
        return {
            "experience_type": "Current Experience",
            "emotion_type": "anxiety",
            "problem_type": "job crisis",
            "situation": " I am  worried ",
            "survey_score": {
                "seeker": {"initial_emotion_intensity": "5", "final_emotion_intensity": "2"},
                "supporter": {},
            },
            "dialog": [
                {"speaker": "seeker", "annotation": {}, "content": f"{text}\n"},
                {"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": "Why?"},
                {"speaker": "seeker", "annotation": {"feedback": "4"}, "content": "Because."},
            ],
        }

    corpus = [dialog("Hello  there"), dialog("Hi")]
    (tmp_path / "ESConv.json").write_text(json.dumps(corpus))

    def official(d):
        return {"dialog": [{"text": " ".join(u["content"].split())} for u in d["dialog"]]}

    (tmp_path / "train.txt").write_text(json.dumps(official(corpus[0])) + "\n")
    (tmp_path / "valid.txt").write_text(json.dumps(official(corpus[1])) + "\n")
    (tmp_path / "test.txt").write_text("")
    return corpus


def test_esconv_load(tmp_path):
    _esconv_raw(tmp_path)
    ds = esconv.load(tmp_path)
    assert [d.split for d in ds] == ["train", "valid"]
    d = ds[0]
    assert d.turns[0].text == "Hello there" and d.situation == "I am worried"
    assert d.turns[1].role == "supporter" and d.turns[1].strategy == "Question"
    assert d.turns[2].feedback == 4
    assert (d.initial_intensity, d.final_intensity) == (5, 2)


def test_esconv_unmatched_dialogue_fails(tmp_path):
    _esconv_raw(tmp_path)
    (tmp_path / "valid.txt").write_text("")
    with pytest.raises(ValueError, match="not found"):
        esconv.load(tmp_path)


def test_empathetic_dialogues_load(tmp_path):
    base = tmp_path / "empatheticdialogues"
    base.mkdir()
    header = "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags"
    rows = [
        "hit:1_conv:2,2,sad,My dog died_comma_ sadly.,7,I am so sorry.,5|5|5_2|2|5,",
        "hit:1_conv:2,1,sad,My dog died_comma_ sadly.,3,My dog died_comma_ it hurts.,5|5|5_2|2|5,",
        # 9-field row: trailing header-less candidates column must be ignored
        "hit:1_conv:2,3,sad,My dog died_comma_ sadly.,3,Thanks.,5|5|5_2|2|5,,cand a|cand b",
    ]
    (base / "train.csv").write_text("\n".join([header, *rows]) + "\n")
    for split in ("valid", "test"):
        (base / f"{split}.csv").write_text(header + "\n")
    [d] = empathetic_dialogues.load(tmp_path)
    assert d.situation == "My dog died, sadly." and d.emotion == "sad"
    assert [t.role for t in d.turns] == ["seeker", "supporter", "seeker"]
    assert d.turns[0].text == "My dog died, it hurts."


def test_jsonl_roundtrip_and_stats(tmp_path):
    _esconv_raw(tmp_path)
    ds = esconv.load(tmp_path)
    write_jsonl(ds, tmp_path / "out")
    back = read_jsonl(tmp_path / "out" / "train.jsonl")
    assert back == [ds[0]]
    s = stats(ds)
    assert s["n_dialogues"] == 2 and s["intensity_drop_mean"] == 3 and s["share_deescalated"] == 1.0


def test_fetch_rejects_bad_hash(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("data")
    good = sha256(src / "f.txt")
    cfg = {"raw_dir": str(tmp_path / "raw"), "source": {"base_url": src.as_uri(), "files": {"f.txt": good}}}
    fetch(cfg)
    assert (tmp_path / "raw" / "f.txt").read_text() == "data"
    cfg["source"]["files"]["f.txt"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch(cfg)
