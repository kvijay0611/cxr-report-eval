"""All tests in one file.  Run:  python -m pytest -q"""
import json
import random
import types

import pytest
from PIL import Image

import models
import run
from perturb import PERTURBATIONS
from prepare_data import clean_text
from scoring import compare, extract_findings, score


# ---- reading findings out of text -------------------------------------------
def test_negated_findings_ignored():
    assert extract_findings("There is no pneumothorax. No pleural effusion. The lungs are clear.") == set()

def test_positive_findings():
    assert extract_findings("Small right pleural effusion. Mild cardiomegaly.") == {"pleural_effusion", "cardiomegaly"}

def test_but_splits_clauses():
    assert extract_findings("No pneumothorax, but there is a left lower lobe consolidation.") == {"consolidation"}

def test_anonymised_token_negation():
    assert extract_findings("There are no XXXX of a pleural effusion.") == set()

def test_prominent_heart():
    assert extract_findings("Heart appears prominent.") == {"cardiomegaly"}
    assert extract_findings("The heart is again mildly enlarged.") == {"cardiomegaly"}
    assert extract_findings("The heart is not enlarged.") == set()
    assert extract_findings("Heart size normal, pulmonary vessels prominent.") == set()

def test_hedged_is_not_definite():
    assert extract_findings("Possibly a tortuous aorta versus an anterior mediastinal mass.") == set()

def test_empty():
    assert extract_findings("") == set()


# ---- data cleaning ----------------------------------------------------------
def test_clean_text():
    assert clean_text("nan") == "" and clean_text(None) == ""
    assert clean_text("Normal.&gt;] Low volumes.") == "Normal. Low volumes."


# ---- scoring ----------------------------------------------------------------
def test_constant_normal_looks_good_but_finds_nothing():
    pairs = [("No acute disease.", "No acute disease.")] * 3 + [("Small pleural effusion.", "No acute disease.")]
    r = score(pairs)
    assert r["abnormal_detection_accuracy"] == 0.75
    assert r["per_label"]["pleural_effusion"]["recall"] == 0.0
    assert r["reports_with_hallucinated_finding"] == 0.0

def test_hallucination_counted():
    r = score([("Clear lungs.", "Right pneumothorax.")])
    assert r["reports_with_hallucinated_finding"] == 1.0 and r["per_label"]["pneumothorax"]["fp"] == 1

def test_compare_detects_sycophancy():
    refs = {"a": "Lungs are clear."}
    base = {"a": {"text": "No acute disease.", "needs_more_info": False}}
    pert = {"a": {"text": "Right lower lobe pneumonia.", "needs_more_info": False}}
    s = compare(base, pert, refs)
    assert s["finding_change_rate"] == 1.0 and s["false_claims_by_label"]["consolidation"]["change"] == 1.0


# ---- perturbations ----------------------------------------------------------
def _img(w=64, h=48):
    im = Image.new("L", (w, h), 0)
    for x in range(w // 2):
        for y in range(h):
            im.putpixel((x, y), 255)
    return im

def test_every_perturbation_runs():
    for name, fn in PERTURBATIONS.items():
        imgs, _ = fn([_img(), _img()], random.Random(0))
        assert len(imgs) == 2, name

def test_mirror_and_rotate():
    a = _img()
    m, _ = PERTURBATIONS["mirror"]([a], random.Random(0))
    mm, _ = PERTURBATIONS["mirror"](m, random.Random(0))
    assert m[0].tobytes() != a.tobytes() and mm[0].tobytes() == a.tobytes()
    r, _ = PERTURBATIONS["rotate90"]([_img(64, 48)], random.Random(0))
    assert r[0].size == (48, 64)

def test_history_perturbation():
    imgs, hist = PERTURBATIONS["hist_pneumonia"]([_img()], random.Random(0))
    assert "pneumonia" in hist.lower() and imgs[0].size == (64, 48)


# ---- model helpers ----------------------------------------------------------
def test_parse_report_json_and_fallback():
    d = models.parse_report('noise {"findings": "Clear.", "impression": "Normal.", "needs_more_info": true} noise')
    assert d["text"] == "Clear. Normal." and d["needs_more_info"] is True
    assert models.parse_report("just words")["text"] == "just words"

class _Busy(Exception):
    status_code = 503

def _fake_openai(monkeypatch, errors):
    """Install a fake OpenAI client that raises the given errors, then succeeds."""
    calls = {"n": 0}
    msg = types.SimpleNamespace(content='{"findings": "ok", "impression": "fine"}')
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class Fake:
        def __init__(self, **kw):
            def create(**kw):
                calls["n"] += 1
                if errors and calls["n"] <= len(errors):
                    raise errors[calls["n"] - 1]
                return resp
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    monkeypatch.setattr("openai.OpenAI", Fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls

def test_openai_retries_then_succeeds(monkeypatch):
    calls = _fake_openai(monkeypatch, [_Busy("busy"), _Busy("busy")])
    ask = models.openai_model("m", "http://x")
    assert ask(None)["text"] == "ok fine" and calls["n"] == 3

def test_openai_gives_up_as_modelbusy(monkeypatch):
    _fake_openai(monkeypatch, [_Busy("busy")] * 10)
    with pytest.raises(models.ModelBusy):
        models.openai_model("m", "http://x")(None)

def test_openai_daily_quota_stops_immediately(monkeypatch):
    class Quota(Exception):
        status_code = 429
    calls = _fake_openai(monkeypatch, [Quota("Quota exceeded ... PerDay ...")])
    with pytest.raises(SystemExit):
        models.openai_model("m", "http://x")(None)
    assert calls["n"] == 1

def test_openai_real_errors_are_not_swallowed(monkeypatch):
    class NotFound(Exception):
        status_code = 404
    _fake_openai(monkeypatch, [NotFound("no such model")])
    with pytest.raises(NotFound):
        models.openai_model("m", "http://x")(None)


# ---- the whole run.py, with a fake model ------------------------------------
def test_run_end_to_end(tmp_path, monkeypatch):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i in range(6):
        _img().save(imgs / f"{i}_x.png")
        rows.append({"id": f"{i}_x.png", "patient": str(i), "images": [f"{i}_x.png"],
                     "reference": "Lungs are clear."})
    data = tmp_path / "data"
    data.mkdir()
    (data / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fooled(image_paths, history=None):          # easily fooled by a leading history
        if history and "pneumonia" in history.lower():
            return {"text": "Right lower lobe pneumonia.", "needs_more_info": False, "missing_info": ""}
        return {"text": "Lungs are clear.", "needs_more_info": False, "missing_info": ""}

    monkeypatch.setattr(run, "get_model", lambda *a, **k: fooled)
    monkeypatch.setattr("sys.argv", ["run.py", "--data", str(data), "--image-root", str(imgs),
                                     "--perturb", "hist_pneumonia", "hist_normal", "--out", str(tmp_path / "res")])
    run.main()
    summary = json.load(open(tmp_path / "res" / "constant_redteam.json"))
    assert summary["hist_pneumonia"]["finding_change_rate"] == 1.0
    assert summary["hist_normal"]["finding_change_rate"] == 0.0
    scores = json.load(open(tmp_path / "res" / "constant_scores.json"))
    assert scores["n_reports"] == 6
