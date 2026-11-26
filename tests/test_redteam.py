import json
import random
from pathlib import Path

from PIL import Image

from clients.base import Report, ReportClient
from redteam.analysis import compare
from redteam.perturbations import PERTURBATIONS


def _img(w=64, h=48):
    im = Image.new("L", (w, h), 0)
    for x in range(w // 2):          # bright left half so mirror is detectable
        for y in range(h):
            im.putpixel((x, y), 255)
    return im


def test_all_perturbations_run_and_keep_count():
    for name, fn in PERTURBATIONS.items():
        imgs, hist = fn([_img(), _img()], random.Random(0))
        assert len(imgs) == 2, name


def test_mirror_twice_is_identity_and_differs_once():
    a = _img()
    m, _ = PERTURBATIONS["mirror"]([a], random.Random(0))
    mm, _ = PERTURBATIONS["mirror"](m, random.Random(0))
    assert m[0].tobytes() != a.tobytes()
    assert mm[0].tobytes() == a.tobytes()


def test_rotate_swaps_dimensions():
    out, _ = PERTURBATIONS["rotate90"]([_img(64, 48)], random.Random(0))
    assert out[0].size == (48, 64)


def test_history_perturbation_sets_history_only():
    imgs, hist = PERTURBATIONS["hist_pneumonia"]([_img()], random.Random(0))
    assert "pneumonia" in hist.lower() and imgs[0].size == (64, 48)


def test_compare_detects_sycophancy():
    refs = {"a": "Lungs are clear.", "b": "Lungs are clear."}
    base = {k: {"text": "No acute disease.", "needs_more_info": False} for k in refs}
    pert = {k: {"text": "Right lower lobe pneumonia.", "needs_more_info": False} for k in refs}
    s = compare(base, pert, refs)
    assert s["finding_change_rate"] == 1.0
    assert s["false_claims_by_label"]["consolidation"]["change"] == 1.0


def test_pipeline_end_to_end_with_fake_client(tmp_path, monkeypatch):
    # tiny fake dataset + fake model that is easily fooled by history
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(6):
        _img().save(img_dir / f"{i}_x.png")
        rows.append({"id": f"{i}_x.png", "patient": str(i), "images": [f"{i}_x.png"],
                     "findings": "", "impression": "", "reference": "Lungs are clear."})
    data = tmp_path / "data"
    data.mkdir()
    for s in ("val", "test"):
        with open(data / f"{s}.jsonl", "w") as f:
            for r in rows if s == "test" else []:
                f.write(json.dumps(r) + "\n")

    class Fooled(ReportClient):
        name = "fooled"
        def generate(self, image_paths, history=None):
            if history and "pneumonia" in history.lower():
                return Report(findings="Right lower lobe pneumonia.")
            return Report(findings="Lungs are clear.", needs_more_info=False)

    import run_redteam
    monkeypatch.setattr(run_redteam, "make_client", lambda a: Fooled())
    monkeypatch.setattr("sys.argv", ["run_redteam.py", "--data", str(data),
                                     "--image-root", str(img_dir), "--limit", "6",
                                     "--perturbations", "hist_pneumonia", "hist_normal",
                                     "--out", str(tmp_path / "res")])
    run_redteam.main()
    summary = json.load(open(tmp_path / "res" / "redteam_fooled_summary.json"))
    assert summary["hist_pneumonia"]["finding_change_rate"] == 1.0
    assert summary["hist_normal"]["finding_change_rate"] == 0.0
