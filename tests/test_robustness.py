import json

from clients.base import Report, ReportClient, TransientModelError
import clients.openai_compat as oc
from clients.openai_compat import OpenAICompatClient


def test_openai_client_retries_then_succeeds(monkeypatch):
    c = OpenAICompatClient(model="m", base_url="http://x", api_key="k")
    calls = {"n": 0}

    class Msg:
        content = '{"findings": "ok", "impression": "fine"}'

    class Choice:
        message = Msg

    class Resp:
        choices = [Choice]

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("busy")
        return Resp

    monkeypatch.setattr(oc, "TRANSIENT_ERRORS", (RuntimeError,))
    monkeypatch.setattr(c.client.chat.completions, "create", fake_create)
    monkeypatch.setattr("time.sleep", lambda s: None)
    rep = c.generate(None)
    assert rep.findings == "ok" and calls["n"] == 3


def test_openai_client_gives_up_with_transient_error(monkeypatch):
    c = OpenAICompatClient(model="m", base_url="http://x", api_key="k")

    def always_fail(**kw):
        raise RuntimeError("busy")

    monkeypatch.setattr(oc, "TRANSIENT_ERRORS", (RuntimeError,))
    monkeypatch.setattr(c.client.chat.completions, "create", always_fail)
    monkeypatch.setattr("time.sleep", lambda s: None)
    try:
        c.generate(None)
        assert False, "should have raised"
    except TransientModelError:
        pass


def test_run_eval_skips_failed_studies(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"id": f"{i}.png", "patient": str(i), "images": [f"{i}.png"], "findings": "",
             "impression": "", "reference": "Small pleural effusion."} for i in range(4)]
    with open(data / "test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    class Flaky(ReportClient):
        name = "flaky"
        def generate(self, image_paths, history=None):
            if str(image_paths[0]).endswith("1.png"):
                raise TransientModelError("busy")
            return Report(findings="Small pleural effusion.")

    import run_eval
    monkeypatch.setattr(run_eval, "make_client", lambda a: Flaky())
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--client", "openai", "--data", str(data),
                                     "--split", "test", "--out", str(tmp_path / "res")])
    run_eval.main()
    scores = json.load(open(tmp_path / "res" / "flaky_test_scores.json"))
    assert scores["n_reports"] == 3
