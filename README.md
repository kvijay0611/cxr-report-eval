# cxr-report-eval

Research harness for evaluating vision-language models on chest X-ray report
drafting, plus a red-team suite that tests when they hallucinate or should
have asked for more information.

> Research only. Not a medical device. Not for clinical use.

## Status
Early scaffold. Implemented: data cleaning (patient-level split), model client
interface, a constant-"normal" control, a rule-based finding extractor, and
per-finding scoring. Results and model comparisons will be added as they are run.

## Setup
    python -m venv .venv
    .venv\Scripts\Activate.ps1        # Windows PowerShell (Mac/Linux: source .venv/bin/activate)
    pip install -r requirements.txt
    python data/clean_data.py --src data/instruction_data.json --out data/processed
    python -m pytest -q
    python run_eval.py --client constant --split test

## Layout
    data/       cleaning script (raw images are not committed)
    clients/    ReportClient interface, baselines, OpenAI-compatible client
    generate/   structured JSON report prompt and parser
    metrics/    finding extraction and scoring
    results/    raw outputs and score files
    tests/      unit tests

## Known limitations
- Reference reports come from Open-I and contain anonymisation tokens (XXXX).
- The finding extractor is rule-based; it will be compared against CheXbert later.
- Reference reports are imperfect: a "hallucinated" finding may be a real one the
  original report omitted. Spot-check samples by hand.
