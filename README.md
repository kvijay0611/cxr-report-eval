# cxr-report-eval

Measures how well a vision-language model drafts chest X-ray reports, and what
happens when you change its inputs (blurred image, misleading history, not an X-ray).

> Research only. Not a medical device. Not for clinical use.

## How it works
1. `prepare_data.py` cleans the Open-I reports and splits studies by patient (run once).
2. `run.py` asks a model (`models.py`) to write a report for each study.
3. `scoring.py` turns each report into a set of findings and compares model vs real report.
4. `perturb.py` changes the input; `run.py --perturb ...` shows how answers change.

## Use
    pip install -r requirements.txt
    python prepare_data.py
    python -m pytest -q
    python run.py --model constant                       # control: always says "normal"
    python run.py --model openai --name MODEL --base-url URL --limit 30
    python run.py --model hf --limit 50 --perturb all    # needs a GPU (e.g. Kaggle)

## Known limitations
- Reference reports (Open-I) contain anonymisation tokens (XXXX) and are imperfect:
  a "hallucinated" finding may be real but missing from the reference.
- Findings are read by keyword rules for 8 labels only; hedged mentions are ignored.
- The hf model reads one image per study.
