"""Step 1 (run once): turn data/instruction_data.json into clean train/val/test files.

Each row is one STUDY (frontal + lateral views share one report), and the split is
by patient so nobody appears in two splits.

    python prepare_data.py
"""
import argparse
import html
import json
import random
import re
from collections import OrderedDict
from pathlib import Path


def clean_text(text):
    """Fix export junk: HTML entities, stray '>]', and a literal 'nan' for empty fields."""
    text = html.unescape(text or "")
    text = re.sub(r"\s*>\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() == "nan" else text


def read_reports(item):
    """Pull (findings, impression) out of one conversation."""
    findings, impression, asked_impression = "", "", False
    for turn in item["conversation"]:
        if turn["role"] == "user":
            asked_impression = "impression" in turn["content"].lower()
        elif turn["role"] == "assistant":
            if asked_impression:
                impression = clean_text(turn["content"])
            elif not findings:
                findings = clean_text(turn["content"])
    return findings, impression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/instruction_data.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    studies = OrderedDict()
    for item in json.load(open(args.src, encoding="utf-8")):
        name = Path(item["image"].replace("\\", "/")).name
        findings, impression = read_reports(item)
        if not findings and not impression:
            continue
        patient = name.split("_")[0]          # number before "_" = patient id
        key = (patient, findings, impression)
        if key not in studies:
            studies[key] = {"id": name, "patient": patient, "images": [],
                            "reference": clean_text(f"{findings} {impression}")}
        if name not in studies[key]["images"]:
            studies[key]["images"].append(name)
    rows = list(studies.values())

    patients = sorted({r["patient"] for r in rows})
    random.Random(args.seed).shuffle(patients)
    n = len(patients)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    split_of = {p: "train" if i < cut1 else "val" if i < cut2 else "test"
                for i, p in enumerate(patients)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    counts = dict.fromkeys(files, 0)
    for r in rows:
        s = split_of[r["patient"]]
        files[s].write(json.dumps(r) + "\n")
        counts[s] += 1
    for f in files.values():
        f.close()
    print(f"{sum(len(r['images']) for r in rows)} images in {len(rows)} studies, "
          f"{n} patients -> {counts}")


if __name__ == "__main__":
    main()
