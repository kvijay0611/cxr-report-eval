"""Turn the old instruction_data.json (Open-I conversations) into clean
train/val/test JSONL files, split BY PATIENT so the same person never
appears in two splits.

Usage:
    python data/clean_data.py --src data/instruction_data.json --out data/processed
"""
import argparse
import html
import json
import random
import re
from collections import OrderedDict
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalise whitespace, decode HTML entities, drop export artifacts
    (a literal 'nan' from an empty field, stray '>]' fragments)."""
    text = html.unescape(text or "")
    text = re.sub(r"\s*>\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() == "nan" else text


norm = clean_text


def image_name(path: str) -> str:
    # old paths use Windows backslashes: data/openi\\images\\...\\1_IM-0001-4001.dcm.png
    return Path(path.replace("\\", "/")).name


def patient_id(fname: str) -> str:
    # file names look like 1_IM-0001-4001.dcm.png; the number before "_" is
    # treated as the patient/study id. Verify this on your copy of the data.
    return fname.split("_")[0]


def parse_item(item: dict):
    findings, impression, want_impression = "", "", False
    for turn in item["conversation"]:
        if turn["role"] == "user":
            want_impression = "impression" in turn["content"].lower()
        elif turn["role"] == "assistant":
            if want_impression:
                impression = norm(turn["content"])
            elif not findings:
                findings = norm(turn["content"])
    return findings, impression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.load(open(args.src, encoding="utf-8"))

    # Group images into STUDIES. Frontal and lateral views of one exam share
    # the same report, so scoring per image would double-count every report.
    studies = OrderedDict()
    for item in raw:
        name = image_name(item["image"])
        findings, impression = parse_item(item)
        if not findings and not impression:
            continue
        key = (patient_id(name), findings, impression)
        if key not in studies:
            studies[key] = {
                "id": name,
                "patient": key[0],
                "images": [],
                "findings": findings,
                "impression": impression,
                "reference": norm(f"{findings} {impression}"),
            }
        if name not in studies[key]["images"]:
            studies[key]["images"].append(name)
    rows = list(studies.values())

    patients = sorted({r["patient"] for r in rows})
    random.Random(args.seed).shuffle(patients)
    n = len(patients)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    split_of = {}
    for i, p in enumerate(patients):
        split_of[p] = "train" if i < cut1 else "val" if i < cut2 else "test"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in counts}
    for r in rows:
        s = split_of[r["patient"]]
        files[s].write(json.dumps(r) + "\n")
        counts[s] += 1
    for f in files.values():
        f.close()
    n_img = sum(len(r["images"]) for r in rows)
    print(f"{n_img} images in {len(rows)} studies, {n} patients -> {counts} (studies)")


if __name__ == "__main__":
    main()
