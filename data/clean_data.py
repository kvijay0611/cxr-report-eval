"""Turn the old instruction_data.json (Open-I conversations) into clean
train/val/test JSONL files, split BY PATIENT so the same person never
appears in two splits.

Usage:
    python data/clean_data.py --src data/instruction_data.json --out data/processed
"""
import argparse
import json
import random
import re
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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
    seen, rows = set(), []
    for item in raw:
        name = image_name(item["image"])
        if name in seen:
            continue
        findings, impression = parse_item(item)
        if not findings and not impression:
            continue
        seen.add(name)
        rows.append({
            "id": name,
            "patient": patient_id(name),
            "image": name,
            "findings": findings,
            "impression": impression,
            "reference": norm(f"{findings} {impression}"),
        })

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
    print(f"{len(rows)} images, {n} patients -> {counts}")


if __name__ == "__main__":
    main()
