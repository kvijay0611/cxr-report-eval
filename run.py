"""One command does everything:  ask the model -> score it -> (optionally) stress-test it.

    python run.py --model constant
    python run.py --model openai --name MODEL --base-url URL --limit 30
    python run.py --model hf --limit 50 --perturb all

First it runs the studies unchanged and scores them against the real reports.
Then, for each --perturb option, it changes the input and shows how the answers change.
Finished studies are saved line by line, so a rerun continues where it stopped.
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

from models import ModelBusy, get_model
from perturb import PERTURBATIONS, SHOULD_ABSTAIN
from scoring import compare, score


def load_done(path):
    if not path.exists():
        return {}
    return {d["id"]: d for d in map(json.loads, open(path, encoding="utf-8")) if "text" in d}


def run_one(ask, rows, perturbation, args, path, tmp):
    """Run every study under one perturbation. Returns id -> answer."""
    from PIL import Image
    done = load_done(path)
    change = PERTURBATIONS[perturbation]
    with open(path, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            paths = [Path(args.image_root) / n for n in row["images"]]
            history = None
            if perturbation != "none":
                rng = random.Random(f"{args.seed}-{perturbation}-{row['id']}")
                imgs, history = change([Image.open(p) for p in paths], rng)
                paths = []
                for k, im in enumerate(imgs):
                    p = Path(tmp) / f"{perturbation}_{row['id']}_{k}.png"
                    im.save(p)
                    paths.append(p)
            try:
                ans = ask(paths, history)
            except ModelBusy as e:
                print(f"  skipped {row['id']}: {e}")
                continue
            rec = {"id": row["id"], "text": ans["text"], "needs_more_info": ans["needs_more_info"]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[row["id"]] = rec
            if i % 10 == 0:
                print(f"  {perturbation}: {i}/{len(rows)}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["constant", "openai", "hf"], default="constant")
    ap.add_argument("--name", default="", help="model id, e.g. gemini-3.8-flash or google/medgemma-4b-it")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--dtype", default="bfloat16", help="hf only: bfloat16 or float16")
    ap.add_argument("--max-images", type=int, default=1, help="hf only: images per study")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--image-root", default="data/raw/images")
    ap.add_argument("--limit", type=int, default=0, help="0 = all studies")
    ap.add_argument("--perturb", nargs="*", default=[], help="names from perturb.py, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = [json.loads(l) for s in args.splits
            for l in open(Path(args.data) / f"{s}.jsonl", encoding="utf-8")]
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    refs = {r["id"]: r["reference"] for r in rows}
    perturbs = [p for p in PERTURBATIONS if p != "none"] if args.perturb == ["all"] else args.perturb

    ask = get_model(args.model, args.name, args.base_url, args.dtype, args.max_images)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = (args.name or args.model).replace("/", "_")

    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in ["none"] + perturbs:
            runs[p] = run_one(ask, rows, p, args, out / f"{tag}_{p}.jsonl", tmp)

    base = runs["none"]
    result = score([(refs[i], base[i]["text"]) for i in base])
    json.dump(result, open(out / f"{tag}_scores.json", "w"), indent=2)
    print(f"\n{tag}: {result['n_reports']} studies scored")
    for k in ("reference_abnormal_rate", "abnormal_detection_accuracy", "normal_agreement_rate",
              "reports_with_hallucinated_finding", "abnormal_reports_with_missed_finding"):
        print(f"  {k}: {result[k]}")
    print("  per finding (support / precision / recall):")
    for l, s in result["per_label"].items():
        print(f"    {l:<18} {s['support']:>4}  {s['precision']:.2f}  {s['recall']:.2f}")

    if perturbs:
        summary = {p: compare(base, runs[p], refs) for p in perturbs}
        json.dump(summary, open(out / f"{tag}_redteam.json", "w"), indent=2)
        print(f"\nStress test (compared with the unchanged run)")
        print(f"{'change':<16}{'answers changed':>16}{'abstained':>11}{'any finding':>13}")
        for p, s in summary.items():
            print(f"{p:<16}{s['finding_change_rate']:>16}{s['abstain_rate']:>11}{s['any_finding_rate']:>13}")
        for p, s in summary.items():
            if p in SHOULD_ABSTAIN:
                print(f"\n{p}: a safe model abstains and reports nothing. "
                      f"abstained={s['abstain_rate']}, invented findings in {s['any_finding_rate']} of reports")
            if p.startswith("hist_"):
                top = sorted(s["false_claims_by_label"].items(), key=lambda kv: -kv[1]["change"])[:3]
                print(f"\n{p}: biggest rise in findings stated but not in the real report:")
                for l, v in top:
                    print(f"    {l:<18} {v['base']} -> {v['perturbed']}")


if __name__ == "__main__":
    main()
