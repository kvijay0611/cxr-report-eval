"""Run the red-team suite: same studies, perturbed inputs, compare to baseline.

Example (no model needed, just a plumbing check):
    python run_redteam.py --client constant --image-root data/raw/images --limit 20

Example (real model behind an OpenAI-compatible server):
    python run_redteam.py --client openai --model MODEL --base-url URL \
        --image-root data/raw/images --limit 100
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

from PIL import Image

from redteam.analysis import compare
from redteam.perturbations import PERTURBATIONS, SHOULD_ABSTAIN
from run_eval import make_client


def load_rows(data_dir, splits):
    rows = []
    for s in splits:
        rows += [json.loads(l) for l in open(Path(data_dir) / f"{s}.jsonl", encoding="utf-8")]
    return rows


def read_done(path):
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            done[d["id"]] = d
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=["constant", "openai"], default="constant")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--image-root", default="data/raw/images")
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    ap.add_argument("--limit", type=int, default=100, help="number of studies (keeps cost down)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if "baseline" not in args.perturbations:
        args.perturbations = ["baseline"] + args.perturbations
    rows = load_rows(args.data, args.splits)
    rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    refs = {r["id"]: r["reference"] for r in rows}

    client = make_client(args)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = client.name.replace("/", "_")

    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for pname in args.perturbations:
            fn = PERTURBATIONS[pname]
            path = out_dir / f"redteam_{tag}_{pname}.jsonl"
            done = read_done(path)          # resume after a crash
            with open(path, "a", encoding="utf-8") as f:
                for i, row in enumerate(rows, 1):
                    if row["id"] in done:
                        continue
                    rng = random.Random(f"{args.seed}-{pname}-{row['id']}")
                    imgs = [Image.open(Path(args.image_root) / n) for n in row["images"]]
                    imgs, history = fn(imgs, rng)
                    paths = []
                    for k, im in enumerate(imgs):
                        p = Path(tmp) / f"{pname}_{row['id']}_{k}.png"
                        im.save(p)
                        paths.append(p)
                    rep = client.generate(paths, history)
                    rec = {"id": row["id"], "text": rep.text,
                           "needs_more_info": rep.needs_more_info,
                           "missing_info": rep.missing_info}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[row["id"]] = rec
                    if i % 10 == 0:
                        print(f"  {pname}: {i}/{len(rows)}")
            runs[pname] = done

    summary = {p: compare(runs["baseline"], runs[p], refs)
               for p in args.perturbations if p != "baseline"}
    json.dump(summary, open(out_dir / f"redteam_{tag}_summary.json", "w"), indent=2)

    print(f"\nRed-team summary: {client.name}, {len(rows)} studies")
    print(f"{'perturbation':<16}{'changed':>9}{'added':>8}{'dropped':>9}{'abstain':>9}{'any_find':>10}")
    for p, s in summary.items():
        print(f"{p:<16}{s['finding_change_rate']:>9}{s['findings_added_per_study']:>8}"
              f"{s['findings_dropped_per_study']:>9}{s['abstain_rate']:>9}{s['any_finding_rate']:>10}")
    for p in summary:
        if p in SHOULD_ABSTAIN:
            s = summary[p]
            print(f"\n{p}: should abstain and report no findings. "
                  f"abstained={s['abstain_rate']}, invented findings in {s['any_finding_rate']} of reports")
        if p.startswith("hist_"):
            fc = summary[p]["false_claims_by_label"]
            top = sorted(fc.items(), key=lambda kv: -kv[1]["change"])[:3]
            print(f"\n{p}: biggest rise in false claims (finding stated but not in reference):")
            for l, v in top:
                print(f"    {l:<18} base {v['base']}  ->  with history {v['perturbed']}")


if __name__ == "__main__":
    main()
