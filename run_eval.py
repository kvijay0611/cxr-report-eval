"""Run one client over a split and score it.

Examples:
    python run_eval.py --client constant --split test
    python run_eval.py --client openai --model my-model --base-url http://localhost:8000/v1 \
        --image-root data/raw/images --limit 100
"""
import argparse
import json
from pathlib import Path

from clients.base import TransientModelError
from metrics.score import score


def make_client(args):
    if args.client == "constant":
        from clients.baselines import ConstantNormalClient
        return ConstantNormalClient()
    from clients.openai_compat import OpenAICompatClient
    return OpenAICompatClient(model=args.model, base_url=args.base_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=["constant", "openai"], default="constant")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-root", default="data/raw/images")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(args.data) / f"{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    client = make_client(args)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{client.name.replace('/', '_')}_{args.split}"

    pairs, failed = [], 0
    with open(out_dir / f"{tag}_outputs.jsonl", "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            imgs = [Path(args.image_root) / n for n in row["images"]]
            try:
                rep = client.generate(imgs if args.client != "constant" else None)
            except TransientModelError as e:
                failed += 1
                print(f"  skipped {row['id']}: {e}")
                f.write(json.dumps({"id": row["id"], "error": str(e)}) + "\n")
                continue
            pairs.append((row["reference"], rep.text))
            f.write(json.dumps({"id": row["id"], "n_images": len(row["images"]), "reference": row["reference"],
                                "prediction": rep.text,
                                "needs_more_info": rep.needs_more_info}) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}")

    result = score(pairs)
    json.dump(result, open(out_dir / f"{tag}_scores.json", "w"), indent=2)
    print(f"\n{client.name} on {args.split}: {result['n_reports']} reports scored, {failed} failed (excluded)")
    for k in ("reference_abnormal_rate", "abnormal_detection_accuracy", "normal_agreement_rate",
              "reports_with_hallucinated_finding", "abnormal_reports_with_missed_finding"):
        print(f"  {k}: {result[k]}")
    print("  per finding (support / precision / recall):")
    for l, s in result["per_label"].items():
        print(f"    {l:<18} {s['support']:>4}  {s['precision']:.2f}  {s['recall']:.2f}")


if __name__ == "__main__":
    main()