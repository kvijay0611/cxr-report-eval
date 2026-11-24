from .findings import LABELS, extract_findings


def _div(a, b):
    return a / b if b else 0.0


def score(pairs):
    """pairs: list of (reference_text, predicted_text). Returns a dict of metrics."""
    stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in LABELS}
    any_fp = any_fn = abnormal_refs = normal_correct = binary_correct = 0
    for ref, pred in pairs:
        r, p = extract_findings(ref), extract_findings(pred)
        if r:
            abnormal_refs += 1
        if not r and not p:
            normal_correct += 1
        binary_correct += bool(r) == bool(p)
        if p - r:
            any_fp += 1
        if r and (r - p):
            any_fn += 1
        for l in LABELS:
            stats[l]["tp"] += l in r and l in p
            stats[l]["fp"] += l in p and l not in r
            stats[l]["fn"] += l in r and l not in p

    per_label = {}
    for l, s in stats.items():
        prec, rec = _div(s["tp"], s["tp"] + s["fp"]), _div(s["tp"], s["tp"] + s["fn"])
        per_label[l] = {**s, "support": s["tp"] + s["fn"],
                        "precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(_div(2 * prec * rec, prec + rec), 3)}
    n = len(pairs)
    return {
        "n_reports": n,
        "reference_abnormal_rate": round(_div(abnormal_refs, n), 3),
        "abnormal_detection_accuracy": round(_div(binary_correct, n), 3),
        "normal_agreement_rate": round(_div(normal_correct, n - abnormal_refs), 3),
        "reports_with_hallucinated_finding": round(_div(any_fp, n), 3),
        "abnormal_reports_with_missed_finding": round(_div(any_fn, abnormal_refs), 3),
        "per_label": per_label,
    }
