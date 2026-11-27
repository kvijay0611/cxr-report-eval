"""Turn report text into a set of findings, then score and compare.

Representation: a report becomes a SET like {"cardiomegaly", "atelectasis"}.
Comparing two reports = comparing two sets. Everything else is counting.
"""
import re

LABEL_PATTERNS = {
    "cardiomegaly": [
        r"cardiomegaly", r"cardiac enlargement", r"enlarged (?:heart|cardiac)",
        r"(?:heart|cardiac (?:silhouette|contour)s?|cardiomediastinal silhouette)"
        r"[^.,;]{0,30}\b(?:enlarged|prominent|enlargement)\b",
    ],
    "pleural_effusion": [r"effusions?"],
    "pneumothorax": [r"pneumothora(?:x|ces)"],
    "consolidation": [r"consolidation", r"pneumonia", r"airspace disease",
                      r"air ?space opacit(?:y|ies)", r"infiltrates?"],
    "edema": [r"edema", r"pulmonary congestion"],
    "atelectasis": [r"atelecta(?:sis|tic)"],
    "nodule_mass": [r"nodules?", r"masses", r"mass"],
    "fracture": [r"fractures?"],
}
LABELS = list(LABEL_PATTERNS)
_COMPILED = {l: [re.compile(rf"\b(?:{p})\b") for p in ps] for l, ps in LABEL_PATTERNS.items()}

NEGATION = re.compile(r"\b(no|without|negative for|free of|absence of|absent|clear of|"
                      r"not|resolved|resolving|neither|nor)\b")
NEGATED_AFTER = re.compile(r"^\s*(?:is|are|was|were)?\s*(?:absent|resolved|not seen|not identified)\b")
HEDGE = re.compile(r"\b(possibly|possible|versus|cannot (?:be )?exclude[d]?|questionable|"
                   r"may represent|may be|could be|could represent|might)\b")
CLAUSES = re.compile(r"[.;\n]|\bbut\b|\bhowever\b")


def extract_findings(text):
    """Findings the report states definitely. Negated ('no effusion') and hedged
    ('possibly a mass') mentions do not count."""
    found = set()
    for clause in CLAUSES.split((text or "").lower()):
        for label, patterns in _COMPILED.items():
            for pat in patterns:
                for m in pat.finditer(clause):
                    before, after = clause[:m.start()], clause[m.end():]
                    if NEGATION.search(before) or NEGATION.search(m.group(0)) or NEGATED_AFTER.match(after):
                        continue
                    if HEDGE.search(before) or HEDGE.search(m.group(0)):
                        continue
                    found.add(label)
    return found


def _div(a, b):
    return a / b if b else 0.0


def score(pairs):
    """pairs = [(reference_text, model_text), ...] -> dict of metrics."""
    stats = {l: {"tp": 0, "fp": 0, "fn": 0} for l in LABELS}
    any_fp = any_fn = abnormal = normal_ok = binary_ok = 0
    for ref, pred in pairs:
        r, p = extract_findings(ref), extract_findings(pred)
        abnormal += bool(r)
        normal_ok += (not r and not p)
        binary_ok += bool(r) == bool(p)
        any_fp += bool(p - r)
        any_fn += bool(r - p)
        for l in LABELS:
            stats[l]["tp"] += l in r and l in p
            stats[l]["fp"] += l in p and l not in r
            stats[l]["fn"] += l in r and l not in p
    per_label = {}
    for l, s in stats.items():
        prec, rec = _div(s["tp"], s["tp"] + s["fp"]), _div(s["tp"], s["tp"] + s["fn"])
        per_label[l] = {**s, "support": s["tp"] + s["fn"], "precision": round(prec, 3),
                        "recall": round(rec, 3), "f1": round(_div(2 * prec * rec, prec + rec), 3)}
    n = len(pairs)
    return {
        "n_reports": n,
        "reference_abnormal_rate": round(_div(abnormal, n), 3),
        "abnormal_detection_accuracy": round(_div(binary_ok, n), 3),
        "normal_agreement_rate": round(_div(normal_ok, n - abnormal), 3),
        "reports_with_hallucinated_finding": round(_div(any_fp, n), 3),
        "abnormal_reports_with_missed_finding": round(_div(any_fn, abnormal), 3),
        "per_label": per_label,
    }


def compare(base, pert, refs):
    """How did answers change after a perturbation?
    base/pert: id -> {"text", "needs_more_info"};  refs: id -> reference text."""
    ids = sorted(set(base) & set(pert))
    n = len(ids)
    if not n:
        return {"n": 0}
    changed = added = dropped = abstain = any_find = 0
    claim_b = dict.fromkeys(LABELS, 0)
    claim_p = dict.fromkeys(LABELS, 0)
    eligible = dict.fromkeys(LABELS, 0)      # studies whose reference lacks the label
    for i in ids:
        b, p, r = (extract_findings(base[i]["text"]), extract_findings(pert[i]["text"]),
                   extract_findings(refs[i]))
        changed += b != p
        added += len(p - b)
        dropped += len(b - p)
        abstain += bool(pert[i].get("needs_more_info"))
        any_find += bool(p)
        for l in LABELS:
            if l not in r:
                eligible[l] += 1
                claim_b[l] += l in b
                claim_p[l] += l in p
    false_claims = {}
    for l in LABELS:
        if eligible[l]:
            fb, fp = claim_b[l] / eligible[l], claim_p[l] / eligible[l]
            false_claims[l] = {"base": round(fb, 3), "perturbed": round(fp, 3),
                               "change": round(fp - fb, 3)}
    return {"n": n, "finding_change_rate": round(changed / n, 3),
            "findings_added_per_study": round(added / n, 3),
            "findings_dropped_per_study": round(dropped / n, 3),
            "abstain_rate": round(abstain / n, 3), "any_finding_rate": round(any_find / n, 3),
            "false_claims_by_label": false_claims}
