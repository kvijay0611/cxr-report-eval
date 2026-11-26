"""Compare a perturbed run against the unperturbed baseline run, per study."""
from metrics.findings import LABELS, extract_findings


def _r(x, nd=3):
    return round(x, nd)


def compare(base: dict, pert: dict, refs: dict) -> dict:
    """base / pert: id -> {"text": str, "needs_more_info": bool}; refs: id -> reference text."""
    ids = sorted(set(base) & set(pert))
    n = len(ids)
    if n == 0:
        return {"n": 0}
    changed = added = dropped = abstain = base_abstain = any_finding = 0
    claims_base = {l: 0 for l in LABELS}
    claims_pert = {l: 0 for l in LABELS}
    eligible = {l: 0 for l in LABELS}   # studies whose reference lacks the label
    for i in ids:
        b, p = extract_findings(base[i]["text"]), extract_findings(pert[i]["text"])
        r = extract_findings(refs[i])
        changed += b != p
        added += len(p - b)
        dropped += len(b - p)
        abstain += bool(pert[i].get("needs_more_info"))
        base_abstain += bool(base[i].get("needs_more_info"))
        any_finding += bool(p)
        for l in LABELS:
            if l not in r:
                eligible[l] += 1
                claims_base[l] += l in b
                claims_pert[l] += l in p
    false_claims = {}
    for l in LABELS:
        if eligible[l]:
            fb, fp = claims_base[l] / eligible[l], claims_pert[l] / eligible[l]
            false_claims[l] = {"base": _r(fb), "perturbed": _r(fp), "change": _r(fp - fb)}
    return {
        "n": n,
        "finding_change_rate": _r(changed / n),
        "findings_added_per_study": _r(added / n),
        "findings_dropped_per_study": _r(dropped / n),
        "abstain_rate": _r(abstain / n),
        "baseline_abstain_rate": _r(base_abstain / n),
        "any_finding_rate": _r(any_finding / n),
        "false_claims_by_label": false_claims,
    }
