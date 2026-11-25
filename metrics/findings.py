"""Rule-based finding extractor with negation and hedging handling.

Deliberately small so it runs on a CPU laptop. Later, swap in CheXbert or
RadGraph and compare. Every label definition here is a judgement call:
keep them written down in the README.

Returns DEFINITE findings only. Hedged mentions ("possibly", "versus",
"cannot exclude" ...) are returned separately by extract_with_uncertainty().
"""
import re
from typing import Set, Tuple

# word-boundary regexes; phrases are matched inside one clause only
LABEL_PATTERNS = {
    "cardiomegaly": [
        r"cardiomegaly",
        r"cardiac enlargement",
        r"enlarged (?:heart|cardiac)",
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

NEG_BEFORE = re.compile(
    r"\b(no|without|negative for|free of|absence of|absent|clear of|"
    r"not|resolved|resolving|neither|nor)\b"
)
NEG_AFTER = re.compile(r"^\s*(?:is|are|was|were)?\s*(?:absent|resolved|not seen|not identified)\b")
HEDGE = re.compile(
    r"\b(possibly|possible|versus|cannot (?:be )?exclude[d]?|questionable|"
    r"may represent|may be|could be|could represent|might)\b"
)
SPLIT = re.compile(r"[.;\n]|\bbut\b|\bhowever\b")


def extract_with_uncertainty(text: str) -> Tuple[Set[str], Set[str]]:
    definite, uncertain = set(), set()
    for clause in SPLIT.split((text or "").lower()):
        for label, patterns in _COMPILED.items():
            for pat in patterns:
                for m in pat.finditer(clause):
                    before, after = clause[:m.start()], clause[m.end():]
                    # negation can sit before the match or inside it ("heart is not enlarged")
                    if NEG_BEFORE.search(before) or NEG_BEFORE.search(m.group(0)) \
                            or NEG_AFTER.match(after):
                        continue
                    if HEDGE.search(before) or HEDGE.search(m.group(0)):
                        uncertain.add(label)
                    else:
                        definite.add(label)
    return definite, uncertain - definite


def extract_findings(text: str) -> Set[str]:
    return extract_with_uncertainty(text)[0]
