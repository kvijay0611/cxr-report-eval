"""Rule-based finding extractor with simple negation handling.

This is a deliberately small v0 so everything runs on a CPU laptop.
Later you can swap in CheXbert or RadGraph and compare against it.
"""
import re

LABELS = {
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement",
                     "heart is enlarged", "heart size is enlarged",
                     "cardiac silhouette is enlarged", "enlarged cardiac silhouette"],
    "pleural_effusion": ["effusion", "effusions"],
    "pneumothorax": ["pneumothorax", "pneumothoraces"],
    "consolidation": ["consolidation", "pneumonia", "airspace disease",
                      "airspace opacity", "infiltrate", "infiltrates"],
    "edema": ["edema", "pulmonary congestion"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "nodule_mass": ["nodule", "nodules", "mass", "masses"],
    "fracture": ["fracture", "fractures"],
}

NEG_BEFORE = re.compile(
    r"\b(no|without|negative for|free of|absence of|absent|clear of|"
    r"not|resolved|resolving|neither|nor)\b"
)
NEG_AFTER = re.compile(r"\b(is|are|was|were)?\s*(absent|resolved|not seen|not identified)\b")
SPLIT = re.compile(r"[.;\n]|\bbut\b|\bhowever\b")


def extract_findings(text: str) -> set:
    present = set()
    for clause in SPLIT.split((text or "").lower()):
        for label, words in LABELS.items():
            for w in words:
                for m in re.finditer(rf"\b{re.escape(w)}\b", clause):
                    before, after = clause[:m.start()], clause[m.end():]
                    if NEG_BEFORE.search(before) or NEG_AFTER.match(after.strip()):
                        continue
                    present.add(label)
    return present
