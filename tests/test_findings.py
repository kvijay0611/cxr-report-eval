from metrics.findings import extract_findings


def test_negated_findings_are_ignored():
    t = "There is no pneumothorax. No pleural effusion. The lungs are clear."
    assert extract_findings(t) == set()


def test_positive_findings_detected():
    t = "Small right pleural effusion. Mild cardiomegaly."
    assert extract_findings(t) == {"pleural_effusion", "cardiomegaly"}


def test_mixed_sentence():
    t = "No pneumothorax, but there is a left lower lobe consolidation."
    assert extract_findings(t) == {"consolidation"}


def test_anonymised_token_negation():
    assert extract_findings("There are no XXXX of a pleural effusion.") == set()


def test_none_input():
    assert extract_findings("") == set()
