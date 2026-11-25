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


def test_prominent_heart_counts_as_cardiomegaly():
    assert extract_findings("Heart appears prominent.") == {"cardiomegaly"}
    assert extract_findings("The heart is again mildly enlarged.") == {"cardiomegaly"}


def test_negated_enlargement():
    assert extract_findings("The heart is not enlarged.") == set()


def test_no_leak_across_comma():
    assert extract_findings("Heart size normal, pulmonary vessels prominent.") == set()


def test_hedged_differential_is_not_definite():
    from metrics.findings import extract_with_uncertainty
    t = "Possibly a tortuous aorta versus an anterior mediastinal mass."
    definite, uncertain = extract_with_uncertainty(t)
    assert definite == set() and uncertain == {"nodule_mass"}
