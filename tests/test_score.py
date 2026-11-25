from metrics.score import score


def test_constant_normal_scores_zero_recall_but_high_binary_accuracy():
    pairs = [("No acute disease.", "No acute disease."),
             ("No acute disease.", "No acute disease."),
             ("Small pleural effusion.", "No acute disease."),
             ("No acute disease.", "No acute disease.")]
    r = score(pairs)
    assert r["abnormal_detection_accuracy"] == 0.75
    assert r["per_label"]["pleural_effusion"]["recall"] == 0.0
    assert r["reports_with_hallucinated_finding"] == 0.0


def test_hallucination_counted():
    r = score([("Clear lungs.", "Right pneumothorax.")])
    assert r["reports_with_hallucinated_finding"] == 1.0
    assert r["per_label"]["pneumothorax"]["fp"] == 1
