from data.clean_data import clean_text


def test_nan_becomes_empty():
    assert clean_text("nan") == ""
    assert clean_text(None) == ""


def test_html_entities_and_artifacts():
    assert clean_text("Pulmonary vascularity is normal.&gt;] Lung volumes low.") == \
        "Pulmonary vascularity is normal. Lung volumes low."
