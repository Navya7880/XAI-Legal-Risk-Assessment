from io import BytesIO

import pandas as pd
from pypdf import PdfReader

from src.compliance_reporting import generate_pdf_report
from src.data_processing import (
    DATASET_CONFIG,
    HIRING_FEATURES,
    HIRING_SENSITIVE,
    HIRING_TARGET,
    LEGAL_FEATURES,
    LEGAL_TARGET,
    generate_hiring_dataset,
    generate_legal_dataset,
)
from src.dice_explainer import counterfactual_quality_metrics, generate_counterfactuals
from src.fairness_audit import audit_fairness, mitigate_fairness
from src.model_training import train_model


def _read_report(**kwargs):
    pdf = generate_pdf_report(**kwargs)
    reader = PdfReader(BytesIO(pdf))
    text = " ".join(
        (page.extract_text() or "") for page in reader.pages
    ).split()
    return " ".join(text), reader


def test_pdf_report_contains_required_sections_without_session_artifacts():
    data = generate_legal_dataset(120)
    config = DATASET_CONFIG["Legal risk (synthetic)"]
    model_result = train_model(
        data, LEGAL_FEATURES, LEGAL_TARGET, "Logistic Regression"
    )
    text, reader = _read_report(
        dataset_name="Legal risk (synthetic)",
        model_name=model_result["model_name"],
        metrics=model_result["metrics"],
        dataset_data=data,
        target_column=config["target"],
        feature_columns=config["features"],
        model_result=model_result,
    )
    assert len(reader.pages) >= 1
    for heading in (
        "Study overview",
        "Dataset information",
        "Model details",
        "Evaluation results",
        "DiCE results",
        "Fairness results",
        "Transparency checklist",
        "Risks and limitations",
        "Human-oversight recommendations",
    ):
        assert heading in text
    assert "does not certify GDPR" in text
    assert "No DiCE run was recorded" in text
    assert "no fairness audit exists" in text.lower()
    assert "Confusion matrix" in text


def test_pdf_report_handles_wide_counterfactuals_and_fairness_artifacts():
    data = generate_hiring_dataset(700, random_state=37)
    config = DATASET_CONFIG["Hiring (synthetic)"]
    model_result = train_model(
        data, HIRING_FEATURES, HIRING_TARGET, "Random Forest"
    )
    groups = data[HIRING_SENSITIVE]
    test_groups = groups.loc[model_result["y_test"].index]
    before = audit_fairness(
        model_result["y_test"], model_result["predictions"], test_groups
    )
    mitigation = mitigate_fairness(
        model_result, groups, "Demographic parity"
    )
    audit = audit_fairness(
        model_result["y_test"], mitigation["predictions"], test_groups
    )
    reference_result = train_model(
        data,
        HIRING_FEATURES + [HIRING_SENSITIVE],
        HIRING_TARGET,
        "Random Forest",
    )
    reference_groups = groups.loc[reference_result["y_test"].index]
    reference_audit = audit_fairness(
        reference_result["y_test"],
        reference_result["predictions"],
        reference_groups,
    )
    comparison = pd.DataFrame(
        [
            {
                "Model": "Before mitigation",
                "Accuracy": before["accuracy"],
                "Demographic parity difference": before[
                    "demographic_parity_difference"
                ],
                "Equalized odds difference": before["equalized_odds_difference"],
            },
            {
                "Model": "After mitigation",
                "Accuracy": audit["accuracy"],
                "Demographic parity difference": audit[
                    "demographic_parity_difference"
                ],
                "Equalized odds difference": audit["equalized_odds_difference"],
            },
        ]
    )
    reference_comparison = pd.DataFrame(
        [
            {
                "Model": "Main, group excluded",
                "Accuracy": before["accuracy"],
                "Demographic parity difference": before[
                    "demographic_parity_difference"
                ],
                "Equalized odds difference": before["equalized_odds_difference"],
            },
            {
                "Model": "Reference, group included",
                "Accuracy": reference_audit["accuracy"],
                "Demographic parity difference": reference_audit[
                    "demographic_parity_difference"
                ],
                "Equalized odds difference": reference_audit[
                    "equalized_odds_difference"
                ],
            },
        ]
    )
    fairness = {
        "comparison": comparison,
        "after": audit,
        "constraint": "Demographic parity",
        "reference_comparison": reference_comparison,
    }
    record = data.iloc[0][config["features"]].to_dict()
    counterfactuals = generate_counterfactuals(
        model_result, record, total_cfs=1, method="random"
    )
    quality = counterfactual_quality_metrics(
        model_result, record, counterfactuals, 1
    )
    text, reader = _read_report(
        dataset_name="Hiring (synthetic)",
        model_name=model_result["model_name"],
        metrics=model_result["metrics"],
        fairness=fairness,
        prediction={
            "prediction": str(model_result["pipeline"].predict(
                pd.DataFrame([record])
            )[0])
        },
        counterfactuals=counterfactuals,
        dataset_data=data,
        target_column=config["target"],
        feature_columns=config["features"],
        model_result=model_result,
        counterfactual_quality=quality,
        counterfactual_run=True,
    )
    assert len(reader.pages) >= 1
    assert "referral_network_score" in text
    assert "Proximity" in text
    assert "Before mitigation" in text
    assert "Reference model including group attribute" in text
    assert "Selection rate CI lower" in text
    assert "Group A" in text
