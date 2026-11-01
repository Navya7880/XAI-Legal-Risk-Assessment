import pandas as pd
import pytest

from src.data_processing import (
    DATASET_CONFIG,
    HIRING_FEATURES,
    HIRING_SENSITIVE,
    HIRING_TARGET,
    generate_hiring_dataset,
)
from src.fairness_audit import (
    audit_fairness,
    bootstrap_fairness_intervals,
    mitigate_fairness,
)
from src.model_training import train_model


def test_fairlearn_metrics_match_hand_computed_values():
    truth = pd.Series(["Yes", "Yes", "No", "No", "Yes", "No"])
    prediction = pd.Series(["Yes", "No", "Yes", "No", "Yes", "Yes"])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    audit = audit_fairness(truth, prediction, groups)
    assert audit["by_group"].loc["A", "Selection rate"] == pytest.approx(2 / 3)
    assert audit["by_group"].loc["A", "True positive rate"] == pytest.approx(0.5)
    assert audit["by_group"].loc["A", "False positive rate"] == pytest.approx(1.0)
    assert audit["by_group"].loc["B", "Selection rate"] == pytest.approx(2 / 3)
    assert audit["by_group"].loc["B", "True positive rate"] == pytest.approx(1.0)
    assert audit["by_group"].loc["B", "False positive rate"] == pytest.approx(0.5)
    assert audit["demographic_parity_difference"] == pytest.approx(0.0)
    assert audit["equalized_odds_difference"] == pytest.approx(0.5)


def test_audit_handles_synthetic_hiring_and_correlated_proxy():
    data = generate_hiring_dataset(600)
    config = DATASET_CONFIG["Hiring (synthetic)"]
    result = train_model(data, HIRING_FEATURES, HIRING_TARGET, "Logistic Regression")
    groups = data.loc[result["y_test"].index, HIRING_SENSITIVE]
    audit = audit_fairness(result["y_test"], result["predictions"], groups)
    assert audit["n_audited"] == len(result["y_test"])
    assert 0 <= audit["demographic_parity_difference"] <= 1
    assert 0 <= audit["equalized_odds_difference"] <= 1
    assert HIRING_SENSITIVE not in result["features"]
    proxy_correlation = data["referral_network_score"].corr(
        data[HIRING_SENSITIVE].eq("Group A").astype(int)
    )
    assert proxy_correlation > 0.6
    assert config["feature_actionability"]["referral_network_score"]["actionable"] is False
    assert config["feature_actionability"]["referral_network_score"]["immutable"] is True


@pytest.mark.parametrize("constraint", ["Demographic parity", "Equalized odds"])
def test_threshold_optimizer_mitigation_runs(constraint):
    data = generate_hiring_dataset(500)
    result = train_model(data, HIRING_FEATURES, HIRING_TARGET, "Logistic Regression")
    mitigation = mitigate_fairness(
        result,
        data[HIRING_SENSITIVE],
        constraint,
    )
    groups_test = data.loc[result["y_test"].index, HIRING_SENSITIVE]
    audit = audit_fairness(
        result["y_test"], mitigation["predictions"], groups_test
    )
    assert len(mitigation["predictions"]) == len(result["y_test"])
    assert 0 <= audit["accuracy"] <= 1
    assert 0 <= audit["demographic_parity_difference"] <= 1
    assert 0 <= audit["equalized_odds_difference"] <= 1


def test_bootstrap_intervals_are_reproducible_and_ordered():
    truth = pd.Series(["Yes", "Yes", "No", "No", "Yes", "No"])
    prediction = pd.Series(["Yes", "No", "Yes", "No", "Yes", "Yes"])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    first = bootstrap_fairness_intervals(
        truth, prediction, groups, n_resamples=500, random_state=19
    )
    second = bootstrap_fairness_intervals(
        truth, prediction, groups, n_resamples=500, random_state=19
    )
    assert first == second
    for lower, upper in first["selection_rate"].values():
        assert lower <= upper
    lower, upper = first["demographic_parity_difference"]
    assert lower <= upper


@pytest.mark.parametrize(
    ("truth", "prediction", "groups"),
    [
        ([], [], []),
        (["Yes", "No"], ["Yes", "No"], ["A", "A"]),
    ],
)
def test_audit_rejects_empty_or_single_group_data(truth, prediction, groups):
    with pytest.raises(ValueError, match="at least two groups"):
        audit_fairness(pd.Series(truth), pd.Series(prediction), pd.Series(groups))
