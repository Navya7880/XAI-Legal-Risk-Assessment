import pandas as pd
import pytest

from src.data_processing import (
    DATASET_CONFIG,
    HIRING_SENSITIVE,
    HIRING_TARGET,
    LEGAL_FEATURES,
    LEGAL_TARGET,
    generate_hiring_dataset,
    generate_legal_dataset,
)
from src.dice_explainer import (
    FeasibilityChecker,
    counterfactual_quality_metrics,
    cost_weighted_proximity,
    generate_counterfactuals,
)
from src.model_training import train_model
from src.prediction import predict_record


@pytest.fixture(scope="module")
def legal_model_and_data():
    data = generate_legal_dataset(700, random_state=31)
    result = train_model(data, LEGAL_FEATURES, LEGAL_TARGET, "Random Forest")
    return result, data


@pytest.fixture(scope="module")
def hiring_model_and_data():
    config = DATASET_CONFIG["Hiring (synthetic)"]
    data = generate_hiring_dataset(700, random_state=37)
    result = train_model(data, config["features"], config["target"], "Random Forest")
    return result, data


def test_dice_returns_only_verified_counterfactuals(legal_model_and_data):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    counterfactuals = generate_counterfactuals(result, record, total_cfs=2)
    assert counterfactuals.attrs["method_used"] in {"genetic", "random"}
    original = predict_record(result, record)["prediction"]
    for _, counterfactual in counterfactuals.iterrows():
        cf_record = counterfactual[LEGAL_FEATURES].to_dict()
        assert predict_record(result, cf_record)["prediction"] != original


def test_legal_constraints_and_integer_features(legal_model_and_data):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    counterfactuals = generate_counterfactuals(
        result, record, total_cfs=2, method="random"
    )
    observed_employment = set(data["employment_status"].unique())
    for _, counterfactual in counterfactuals.iterrows():
        assert counterfactual["age"] == record["age"]
        assert counterfactual["offense_category"] == record["offense_category"]
        assert isinstance(counterfactual["age"], int)
        assert isinstance(counterfactual["prior_offenses"], int)
        assert 0 <= counterfactual["prior_offenses"] <= 6
        assert counterfactual["employment_status"] in observed_employment


def test_empty_vary_list_returns_empty_dataframe(legal_model_and_data):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    counterfactuals = generate_counterfactuals(
        result, record, features_to_vary=[], method="random"
    )
    assert isinstance(counterfactuals, pd.DataFrame)
    assert counterfactuals.empty


def test_immutable_features_cannot_be_requested_to_vary(legal_model_and_data):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    with pytest.raises(ValueError, match="immutable features"):
        generate_counterfactuals(
            result,
            record,
            features_to_vary=["age"],
            method="random",
        )


def test_quality_metrics_are_in_valid_ranges(legal_model_and_data):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    counterfactuals = generate_counterfactuals(
        result, record, total_cfs=2, method="random"
    )
    metrics = counterfactual_quality_metrics(result, record, counterfactuals, 2)
    assert 0 <= metrics["validity"] <= 1
    assert 0 <= metrics["proximity"] <= 1
    assert metrics["sparsity"] >= 0
    assert 0 <= metrics["diversity"] <= 1


def test_hiring_group_is_not_a_model_or_changed_feature(hiring_model_and_data):
    result, data = hiring_model_and_data
    config = DATASET_CONFIG["Hiring (synthetic)"]
    record = data.iloc[0][config["features"]].to_dict()
    assert HIRING_SENSITIVE not in result["features"]
    assert result["target"] == HIRING_TARGET
    counterfactuals = generate_counterfactuals(
        result, record, total_cfs=2, method="random"
    )
    assert HIRING_SENSITIVE not in counterfactuals.columns
    assert not counterfactuals.empty
    original_prediction = predict_record(result, record)["prediction"]
    for _, counterfactual in counterfactuals.iterrows():
        assert HIRING_SENSITIVE not in counterfactual.index
        cf_record = counterfactual[config["features"]].to_dict()
        assert predict_record(result, cf_record)["prediction"] != original_prediction
        assert all(
            counterfactual[feature] == record[feature]
            for feature in config["immutable_features"]
            if feature in record
        )
        for feature, allowed in config["permitted_range"].items():
            if allowed == "observed":
                assert counterfactual[feature] in set(data[feature].unique())
            else:
                assert allowed[0] <= counterfactual[feature] <= allowed[1]


@pytest.mark.parametrize("method", ["random", "genetic", "kdtree"])
def test_supported_methods_or_visible_random_fallback(legal_model_and_data, method):
    result, data = legal_model_and_data
    record = data.iloc[0][LEGAL_FEATURES].to_dict()
    counterfactuals = generate_counterfactuals(
        result, record, total_cfs=1, method=method
    )
    assert counterfactuals.attrs["method_used"] in {method, "random"}
    if counterfactuals.attrs["method_used"] != method:
        assert counterfactuals.attrs["method_notice"]


def test_monotonic_education_may_not_decrease():
    spec = DATASET_CONFIG["Hiring (synthetic)"]["feasibility_spec"]
    checker = FeasibilityChecker(spec)
    original = {"education": "masters"}
    passed, violations = checker.check(
        {"education": "high_school"},
        original,
    )
    assert not passed
    assert "education_may_only_increase" in violations


def test_monotonic_experience_may_not_decrease():
    spec = DATASET_CONFIG["Hiring (synthetic)"]["feasibility_spec"]
    checker = FeasibilityChecker(spec)
    passed, violations = checker.check(
        {"experience_years": 2},
        {"experience_years": 5},
    )
    assert not passed
    assert "experience_years_may_only_increase" in violations


def test_work_experience_cannot_exceed_age_minus_16():
    spec = {
        "dependency_rules": [
            {
                "name": "experience_cannot_exceed_age_minus_16",
                "type": "work_experience_age",
                "experience_feature": "experience_years",
                "age_feature": "age",
                "minimum_working_age": 16,
            }
        ]
    }
    checker = FeasibilityChecker(spec)
    passed, violations = checker.check(
        {"experience_years": 30, "age": 40},
        {"experience_years": 5, "age": 40},
    )
    assert not passed
    assert "experience_cannot_exceed_age_minus_16" in violations


def test_cost_weighted_proximity_increases_for_expensive_change():
    data = pd.DataFrame({"technical_score": [0, 50, 100]})
    original = {"technical_score": 50}
    small = {"technical_score": 51}
    large = {"technical_score": 100}
    weights = {"technical_score": 10.0}
    small_cost = cost_weighted_proximity(original, small, data, weights)
    large_cost = cost_weighted_proximity(original, large, data, weights)
    assert small_cost < large_cost
    assert 0 <= large_cost <= 1
