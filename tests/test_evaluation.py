import pandas as pd

from src.data_processing import LEGAL_FEATURES, LEGAL_TARGET, generate_legal_dataset
from src.evaluation.methods import METHODS, nearest_unlike_neighbor, wachter_search
from src.evaluation.metrics import evaluate_counterfactuals
from src.evaluation.runner import confidence_interval
from src.evaluation.robustness import changed_feature_jaccard, perturb_record
from src.model_training import train_model


def test_evaluation_methods_are_defined():
    assert set(METHODS) == {
        "dice_random", "dice_genetic", "dice_kdtree",
        "wachter_search", "nearest_unlike_neighbor",
    }


def test_evaluation_baselines_return_valid_frame_and_metrics():
    data = generate_legal_dataset(600, random_state=31)
    result = train_model(data, LEGAL_FEATURES, LEGAL_TARGET, "Random Forest", 31)
    record = result["x_test"].iloc[0].to_dict()

    for method in (nearest_unlike_neighbor, wachter_search):
        counterfactuals = method(result, record, total_cfs=3, seed=31)
        assert isinstance(counterfactuals, pd.DataFrame)
        metrics = evaluate_counterfactuals(result, record, counterfactuals, 3)
        assert 0 <= metrics["validity"] <= 1
        assert 0 <= metrics["proximity"] <= 1
        assert 0 <= metrics["diversity"] <= 1
        assert 0 <= metrics["plausibility"] <= 1
        assert 0 <= metrics["feasibility_rate"] <= 1


def test_confidence_interval_is_ordered():
    low, high = confidence_interval([0.5, 0.6, 0.7, 0.8, 0.9])
    assert low <= 0.7 <= high


def test_perturb_record_changes_supported_features():
    record = {"age": 30, "prior_offenses": 2, "employment_status": "employed"}
    perturbed = perturb_record(record, ["age", "prior_offenses"])
    assert perturbed["age"] in (29, 31)
    assert perturbed["prior_offenses"] in (1, 3)
    assert perturbed["employment_status"] == "employed"


def test_changed_feature_jaccard_is_bounded():
    original = {"age": 30, "prior_offenses": 0}
    left = pd.DataFrame([{"age": 31, "prior_offenses": 0}])
    right = pd.DataFrame([{"age": 31, "prior_offenses": 1}])
    value = changed_feature_jaccard(original, left, right, ["age", "prior_offenses"])
    assert 0.0 <= value <= 1.0
