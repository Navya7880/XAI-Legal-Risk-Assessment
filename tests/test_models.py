import pandas as pd
import pytest

from src.data_processing import (
    DATASET_CONFIG,
    LEGAL_FEATURES,
    LEGAL_TARGET,
    generate_hiring_dataset,
    generate_legal_dataset,
    load_public_legal_dataset,
)
from src.model_training import train_model
from src.model_training import (
    cross_validate_model,
    load_model,
    save_model,
)
from src.prediction import predict_record


@pytest.mark.parametrize(
    ("name", "generator"),
    [
        ("Legal risk (synthetic)", generate_legal_dataset),
        ("Hiring (synthetic)", generate_hiring_dataset),
    ],
)
def test_training_and_prediction(name, generator):
    config = DATASET_CONFIG[name]
    data = generator(500)
    result = train_model(data, config["features"], config["target"], "Logistic Regression")
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["metrics"]["f1"] > 0
    assert len(result["metrics"]["confusion_matrix"]) == 2
    record = data.iloc[0][config["features"]].to_dict()
    assert predict_record(result, record)["prediction"] in result["metrics"]["labels"]


def test_training_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        train_model(pd.DataFrame({"x": [1, 2]}), ["x"], "target")


def test_training_rejects_single_class_and_invalid_columns():
    data = generate_legal_dataset(200)
    data["single_target"] = "only"
    with pytest.raises(ValueError, match="at least two target classes"):
        train_model(data, LEGAL_FEATURES, "single_target")
    with pytest.raises(ValueError, match="Missing required dataset columns"):
        train_model(data, ["not_a_column"], LEGAL_TARGET)


def test_preprocessing_is_fitted_only_on_training_rows():
    data = generate_legal_dataset(500)
    result = train_model(data, LEGAL_FEATURES, LEGAL_TARGET, "Logistic Regression")
    preprocessor = result["pipeline"].named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    train_count = len(result["training_data"]) - len(result["x_test"])
    assert scaler.n_samples_seen_ == train_count


def test_model_save_load_round_trip(tmp_path):
    data = generate_legal_dataset(400)
    result = train_model(data, LEGAL_FEATURES, LEGAL_TARGET, "Logistic Regression")
    path = save_model(result, tmp_path / "roundtrip.joblib")
    loaded = load_model(path)
    pd.testing.assert_series_equal(
        pd.Series(loaded["pipeline"].predict(result["x_test"]), index=result["x_test"].index),
        pd.Series(result["pipeline"].predict(result["x_test"]), index=result["x_test"].index),
    )


def test_five_fold_cross_validation_returns_required_summaries():
    data = generate_legal_dataset(300)
    results = cross_validate_model(
        data, LEGAL_FEATURES, LEGAL_TARGET, "Logistic Regression"
    )
    assert set(results) == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= metric["mean"] <= 1 for metric in results.values())
    assert all(metric["std"] >= 0 for metric in results.values())


def test_public_legal_dataset_uses_explicit_column_mapping(tmp_path):
    source = pd.DataFrame(
        {
            "age_years": [30, 40],
            "priors_count": [1, 2],
            "charge": ["F", "M"],
            "work_status": ["employed", "student"],
            "outcome": [0, 1],
        }
    )
    path = tmp_path / "public.csv"
    source.to_csv(path, index=False)
    mapping = {
        "age": "age_years",
        "prior_offenses": "priors_count",
        "offense_category": "charge",
        "employment_status": "work_status",
        "risk_category": "outcome",
    }
    loaded = load_public_legal_dataset(path, mapping)
    assert list(loaded.columns) == [
        "age",
        "prior_offenses",
        "offense_category",
        "employment_status",
        "risk_category",
    ]
    assert len(loaded) == 2
