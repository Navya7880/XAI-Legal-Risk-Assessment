
"""Leakage-conscious model training, evaluation, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_FACTORIES = {
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
    ),

    "Gradient Boosting": lambda seed: GradientBoostingClassifier(
        random_state=seed
    ),

    "Logistic Regression": lambda seed: LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        random_state=seed,
    ),
}


# --------------------------------------------------
# MODEL ARTIFACT PATH
# --------------------------------------------------

def model_artifact_path(dataset_name: str, model_name: str) -> Path:
    """Return a stable repository-local path for a trained model."""

    root = Path(__file__).resolve().parents[1]

    dataset_slug = "_".join(
        dataset_name.lower()
        .replace("(", "")
        .replace(")", "")
        .split()
    )

    model_slug = "_".join(model_name.lower().split())

    return root / "models" / f"{dataset_slug}_{model_slug}.joblib"


# --------------------------------------------------
# PREPROCESSING PIPELINE
# --------------------------------------------------

def _make_pipeline(
    frame: pd.DataFrame,
    model_name: str,
    random_state: int,
) -> Pipeline:

    categorical = frame.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric = [
        column
        for column in frame.columns
        if column not in categorical
    ]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric:
        transformers.append(
            ("numeric", numeric_pipe, numeric)
        )

    if categorical:
        transformers.append(
            ("categorical", categorical_pipe, categorical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    try:
        estimator = MODEL_FACTORIES[model_name](random_state)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model: {model_name}"
        ) from exc

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str = "Random Forest",
    random_state: int = 42,
    test_size: float = 0.25,
) -> dict[str, Any]:

    """Split before fitting preprocessing, then report held-out metrics."""

    required = feature_columns + [target_column]

    if not feature_columns:
        raise ValueError(
            "At least one model feature is required."
        )

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError(
            "Model feature columns must be unique."
        )

    if target_column in feature_columns:
        raise ValueError(
            "The target column cannot be included as a model feature."
        )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {', '.join(missing)}"
        )

    clean = data[required].dropna(
        subset=[target_column]
    ).copy()

    y = clean[target_column].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "Training requires at least two target classes."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class must contain at least 2 samples."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        clean[feature_columns],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = _make_pipeline(
        x_train,
        model_name,
        random_state,
    )

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    labels = [
        str(label)
        for label in pipeline.classes_
    ]

    metrics: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),

        "labels": labels,

        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(labels) == 2:
        positive_scores = pipeline.predict_proba(x_test)[:, 1]

        metrics["roc_auc"] = float(
            roc_auc_score(y_test, positive_scores)
        )

    else:
        probabilities = pipeline.predict_proba(x_test)

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "x_test": x_test,
        "y_test": y_test,

        "predictions": pd.Series(
            predictions,
            index=y_test.index,
            name="prediction",
        ),

        "features": feature_columns,
        "target": target_column,
        "model_name": model_name,
        "random_state": random_state,
        "training_data": clean,
    }


# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

def cross_validate_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str = "Random Forest",
    random_state: int = 42,
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:

    required = feature_columns + [target_column]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {', '.join(missing)}"
        )

    clean = data[required].dropna(
        subset=[target_column]
    ).copy()

    y = clean[target_column].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "Cross-validation requires at least two target classes."
        )

    if y.value_counts().min() < n_splits:
        raise ValueError(
            f"Each target class must contain at least {n_splits} samples."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    roc_auc_scoring = (
        "roc_auc"
        if y.nunique() == 2
        else "roc_auc_ovr_weighted"
    )

    scores = cross_validate(
        _make_pipeline(
            clean[feature_columns],
            model_name,
            random_state,
        ),

        clean[feature_columns],
        y,

        cv=cv,

        scoring={
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "roc_auc": roc_auc_scoring,
        },

        error_score="raise",
    )

    return {
        metric: {
            "mean": float(
                np.mean(scores[f"test_{metric}"])
            ),

            "std": float(
                np.std(scores[f"test_{metric}"])
            ),
        }

        for metric in (
            "accuracy",
            "f1",
            "roc_auc",
        )
    }


# --------------------------------------------------
# PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

def test_set_permutation_importance(
    model_result: dict[str, Any],
    n_repeats: int = 10,
) -> pd.DataFrame:

    importance = permutation_importance(
        model_result["pipeline"],
        model_result["x_test"],
        model_result["y_test"],
        scoring="accuracy",
        n_repeats=n_repeats,
        random_state=model_result["random_state"],
    )

    return pd.DataFrame(
        {
            "feature": model_result["features"],

            "importance_mean": importance.importances_mean,

            "importance_std": importance.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
        ignore_index=True,
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

def save_model(
    result: dict[str, Any],
    path: str | Path,
) -> Path:

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(result, destination)

    return destination


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(
    path: str | Path,
) -> dict[str, Any]:

    return joblib.load(path)
