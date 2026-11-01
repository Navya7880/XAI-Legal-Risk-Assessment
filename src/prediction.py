"""Validated prediction helpers for trained sklearn pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd


def predict_record(model_result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    features = model_result["features"]
    missing = [feature for feature in features if feature not in record]
    if missing:
        raise ValueError(f"Missing prediction inputs: {', '.join(missing)}")
    row = pd.DataFrame([{feature: record[feature] for feature in features}])
    pipeline = model_result["pipeline"]
    prediction = str(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0] if hasattr(pipeline, "predict_proba") else None
    return {
        "prediction": prediction,
        "probabilities": (
            {str(label): float(probability) for label, probability in zip(pipeline.classes_, probabilities)}
            if probabilities is not None
            else None
        ),
    }
