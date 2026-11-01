from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from src.dice_explainer import counterfactual_quality_metrics
from src.evaluation.methods import gower_distance


def evaluate_counterfactuals(model_result: dict[str, Any], record: dict[str, Any],
                             counterfactuals: pd.DataFrame, requested_count: int = 3) -> dict[str, float]:
    base = counterfactual_quality_metrics(
        model_result, record, counterfactuals, requested_count=requested_count
    )
    if counterfactuals.empty:
        plausibility = 0.0
    else:
        features = model_result["features"]
        distances = [
            min(gower_distance(row, training_row, features, model_result["training_data"])
                for _, training_row in model_result["training_data"][features].iterrows())
            for _, row in counterfactuals[features].iterrows()
        ]
        plausibility = float(np.mean([1.0 - distance for distance in distances]))
    return {
        "validity": float(base["validity"]),
        "proximity": float(base["proximity"]),
        "sparsity": float(base["sparsity"]),
        "diversity": float(base["diversity"]),
        "plausibility": float(np.clip(plausibility, 0.0, 1.0)),
        "feasibility_rate": float(base["feasibility_rate"]),
    }
