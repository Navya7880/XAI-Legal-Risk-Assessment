from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data_processing import DATASET_CONFIG
from src.dice_explainer import FeasibilityChecker, generate_counterfactuals


DICE_METHODS = ("random", "genetic", "kdtree")
METHODS = ("dice_random", "dice_genetic", "dice_kdtree", "wachter_search", "nearest_unlike_neighbor")


def _dataset_config(model_result: dict[str, Any]) -> dict[str, Any]:
    target = model_result["target"]
    for config in DATASET_CONFIG.values():
        if config["target"] == target:
            return config
    return {}


def _same_value(left: Any, right: Any) -> bool:
    try:
        return bool(pd.isna(left) and pd.isna(right)) if pd.isna(left) or pd.isna(right) else left == right
    except Exception:
        return False


def gower_distance(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any],
                   features: Iterable[str], training: pd.DataFrame) -> float:
    distances = []
    for feature in features:
        a, b = left[feature], right[feature]
        if _same_value(a, b):
            distances.append(0.0)
        elif pd.api.types.is_numeric_dtype(training[feature]):
            span = float(training[feature].max() - training[feature].min())
            distances.append(abs(float(a) - float(b)) / span if span > 0 else 1.0)
        else:
            distances.append(1.0)
    return float(np.mean(distances))


def _requested_candidates(model_result: dict[str, Any], record: dict[str, Any],
                          candidate_pool: pd.DataFrame, total_cfs: int = 3) -> pd.DataFrame:
    features = list(model_result["features"])
    config = _dataset_config(model_result)
    immutable = {f for f in config.get("immutable_features", []) if f in features}
    checker = FeasibilityChecker(config.get("feasibility_spec", {}))
    original_pred = model_result["pipeline"].predict(pd.DataFrame([record], columns=features))[0]

    valid = []
    for _, row in candidate_pool.iterrows():
        candidate = row[features].copy()
        if any(not _same_value(candidate[f], record[f]) for f in immutable):
            continue
        if pipeline_predict(model_result, candidate) == original_pred:
            continue
        if not checker.check(candidate, record)[0]:
            continue
        valid.append(candidate)

    return pd.DataFrame(valid, columns=features).drop_duplicates().head(total_cfs)


def pipeline_predict(model_result: dict[str, Any], candidate: pd.Series | dict[str, Any]) -> Any:
    features = model_result["features"]
    frame = pd.DataFrame([candidate], columns=features)
    return model_result["pipeline"].predict(frame)[0]


def nearest_unlike_neighbor(model_result: dict[str, Any], record: dict[str, Any],
                            total_cfs: int = 3, seed: int = 42) -> pd.DataFrame:
    features = model_result["features"]
    training = model_result["training_data"][features + [model_result["target"]]].copy()
    original_pred = pipeline_predict(model_result, record)
    pool = training[training[model_result["target"]].astype(str) != str(original_pred)].copy()
    pool["__distance__"] = pool.apply(
        lambda row: gower_distance(record, row, features, training), axis=1
    )
    pool = pool.sort_values("__distance__", kind="stable")
    candidates = pool.head(max(total_cfs * 20, 100)).drop(columns=["__distance__", model_result["target"]], errors="ignore")
    return _requested_candidates(model_result, record, candidates, total_cfs)


def wachter_search(model_result: dict[str, Any], record: dict[str, Any],
                   total_cfs: int = 3, seed: int = 42) -> pd.DataFrame:
    """Wachter-style balanced loss search approximated by interpolation toward NUN targets."""
    nun = nearest_unlike_neighbor(model_result, record, total_cfs=max(total_cfs * 4, 12), seed=seed)
    features = model_result["features"]
    config = _dataset_config(model_result)
    immutable = set(config.get("immutable_features", []))
    original_pred = pipeline_predict(model_result, record)
    checker = FeasibilityChecker(config.get("feasibility_spec", {}))
    candidates = []

    targets = nun.to_dict("records") if not nun.empty else []
    if not targets:
        targets = nearest_unlike_neighbor(model_result, record, total_cfs=1, seed=seed).to_dict("records")

    for target in targets:
        for step in np.linspace(0.05, 1.0, 20):
            candidate = dict(record)
            for feature in features:
                if feature in immutable:
                    continue
                if pd.api.types.is_numeric_dtype(model_result["training_data"][feature]):
                    candidate[feature] = float(record[feature]) + step * (
                        float(target[feature]) - float(record[feature])
                    )
                    if str(feature).endswith("years") or feature in {"prior_offenses"}:
                        candidate[feature] = round(candidate[feature])
                    else:
                        candidate[feature] = round(candidate[feature], 3)
                else:
                    candidate[feature] = target[feature]
            frame = pd.DataFrame([candidate], columns=features)
            if pipeline_predict(model_result, candidate) != original_pred and checker.check(candidate, record)[0]:
                candidates.append(candidate)
                break

    frame = pd.DataFrame(candidates, columns=features).drop_duplicates()
    if not frame.empty:
        frame["__distance__"] = frame.apply(
            lambda row: gower_distance(record, row, features, model_result["training_data"]), axis=1
        )
        frame = frame.sort_values("__distance__", kind="stable").drop(columns="__distance__")
    return frame.head(total_cfs)


def generate_method_counterfactuals(method: str, model_result: dict[str, Any],
                                    record: dict[str, Any], seed: int, total_cfs: int = 3) -> pd.DataFrame:
    if method.startswith("dice_"):
        random.seed(seed)
        np.random.seed(seed)
        return generate_counterfactuals(
            model_result, record, total_cfs=total_cfs,
            method=method.removeprefix("dice_"),
            constraint_mode="full",
        )
    if method == "nearest_unlike_neighbor":
        return nearest_unlike_neighbor(model_result, record, total_cfs, seed)
    if method == "wachter_search":
        return wachter_search(model_result, record, total_cfs, seed)
    raise ValueError(f"Unknown counterfactual method: {method}")
