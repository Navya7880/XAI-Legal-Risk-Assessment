from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import DATASET_CONFIG, ensure_datasets
from src.evaluation.methods import (
    METHODS,
    generate_method_counterfactuals,
    gower_distance,
)
from src.model_training import train_model
from src.ui import MODEL_NAMES


def perturb_record(record: dict[str, Any], features: list[str]) -> dict[str, Any]:
    perturbed = dict(record)
    if "age" in features and "age" in perturbed:
        perturbed["age"] = max(18, min(75, float(perturbed["age"]) + np.random.choice([-1, 1])))
    if "prior_offenses" in features and "prior_offenses" in perturbed:
        perturbed["prior_offenses"] = max(0, float(perturbed["prior_offenses"]) + np.random.choice([-1, 1]))
    return perturbed


def changed_feature_jaccard(original: dict[str, Any], left: pd.DataFrame, right: pd.DataFrame,
                            features: list[str]) -> float:
    def changed(frame: pd.DataFrame) -> set[str]:
        if frame.empty:
            return set()
        row = frame.iloc[0]
        changed_features = set()
        for feature in features:
            try:
                differs = not bool(row[feature] == original[feature])
            except Exception:
                differs = True
            if differs:
                changed_features.add(feature)
        return changed_features

    left_changed, right_changed = changed(left), changed(right)
    union = left_changed | right_changed
    return len(left_changed & right_changed) / len(union) if union else 1.0


def mean_cf_distance(original: dict[str, Any], left: pd.DataFrame, right: pd.DataFrame,
                     features: list[str], training: pd.DataFrame) -> float:
    distances = []
    for _, row in left.iterrows():
        candidates = [gower_distance(row, other, features, training) for _, other in right.iterrows()]
        if candidates:
            distances.append(min(candidates))
    return float(np.mean(distances)) if distances else 0.0


def run_robustness(seeds: list[int] | None = None, instances: int = 20) -> pd.DataFrame:
    seeds = seeds or [42, 43, 44, 45, 46]
    datasets = ensure_datasets()
    root = Path(__file__).resolve().parents[2]
    rows = []

    for dataset_name, data in datasets.items():
        config = DATASET_CONFIG[dataset_name]
        features, target = config["features"], config["target"]
        for model_name in MODEL_NAMES:
            base_seed = seeds[0]
            result = train_model(data, features, target, model_name, base_seed)
            test_records = result["x_test"].head(instances).to_dict("records")

            for method in METHODS:
                original_cfs = []
                perturbed_cfs = []
                for index, record in enumerate(test_records):
                    np.random.seed(base_seed + index)
                    original = generate_method_counterfactuals(method, result, record, base_seed, 3)
                    perturbed_record = perturb_record(record, features)
                    perturbed = generate_method_counterfactuals(method, result, perturbed_record, base_seed, 3)
                    original_cfs.append(original)
                    perturbed_cfs.append(perturbed)
                    rows.append({
                        "analysis": "input_perturbation",
                        "dataset": dataset_name, "model": model_name, "method": method,
                        "seed": base_seed,
                        "jaccard_similarity": changed_feature_jaccard(record, original, perturbed, features),
                        "mean_distance": mean_cf_distance(record, original, perturbed, features, result["training_data"]),
                    })

                # Stability across DiCE seed choices.
                if method.startswith("dice_"):
                    seed_outputs = {}
                    for seed in seeds:
                        outputs = []
                        for record in test_records:
                            outputs.append(generate_method_counterfactuals(method, result, record, seed, 3))
                        seed_outputs[seed] = outputs
                    for left_index, left_seed in enumerate(seeds[:-1]):
                        right_seed = seeds[left_index + 1]
                        similarities = [
                            changed_feature_jaccard(record, seed_outputs[left_seed][i], seed_outputs[right_seed][i], features)
                            for i, record in enumerate(test_records)
                        ]
                        rows.append({
                            "analysis": "dice_seed_stability", "dataset": dataset_name,
                            "model": model_name, "method": method, "seed": left_seed,
                            "jaccard_similarity": float(np.mean(similarities)),
                            "mean_distance": 0.0,
                        })

                # Bootstrap retraining validity.
                valid_rates = []
                for record in test_records:
                    base_cf = generate_method_counterfactuals(method, result, record, base_seed, 3)
                    if base_cf.empty:
                        continue
                    bootstrap_validities = []
                    for bootstrap_seed in seeds:
                        sample = data.sample(frac=1.0, replace=True, random_state=bootstrap_seed)
                        bootstrap = train_model(sample, features, target, model_name, bootstrap_seed)
                        original_pred = bootstrap["pipeline"].predict(
                            pd.DataFrame([record], columns=features)
                        )[0]
                        candidate_valid = [
                            bool(bootstrap["pipeline"].predict(
                                pd.DataFrame([candidate], columns=features)
                            )[0] != original_pred)
                            for candidate in base_cf.to_dict("records")
                        ]
                        bootstrap_validities.append(float(np.mean(candidate_valid)) if candidate_valid else 0.0)
                    valid_rates.extend(bootstrap_validities)
                rows.append({
                    "analysis": "bootstrap_retraining",
                    "dataset": dataset_name, "model": model_name, "method": method, "seed": base_seed,
                    "validity_rate": float(np.mean(valid_rates)) if valid_rates else 0.0,
                    "mean_distance": 0.0,
                })

    report = pd.DataFrame(rows)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports / "robustness.csv", index=False)

    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    for analysis, filename, value, title in [
        ("input_perturbation", "robustness_input.png", "jaccard_similarity", "Input-perturbation feature stability"),
        ("dice_seed_stability", "robustness_seed.png", "jaccard_similarity", "DiCE stability across seeds"),
        ("bootstrap_retraining", "robustness_retraining.png", "validity_rate", "Counterfactual validity after retraining"),
    ]:
        subset = report[report["analysis"] == analysis].dropna(subset=[value])
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="method", columns=["dataset", "model"], values=value, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind="bar", ax=ax, rot=20, capsize=3)
        ax.set_title(title)
        ax.set_ylabel(value.replace("_", " ").title())
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return report
