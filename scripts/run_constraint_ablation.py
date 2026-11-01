"""Run a real 200-case feasibility ablation and write report artifacts."""

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import DATASET_CONFIG, LEGAL_FEATURES, LEGAL_TARGET
from src.data_processing import generate_legal_dataset
from src.dice_explainer import (
    FeasibilityChecker,
    counterfactual_quality_metrics,
    generate_counterfactuals,
)
from src.model_training import train_model


REPORTS = ROOT / "reports"


def is_plausible(
    candidate: pd.Series,
    training_data: pd.DataFrame,
    features: list[str],
) -> bool:
    for feature in features:
        if feature not in candidate:
            continue
        series = training_data[feature].dropna()
        if pd.api.types.is_numeric_dtype(series):
            if not (float(series.min()) <= float(candidate[feature]) <= float(series.max())):
                return False
        elif candidate[feature] not in set(series.unique()):
            return False
    return True


def main() -> None:
    dataset_name = "Legal risk (synthetic)"
    data = generate_legal_dataset(1200, random_state=31)
    model_result = train_model(
        data,
        LEGAL_FEATURES,
        LEGAL_TARGET,
        "Random Forest",
        random_state=31,
    )
    test_cases = model_result["x_test"]
    test_cases = test_cases.iloc[:200]
    config = DATASET_CONFIG[dataset_name]
    checker = FeasibilityChecker(config["feasibility_spec"])
    rows = []

    for mode in ("none", "ranges", "full"):
        validity = []
        proximity = []
        plausibility = []
        feasibility = []

        for _, case in test_cases.iterrows():
            record = {
                feature: case[feature]
                for feature in LEGAL_FEATURES
            }
            counterfactuals = generate_counterfactuals(
                model_result,
                record,
                total_cfs=3,
                method="random",
                constraint_mode=mode,
            )
            quality = counterfactual_quality_metrics(
                model_result,
                record,
                counterfactuals,
                3,
                permitted_range=config["permitted_range"],
            )
            validity.append(quality["validity"])
            proximity.append(quality["proximity"])

            candidate_plausibility = []
            candidate_feasibility = []
            for _, candidate in counterfactuals.iterrows():
                candidate_plausibility.append(
                    is_plausible(candidate, model_result["training_data"], LEGAL_FEATURES)
                )
                candidate_feasibility.append(
                    checker.check(candidate, record)[0]
                )
            plausibility.append(
                sum(candidate_plausibility) / len(candidate_plausibility)
                if candidate_plausibility else 0.0
            )
            feasibility.append(
                sum(candidate_feasibility) / len(candidate_feasibility)
                if candidate_feasibility else 0.0
            )

        rows.append({
            "mode": mode,
            "test_cases": len(test_cases),
            "validity": sum(validity) / len(validity),
            "proximity": sum(proximity) / len(proximity),
            "plausibility": sum(plausibility) / len(plausibility),
            "feasibility_rate": sum(feasibility) / len(feasibility),
        })

    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORTS / "ablation_constraints.csv", index=False)

    figure = go.Figure()
    for metric in ("validity", "proximity", "plausibility", "feasibility_rate"):
        figure.add_trace(go.Bar(
            name=metric,
            x=frame["mode"],
            y=frame[metric],
        ))
    figure.update_layout(
        title="Counterfactual constraint ablation (200 legal-risk test cases)",
        yaxis_title="Rate or normalized score",
        barmode="group",
        height=500,
        font=dict(size=16),
        title_font_size=22,
        legend_font_size=16,
    )
    figure.update_xaxes(tickfont=dict(size=16), title_font=dict(size=16))
    figure.update_yaxes(tickfont=dict(size=16), title_font=dict(size=16))
    figure.write_html(REPORTS / "ablation_constraints.html", include_plotlyjs="cdn")

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
