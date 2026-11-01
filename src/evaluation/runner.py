from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data_processing import DATASET_CONFIG, ensure_datasets
from src.evaluation.methods import METHODS, generate_method_counterfactuals
from src.evaluation.metrics import evaluate_counterfactuals
from src.model_training import train_model
from src.ui import MODEL_NAMES


METRIC_NAMES = [
    "validity", "proximity", "sparsity", "diversity",
    "plausibility", "feasibility_rate", "runtime",
]
ROOT = Path(__file__).resolve().parents[2]


def confidence_interval(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values[0]), float(values[0])
    center = float(np.mean(values))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return center - half_width, center + half_width


def run_evaluation(seeds: list[int] | None = None, instances: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seeds or [42, 43, 44, 45, 46]
    datasets = ensure_datasets()
    raw_rows = []

    for dataset_name, data in datasets.items():
        config = DATASET_CONFIG[dataset_name]
        for model_name in MODEL_NAMES:
            for seed in seeds:
                result = train_model(
                    data, config["features"], config["target"],
                    model_name=model_name, random_state=seed,
                )
                x_test = result["x_test"].head(instances)
                for method in METHODS:
                    values = {metric: [] for metric in METRIC_NAMES if metric != "runtime"}
                    start = time.perf_counter()
                    for _, row in x_test.iterrows():
                        record = row.to_dict()
                        try:
                            counterfactuals = generate_method_counterfactuals(
                                method, result, record, seed, total_cfs=3
                            )
                            metrics = evaluate_counterfactuals(
                                result, record, counterfactuals, requested_count=3
                            )
                        except Exception as exc:
                            counterfactuals = pd.DataFrame()
                            metrics = {
                                "validity": 0.0, "proximity": 0.0, "sparsity": 0.0,
                                "diversity": 0.0, "plausibility": 0.0, "feasibility_rate": 0.0,
                            }
                        for metric, value in metrics.items():
                            values[metric].append(value)
                    runtime = time.perf_counter() - start
                    raw_rows.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": method,
                        "seed": seed,
                        "instances": len(x_test),
                        **{metric: float(np.mean(items)) for metric, items in values.items()},
                        "runtime": runtime,
                    })

    raw = pd.DataFrame(raw_rows)
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw.to_csv(reports / "evaluation_runs.csv", index=False)

    summary_rows = []
    for keys, group in raw.groupby(["dataset", "model", "method"], sort=True):
        dataset, model, method = keys
        for metric in METRIC_NAMES:
            values = group[metric].tolist()
            low, high = confidence_interval(values)
            summary_rows.append({
                "dataset": dataset, "model": model, "method": method, "metric": metric,
                "mean": float(np.mean(values)), "ci95_low": low, "ci95_high": high,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports / "evaluation_summary.csv", index=False)

    significance_rows = []
    grouping = ["dataset", "model"]
    for (dataset, model), group in raw.groupby(grouping):
        for metric in METRIC_NAMES:
            pivot = group.pivot(index="seed", columns="method", values=metric).dropna()
            comparisons = list(itertools.combinations(METHODS, 2))
            alpha = 0.05 / len(comparisons)
            for left, right in comparisons:
                a = pivot[left].to_numpy()
                b = pivot[right].to_numpy()
                differences = a - b
                try:
                    statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox")
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                nonzero = differences[differences != 0]
                rank_biserial = (
                    (np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero)
                    if len(nonzero) else 0.0
                )
                significance_rows.append({
                    "dataset": dataset, "model": model, "metric": metric,
                    "method_a": left, "method_b": right, "statistic": float(statistic),
                    "p_value": float(p_value), "bonferroni_alpha": alpha,
                    "bonferroni_significant": bool(p_value < alpha),
                    "effect_size_rank_biserial": float(rank_biserial),
                })
    significance = pd.DataFrame(significance_rows)
    significance.to_csv(reports / "evaluation_significance.csv", index=False)

    publication_latex(summary, reports / "evaluation_summary.tex")
    publication_figures(summary, raw, figures)
    progress = ROOT / "docs" / "PROGRESS.md"
    progress_text = progress.read_text(encoding="utf-8").rstrip()
    marker = "## Rigorous Counterfactual Evaluation"
    section = (
        "\n\n## Rigorous Counterfactual Evaluation\n\n"
        + f"The evaluation was run over {len(seeds)} random seeds and "
        + f"{instances} held-out instances per method, dataset, and model. "
        + "Raw seed-level measurements, confidence intervals, significance tests, and figures "
        + "are in `reports/evaluation_runs.csv`, `reports/evaluation_summary.csv`, "
        + "`reports/evaluation_significance.csv`, and `reports/figures/`.\n\n"
        + "Reproduce the measured results with `make evaluate`.\n"
    )
    if marker in progress_text:
        progress_text = progress_text[:progress_text.index(marker)].rstrip()
    progress.write_text(progress_text + section, encoding="utf-8")
    return raw, summary


def publication_latex(summary: pd.DataFrame, output: Path) -> None:
    table = summary.pivot_table(
        index=["dataset", "model", "method"], columns="metric",
        values=["mean", "ci95_low", "ci95_high"], aggfunc="first",
    )
    lines = []
    for (dataset, model, method), values in table.iterrows():
        cells = []
        for metric in METRIC_NAMES:
            mean = values.get(("mean", metric), np.nan)
            low = values.get(("ci95_low", metric), np.nan)
            high = values.get(("ci95_high", metric), np.nan)
            cells.append(f"{mean:.3f} [{low:.3f}, {high:.3f}]")
        lines.append(" & ".join([dataset, model, method] + cells) + r" \\")
    headers = "dataset & model & method & " + " & ".join(m.replace("_", " ") for m in METRIC_NAMES) + r" \\"
    latex = (
        "\\begin{tabular}{lll" + "c" * len(METRIC_NAMES) + "}\n"
        "\\toprule\n"
        + headers
        + "\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    output.write_text(latex, encoding="utf-8")


def publication_figures(summary: pd.DataFrame, raw: pd.DataFrame, figures: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for metric in METRIC_NAMES:
        subset = raw[["dataset", "model", "method", metric]]
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = METHODS
        data = [subset.loc[subset["method"] == method, metric].dropna().to_numpy() for method in labels]
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures / f"boxplot_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    bar = summary[summary["metric"] == "validity"]
    pivot = bar.pivot_table(index=["dataset", "model"], columns="method", values="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    low = bar.pivot_table(index=["dataset", "model"], columns="method", values="ci95_low")
    high = bar.pivot_table(index=["dataset", "model"], columns="method", values="ci95_high")
    x_positions = np.arange(len(pivot.index))
    width = 0.15
    for index, method in enumerate(METHODS):
        offset = (index - (len(METHODS) - 1) / 2) * width
        means = pivot[method].to_numpy()
        yerr = np.vstack([
            means - low[method].to_numpy(),
            high[method].to_numpy() - means,
        ])
        ax.bar(
            x_positions + offset, means, width=width, label=method,
            yerr=yerr, capsize=2,
        )
    ax.set_xticks(x_positions, [f"{dataset}\n{model}" for dataset, model in pivot.index], rotation=20)
    ax.legend()
    ax.set_ylabel("Validity")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "bar_validity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
