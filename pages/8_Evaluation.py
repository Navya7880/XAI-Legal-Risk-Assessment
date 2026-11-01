from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import configure_navigation, page_header, setup_page


setup_page("Legal Risk Assessment | Evaluation")


@st.cache_data
def load_evaluation():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    summary = pd.read_csv(reports / "evaluation_summary.csv") if (reports / "evaluation_summary.csv").exists() else pd.DataFrame()
    significance = pd.read_csv(reports / "evaluation_significance.csv") if (reports / "evaluation_significance.csv").exists() else pd.DataFrame()
    return summary, significance


@st.cache_data
def load_robustness():
    path = Path(__file__).resolve().parents[1] / "reports" / "robustness.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def render_evaluation():
    page_header("Rigorous Counterfactual Evaluation", "Measured comparisons across datasets, models, methods, and seeds.")
    summary, significance = load_evaluation()
    robustness = load_robustness()
    if summary.empty:
        st.warning("No evaluation report found. Run `make evaluate` to generate actual results.")
        st.code("make evaluate", language="bash")
        return

    st.subheader("How to read this page")
    st.write(
        "Each value summarizes five random seeds; the bracket is a 95% confidence interval. "
        "Validity measures opposite predictions. Proximity, sparsity, diversity, plausibility, "
        "and feasibility describe how close, sparse, varied, realistic, and allowed explanations are. "
        "Runtime is generation seconds for 200 instances."
    )
    evaluation_tab, robustness_tab = st.tabs(["Evaluation", "Robustness"])

    with evaluation_tab:
        dataset = st.selectbox("Dataset", sorted(summary["dataset"].unique()))
        model = st.selectbox("Model", sorted(summary["model"].unique()))
        subset = summary[(summary["dataset"] == dataset) & (summary["model"] == model)]
        st.subheader("Mean and 95% confidence interval")
        st.dataframe(
            subset.pivot(index="method", columns="metric", values="mean").round(4),
            width="stretch",
        )

        metric = st.selectbox("Chart metric", list(subset["metric"].unique()))
        chart_data = subset[subset["metric"] == metric]
        fig = px.bar(
            chart_data, x="method", y="mean",
            error_y=chart_data["ci95_high"] - chart_data["mean"],
            error_y_minus=chart_data["mean"] - chart_data["ci95_low"],
            title=f"{metric.replace('_', ' ').title()} by method",
        )
        st.plotly_chart(fig, width="stretch")

        image = Path(__file__).resolve().parents[1] / "reports" / "figures" / f"boxplot_{metric}.png"
        if image.exists():
            st.image(str(image), caption=f"{metric.replace('_', ' ').title()} distribution")

        st.subheader("Pairwise significance")
        if not significance.empty:
            sig_subset = significance[
                (significance["dataset"] == dataset) & (significance["model"] == model)
            ]
            st.dataframe(
                sig_subset[
                    ["metric", "method_a", "method_b", "p_value",
                     "bonferroni_significant", "effect_size_rank_biserial"]
                ].sort_values(["metric", "p_value"]).round(5),
                width="stretch",
            )

    with robustness_tab:
        if robustness.empty:
            st.warning("No robustness report found. Run `make evaluate` to generate it.")
            st.code("make evaluate", language="bash")
        else:
            bootstrap = robustness[robustness["analysis"] == "bootstrap_retraining"]
            validity = float(bootstrap["validity_rate"].mean()) if "validity_rate" in bootstrap else 0.0
            st.success(
                f"About {validity:.1%} of counterfactual suggestions remain valid after retraining across "
                "the tested bootstrap samples."
            )
            st.write(
                "A low retained-validity rate means that the suggested recourse is sensitive to model "
                "changes. Deploying such advice without monitoring and refreshes can make a previously "
                "valid explanation unreliable or potentially harmful."
            )
            st.dataframe(robustness.round(5), width="stretch")


configure_navigation(render_evaluation)
