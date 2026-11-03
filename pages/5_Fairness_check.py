
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import (
    DATASET_CONFIG,
    HIRING_GENDER,
    HIRING_SENSITIVE,
    HIRING_TARGET,
    LEGAL_PROTECTED,
)
from src.fairness_audit import audit_fairness, mitigate_fairness
from src.ui import (
    display_plotly_chart,
    get_datasets,
    get_model,
    page_header,
    setup_page,
)

setup_page("Fairness check")

dataset_name = page_header(
    "Fairness check",
    "Compare model outcomes across groups in the selected synthetic scenario.",
)

st.caption("Keep one scenario selected in the sidebar.")

model_name = st.selectbox(
    "Model to review",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"],
)

constraint = st.selectbox(
    "Optional mitigation approach",
    ["Demographic parity", "Equalized odds"],
    help="This post-processing approach changes model thresholds to target a chosen group metric.",
)

data = get_datasets()[dataset_name]
result = get_model(dataset_name, model_name)
config = DATASET_CONFIG[dataset_name]

protected_attribute = (
    LEGAL_PROTECTED
    if dataset_name == "Legal risk (synthetic)"
    else HIRING_GENDER
)

if dataset_name == "Hiring (synthetic)":
    st.caption("Protected attribute: gender (synthetic).")
else:
    st.caption(
        "Protected attribute: synthetic_group. This is a synthetic teaching attribute, "
        "not a real demographic characteristic."
    )

if protected_attribute in data.columns:
    groups = data[protected_attribute]
else:
    groups = pd.Series(
        [
            "Group A" if position % 2 == 0 else "Group B"
            for position in range(len(data))
        ],
        index=data.index,
        name=protected_attribute,
    )

groups_test = groups.loc[result["y_test"].index]

before = audit_fairness(
    result["y_test"],
    result["predictions"],
    groups_test,
)

gap = before["demographic_parity_difference"]

per_group = before["by_group"].reset_index()

per_group = per_group.rename(
    columns={per_group.columns[0]: protected_attribute}
)

per_group = per_group[
    [protected_attribute, "Selection rate", "Accuracy"]
]

# Per-group results
st.subheader("Per-group results")

st.dataframe(
    per_group,
    hide_index=True,
    use_container_width=True,
)

# Fairness summary
if gap < 0.10:
    summary, summary_color = "Small gap", "#e8f5ee"
elif gap < 0.20:
    summary, summary_color = "Moderate gap", "#fff5df"
else:
    summary, summary_color = "Large gap", "#fff0ed"

st.markdown(
    f'<div class="xai-card" style="background:{summary_color}">'
    f"<h3>{summary}</h3>"
    f"<p>Measured selection-rate difference: <strong>{gap:.3f}</strong>. "
    "This is a teaching summary for this synthetic test sample, not a fairness verdict.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Teaching thresholds: small < 0.10; moderate 0.10–<0.20; large ≥ 0.20, "
    "using the absolute demographic-parity difference."
)

# Before and after comparison
st.subheader("Before and after comparison")

try:

    with st.spinner("Comparing the selected mitigation approach…"):

        mitigation = mitigate_fairness(
            result,
            groups,
            constraint,
        )

        after = audit_fairness(
            result["y_test"],
            mitigation["predictions"],
            groups_test,
        )

    comparison = pd.DataFrame(
        [
            {
                "Stage": "Before mitigation",
                "Accuracy": before["accuracy"],
                "Selection-rate difference": before[
                    "demographic_parity_difference"
                ],
                "True/false-positive difference": before[
                    "equalized_odds_difference"
                ],
            },
            {
                "Stage": "After mitigation",
                "Accuracy": after["accuracy"],
                "Selection-rate difference": after[
                    "demographic_parity_difference"
                ],
                "True/false-positive difference": after[
                    "equalized_odds_difference"
                ],
            },
        ]
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    # Prepare group metric data
    metric_rows = []

    for stage, audit in (
        ("Before", before),
        ("After", after),
    ):

        for group, row in audit["by_group"].iterrows():

            for label, key in (
                ("Selection rate", "Selection rate"),
                ("True positive rate", "True positive rate"),
                ("False positive rate", "False positive rate"),
            ):

                value = float(row[key])

                low_error = 0.0
                high_error = 0.0

                if label == "Selection rate":

                    low, high = audit["bootstrap"]["selection_rate"][group]

                    low_error = max(0.0, value - low)
                    high_error = max(0.0, high - value)

                metric_rows.append(
                    {
                        protected_attribute: group,
                        "Stage": stage,
                        "Measure": label,
                        "Value": value,
                        "Lower error": low_error,
                        "Upper error": high_error,
                    }
                )

    # Group comparison chart
    display_plotly_chart(
        px.bar(
            pd.DataFrame(metric_rows),
            x=protected_attribute,
            y="Value",
            color="Measure",
            facet_col="Stage",
            barmode="group",
            error_y="Upper error",
            error_y_minus="Lower error",
            labels={
                protected_attribute: protected_attribute.replace(
                    "_", " "
                ).title()
            },
            title="Group selection, true-positive, and false-positive rates",
        ),
        font_size=st.session_state.font_px,
        width="stretch",
    )

    st.caption(
        "Selection-rate bars show bootstrap 95% intervals (500 resamples, fixed seed). "
        f"After-mitigation selection-rate difference interval: "
        f"{after['demographic_parity_difference_ci'][0]:.3f}–"
        f"{after['demographic_parity_difference_ci'][1]:.3f}."
    )

    # Save fairness results
    st.session_state["latest_fairness"] = {
        "before": before,
        "after": after,
        "comparison": comparison,
        "constraint": constraint,
        "groups": groups_test,
        "y_true": result["y_test"],
        "predictions_before": result["predictions"],
        "predictions_after": mitigation["predictions"],
        "reference_group_model": None,
    }

    st.session_state["latest_fairness_dataset"] = dataset_name
    st.session_state["latest_fairness_model"] = model_name

    # Technical details
    with st.expander("Technical details · group metrics"):

        st.dataframe(
            after["by_group"],
            use_container_width=True,
        )

        st.write(
            {
                "Before accuracy": before["accuracy"],
                "After accuracy": after["accuracy"],
                "Before selection-rate difference": before[
                    "demographic_parity_difference"
                ],
                "After selection-rate difference": after[
                    "demographic_parity_difference"
                ],
                "Before equalized-odds difference": before[
                    "equalized_odds_difference"
                ],
                "After equalized-odds difference": after[
                    "equalized_odds_difference"
                ],
            }
        )

except (ValueError, RuntimeError, TypeError) as error:

    st.error(
        f"This comparison could not be completed. Try another model or constraint. "
        f"Details: {error}"
    )

    after = None

    st.session_state["latest_fairness"] = {
        "before": before,
        "after": None,
        "comparison": pd.DataFrame(
            [
                {
                    "Stage": "Before mitigation",
                    "Accuracy": before["accuracy"],
                    "Selection-rate difference": before[
                        "demographic_parity_difference"
                    ],
                    "True/false-positive difference": before[
                        "equalized_odds_difference"
                    ],
                }
            ]
        ),
        "mitigation_error": str(error),
        "constraint": constraint,
        "groups": groups_test,
        "y_true": result["y_test"],
        "predictions_before": result["predictions"],
        "predictions_after": None,
    }

    st.session_state["latest_fairness_dataset"] = dataset_name
    st.session_state["latest_fairness_model"] = model_name


# Protected-attribute flip test
st.subheader("Protected-attribute flip test")

flipped = result["x_test"].copy()

changed_predictions = 0

flipped[protected_attribute] = groups_test.reindex(
    flipped.index
)

values = pd.unique(groups_test.dropna())

if len(values) >= 2:

    value_map = {
        values[0]: values[-1],
        values[-1]: values[0],
    }

    flipped[protected_attribute] = (
        flipped[protected_attribute]
        .map(value_map)
        .fillna(flipped[protected_attribute])
    )

    flipped_predictions = result["pipeline"].predict(flipped)

    changed_predictions = int(
        sum(
            before_prediction != after_prediction
            for before_prediction, after_prediction in zip(
                result["predictions"],
                flipped_predictions,
            )
        )
    )

flip_share = changed_predictions / max(
    len(result["predictions"]),
    1,
)

st.metric(
    "Share of predictions changed by flipping the attribute",
    f"{flip_share:.1%}",
)

st.caption(
    "The protected attribute is excluded from model inputs, so this check is also "
    "a useful evidence of model dependence rather than a guaranteed real-world test."
)

# Explanation of measures
with st.expander("What do these measures mean?"):

    st.markdown(
        "**Selection rate:** share of each group receiving the positive model result.  \n"
        "**True-positive rate:** share of actual positive examples correctly identified.  \n"
        "**False-positive rate:** share of actual negative examples incorrectly marked positive.  \n"
        "**Demographic-parity difference:** gap between groups’ selection rates.  \n"
        "**Equalized-odds difference:** largest group gap across true- and false-positive rates."
    )

st.caption(
    "Fairness measures can conflict with each other and with accuracy. Mitigation does not "
    "prove fairness for real people or establish legal compliance."
)
