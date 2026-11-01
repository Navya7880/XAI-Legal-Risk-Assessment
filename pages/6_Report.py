import pandas as pd
import streamlit as st
from html import escape

from src.compliance_reporting import generate_pdf_report
from src.data_processing import DATASET_CONFIG, HIRING_GENDER, LEGAL_PROTECTED
from src.fairness_audit import audit_fairness
from src.ui import get_datasets, get_model, page_header, setup_page

setup_page("Report")

dataset_name = page_header(
    "Report",
    "Review session results and generate a transparent PDF summary.",
)
model_name = st.selectbox("Model for this report", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
data = get_datasets()[dataset_name]
config = DATASET_CONFIG[dataset_name]
try:
    with st.spinner("Loading the saved model and report inputs…"):
        result = get_model(dataset_name, model_name)
except Exception as error:
    st.error(
        f"The report could not load the saved model: {error}. "
        "Developer: run `python -m src.models.train`."
    )
    st.stop()

latest_dataset = st.session_state.get("latest_dataset")
prediction = (
    st.session_state.get("latest_prediction")
    if latest_dataset == dataset_name
    and st.session_state.get("latest_model") == model_name
    else None
)
counterfactual_run = (
    st.session_state.get("latest_counterfactual_dataset") == dataset_name
    and st.session_state.get("latest_counterfactual_model") == model_name
    and st.session_state.get("latest_counterfactuals") is not None
)
counterfactuals = (
    st.session_state.get("latest_counterfactuals") if counterfactual_run else None
)
counterfactual_quality = (
    st.session_state.get("latest_counterfactual_quality")
    if counterfactual_run
    else None
)
@st.cache_data
def _cached_fairness(dataset_name: str, model_name: str, y_true, predictions, groups):
    return audit_fairness(y_true, predictions, groups)

with st.spinner("Computing fairness results for this scenario…"):
    protected_attribute = LEGAL_PROTECTED if dataset_name == "Legal risk (synthetic)" else HIRING_GENDER
    full_data = get_datasets()[dataset_name]
    if protected_attribute in full_data.columns:
        groups = full_data.loc[result["y_test"].index, protected_attribute]
    else:
        groups = pd.Series(
            ["Group A" if position % 2 == 0 else "Group B"
             for position in range(len(result["y_test"]))],
            index=result["y_test"].index,
            name=protected_attribute,
        )
    before = _cached_fairness(
        dataset_name,
        model_name,
        result["y_test"],
        result["predictions"],
        groups,
    )
fairness = {
    "before": before,
    "after": None,
    "comparison": pd.DataFrame([{
        "Stage": "Before mitigation",
        "Accuracy": before["accuracy"],
        "Selection-rate difference": before["demographic_parity_difference"],
        "True/false-positive difference": before["equalized_odds_difference"],
    }]),
    "groups": groups,
    "y_true": result["y_test"],
    "predictions_before": result["predictions"],
}

st.subheader("1. Review the summary")
summary_columns = st.columns(3)
summary_columns[0].metric("Dataset", dataset_name.replace(" (synthetic)", ""))
summary_columns[1].metric("Examples", f"{len(data):,}")
summary_columns[2].metric("Model accuracy", f"{result['metrics']['accuracy']:.1%}")

st.subheader("2. Check the checklist")
checklist_items = [
    ("Dataset details", True, f"{len(data):,} rows and {len(config['features'])} input fields are available."),
    ("Held-out model metrics", bool(result.get("metrics")), "Real held-out model results are available." if result.get("metrics") else "No model metrics are available."),
    ("Prediction", prediction is not None, "A prediction was made in this session." if prediction is not None else "Run a prediction first to include one."),
    ("DiCE explanation", counterfactual_run, "A DiCE search was run in this session." if counterfactual_run else "Run an explanation first to include one."),
    ("Fairness comparison", fairness is not None, "Fairness results were computed from the selected model's held-out test rows." if fairness is not None else "Fairness results are unavailable."),
    ("Legal/privacy impact review", False, "A qualified, context-specific assessment is not part of this academic demo."),
]
for label, present, note in checklist_items:
    status = "Available" if present else "Not available"
    st.markdown(f"**{label} · {status}** — {note}")

st.subheader("Compliance mapping")
# TODO: verify wording with qualified legal counsel before deployment.
st.caption("Prototype mapping, not legal advice")
compliance_rows = [
    {"Framework": "EU AI Act", "Requirement area": "Transparency / user information", "App output": "Prediction, limitations, model information and report", "Status": "Prototype mapping, not legal advice"},
    {"Framework": "EU AI Act", "Requirement area": "Human oversight", "App output": "Case review, editable inputs and non-automated report checklist", "Status": "Prototype mapping, not legal advice"},
    {"Framework": "GDPR Art. 13-15", "Requirement area": "Information and access", "App output": "Purpose, synthetic-data limits and downloadable report", "Status": "Prototype mapping, not legal advice"},
    {"Framework": "GDPR Art. 22", "Requirement area": "Automated individual decisions", "App output": "Model output is marked as non-binding; human review remains required", "Status": "Prototype mapping, not legal advice"},
]
st.table(pd.DataFrame(compliance_rows))

st.subheader("3. Generate and download the PDF")
st.caption(
    "The report supports a preliminary assessment and does not certify GDPR or EU AI Act compliance."
)
report_key = f"report_pdf_{dataset_name}_{model_name}"
if st.button("Generate PDF report", type="primary"):
    try:
        with st.spinner("Putting the available session results into a report…"):
            pdf = generate_pdf_report(
                dataset_name,
                model_name,
                result["metrics"],
                fairness=fairness,
                prediction=prediction,
                counterfactuals=counterfactuals,
                dataset_data=data,
                target_column=config["target"],
                feature_columns=config["features"],
                model_result=result,
                counterfactual_quality=counterfactual_quality,
                counterfactual_run=counterfactual_run,
            )
        st.session_state[report_key] = pdf
        st.success("Your report is ready. Download it below.")
    except (ValueError, TypeError, OSError) as error:
        st.error(f"The PDF could not be prepared. Review the session results and try again. {error}")

if st.session_state.get(report_key):
    st.download_button(
        "Download PDF report",
        st.session_state[report_key],
        file_name="xai_transparency_assessment.pdf",
        mime="application/pdf",
        width="stretch",
    )

st.subheader("Per-decision HTML report")
if st.button("Prepare HTML decision report", type="secondary"):
    try:
        with st.spinner("Preparing the per-decision report…"):
            prediction_text = "No prediction available"
            if prediction:
                prediction_text = str(prediction.get("prediction", prediction))

            factors_text = "No factor summary is available from the current session."
            if counterfactuals is not None and not counterfactuals.empty and prediction:
                original = prediction.get("record", {})
                factor_counts = {}
                for _, row in counterfactuals.head(3).iterrows():
                    for feature in config["features"]:
                        if feature in row.index and original.get(feature) != row[feature]:
                            factor_counts[feature] = factor_counts.get(feature, 0) + 1
                if factor_counts:
                    factors_text = "; ".join(
                        f"{config.get('display_labels', {}).get(feature, feature.replace('_', ' ').title())} changed {count} time(s)"
                        for feature, count in sorted(
                            factor_counts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:5]
                    )
            elif prediction and prediction.get("probabilities"):
                ordered_factors = sorted(
                    prediction["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                factors_text = "; ".join(
                    f"{name}: {value:.1%}" for name, value in ordered_factors
                )

            cf_html = "<li>No counterfactuals were generated in this session.</li>"
            if counterfactuals is not None and not counterfactuals.empty:
                items = []
                original = prediction.get("record") if prediction else None
                for _, row in counterfactuals.head(3).iterrows():
                    if original:
                        changes = [
                            f"{escape(config.get('display_labels', {}).get(feature, feature))}: "
                            f"{escape(config.get('display_options', {}).get(feature, {}).get(str(original.get(feature)), str(original.get(feature))))} → "
                            f"{escape(config.get('display_options', {}).get(feature, {}).get(str(row[feature]), str(row[feature])))}"
                            for feature in config["features"]
                            if feature in row.index and original.get(feature) != row[feature]
                        ]
                    else:
                        changes = [
                            f"{escape(str(feature))} → {escape(str(row[feature]))}"
                            for feature in config["features"]
                            if feature in row.index
                        ]
                    items.append("<li>" + ("; ".join(changes) if changes else "No changed fields") + "</li>")
                cf_html = "".join(items)

            fairness_text = "Fairness check not run in this session."
            if fairness:
                fairness_text = (
                    f"Demographic-parity difference: {fairness.get('demographic_parity_difference', 'N/A')}. "
                    "This is a synthetic fairness result, not a legal finding."
                )

            html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Per-decision report</title></head>
<body><h1>Per-decision report</h1>
<p><strong>Scenario:</strong> {escape(str(dataset_name.replace(" (synthetic)", "")))}</p>
<p><strong>Prediction:</strong> {escape(prediction_text)}</p>
<h2>Top factors</h2><p>{escape(factors_text)}</p>
<h2>Plain-language counterfactuals</h2><ol>{cf_html}</ol>
<h2>Fairness result</h2><p>{escape(fairness_text)}</p>
<h2>Limitations</h2><ul>
<li>The data is synthetic and does not represent real legal or employment populations.</li>
<li>This is not a legal decision tool and must not be used to decide about real people.</li>
<li>Counterfactuals are explanatory examples, not causal advice.</li>
</ul>
<p><strong>Research-prototype disclaimer:</strong> Outputs are not validated decisions and require qualified human review.</p>
</body></html>"""
            st.session_state[f"{report_key}_html"] = html
            st.success("Your HTML decision report is ready.")
    except Exception as error:
        st.error(f"The decision report could not be prepared. Please review the inputs and try again. {error}")

if st.session_state.get(f"{report_key}_html"):
    st.download_button(
        "Download HTML decision report",
        st.session_state[f"{report_key}_html"].encode("utf-8"),
        file_name="per_decision_report.html",
        mime="text/html",
        width="stretch",
    )

with st.expander("Technical details · report inputs"):
    metric_table = pd.DataFrame(
        [
            {"Measure": key.replace("_", " ").title(), "Value": value}
            for key, value in result["metrics"].items()
            if isinstance(value, (int, float))
        ]
    )
    st.dataframe(metric_table, hide_index=True)
    st.write(
        "Dataset input fields:",
        [
            config.get("display_labels", {}).get(feature, feature)
            for feature in config["features"]
        ],
    )
