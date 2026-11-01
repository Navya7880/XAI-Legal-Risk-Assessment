import pandas as pd
import plotly.express as px
import streamlit as st
import time

from src.data_processing import DATASET_CONFIG
from src.prediction import predict_record
from src.ui import (
    FEATURE_LABELS,
    display_plotly_chart,
    feature_form,
    get_datasets,
    get_model,
    page_header,
    show_experimental_disclaimer,
    setup_page,
)

setup_page("What could change this?")

dataset_name = page_header(
    "What could change this?",
    "Explore constrained input changes that can alter the model's prediction.",
)
show_experimental_disclaimer()
data = get_datasets()[dataset_name]
config = DATASET_CONFIG[dataset_name]
model_name = (
    st.session_state.get("latest_model")
    if st.session_state.get("latest_dataset") == dataset_name
    else st.session_state.get("best_model_by_dataset", {}).get(
        dataset_name, "Random Forest"
    )
)
model_result = get_model(dataset_name, model_name)
features = config["features"]
immutable_features = config["immutable_features"]
actionable_features = [
    feature for feature in features if feature not in immutable_features
]

case_from_prediction = (
    st.session_state.get("latest_record")
    if st.session_state.get("latest_dataset") == dataset_name
    else None
)
case_from_prediction = case_from_prediction or (
    st.session_state.get("prediction_case")
    if st.session_state.get("prediction_case_dataset") == dataset_name
    else None
)
if case_from_prediction:
    record = {feature: case_from_prediction[feature] for feature in features}
else:
    record = data.iloc[0][features].to_dict()
original_prediction = predict_record(model_result, record)
prediction_label = str(original_prediction["prediction"])
if dataset_name == "Legal risk (synthetic)" and prediction_label.lower() == "lower":
    st.subheader("What could raise this?")
    st.info("The starting result is Lower. These alternatives show changes that may worsen the outcome toward Elevated.")
else:
    st.subheader("What could change this?")
    if dataset_name == "Legal risk (synthetic)":
        st.info("The starting result is Elevated. This search looks for realistic changes that may lower the outcome.")
input_parts = []
for name, value in record.items():
    label = config.get("display_labels", {}).get(name, FEATURE_LABELS.get(name, name))
    display_value = config.get("display_options", {}).get(name, {}).get(str(value), str(value))
    input_parts.append(f"{label}: {display_value}")
st.markdown(
    '<div class="xai-card"><h3>Starting case</h3>'
    f"<p><strong>Model result:</strong> {original_prediction['prediction']}<br>"
    f"<strong>Inputs:</strong> {' · '.join(input_parts)}"
    "</p></div>",
    unsafe_allow_html=True,
)
if not case_from_prediction:
    st.info("This is a real example row from the selected synthetic dataset.")
if immutable_features:
    locked = [feature for feature in immutable_features if feature in features]
    st.caption(
        f"Held fixed for this search: {', '.join(FEATURE_LABELS.get(item, item) for item in locked) or 'no model inputs'}."
        " These values stay fixed because they are not considered changeable in this example."
    )
    if config.get("sensitive"):
        st.caption(
            f"{config['sensitive'].replace('_', ' ').title()} is excluded from the model "
            "and is never changed."
        )

number_of_options = st.slider(
    "How many alternatives?",
    min_value=1,
    max_value=5,
    value=3,
)
realistic_constraints = st.toggle(
    "Realistic constraints on/off",
    value=True,
)
with st.expander("Advanced settings", expanded=False):
    method = st.selectbox(
        "Search method",
        ["genetic", "random", "kdtree"],
        help="Different search strategies look for model inputs that change the predicted category.",
    )
    vary = st.multiselect(
        "Fields allowed to change",
        actionable_features,
        default=actionable_features,
        format_func=lambda feature: config.get("display_labels", {}).get(
            feature,
            FEATURE_LABELS.get(feature, feature),
        ),
    )
    st.caption("Locked fields stay unchanged. Only the selected fields may vary.")

with st.expander("Edit the starting case", expanded=False):
    record = feature_form(
        data,
        features,
        "dice",
        initial_values=record,
        label_map=config.get("display_labels", {}),
        option_map=config.get("display_options", {}),
        help_map=config.get("field_help", {}),
        fixed_features=set(config.get("immutable_features", [])),
    )
    st.caption("Changing these values creates a what-if from the edited case.")

generate = st.button(
    "Find possible changes",
    type="primary",
    disabled=not bool(vary),
)
if not vary:
    st.warning("Choose at least one field that may change.")

if generate:
    try:
        from src.dice_explainer import (
            compare_counterfactual,
            counterfactual_quality_metrics,
            generate_counterfactuals,
        )

        started_at = time.perf_counter()
        with st.spinner("Searching for alternatives and checking each with the model…"):
            counterfactuals = generate_counterfactuals(
                model_result,
                record,
                total_cfs=number_of_options,
                features_to_vary=vary,
                method=method,
                immutable_features=immutable_features,
                permitted_range=config["permitted_range"],
                constraint_mode="full" if realistic_constraints else "ranges",
            )
            runtime_seconds = time.perf_counter() - started_at
            quality = counterfactual_quality_metrics(
                model_result,
                record,
                counterfactuals,
                number_of_options,
                permitted_range=config["permitted_range"],
            )
            quality["runtime"] = runtime_seconds
        used_method = counterfactuals.attrs.get("method_used", method)
        notice = counterfactuals.attrs.get("method_notice")
        if notice:
            st.warning(f"{notice} You can try another search method in Advanced settings.")
        rejection_rate = counterfactuals.attrs.get("rejection_rate", 0.0)
        rejected_count = counterfactuals.attrs.get("rejected_count", 0)
        st.caption(
            f"Feasibility post-filter: {rejected_count} rejected "
            f"({rejection_rate:.1%} rejection rate)."
        )
        st.session_state["latest_prediction"] = original_prediction
        st.session_state["latest_record"] = record
        st.session_state["latest_dataset"] = dataset_name
        st.session_state["latest_model"] = model_name
        st.session_state["latest_counterfactuals"] = counterfactuals
        st.session_state["latest_counterfactual_dataset"] = dataset_name
        st.session_state["latest_counterfactual_model"] = model_name
        st.session_state["latest_counterfactual_quality"] = quality
        st.session_state["latest_counterfactual_method"] = used_method

        st.subheader("Possible alternatives")
        st.caption(f"Search method used: {used_method}.")
        if counterfactuals.empty:
            st.warning(
                "No realistic change found for this case. Try allowing more fields to vary."
            )
        else:
            changed_counts = []
            chart_rows = []
            for index, row in counterfactuals.iterrows():
                changes = compare_counterfactual(record, row)
                changed_counts.append(len(changes))
                candidate = row[features].to_dict()
                new_prediction = predict_record(model_result, candidate)["prediction"]
                changes_text = "<br>".join(
                    f"<strong>{config.get('display_labels', {}).get(change.feature, FEATURE_LABELS.get(change.feature, change.feature))}:</strong> "
                    f"{config.get('display_options', {}).get(change.feature, {}).get(str(change.original), str(change.original))} → "
                    f"{config.get('display_options', {}).get(change.feature, {}).get(str(change.counterfactual), str(change.counterfactual))}"
                    for change in changes.itertuples(index=False)
                )
                st.markdown(
                    f'<div class="xai-card"><h3>Option {index + 1}</h3>'
                    f"<p>{changes_text}<br><strong>New model result:</strong> "
                    f"{new_prediction}</p></div>",
                    unsafe_allow_html=True,
                )
                for change in changes.itertuples(index=False):
                    label = FEATURE_LABELS.get(change.feature, change.feature)
                    chart_rows.extend(
                        [
                            {"Field": label, "Value": str(change.original), "State": "Before"},
                            {
                                "Field": label,
                                "Value": str(change.counterfactual),
                                "State": "After",
                            },
                        ]
                    )
            smallest = min(changed_counts)
            st.write(
                f"Smallest change needed: **{smallest} "
                f"{'field' if smallest == 1 else 'fields'}**."
            )
            if chart_rows:
                chart = px.scatter(
                    pd.DataFrame(chart_rows),
                    x="Field",
                    y="Value",
                    color="State",
                    symbol="State",
                    title="Before and after values for changed fields",
                )
                chart.update_traces(marker={"size": 13})
                display_plotly_chart(
                    chart,
                    font_size=st.session_state.font_px,
                    width="stretch",
                )

            with st.expander("How good are these alternatives?", expanded=False):
                st.caption(
                    "These measures describe the generated examples, not whether a change "
                    "is practical or advisable."
                )
                metric_rows = [
                    ("Validity", f"{quality['validity']:.3f}", "Share of the requested options that change the model result."),
                    ("Proximity", f"{quality['proximity']:.3f}", "How close the alternatives are to the starting case; lower is closer."),
                    ("Sparsity", f"{quality['sparsity']:.3f}", "Average number of fields changed in each alternative; fewer is simpler."),
                    ("Diversity", f"{quality['diversity']:.3f}", "How different the alternatives are from each other; higher is more varied."),
                    ("Runtime", f"{quality['runtime']:.3f} s", "Time spent generating and checking the alternatives for this case."),
                ]
                st.table(pd.DataFrame(metric_rows, columns=["Metric", "Value", "Plain-language meaning"]))

            shap_section = st.container()
            try:
                import numpy as np
                import shap

                pipeline = model_result["pipeline"]
                preprocessor = pipeline.named_steps["preprocessor"]
                classifier = pipeline.named_steps["classifier"]
                training_frame = model_result["training_data"][features].fillna("")
                background = preprocessor.transform(training_frame.head(200))
                query_frame = pd.DataFrame([record], columns=features)
                query_encoded = preprocessor.transform(query_frame)
                encoded_names = preprocessor.get_feature_names_out(features)

                def original_name(encoded_name: str) -> str:
                    name = str(encoded_name).split("__")[-1]
                    if "_" in name:
                        prefix = name.split("_", 1)[0]
                        if prefix in features:
                            return prefix
                    return next(
                        (feature for feature in features if feature in name),
                        name,
                    )

                shap_explanation = shap.Explainer(classifier.predict_proba, background)(query_encoded)
                shap_values = np.asarray(shap_explanation.values)
                if shap_values.ndim == 3:
                    shap_values = shap_values[0]
                    shap_values = np.mean(np.abs(shap_values), axis=1)
                elif shap_values.ndim == 2:
                    shap_values = shap_values[0]
                else:
                    shap_values = shap_values.reshape(-1)
                shap_scores = {}
                for name, value in zip(encoded_names, np.abs(np.asarray(shap_values).reshape(-1))):
                    label = FEATURE_LABELS.get(original_name(name), original_name(name))
                    shap_scores[label] = shap_scores.get(label, 0.0) + float(value)
                shap_top = sorted(shap_scores.items(), key=lambda item: item[1], reverse=True)[:5]

                with shap_section:
                    st.subheader("SHAP top 5")
                    st.caption("SHAP: estimated contribution size for the same starting case.")
                    st.table(pd.DataFrame(shap_top, columns=["Factor", "Attribution strength"]))
            except Exception as error:
                st.subheader("SHAP top 5")
                st.info(f"SHAP could not be displayed: {error}")

            download_data = counterfactuals.copy()
            for metric, value in quality.items():
                download_data[metric] = value
            st.download_button(
                "Download alternatives as CSV",
                download_data.to_csv(index=False).encode("utf-8"),
                "dice_counterfactuals.csv",
                "text/csv",
            )
        with st.expander("Technical details", expanded=False):
            st.write("Model inputs:", features)
            st.write("Search method:", used_method)
            st.write("Immutable fields:", immutable_features)
            st.dataframe(counterfactuals, hide_index=True, width="stretch")
    except ModuleNotFoundError as error:
        if error.name != "dice_ml":
            raise
        st.error(
            "The explanation engine is not installed in the Python environment running "
            "Streamlit. Activate the supported environment and run "
            "`python -m pip install -r requirements.txt`, then restart the app."
        )
    except (ValueError, TypeError, RuntimeError) as error:
        st.error(
            f"The search could not finish. Check the selected fields and try again. Details: {error}"
        )
else:
    previous = st.session_state.get("latest_counterfactuals")
    if (
        st.session_state.get("latest_counterfactual_dataset") == dataset_name
        and previous is not None
    ):
        st.info("Your most recent alternatives are available below after running a new search.")

st.caption("These are model what-ifs, not advice or guaranteed outcomes.")
