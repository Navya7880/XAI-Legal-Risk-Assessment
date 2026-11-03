
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import DATASET_CONFIG
from src.prediction import predict_record
from src.ui import (
    MODEL_NAMES,
    FEATURE_LABELS,
    display_plotly_chart,
    feature_form,
    get_datasets,
    get_model,
    page_header,
    prediction_example_rows,
    reset_form_values,
    set_form_values,
    show_experimental_disclaimer,
    setup_page,
)

setup_page("Assess a case")

# Page heading
dataset_name = page_header(
    "Assess a case",
    "Enter case details to generate a model prediction.",
)

show_experimental_disclaimer()

# Load dataset and configuration
data = get_datasets()[dataset_name]
config = DATASET_CONFIG[dataset_name]
features = config["features"]

examples = prediction_example_rows(data, config["target"])


# --------------------------------------------------
# 1. Case details
# --------------------------------------------------

st.subheader("1. Enter case details")

st.write("Choose an example or enter the case details manually.")

# Example buttons
example_columns = st.columns(
    len(examples),
    gap="small",
)

for column, (label, row) in zip(example_columns, examples.items()):
    with column:
        st.button(
            label,
            key=f"example_{dataset_name}_{label}",
            on_click=set_form_values,
            args=("risk", row, features, dataset_name),
        )

# Reset button
reset_column = st.columns([5, 1], gap="small")[1]

with reset_column:
    st.button(
        "Reset inputs",
        key=f"reset_{dataset_name}",
        on_click=reset_form_values,
        args=("risk", features),
    )

# Previous form values
initial_record = (
    st.session_state.get("prediction_case")
    if st.session_state.get("prediction_case_dataset") == dataset_name
    else None
)

# Input form
with st.form(f"prediction_form_{dataset_name}"):

    if dataset_name == "Legal risk (synthetic)":

        display_config = config
        label_map = display_config.get("display_labels", {})
        option_map = display_config.get("display_options", {})
        help_map = display_config.get("field_help", {})
        fixed_features = set(config.get("immutable_features", []))

        st.subheader("About the person")

        about_columns = st.columns(3)

        with about_columns[0]:
            age_record = feature_form(
                data, ["age"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        with about_columns[1]:
            offense_record = feature_form(
                data, ["prior_offenses"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        with about_columns[2]:
            type_record = feature_form(
                data, ["offense_category"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        st.subheader("Support and rehabilitation")

        support_columns = st.columns(2)

        with support_columns[0]:
            program_record = feature_form(
                data, ["program_completion"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        with support_columns[1]:
            treatment_record = feature_form(
                data, ["substance_treatment"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        st.subheader("Qualification")

        qualification_columns = st.columns(2)

        with qualification_columns[0]:
            work_record = feature_form(
                data, ["employment_status"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        with qualification_columns[1]:
            education_record = feature_form(
                data, ["education_level"], "risk",
                initial_values=initial_record,
                label_map=label_map,
                option_map=option_map,
                help_map=help_map,
                fixed_features=fixed_features,
            )

        left_record = {}
        left_record.update(age_record)
        left_record.update(offense_record)
        left_record.update(type_record)

        middle_record = {}
        middle_record.update(program_record)
        middle_record.update(treatment_record)
        middle_record.update(work_record)

        right_record = {}
        right_record.update(education_record)

    elif len(features) >= 3:

        left_features = features[:2]
        middle_features = features[2:]
        right_features = []

    else:

        left_features = features
        middle_features = []
        right_features = []

    if dataset_name != "Legal risk (synthetic)":

        left_column, middle_column, right_column = st.columns(
            3, gap="large"
        )

        with left_column:
            left_record = feature_form(
                data, left_features, "risk",
                initial_values=initial_record,
            )

        with middle_column:
            middle_record = feature_form(
                data, middle_features, "risk",
                initial_values=initial_record,
            )

        with right_column:
            right_record = feature_form(
                data, right_features, "risk",
                initial_values=initial_record,
            )

    # Model selection
    best_by_dataset = st.session_state.get(
        "best_model_by_dataset", {}
    )

    default_model = best_by_dataset.get(
        dataset_name, MODEL_NAMES[0]
    )

    model_name = st.selectbox(
        "Select model",
        MODEL_NAMES,
        index=(
            MODEL_NAMES.index(default_model)
            if default_model in MODEL_NAMES
            else 0
        ),
        key=f"prediction_model_{dataset_name}",
    )

    submitted = st.form_submit_button(
        "See the model's prediction",
        type="primary",
    )

    record = {**left_record, **middle_record, **right_record}


# --------------------------------------------------
# 2. Model prediction
# --------------------------------------------------

if submitted:

    try:

        if set(record) != set(features):
            raise ValueError(
                "Choose a value for every input before continuing."
            )

        with st.spinner("Generating prediction..."):

            model_result = get_model(dataset_name, model_name)

            prediction_state = predict_record(
                model_result,
                record,
            )

        # Save prediction details
        st.session_state["latest_prediction"] = prediction_state
        st.session_state["latest_record"] = record.copy()
        st.session_state["latest_dataset"] = dataset_name
        st.session_state["latest_model"] = model_name

        st.session_state["prediction_case"] = record.copy()
        st.session_state["prediction_case_dataset"] = dataset_name

        st.session_state.setdefault(
            "model_results_by_dataset", {}
        ).setdefault(dataset_name, {})[model_name] = model_result

        st.session_state.setdefault(
            "best_model_by_dataset", {}
        ).setdefault(dataset_name, model_name)

    except (ValueError, TypeError) as error:

        st.warning(
            f"Could not generate the prediction. Please review the inputs. {error}"
        )


# --------------------------------------------------
# 3. Prediction result
# --------------------------------------------------

st.divider()
st.subheader("2. Prediction result")

prediction_state = st.session_state.get("latest_prediction")

matching_previous = (
    st.session_state.get("latest_dataset") == dataset_name
    and st.session_state.get("latest_model")
    == st.session_state.get(
        f"prediction_model_{dataset_name}", MODEL_NAMES[0]
    )
)

if matching_previous and prediction_state:

    predicted = prediction_state["prediction"]
    probabilities = prediction_state.get("probabilities") or {}

    # Predicted category
    st.markdown("**Predicted category**")

    category = str(predicted)

    category_class = (
        "result-lower"
        if category.lower() == "lower"
        else "result-elevated"
    )

    st.markdown(
        f'<span class="result-badge {category_class}">{category}</span>',
        unsafe_allow_html=True,
    )

    # Confidence score
    confidence = max(probabilities.values(), default=None)

    if confidence is not None:

        st.write(f"**Confidence score:** {confidence:.1%}")

        # Category scores
        st.markdown("**Model scores for each category**")

        probability_data = pd.DataFrame(
            [
                {
                    "Category": label,
                    "Model score": value,
                }
                for label, value in probabilities.items()
            ]
        )

        display_plotly_chart(
            px.bar(
                probability_data,
                x="Model score",
                y="Category",
                orientation="h",
                range_x=[0, 1],
                text=probability_data["Model score"].map(
                    lambda value: f"{value:.0%}"
                ),
                labels={
                    "Model score": "Model score (0–100%)"
                },
            ),
            font_size=st.session_state.font_px,
            width="stretch",
        )

        st.caption(
            "These scores represent the model's preference between "
            "categories. They are not probabilities of reoffending "
            "or guarantees about an individual."
        )

    # Input summary
    st.divider()
    st.markdown("**Input summary**")

    summary_data = pd.DataFrame(
        [
            {
                "Input": FEATURE_LABELS.get(feature, feature),
                "Chosen value": str(value),
            }
            for feature, value in st.session_state[
                "latest_record"
            ].items()
        ]
    )

    st.dataframe(
        summary_data,
        hide_index=True,
        use_container_width=True,
    )

    # DiCE navigation
    st.divider()

    st.markdown("**Want to understand this prediction?**")

    st.page_link(
        "pages/4_What_could_change_this.py",
        label="Explore DiCE explanations",
    )

else:

    st.info(
        "Enter the case details and click Predict. "
        "Your prediction result will appear here."
    )
