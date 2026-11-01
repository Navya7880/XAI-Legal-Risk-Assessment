import streamlit as st

from src.ui import MODEL_NAMES, get_model, page_header, setup_page

setup_page("Model information")

dataset_name = page_header(
    "Model information",
    "Held-out performance and plain-language limitations for the selected saved model.",
)

model_name = st.selectbox("Model", MODEL_NAMES)
result = get_model(dataset_name, model_name)

metrics = result["metrics"]
columns = st.columns(4)
for column, key, label in zip(
    columns,
    ("accuracy", "precision", "recall", "f1"),
    ("Accuracy", "Precision (weighted average)", "Recall (weighted average)", "F1 (weighted average)"),
):
    column.metric(label, f"{metrics[key]:.1%}" if key != "f1" else f"{metrics[key]:.3f}")

st.subheader("How it was built")
st.write(
    f"The {model_name} was trained on a held-out split of the synthetic "
    f"{dataset_name.replace(' (synthetic)', '').lower()} scenario. "
    "Input preprocessing is fitted only from the training split, and these figures "
    "come from the saved model's held-out test sample."
)

st.subheader("Limitations")
for limitation in (
    "The data is synthetic and does not represent real court, employment, or demographic populations.",
    "This tool is not a legal decision tool and should not be used to assess or decide about real people.",
    "Counterfactual suggestions are explanatory examples, not causal advice or guaranteed routes to an outcome.",
):
    st.write(f"- {limitation}")
