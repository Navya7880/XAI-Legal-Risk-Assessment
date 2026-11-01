
import streamlit as st

from src.data_processing import DATASET_CONFIG
from src.ui import (
    MODEL_NAMES,
    configure_navigation,
    get_datasets,
    page_header,
    get_model,
    setup_page,
)
from src.model_training import model_artifact_path

setup_page("Legal Risk Assessment | DiCE Explainability")


# --------------------------------------------------
# Overview Page
# --------------------------------------------------

def render_overview() -> None:
    page_header("Home", "A plain-language tool for exploring synthetic risk and hiring scenarios.")

    st.markdown(
        """
        <section class="home-hero">
            <h2>Explore model-based outcomes with care</h2>
            <p>This prototype uses saved machine-learning models to assess a hypothetical case and show which factors may be associated with the outcome.</p>
            <p>It can assess a case, explain what could change the result, check fairness, and generate a report.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/3_Assess_a_case.py",
        label="Try an example case",
        icon=":material/arrow_forward:",
        width="content",
    )

    cards = st.columns(4)
    card_content = [
        ("Assess", "Enter a case and review a predicted category."),
        ("Explain", "See what inputs could change the result."),
        ("Check fairness", "Compare outcomes across groups."),
        ("Generate a report", "Download a concise decision summary."),
    ]
    for column, (title, text) in zip(cards, card_content):
        with column:
            st.markdown(
                f'<div class="xai-card"><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.warning(
        "Research prototype using synthetic data. This tool is not a legal decision system "
        "and must not be used to make decisions about real people."
    )


# --------------------------------------------------
# Application Navigation
# --------------------------------------------------

configure_navigation(render_overview).run()
