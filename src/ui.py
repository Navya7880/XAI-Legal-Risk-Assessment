"""Shared Streamlit helpers used by the multipage application."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from src.data_processing import DATASET_CONFIG, ensure_datasets
from src.model_training import load_model, model_artifact_path


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAMES = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
]

FEATURE_LABELS = {
    "age": "Age",
    "prior_offenses": "Number of previous offenses",
    "offense_category": "Type of current offense",
    "employment_status": "Employment status",
    "education": "Education",
    "experience_years": "Years of experience",
    "technical_score": "Technical assessment",
    "assessment_score": "General assessment",
    "referral_network_score": "Referral network score",
    "education_level": "Highest education",
}

FEATURE_HELP = {
    "age": "Choose an age from the range in the synthetic dataset.",
    "prior_offenses": "Past convictions on record. This cannot be reduced.",
    "offense_category": "Choose the category used in this synthetic example.",
    "employment_status": "Choose a synthetic employment category.",
    "education": "Choose the education category used in the synthetic example.",
    "experience_years": "Choose years of experience in the synthetic dataset range.",
    "technical_score": "Choose a technical assessment score from 0 to 100.",
    "assessment_score": "Choose a general assessment score from 0 to 100.",
    "referral_network_score": "A synthetic historical proxy; it is locked for explanations.",
}


# --------------------------------------------------
# FOUR MAIN NAVIGATION SECTIONS
# --------------------------------------------------

PAGE_GROUPS = {
    "": [
        (
            "Home",
            None,
            ":material/home:",
        ),
        (
            "Assess a case",
            "pages/3_Assess_a_case.py",
            ":material/query_stats:",
        ),
        (
            "What could change this?",
            "pages/4_What_could_change_this.py",
            ":material/psychology:",
        ),
        (
            "Fairness check",
            "pages/5_Fairness_check.py",
            ":material/balance:",
        ),
        (
            "Report",
            "pages/6_Report.py",
            ":material/description:",
        ),
        (
            "Model information",
            "pages/7_Model_information.py",
            ":material/info:",
        ),
    ],
}


NAVIGATION_GROUPS = []
CURRENT_PAGE_TITLE = "Overview"


# --------------------------------------------------
# NAVIGATION CONFIGURATION
# --------------------------------------------------

def configure_navigation(overview) -> st.StreamlitPage:
    """Configure four main navigation sections."""

    global NAVIGATION_GROUPS, CURRENT_PAGE_TITLE

    overview_page = st.Page(
        overview,
        title="Home",
        icon=":material/dashboard:",
        default=True,
    )

    page_groups = {
        "": [overview_page]
    }

    NAVIGATION_GROUPS = [
        ("", [(overview_page, "Home", ":material/dashboard:")])
    ]

    for group_name, pages in PAGE_GROUPS.items():

        streamlit_pages = []
        sidebar_pages = []

        for title, path, icon in pages:
            if path is None:
                continue

            page = st.Page(
                path,
                title=title,
                icon=icon,
            )

            streamlit_pages.append(page)
            sidebar_pages.append((page, title, icon))

        if group_name == "":
            page_groups[""] = [overview_page] + streamlit_pages
        else:
            page_groups[group_name] = streamlit_pages

        NAVIGATION_GROUPS.append(
            (group_name, sidebar_pages)
        )

    current_page = st.navigation(
        page_groups,
        position="hidden",
    )

    CURRENT_PAGE_TITLE = current_page.title

    return current_page


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

@st.cache_data
def get_datasets():
    return ensure_datasets()


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model(
    dataset_name: str,
    model_name: str = "Random Forest",
):
    config = DATASET_CONFIG[dataset_name]

    artifact = model_artifact_path(
        dataset_name,
        model_name,
    )

    if artifact.exists():

        loaded = load_model(artifact)

        if (
            loaded.get("features") != config["features"]
            or loaded.get("target") != config["target"]
            or loaded.get("model_name") != model_name
        ):
            raise ValueError(
                f"Saved model at {artifact} does not match this workflow."
            )

        return loaded

    st.error(
        "This saved model is not available. "
        "Developer: run `python -m src.models.train` to train and save the models."
    )
    st.stop()
    # --------------------------------------------------
# SHARED STYLES – PREMIUM CORPORATE THEME
# --------------------------------------------------

def inject_shared_styles() -> None:
    """Apply the shared Premium Corporate theme to the application."""

    palette = [
        "#365D86",
        "#6B86A3",
        "#172B45",
        "#A9B8C8",
        "#64748B",
    ]

    # Plotly theme
    pio.templates["legal_risk_assessment"] = pio.templates["plotly_white"]

    pio.templates["legal_risk_assessment"].update(
        layout={
            "colorway": palette,
            "font": {
                "family": "Inter, Segoe UI, Arial, sans-serif",
                "color": "#26364A",
            },
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "xaxis": {
                "gridcolor": "#E2E8F0",
                "zerolinecolor": "#CBD5E1",
            },
            "yaxis": {
                "gridcolor": "#E2E8F0",
                "zerolinecolor": "#CBD5E1",
            },
        }
    )

    px.defaults.template = "legal_risk_assessment"
    px.defaults.color_discrete_sequence = palette

    px.defaults.color_continuous_scale = [
        "#EAF0F6",
        "#6B86A3",
        "#365D86",
        "#172B45",
    ]

    # Streamlit CSS
    st.markdown(
        """
        <style>

        /* -----------------------------------------
           PREMIUM CORPORATE COLOUR VARIABLES
        ----------------------------------------- */

        :root {
            --primary: #365D86;
            --primary-dark: #294967;
            --accent-soft: #EAF0F6;

            --surface: #FFFFFF;
            --surface-soft: #F7F8FA;

            --text: #26364A;
            --text-muted: #64748B;

            --border: #E2E8F0;
        }

        /* -----------------------------------------
           MAIN APPLICATION
        ----------------------------------------- */

        .stApp {
            background-color: var(--surface-soft);
            color: var(--text);
        }

        .main .block-container {
            max-width: none;
            width: 100%;
            padding-top: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 650;
        }

        h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }

        h2 {
            font-size: 1.35rem;
        }

        h3 {
            font-size: 1.15rem;
        }

        p, label, span {
            color: inherit;
            font-size: 1rem;
        }

        .stCaption,
        [data-testid="stCaptionContainer"],
        small {
            font-size: 0.95rem !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [role="radiogroup"] label,
        [data-testid="stWidgetLabel"] p {
            font-size: 1rem;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stPageLink > a,
        .stFormSubmitButton > button {
            font-size: 1rem !important;
            min-height: 44px;
        }

        [data-testid="stSlider"] label,
        [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
            font-size: 1rem;
        }

        /* -----------------------------------------
           SIDEBAR
        ----------------------------------------- */

        [data-testid="stSidebar"] {
            background-color: var(--surface);
            border-right: 1px solid var(--border);
            width: 16rem !important;
            min-width: 16rem !important;
        }

        [data-testid="stSidebar"] > div {
            width: 16rem !important;
            min-width: 16rem !important;
            padding-top: 1rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        [data-testid="stSidebar"] h2 {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
        }

        [data-testid="stSidebar"] a {
            min-height: 48px;
            padding: 0.65rem 0.75rem;
            font-size: 1rem;
            line-height: 1.2;
        }

        [data-testid="stSidebar"] .xai-sidebar-heading {
            margin-top: 0.85rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-muted);
        }

        .xai-sidebar-rule {
            height: 1px;
            background-color: var(--border);
            margin: 1rem 0;
        }

        .xai-sidebar-heading {
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-top: 1.4rem;
            margin-bottom: 0.45rem;
        }

        [data-testid="stSidebar"] a {
            border-radius: 6px;
            color: var(--text);
            transition: background-color 0.15s ease;
        }

        [data-testid="stSidebar"] a:hover {
            background-color: var(--accent-soft);
            color: var(--primary-dark);
        }

        [data-testid="stSidebar"] a[aria-current="page"] {
            background-color: var(--accent-soft);
            color: var(--primary-dark);
            border-left: 3px solid var(--primary);
            font-weight: 650;
        }

        /* -----------------------------------------
           METRIC CARDS
        ----------------------------------------- */

        [data-testid="stMetric"] {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 700;
        }

        /* -----------------------------------------
           INFORMATION CARDS
        ----------------------------------------- */

        .xai-card,
        .xai-feature-card,
        .xai-dataset-note {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.8rem;
        }

        .xai-card h3 {
            color: var(--text);
            font-size: 1rem;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }

        .xai-card p,
        .xai-feature-card p,
        .xai-dataset-note p {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
            margin-bottom: 0;
        }

        /* -----------------------------------------
           RESULT AND STEP CONTAINERS
        ----------------------------------------- */

        .xai-result,
        .xai-step {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        /* -----------------------------------------
           WELCOME SECTION
        ----------------------------------------- */

        .xai-welcome {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
        }

        .xai-eyebrow {
            color: var(--primary);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .xai-welcome h2 {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 0.6rem;
        }

        .xai-welcome p {
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }

        .xai-welcome-note {
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top: 0.8rem;
        }

        /* -----------------------------------------
           BUTTONS
        ----------------------------------------- */

        .stButton > button,
        .stDownloadButton > button {
            background-color: var(--primary);
            color: #FFFFFF;
            border: 1px solid var(--primary);
            border-radius: 6px;
            font-weight: 600;
            min-height: 44px;
            transition: background-color 0.15s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background-color: var(--primary-dark);
            color: #FFFFFF;
            border-color: var(--primary-dark);
        }

        /* -----------------------------------------
           INPUTS
        ----------------------------------------- */

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] {
            border-color: var(--border);
            border-radius: 6px;
        }

        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"] > div:focus-within {
            border-color: var(--primary);
        }

        [data-testid="stSelectbox"],
        [data-testid="stSlider"],
        [data-testid="stRadio"] {
            margin-bottom: 0.45rem;
        }

        .stTabs [role="tab"] {
            min-height: 44px;
            font-size: 0.95rem;
            font-weight: 600;
        }

        [data-testid="stForm"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            background-color: var(--surface);
        }

        /* -----------------------------------------
           DATA TABLES
        ----------------------------------------- */

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        /* -----------------------------------------
           ALERTS
        ----------------------------------------- */

        [data-testid="stAlert"] {
            border-radius: 7px;
            border: 1px solid var(--border);
        }

        /* -----------------------------------------
           DIVIDERS
        ----------------------------------------- */

        hr {
            border-color: var(--border);
        }

        /* -----------------------------------------
           RESPONSIVE LAYOUT
        ----------------------------------------- */

        @media (max-width: 720px) {

            .main .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .xai-welcome {
                padding: 1rem;
            }

            .xai-welcome h2 {
                font-size: 1.25rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_page(title: str) -> None:
    """Configure one Streamlit page and its shared display settings."""

    st.set_page_config(
        page_title=f"{title} | Legal Risk Assessment",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_shared_styles()

    st.session_state.setdefault("font_px", 18)

    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            font-size: 18px !important;
            line-height: 1.6 !important;
            font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
        }}
        .main .block-container {{
            max-width: none !important;
            width: 100% !important;
            padding-top: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            box-sizing: border-box !important;
        }}
        .stApp, [data-testid="stAppViewContainer"] {{
            overflow-x: hidden !important;
        }}
        section[data-testid="stMain"] {{
            width: 100% !important;
            max-width: none !important;
        }}
        p, li, ul, ol,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stAlert"],
        [data-testid="stAlert"] *,
        div[data-testid="stNotification"],
        div[data-testid="stNotification"] * {{
            font-size: 18px !important;
            line-height: 1.6 !important;
        }}
        .stButton > button,
        .stDownloadButton > button,
        .stPageLink > a,
        .stFormSubmitButton > button,
        button[kind] {{
            font-size: 17px !important;
            min-height: 2.8rem !important;
            padding: 0.5rem 1.2rem !important;
        }}
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [role="radiogroup"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label {{
            font-size: 17px !important;
            line-height: 1.5 !important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {{
            font-size: 17px !important;
            min-height: 2.8rem !important;
        }}
        [data-testid="stSidebar"] a {{
            font-size: 18px !important;
            min-height: 44px !important;
            padding: 0.6rem 1rem !important;
            line-height: 1.35 !important;
            font-weight: 650 !important;
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {{
            width: 16rem !important;
            min-width: 16rem !important;
            max-width: 16rem !important;
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {{
            overflow-wrap: anywhere !important;
            white-space: normal !important;
        }}
        [data-testid="stSidebar"] {{
            height: 100vh !important;
            overflow-y: auto !important;
        }}
        .stCaption,
        [data-testid="stCaptionContainer"],
        small,
        [data-testid="stCaptionContainer"] * {{
            font-size: 15px !important;
            line-height: 1.45 !important;
        }}
        [data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stSlider"] svg text,
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
            font-size: 16px !important;
        }}
        [data-testid="stSelectbox"] *,
        [data-baseweb="select"] *,
        [data-baseweb="menu"] *,
        [role="option"] * {{
            font-size: 17px !important;
        }}
        input[type="radio"],
        input[type="checkbox"] {{
            transform: scale(1.2) !important;
            transform-origin: center !important;
        }}
        h1 {{
            font-size: 40px !important;
        }}
        h2 {{
            font-size: 32px !important;
        }}
        h3 {{
            font-size: 26px !important;
        }}
        h4 {{
            font-size: 22px !important;
        }}
        h1, h2, h3, h4 {{
            font-weight: 600 !important;
            line-height: 1.25 !important;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {{
            font-size: 16px !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 32px !important;
            font-weight: 700 !important;
        }}
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"],
        [data-testid="stTable"] * {{
            font-size: 16px !important;
        }}
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stDataFrame"] td,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {{
            min-height: 38px !important;
            height: 38px !important;
        }}
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stTable"] th {{
            font-size: 16px !important;
            font-weight: 700 !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {{
            font-size: 17px !important;
        }}
        .stTabs [role="tab"],
        .stTabs [role="tab"] p {{
            font-size: 18px !important;
            padding: 0.6rem 1.2rem !important;
        }}
        [role="tooltip"],
        [role="tooltip"] *,
        [data-baseweb="tooltip"],
        [data-baseweb="tooltip"] * {{
            font-size: 16px !important;
        }}
        .xai-sidebar-heading,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-size: 16px !important;
            font-weight: 700 !important;
        }}
        @media (max-width: 720px) {{
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div {{
                width: min(16rem, 85vw) !important;
                min-width: min(16rem, 85vw) !important;
                max-width: min(16rem, 85vw) !important;
            }}
            .main .block-container {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
            }}
            [data-testid="stSidebar"] a {{
                min-height: 52px !important;
            }}
            [data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
            }}
        }}
        @media (min-width: 721px) and (max-width: 1100px) {{
            .main .block-container {{
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }}
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div {{
                width: 16rem !important;
                min-width: 16rem !important;
                max-width: 16rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    style_path = Path(__file__).resolve().parents[1] / "app" / "style.py"
    style_spec = importlib.util.spec_from_file_location("_app_shared_style", style_path)
    if style_spec and style_spec.loader:
        style_module = importlib.util.module_from_spec(style_spec)
        style_spec.loader.exec_module(style_module)
        style_module.apply_style()

    # --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

def render_sidebar() -> str:

    inject_shared_styles()

    dataset_names = list(DATASET_CONFIG)

    if not dataset_names:
        raise ValueError("No datasets are configured.")

    if st.session_state.get("selected_dataset") not in dataset_names:
        st.session_state["selected_dataset"] = dataset_names[0]

    with st.sidebar:

        st.markdown("## Legal Risk Assessment")
        st.caption("Model Explainability")

        st.markdown(
            '<div class="xai-sidebar-rule"></div>',
            unsafe_allow_html=True,
        )

        # Scenario selection
        scenario_labels = {
            "Legal risk (synthetic)": "Legal risk",
            "Hiring (synthetic)": "Hiring",
        }
        current_label = scenario_labels.get(
            st.session_state["selected_dataset"],
            st.session_state["selected_dataset"],
        )
        selected_label = st.selectbox(
            "Scenario",
            list(scenario_labels.values()),
            index=list(scenario_labels.values()).index(current_label),
            key="selected_scenario",
            label_visibility="collapsed",
        )
        selected_dataset = next(
            (key for key, label in scenario_labels.items() if label == selected_label),
            st.session_state["selected_dataset"],
        )
        st.session_state["selected_dataset"] = selected_dataset

        # Four main navigation sections
        for group_name, pages in NAVIGATION_GROUPS:
            for page, label, icon in pages:

                st.page_link(
                    page,
                    label=label,
                    icon=icon,
                )

    return st.session_state["selected_dataset"]


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

def page_header(
    title: str,
    description: str,
    step: int | None = None,
) -> str:

    dataset = render_sidebar()

    st.markdown(
        f"""
        <div class="page-title-banner">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return dataset


# --------------------------------------------------
# EXPERIMENTAL DISCLAIMER
# --------------------------------------------------

def show_experimental_disclaimer() -> None:

    st.caption(
        "Synthetic data for research and demonstration. "
        "Outputs are not legal findings or decisions about real people."
    )


# --------------------------------------------------
# PREDICTION EXAMPLES AND FORM STATE
# --------------------------------------------------

def prediction_example_rows(
    data: pd.DataFrame,
    target: str,
) -> dict[str, dict[str, Any]]:
    """Return representative real rows for quick prediction examples."""

    if target == "risk_category":
        labels = {
            "Lower": "Low-risk example",
            "Elevated": "Elevated-risk example",
        }
        rows: dict[str, dict[str, Any]] = {}
        for outcome, label in labels.items():
            matches = data[data[target] == outcome]
            if not matches.empty:
                rows[label] = matches.iloc[0].drop(labels=[target]).to_dict()
        if rows:
            return rows

    if target == "hired":
        labels = {
            "No": "Likely-not-hired example",
            "Yes": "Likely-hired example",
        }
        rows = {}
        for outcome, label in labels.items():
            matches = data[data[target] == outcome]
            if not matches.empty:
                rows[label] = matches.iloc[0].drop(labels=[target]).to_dict()
        if rows:
            return rows

    return {
        "Representative example": data.iloc[0].drop(
            labels=[target],
            errors="ignore",
        ).to_dict()
    }


def set_form_values(
    prefix: str,
    row: dict[str, Any],
    features: list[str],
    dataset_name: str,
) -> None:
    """Populate prediction controls from an example row."""

    st.session_state["prediction_case"] = {
        feature: row[feature]
        for feature in features
        if feature in row
    }
    st.session_state["prediction_case_dataset"] = dataset_name
    for feature in features:
        if feature in row:
            st.session_state[f"{prefix}_{feature}"] = row[feature]


def reset_form_values(prefix: str, features: list[str]) -> None:
    """Reset prediction controls without changing the selected dataset."""

    st.session_state.pop("prediction_case", None)
    st.session_state.pop("prediction_case_dataset", None)
    for feature in features:
        st.session_state.pop(f"{prefix}_{feature}", None)


# --------------------------------------------------
# INFORMATION CARD
# --------------------------------------------------

def info_card(
    title: str,
    description: str,
    icon: str = "",
) -> None:

    st.markdown(
        f'<div class="xai-card"><h3>{icon} {title}</h3>'
        f"<p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def display_plotly_chart(
    figure,
    *,
    font_size: int | None = None,
    **streamlit_kwargs,
) -> None:
    """Display a Plotly figure with the shared readable font size."""
    figure.update_layout(
        font=dict(size=16),
        title_font_size=22,
        legend_font_size=16,
        height=500,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        margin={"l": 80, "r": 40, "t": 80, "b": 80},
    )
    st.plotly_chart(figure, **streamlit_kwargs)


# --------------------------------------------------
# FEATURE INPUT FORM
# --------------------------------------------------

def feature_form(
    data: pd.DataFrame,
    features: list[str],
    prefix: str,
    initial_values: dict[str, Any] | None = None,
    label_map: dict[str, str] | None = None,
    option_map: dict[str, dict[str, str]] | None = None,
    help_map: dict[str, str] | None = None,
    fixed_features: set[str] | None = None,
) -> dict[str, Any]:
    """Render dataset-bounded controls and return their current values."""

    values: dict[str, Any] = {}
    initial_values = initial_values or {}
    label_map = label_map or {}
    option_map = option_map or {}
    help_map = help_map or {}
    fixed_features = fixed_features or set()

    for feature in features:

        series = data[feature]

        label = label_map.get(feature, FEATURE_LABELS.get(
            feature,
            feature.replace("_", " ").title(),
        ))
        help_text = help_map.get(feature, FEATURE_HELP.get(feature))

        key = f"{prefix}_{feature}"

        # Categorical features
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
        ):

            options = sorted(
                series.dropna().astype(str).unique().tolist()
            )
            value_labels = (option_map or {}).get(feature, {})
            if value_labels:
                options = [value for value in value_labels if value in options]

            if not options:
                raise ValueError(
                    f"No valid options available for {feature}."
                )

            default = str(
                initial_values.get(feature, options[0])
            )
            display_options = [
                value_labels.get(option, option) for option in options
            ]
            default_display = value_labels.get(default, default)
            if key not in st.session_state:
                st.session_state[key] = default_display
            elif str(st.session_state[key]) in value_labels:
                st.session_state[key] = value_labels[str(st.session_state[key])]

            if len(options) <= 5:

                values[feature] = st.radio(
                    label,
                    display_options,
                    key=key,
                    help=help_text,
                    horizontal=True,
                )
                values[feature] = next(
                    (option for option, display in value_labels.items() if display == values[feature]),
                    values[feature],
                )

            else:

                values[feature] = st.selectbox(
                    label,
                    display_options,
                    key=key,
                    help=help_text,
                )
                values[feature] = next(
                    (option for option, display in value_labels.items() if display == values[feature]),
                    values[feature],
                )
            if feature in fixed_features:
                st.caption("Fixed")
            else:
                st.caption("Can change")

        # Integer features
        elif (
            feature in ("age", "prior_offenses")
            or pd.api.types.is_integer_dtype(series.dtype)
        ):

            minimum = int(series.min())
            maximum = int(series.max())

            default = int(
                initial_values.get(
                    feature,
                    round(float(series.median())),
                )
            )

            value = min(
                max(default, minimum),
                maximum,
            )
            if key not in st.session_state:
                st.session_state[key] = value
            else:
                st.session_state[key] = min(
                    max(int(st.session_state[key]), minimum),
                    maximum,
                )

            values[feature] = st.slider(
                label,
                min_value=minimum,
                max_value=maximum,
                step=1,
                key=key,
                help=help_text,
                format="%d",
            )
            st.caption("Fixed" if feature in fixed_features else "Can change")

        # Continuous numeric features
        else:

            minimum = float(series.min())
            maximum = float(series.max())

            default = float(
                initial_values.get(
                    feature,
                    series.median(),
                )
            )

            step = (
                1.0
                if float(series.dropna().max()).is_integer()
                else 0.1
            )
            value = min(max(default, minimum), maximum)
            if key not in st.session_state:
                st.session_state[key] = value
            else:
                st.session_state[key] = min(
                    max(float(st.session_state[key]), minimum),
                    maximum,
                )

            values[feature] = st.slider(
                label,
                min_value=minimum,
                max_value=maximum,
                step=step,
                key=key,
                help=help_text,
            )
            st.caption("Fixed" if feature in fixed_features else "Can change")

    return values
# --------------------------------------------------
# FEATURE INPUT VALIDATION
# --------------------------------------------------

def validate_feature_values(
    data: pd.DataFrame,
    features: list[str],
    values: dict[str, Any],
) -> tuple[bool, str]:

    for feature in features:

        if feature not in values:
            return False, f"Missing feature: {feature}"

        if pd.api.types.is_numeric_dtype(data[feature]):

            try:
                value = float(values[feature])
            except (TypeError, ValueError):
                return False, f"Invalid numeric value for {feature}."

            minimum = float(data[feature].min())
            maximum = float(data[feature].max())

            if not minimum <= value <= maximum:
                return (
                    False,
                    f"{feature} must be between {minimum} and {maximum}.",
                )

        else:

            options = (
                data[feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if str(values[feature]) not in options:
                return (
                    False,
                    f"Invalid selection for {feature}.",
                )

    return True, ""


# --------------------------------------------------
# FEATURE SUMMARY
# --------------------------------------------------

def feature_summary(
    values: dict[str, Any],
) -> None:

    if not values:
        return

    st.markdown("### Selected Applicant Details")

    summary_data = []

    for feature, value in values.items():

        label = FEATURE_LABELS.get(
            feature,
            feature.replace("_", " ").title(),
        )

        summary_data.append({
            "Feature": label,
            "Selected Value": value,
        })

    st.dataframe(
        pd.DataFrame(summary_data),
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# SECTION HEADER
# --------------------------------------------------

def section_header(
    title: str,
    description: str = "",
) -> None:

    st.markdown(f"### {title}")

    if description:
        st.caption(description)


# --------------------------------------------------
# SUCCESS / WARNING / ERROR MESSAGES
# --------------------------------------------------

def show_success(message: str) -> None:

    st.success(message)


def show_warning(message: str) -> None:

    st.warning(message)


def show_error(message: str) -> None:

    st.error(message)


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

def show_empty_state(
    title: str,
    description: str,
) -> None:

    st.info(
        f"**{title}**\n\n{description}"
    )


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

def model_selector(
    key: str = "selected_model",
) -> str:

    if key not in st.session_state:
        st.session_state[key] = MODEL_NAMES[0]

    return st.selectbox(
        "Select Prediction Model",
        MODEL_NAMES,
        key=key,
    )


# --------------------------------------------------
# CONFIRMATION BUTTON
# --------------------------------------------------

def confirmation_button(
    label: str = "Generate Explanation",
    key: str = "generate_explanation",
) -> bool:

    return st.button(
        label,
        key=key,
        type="primary",
        width="stretch",
    )


# --------------------------------------------------
# PAGE FOOTER
# --------------------------------------------------

def page_footer() -> None:

    st.markdown("---")

    st.caption(
        "Model-Agnostic Interpretability for Black-Box Classifiers "
        "using Counterfactual Explanations (DiCE)."
    )

    st.caption(
        "Academic research demonstration | Synthetic data only"
    )
    # --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

def render_prediction_result(
    prediction: Any,
    probability: float | None = None,
) -> None:

    st.markdown("### Prediction Result")

    if prediction is None:
        show_empty_state(
            "No prediction available",
            "Please provide valid input features and run the model.",
        )
        return

    st.success(f"Predicted Class: {prediction}")

    if probability is not None:

        probability = float(probability)

        probability = min(max(probability, 0.0), 1.0)

        st.metric(
            "Prediction Confidence",
            f"{probability * 100:.2f}%",
        )

        st.progress(probability)


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------

def render_model_metrics(
    metrics: dict[str, Any],
) -> None:

    if not metrics:
        show_empty_state(
            "Metrics unavailable",
            "Model performance metrics have not been calculated.",
        )
        return

    st.markdown("### Model Performance")

    columns = st.columns(len(metrics))

    for column, (name, value) in zip(columns, metrics.items()):

        with column:

            if isinstance(value, (int, float)):

                st.metric(
                    name.replace("_", " ").title(),
                    f"{value:.3f}",
                )

            else:

                st.metric(
                    name.replace("_", " ").title(),
                    str(value),
                )


# --------------------------------------------------
# EXPLANATION RESULT
# --------------------------------------------------

def render_explanation(
    explanation: str | None,
    title: str = "Model Explanation",
) -> None:

    st.markdown(f"### {title}")

    if not explanation:

        show_empty_state(
            "Explanation unavailable",
            "Generate an explanation to view the model reasoning.",
        )

        return

    st.info(explanation)


# --------------------------------------------------
# COUNTERFACTUAL RESULTS
# --------------------------------------------------

def render_counterfactuals(
    counterfactuals: pd.DataFrame | None,
) -> None:

    st.markdown("### Counterfactual Explanations")

    if counterfactuals is None or counterfactuals.empty:

        show_empty_state(
            "No counterfactuals generated",
            "Try changing the input features or model settings.",
        )

        return

    st.dataframe(
        counterfactuals,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

def render_feature_importance(
    importance: pd.DataFrame | None,
    feature_column: str = "Feature",
    importance_column: str = "Importance",
) -> None:

    st.markdown("### Feature Importance")

    if importance is None or importance.empty:

        show_empty_state(
            "Feature importance unavailable",
            "No feature importance values are available.",
        )

        return

    required_columns = [
        feature_column,
        importance_column,
    ]

    if not all(
        column in importance.columns
        for column in required_columns
    ):

        show_error(
            "Feature importance data has an unexpected format."
        )

        return

    chart_data = importance[
        required_columns
    ].copy()

    chart_data = chart_data.sort_values(
        importance_column,
        ascending=True,
    )

    st.bar_chart(
        chart_data.set_index(feature_column)
    )


# --------------------------------------------------
# DATAFRAME DISPLAY
# --------------------------------------------------

def render_dataframe(
    data: pd.DataFrame | None,
    title: str = "Data Preview",
) -> None:

    st.markdown(f"### {title}")

    if data is None or data.empty:

        show_empty_state(
            "No data available",
            "There is no data to display.",
        )

        return

    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

def download_dataframe(
    data: pd.DataFrame,
    filename: str = "results.csv",
    button_label: str = "Download Results",
) -> None:

    if data is None or data.empty:
        return

    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=button_label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
    # --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def render_confusion_matrix(
    matrix: Any,
    labels: list[str] | None = None,
) -> None:

    st.markdown("### Confusion Matrix")

    if matrix is None:
        show_empty_state(
            "Confusion matrix unavailable",
            "Run model evaluation to generate the confusion matrix.",
        )
        return

    matrix_df = pd.DataFrame(matrix)

    if labels and len(labels) == len(matrix_df):
        matrix_df.index = labels
        matrix_df.columns = labels

    st.dataframe(
        matrix_df,
        width="stretch",
    )


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

def render_classification_report(
    report: dict[str, Any] | None,
) -> None:

    st.markdown("### Classification Report")

    if not report:
        show_empty_state(
            "Classification report unavailable",
            "Evaluate the model to generate the report.",
        )
        return

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        width="stretch",
    )


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

def render_dataset_info(
    data: pd.DataFrame,
    dataset_name: str = "Selected Dataset",
) -> None:

    st.markdown("### Dataset Information")

    if data is None or data.empty:
        show_empty_state(
            "Dataset unavailable",
            "No dataset records were found.",
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(data):,}")

    with col2:
        st.metric("Total Features", f"{len(data.columns):,}")

    with col3:
        st.metric(
            "Missing Values",
            f"{int(data.isnull().sum().sum()):,}",
        )

    st.caption(f"Dataset: {dataset_name}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

def render_model_info(
    model_name: str,
    description: str = "",
) -> None:

    st.markdown("### Model Information")

    st.markdown(f"**Model:** {model_name}")

    if description:
        st.caption(description)


# --------------------------------------------------
# LOADING MESSAGE
# --------------------------------------------------

def render_loading_message(
    message: str = "Processing your request...",
) -> None:

    st.info(message)


# --------------------------------------------------
# RESULTS TABS
# --------------------------------------------------

def create_result_tabs() -> tuple:

    return st.tabs([
        "Prediction",
        "Explanation",
        "Counterfactuals",
        "Data",
    ])


# --------------------------------------------------
# EXPANDABLE INFORMATION
# --------------------------------------------------

def render_expander(
    title: str,
    content: str,
) -> None:

    with st.expander(title):

        st.write(content)


# --------------------------------------------------
# RESEARCH NOTICE
# --------------------------------------------------

def render_research_notice() -> None:

    st.info(
        """
        **Research and Demonstration Notice**

        This application demonstrates model interpretability
        and counterfactual explanation techniques using DiCE.

        The displayed predictions and explanations are intended
        for academic research and educational purposes only.

        They must not be treated as legal advice, evidence of
        individual risk, or automated legal decisions.
        """
    )
    # --------------------------------------------------
# PREDICTION COMPARISON
# --------------------------------------------------

def render_prediction_comparison(
    original_prediction: Any,
    new_prediction: Any,
) -> None:

    st.markdown("### Prediction Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Prediction",
            str(original_prediction),
        )

    with col2:
        st.metric(
            "New Prediction",
            str(new_prediction),
        )

    if original_prediction != new_prediction:
        st.success(
            "The prediction changed after modifying the input features."
        )
    else:
        st.info(
            "The prediction remained unchanged."
        )


# --------------------------------------------------
# FEATURE CHANGES
# --------------------------------------------------

def render_feature_changes(
    original_values: dict[str, Any],
    new_values: dict[str, Any],
) -> None:

    st.markdown("### Feature Changes")

    changes = []

    for feature, original_value in original_values.items():

        if feature not in new_values:
            continue

        new_value = new_values[feature]

        if original_value != new_value:

            changes.append({
                "Feature": FEATURE_LABELS.get(
                    feature,
                    feature.replace("_", " ").title(),
                ),
                "Original Value": original_value,
                "New Value": new_value,
            })

    if not changes:
        show_empty_state(
            "No feature changes",
            "No input features were modified.",
        )
        return

    st.dataframe(
        pd.DataFrame(changes),
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# COUNTERFACTUAL SUMMARY
# --------------------------------------------------

def render_counterfactual_summary(
    original_prediction: Any,
    counterfactual_prediction: Any,
    changed_features: int = 0,
) -> None:

    st.markdown("### Counterfactual Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Prediction",
            str(original_prediction),
        )

    with col2:
        st.metric(
            "Counterfactual Prediction",
            str(counterfactual_prediction),
        )

    with col3:
        st.metric(
            "Changed Features",
            changed_features,
        )

    if original_prediction != counterfactual_prediction:

        st.success(
            "The counterfactual produced a different model prediction."
        )

    else:

        st.warning(
            "The counterfactual did not change the model prediction."


        )


# --------------------------------------------------
# FEATURE VALUE DISPLAY
# --------------------------------------------------

def render_feature_values(
    values: dict[str, Any],
) -> None:

    st.markdown("### Input Feature Values")

    if not values:
        show_empty_state(
            "No feature values",
            "No input values are available.",
        )
        return

    columns = st.columns(2)

    for index, (feature, value) in enumerate(values.items()):

        label = FEATURE_LABELS.get(
            feature,
            feature.replace("_", " ").title(),
        )

        with columns[index % 2]:

            st.metric(
                label,
                str(value),
            )


# --------------------------------------------------
# EXPLANATION DISCLAIMER
# --------------------------------------------------

def render_explanation_disclaimer() -> None:

    st.caption(
        "Counterfactual explanations describe changes that "
        "may alter a model's output. They do not establish "
        "real-world causation or guarantee a particular outcome."
    )


# --------------------------------------------------
# PAGE SEPARATOR
# --------------------------------------------------

def render_separator() -> None:

    st.markdown("---")
    # --------------------------------------------------
# FAIRNESS METRICS
# --------------------------------------------------

def render_fairness_metrics(
    metrics: dict[str, Any] | None,
) -> None:

    st.markdown("### Fairness Metrics")

    if not metrics:
        show_empty_state(
            "Fairness metrics unavailable",
            "Run the fairness evaluation to view the results.",
        )
        return

    columns = st.columns(min(len(metrics), 3))

    for index, (name, value) in enumerate(metrics.items()):

        with columns[index % len(columns)]:

            if isinstance(value, (int, float)):
                st.metric(
                    name.replace("_", " ").title(),
                    f"{value:.3f}",
                )
            else:
                st.metric(
                    name.replace("_", " ").title(),
                    str(value),
                )


# --------------------------------------------------
# GROUP COMPARISON
# --------------------------------------------------

def render_group_comparison(
    data: pd.DataFrame | None,
    group_column: str,
    target_column: str,
) -> None:

    st.markdown("### Group Comparison")

    if data is None or data.empty:
        show_empty_state(
            "No data available",
            "Load a dataset to compare groups.",
        )
        return

    required_columns = [group_column, target_column]

    if not all(
        column in data.columns
        for column in required_columns
    ):
        show_error(
            "Required group comparison columns are missing."
        )
        return

    comparison = (
        data.groupby(group_column, observed=False)[target_column]
        .mean()
        .reset_index()
    )

    comparison.columns = [
        "Group",
        "Average Target Value",
    ]

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

def render_dataset_statistics(
    data: pd.DataFrame | None,
) -> None:

    st.markdown("### Dataset Statistics")

    if data is None or data.empty:
        show_empty_state(
            "Statistics unavailable",
            "No dataset is available.",
        )
        return

    numeric_data = data.select_dtypes(
        include="number"
    )

    if numeric_data.empty:
        st.info("No numeric columns found in this dataset.")
        return

    st.dataframe(
        numeric_data.describe().transpose(),
        width="stretch",
    )


# --------------------------------------------------
# MISSING VALUE REPORT
# --------------------------------------------------

def render_missing_value_report(
    data: pd.DataFrame | None,
) -> None:

    st.markdown("### Missing Value Report")

    if data is None or data.empty:
        return

    missing = pd.DataFrame({
        "Feature": data.columns,
        "Missing Values": data.isnull().sum().values,
        "Missing Percentage": (
            data.isnull().mean().values * 100
        ).round(2),
    })

    missing = missing.sort_values(
        "Missing Values",
        ascending=False,
    )

    st.dataframe(
        missing,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# EXPLANATION REPORT
# --------------------------------------------------

def render_explanation_report(
    prediction: Any,
    explanation: str,
    feature_values: dict[str, Any] | None = None,
) -> None:

    st.markdown("### Explanation Report")

    st.write("**Model Prediction:**", prediction)

    if feature_values:
        st.markdown("**Input Features**")

        st.json(feature_values)

    st.markdown("**Explanation**")

    if explanation:
        st.write(explanation)
    else:
        st.info("No explanation has been generated.")


# --------------------------------------------------
# FAIRNESS DISCLAIMER
# --------------------------------------------------

def render_fairness_disclaimer() -> None:

    st.caption(
        "Fairness metrics are statistical indicators. "
        "They do not independently establish whether a model "
        "is fair or legally compliant."
    )


# --------------------------------------------------
# APPLICATION INFORMATION
# --------------------------------------------------

def render_application_info() -> None:

    with st.expander("About this application"):

        st.write(
            """
            This application demonstrates model-agnostic
            interpretability for black-box classifiers using
            counterfactual explanation generation with DiCE.

            The application includes:

            - Model prediction
            - Counterfactual explanations
            - Feature importance
            - Fairness analysis
            - Model performance evaluation

            All demonstrations use synthetic or research data.
            """
        )
        # --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------

def render_histogram(
    data: pd.DataFrame,
    column: str,
) -> None:

    st.markdown("### Feature Distribution")

    if data is None or data.empty or column not in data.columns:
        show_empty_state(
            "Distribution unavailable",
            "The selected feature is not available.",
        )
        return

    if not pd.api.types.is_numeric_dtype(data[column]):
        st.info("Histogram is available for numeric features only.")
        return

    st.bar_chart(
        data[column].dropna().value_counts().sort_index()
    )


# --------------------------------------------------
# CORRELATION MATRIX
# --------------------------------------------------

def render_correlation_matrix(
    data: pd.DataFrame,
) -> None:

    st.markdown("### Feature Correlation")

    if data is None or data.empty:
        show_empty_state(
            "Correlation unavailable",
            "No dataset is available.",
        )
        return

    numeric_data = data.select_dtypes(include="number")

    if numeric_data.shape[1] < 2:
        st.info("At least two numeric features are required.")
        return

    correlation = numeric_data.corr()

    st.dataframe(
        correlation.round(3),
        width="stretch",
    )


# --------------------------------------------------
# MODEL EVALUATION SUMMARY
# --------------------------------------------------

def render_evaluation_summary(
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1_score: float | None = None,
) -> None:

    st.markdown("### Model Evaluation Summary")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }

    columns = st.columns(4)

    for column, (name, value) in zip(columns, metrics.items()):

        with column:

            if value is None:
                st.metric(name, "N/A")
            else:
                st.metric(name, f"{float(value):.3f}")


# --------------------------------------------------
# REPORT DOWNLOAD
# --------------------------------------------------

def download_text_report(
    report: str,
    filename: str = "explanation_report.txt",
) -> None:

    if not report:
        st.info("No report available for download.")
        return

    st.download_button(
        label="Download Explanation Report",
        data=report,
        file_name=filename,
        mime="text/plain",
        width="stretch",
    )


# --------------------------------------------------
# JSON RESULT DISPLAY
# --------------------------------------------------

def render_json_result(
    result: dict[str, Any] | None,
    title: str = "Detailed Results",
) -> None:

    st.markdown(f"### {title}")

    if not result:
        show_empty_state(
            "No results available",
            "Run the analysis to generate detailed results.",
        )
        return

    st.json(result)


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

def render_model_status(
    model_name: str,
    loaded: bool,
) -> None:

    st.markdown("### Model Status")

    if loaded:
        st.success(f"{model_name} is loaded.")
    else:
        st.warning(f"{model_name} is not currently loaded.")


# --------------------------------------------------
# FINAL APPLICATION DISCLAIMER
# --------------------------------------------------

def render_final_disclaimer() -> None:

    st.markdown("---")

    st.caption(
        """
        **Disclaimer:** This project is an academic research
        demonstration. Predictions, counterfactuals, and fairness
        metrics are generated for experimental purposes only.
        They must not be used as legal advice or as the sole basis
        for decisions affecting individuals.
        """
    )
    # --------------------------------------------------
# PREDICTION PROBABILITIES
# --------------------------------------------------

def render_prediction_probabilities(
    probabilities: dict[str, float] | None,
) -> None:

    st.markdown("### Prediction Probabilities")

    if not probabilities:
        show_empty_state(
            "Probabilities unavailable",
            "The model has not returned prediction probabilities.",
        )
        return

    probability_df = pd.DataFrame({
        "Class": list(probabilities.keys()),
        "Probability": list(probabilities.values()),
    })

    st.bar_chart(
        probability_df.set_index("Class")
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

def render_model_comparison(
    comparison: pd.DataFrame | None,
) -> None:

    st.markdown("### Model Comparison")

    if comparison is None or comparison.empty:
        show_empty_state(
            "Comparison unavailable",
            "Evaluate multiple models to compare their performance.",
        )
        return

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# FEATURE IMPACT SUMMARY
# --------------------------------------------------

def render_feature_impact(
    impacts: dict[str, float] | None,
) -> None:

    st.markdown("### Feature Impact Summary")

    if not impacts:
        show_empty_state(
            "Feature impact unavailable",
            "No feature impact information is available.",
        )
        return

    impact_df = pd.DataFrame({
        "Feature": list(impacts.keys()),
        "Impact": list(impacts.values()),
    })

    impact_df = impact_df.sort_values(
        "Impact",
        key=lambda column: column.abs(),
        ascending=True,
    )

    st.bar_chart(
        impact_df.set_index("Feature")
    )


# --------------------------------------------------
# ANALYSIS SUMMARY
# --------------------------------------------------

def render_analysis_summary(
    summary: str | None,
) -> None:

    st.markdown("### Analysis Summary")

    if not summary:
        show_empty_state(
            "Summary unavailable",
            "Complete the analysis to generate a summary.",
        )
        return

    st.write(summary)


# --------------------------------------------------
# CSV REPORT EXPORT
# --------------------------------------------------

def export_analysis_csv(
    data: pd.DataFrame | None,
    filename: str = "analysis_results.csv",
) -> None:

    if data is None or data.empty:
        st.info("No analysis results available to export.")
        return

    csv_data = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Export Analysis as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


# --------------------------------------------------
# ANALYSIS PROGRESS
# --------------------------------------------------

def render_analysis_progress(
    progress: float,
    message: str = "Analysis in progress",
) -> None:

    progress = min(
        max(float(progress), 0.0),
        1.0,
    )

    st.progress(progress)

    st.caption(
        f"{message}: {progress * 100:.0f}%"
    )


# --------------------------------------------------
# FINAL PAGE INFORMATION
# --------------------------------------------------

def render_project_information() -> None:

    with st.expander("Project Information"):

        st.markdown(
            """
            **Project Title**

            Model-Agnostic Interpretability for Black-Box
            Classifiers via Counterfactual Explanation Generation
            (DiCE), Applied to Judicial/Legal Risk Assessment.

            **Core Technologies**

            - Python
            - Streamlit
            - Scikit-learn
            - DiCE
            - Pandas
            - NumPy
            - Matplotlib

            **Main Objectives**

            - Explain black-box model predictions.
            - Generate counterfactual explanations.
            - Analyze feature contributions.
            - Evaluate model fairness.
            - Improve transparency in predictive systems.

            **Application Purpose**

            Academic research and educational demonstration
            using synthetic data.
            """
        )
