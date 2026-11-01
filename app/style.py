import streamlit as st


def apply_style() -> None:
    """Apply the shared professional visual theme."""
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #F4F8FB 0%, #E8F1F5 100%) !important;
            color: #1B2733;
        }
        .main .block-container {
            max-width: 1100px !important;
            margin: 0 auto;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
            color: #1B2733;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span {
            background-color: #FFFFFF !important;
            color: #1B2733 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F2A47 0%, #14707A 100%) !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] a *,
        [data-testid="stSidebar"] nav a,
        [data-testid="stSidebar"] nav a * {
            color: #FFFFFF !important;
            text-decoration: none !important;
        }
        [data-testid="stSidebar"] a:hover,
        [data-testid="stSidebar"] a:hover *,
        [data-testid="stSidebar"] nav a:hover,
        [data-testid="stSidebar"] nav a:hover * {
            color: #FFFFFF !important;
            background: rgba(15,42,71,0.58) !important;
        }
        [data-testid="stSidebar"] a[aria-current="page"],
        [data-testid="stSidebar"] a[aria-current="page"] *,
        [data-testid="stSidebar"] nav a[aria-current="page"],
        [data-testid="stSidebar"] nav a[aria-current="page"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #E4F0F3 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #FFFFFF !important;
            border-color: #D9E4EC !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #1B2733 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select"] {
            background: #FFFFFF !important;
            color: #1B2733 !important;
            -webkit-text-fill-color: #1B2733 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] [role="button"],
        [data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
            background: #FFFFFF !important;
            color: #1B2733 !important;
            -webkit-text-fill-color: #1B2733 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            border: 1px solid #D9E4EC !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-baseweb="select"] svg {
            color: #1B2733 !important;
            fill: #1B2733 !important;
            stroke: #1B2733 !important;
            opacity: 1 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        [role="listbox"],
        [role="listbox"] * ,
        [role="option"] {
            background-color: #FFFFFF !important;
            color: #1B2733 !important;
        }
        [role="option"][aria-selected="true"] {
            background-color: #E8F1F5 !important;
            color: #1B2733 !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
            background: #FFFFFF !important;
            color: #1B2733 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #E3F2F1 !important;
            color: #1B2733 !important;
        }
        [data-baseweb="popover"] [role="option"] *,
        [data-baseweb="popover"] [role="option"]:hover *,
        [data-baseweb="popover"] [role="option"][aria-selected="true"] * {
            color: #1B2733 !important;
            -webkit-text-fill-color: #1B2733 !important;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [role="radiogroup"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stSelectbox"] label {
            color: #1B2733 !important;
        }
        [data-testid="stSidebar"] a {
            border-radius: 9px !important;
            margin: 2px 0 !important;
        }
        [data-testid="stSidebar"] a[aria-current="page"] {
            background: rgba(15,42,71,0.72) !important;
            border-left: 4px solid #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        .page-title-banner, .home-hero {
            border-radius: 12px;
            padding: 22px 28px;
            margin: 0 0 20px 0;
            color: #FFFFFF !important;
            background: linear-gradient(135deg, #0F2A47 0%, #14707A 100%);
            box-shadow: 0 8px 20px rgba(15,42,71,0.12);
        }
        .page-title-banner h1, .home-hero h2 {
            color: #FFFFFF !important;
            margin: 0 0 6px 0;
            font-size: 2rem;
        }
        .page-title-banner p, .home-hero p {
            color: #FFFFFF !important;
            margin: 0;
        }
        .stButton > button, .stFormSubmitButton > button,
        .stDownloadButton > button, .stPageLink > a {
            border-radius: 10px !important;
            min-height: 44px;
            font-weight: 700 !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button,
        .stDownloadButton > button, .stPageLink > a {
            background: linear-gradient(135deg, #0F2A47 0%, #14707A 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid transparent !important;
        }
        .stButton > button[kind="primary"] *,
        .stFormSubmitButton > button *,
        .stDownloadButton > button *,
        .stPageLink > a *,
        [data-testid^="stBaseButton-primary"],
        [data-testid^="stBaseButton-primary"] * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primary"]:active,
        [data-testid="stBaseButton-primary"]:disabled,
        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:active,
        .stButton > button[kind="primary"]:disabled,
        .stFormSubmitButton > button,
        .stFormSubmitButton > button:hover,
        .stFormSubmitButton > button:active,
        .stFormSubmitButton > button:disabled {
            background: linear-gradient(135deg, #0F2A47 0%, #14707A 100%) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: 1px solid transparent !important;
            opacity: 1 !important;
        }
        .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #14707A !important;
            border: 1px solid #14707A !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        .stPageLink > a:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 16px rgba(15,42,71,0.18);
        }
        .stButton > button[kind="primary"]:hover *,
        .stButton > button[kind="primary"]:active *,
        .stButton > button[kind="primary"]:disabled *,
        .stFormSubmitButton > button:hover *,
        .stFormSubmitButton > button:active *,
        .stFormSubmitButton > button:disabled *,
        .stDownloadButton > button,
        .stDownloadButton > button:hover,
        .stDownloadButton > button:active,
        .stDownloadButton > button:disabled,
        .stDownloadButton > button *,
        .stPageLink > a,
        .stPageLink > a:hover,
        .stPageLink > a:active,
        .stPageLink > a * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        .xai-card, [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #D9E4EC !important;
            border-radius: 12px !important;
            padding: 18px 20px !important;
            box-shadow: 0 4px 14px rgba(27,39,51,0.06);
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #0F2A47 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        [data-testid="stCaptionContainer"], .stCaption, small {
            font-size: 0.9rem !important;
            color: #4A5B6A !important;
        }
        .result-badge {
            display: inline-block;
            padding: 7px 14px;
            border-radius: 999px;
            font-size: 1rem;
            font-weight: 800;
            border: 1px solid transparent;
        }
        .result-lower {
            color: #205A38 !important;
            background: #E8F4EC !important;
            border-color: #2E8B57 !important;
        }
        .result-elevated {
            color: #922C21 !important;
            background: #FBEAE8 !important;
            border-color: #C0392B !important;
        }
        .warning-badge {
            color: #806000 !important;
            background: #FFF4CE !important;
            border: 1px solid #E0A100 !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [role="tab"] { border-radius: 8px 8px 0 0; }
        @media (max-width: 720px) {
            .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            .page-title-banner, .home-hero { padding: 18px; }
            .page-title-banner h1, .home-hero h2 { font-size: 1.55rem; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
