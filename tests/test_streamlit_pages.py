from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILES = [
    PROJECT_ROOT / "app.py",
    *[
        PROJECT_ROOT / "pages" / page
        for page in (
            "3_Assess_a_case.py",
            "4_What_could_change_this.py",
            "5_Fairness_check.py",
            "6_Report.py",
            "7_Model_information.py",
        )
    ],
]


def test_sidebar_has_one_explicit_icon_navigation_menu():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=120
    ).run()
    links = app.sidebar.get("page_link")

    assert [link.label for link in links] == [
        "Home",
        "Assess a case",
        "What could change this?",
        "Fairness check",
        "Report",
        "Model information",
    ]
    assert all(link.icon.startswith(":material/") for link in links)
    assert not app.exception


@pytest.mark.parametrize("app_path", APP_FILES, ids=lambda path: path.name)
def test_streamlit_page_loads_without_exception(app_path):
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=180
    ).run()
    if app_path.name != "app.py":
        app.switch_page(f"pages/{app_path.name}").run()
    assert not app.exception


def test_integer_feature_input_uses_integer_format():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=60,
    ).run()
    app.switch_page("pages/3_Assess_a_case.py").run()
    age = next(item for item in app.slider if item.label == "Age")
    assert age.proto.format == "%d"


def test_scenario_selector_switches_prediction_workflow():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=90
    ).run()
    app.switch_page("pages/3_Assess_a_case.py").run()
    app.selectbox(key="selected_scenario").set_value("Hiring").run()

    assert not app.exception
    assert app.session_state["selected_dataset"] == "Hiring (synthetic)"
    assert any(item.label == "Technical assessment" for item in app.slider)


def test_example_button_fills_prediction_form_from_real_dataset_row():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=90
    ).run()
    app.switch_page("pages/3_Assess_a_case.py").run()
    example_button = next(
        button for button in app.button if button.label == "Low-risk example"
    )
    example_button.click().run()
    assert app.session_state["risk_age"] == app.session_state["prediction_case"]["age"]
    assert app.session_state["prediction_case_dataset"] == "Legal risk (synthetic)"


def test_prediction_appears_and_case_is_received_by_dice_page():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=90
    ).run()
    app.switch_page("pages/3_Assess_a_case.py").run()
    submit = next(
        button
        for button in app.button
        if button.label == "See the model's prediction"
    )
    submit.click().run()
    assert not app.exception
    assert "latest_prediction" in app.session_state
    assert app.session_state["latest_prediction"]["prediction"]
    case = app.session_state["latest_record"].copy()
    app.switch_page("pages/4_What_could_change_this.py").run()
    assert not app.exception
    assert app.session_state["latest_record"] == case
    assert any("Starting case" in item.value for item in app.markdown)


def test_explain_page_generates_counterfactuals_without_error():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=180
    ).run()
    app.switch_page("pages/4_What_could_change_this.py").run()

    generate = next(
        button for button in app.button if button.label == "Find possible changes"
    )
    generate.click().run()

    assert not app.exception
    assert app.session_state["latest_counterfactual_dataset"] == "Legal risk (synthetic)"
    assert app.session_state["latest_counterfactual_model"] == "Random Forest"
    assert app.session_state["latest_counterfactual_method"] in {
        "genetic",
        "random",
        "kdtree",
    }
