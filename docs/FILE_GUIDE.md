# File Guide

This guide describes the application files that exist in the repository and their direct relationships based on imports and function calls.

## Main data flow

1. `src/data_processing.py` creates or loads the synthetic legal and hiring data and defines their feature lists, labels, immutable fields, permitted ranges, and feasibility metadata. `src/models/train.py` and `src/ui.py` call its dataset functions.
2. `src/models/train.py` trains the configured models through `src/model_training.py`; `src/model_training.py` creates preprocessing pipelines, evaluates held-out data, and saves or loads `models/*.joblib` artifacts used by the Streamlit pages.
3. `src/ui.py`, `pages/3_Assess_a_case.py`, and `pages/5_Fairness_check.py` load datasets and saved models. `pages/3_Assess_a_case.py` sends a case to `src/prediction.py`, whose pipeline returns the predicted category and probabilities.
4. `pages/4_What_could_change_this.py` passes the case and model to `src/dice_explainer.py`; the explainer applies fixed-field and feasibility rules, runs DiCE, and can add complete-search alternatives. `src/evaluation/methods.py` reuses the explainer and nearby-neighbor/Wachter-style methods for batch evaluation.
5. `src/fairness_audit.py` computes group selection and error-rate metrics for `pages/5_Fairness_check.py` and `pages/6_Report.py`; the report page combines prediction, counterfactual, and fairness session data, while `src/compliance_reporting.py` renders the PDF. `scripts/run_evaluation.py` drives `src/evaluation/runner.py` and `src/evaluation/robustness.py`, which write the evaluation and robustness reports consumed by `pages/8_Evaluation.py`.

## `app/style.py`

- `app/style.py` injects shared Streamlit CSS and is imported and called from the application setup in `src/ui.py` as part of `setup_page()`; it affects `app.py` and every page through the shared setup.

## `pages/`

- `pages/3_Assess_a_case.py` builds the legal or hiring case form, runs `src/prediction.py`, stores prediction state, and uses `src/data_processing.py` and shared rendering helpers from `src/ui.py`.
- `pages/4_What_could_change_this.py` runs cached, bounded counterfactual searches through `src/dice_explainer.py` and shared prediction/form helpers from `src/ui.py`, and stores results for `pages/6_Report.py`.
- `pages/5_Fairness_check.py` loads the selected saved model and dataset, computes before/mitigation fairness results through `src/fairness_audit.py`, and renders charts with `src/ui.py`.
- `pages/6_Report.py` combines session prediction and counterfactual state with cached fairness results from `src/fairness_audit.py`, then uses `src/compliance_reporting.py` to generate a PDF.
- `pages/7_Model_information.py` displays model information and metrics loaded through `src/ui.py`; it is reached by the shared navigation configured in `src/ui.py`.
- `pages/8_Evaluation.py` reads CSV and image report outputs from `reports/`, with data loading and page setup supplied by `src/ui.py`.

## `src/`

- `src/__init__.py` marks `src` as a Python package and is used implicitly by imports such as `src.data_processing` and modules under `src/evaluation`.
- `src/compliance_reporting.py` builds HTML/PDF report content and is used by `pages/6_Report.py`.
- `src/data_processing.py` defines datasets, feature metadata, synthetic generators, constraints, and dataset loading; it is used by `src/ui.py`, model training, prediction pages, DiCE, fairness/report pages, scripts, tests, and evaluation modules.
- `src/dice_explainer.py` validates inputs, enforces feasibility, runs DiCE, completes fallback searches, and computes counterfactual quality metrics; it is used by `pages/4_What_could_change_this.py`, evaluation modules, scripts, and tests.
- `src/fairness_audit.py` calculates group fairness metrics, bootstrap intervals, mitigation, and flip-test support; it is used by `pages/5_Fairness_check.py`, `pages/6_Report.py`, and fairness/report tests.
- `src/model_training.py` defines model factories, training/evaluation, permutation importance, and artifact save/load operations; it is used by `src/models/train.py`, `src/ui.py`, scripts, pages, and tests.
- `src/prediction.py` validates a record and calls a saved pipeline to return a category and probabilities; it is used by the assess and counterfactual pages and model tests.
- `src/ui.py` owns Streamlit navigation, shared styles, dataset/model cache loading, forms, result cards, plots, and reusable page renderers; it is used by `app.py`, all application pages, model training entry code, evaluation code, and Streamlit tests.
- `src/visualization.py` creates Plotly distribution, class-count, and confusion-matrix figures; it is available to application and analysis code that imports its chart helpers.

## `src/evaluation/`

- `src/evaluation/__init__.py` marks the evaluation directory as a package and is used implicitly by imports from `src.evaluation`.
- `src/evaluation/methods.py` implements DiCE method dispatch, nearest-unlike-neighbor, and Wachter-style search; it is used by `src/evaluation/runner.py`, robustness analysis, and evaluation tests.
- `src/evaluation/metrics.py` calculates validity, proximity, sparsity, diversity, plausibility, and feasibility metrics; it is used by `src/evaluation/runner.py` and evaluation tests.
- `src/evaluation/robustness.py` measures input perturbation stability, DiCE seed stability, and bootstrap retraining validity, and writes `reports/robustness.csv` plus report figures; it is called by `scripts/run_evaluation.py` and read by `pages/8_Evaluation.py`.
- `src/evaluation/runner.py` runs multi-seed, multi-model, multi-method counterfactual evaluation and writes CSV/LaTeX/figure outputs under `reports/`; it is called by `scripts/run_evaluation.py` and its page reads the generated CSV files.

## `src/models/`

- `src/models/__init__.py` marks `src.models` as a package for imports from `src.models.train`.
- `src/models/train.py` is the standalone training entry point; it calls `ensure_datasets()` from `src/data_processing.py`, trains through `src/model_training.py`, and saves all configured artifacts for `src/ui.py` and the Streamlit pages.

## `scripts/`

- `scripts/run_constraint_ablation.py` trains a real model, compares DiCE constraint modes over 200 cases, and writes constraint ablation CSV/HTML artifacts under `reports/`; it imports dataset, DiCE, and model-training modules.
- `scripts/run_evaluation.py` is the command-line evaluation entry point; it calls `run_evaluation()` and `run_robustness()` and produces files consumed by `pages/8_Evaluation.py`.
- `scripts/time_counterfactual_search.py` loads a saved Random Forest and times DiCE for one fixed legal-risk case; it imports dataset metadata, model loading, and `src/dice_explainer.py`.

## `tests/`

- `tests/__init__.py` marks `tests` as a Python package and is used implicitly by Python test discovery.
- `tests/test_dice.py` validates counterfactual generation, feasibility, quality metrics, fixed fields, and prediction behavior through `src/dice_explainer.py`, `src/prediction.py`, and `src/model_training.py`.
- `tests/test_evaluation.py` validates method implementations, quality metrics, confidence intervals, and perturbation/Jaccard behavior through the `src/evaluation` package and model-training code.
- `tests/test_fairness.py` validates fairness auditing, mitigation, and flip-test-related behavior through `src/fairness_audit.py` and `src/model_training.py`.
- `tests/test_models.py` validates training data, saved-model loading, predictions, and model artifacts through `src/model_training.py`, `src/prediction.py`, and `src/data_processing.py`.
- `tests/test_reports.py` validates PDF/report generation, fairness results, and counterfactual quality through reporting, fairness, DiCE, and training modules.
- `tests/test_streamlit_pages.py` loads the Streamlit app and pages through `AppTest` to check page startup and rendered state.

## `notebooks/`

- `notebooks/01_data_exploration.ipynb` is an interactive exploration notebook that imports `ensure_datasets()` from `src/data_processing.py` and inspects the generated legal and hiring datasets.

## `reports/`

- `reports/ablation_constraints.csv` contains the constraint-ablation measurements written by `scripts/run_constraint_ablation.py`.
- `reports/ablation_constraints.html` contains the HTML version of the constraint-ablation chart/table written by `scripts/run_constraint_ablation.py`; it is a standalone report artifact.
