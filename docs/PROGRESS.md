# Progress: Feasibility and Causal Constraints

## User-facing Streamlit restructure

- Simplified the sidebar to Home, Assess a case, What could change this?, Fairness check, Report, and Model information.
- Renamed the application scenarios as “Legal risk” and “Hiring”; they are presented as scenarios rather than datasets.
- Removed Dataset Explorer, Model Training, and About from the Streamlit app.
- Moved exploratory analysis to `notebooks/01_data_exploration.ipynb`.
- Added `python -m src.models.train` to train and save models under `models/`; the app loads saved artifacts only and shows developer instructions if an artifact is missing.

## Implemented

- Added per-dataset feasibility specifications to `src/data_processing.py`.
- Added monotonic feature rules, dependency-rule hooks, and real-world feature cost weights.
- Added `FeasibilityChecker` as a post-generation filter in `src/dice_explainer.py`.
- Added feasibility rejection counts and rejection rates to generated counterfactual metadata.
- Added cost-weighted normalized proximity as an additional quality metric.
- Added a `Realistic constraints on/off` toggle to the DiCE page.
- Added rule-specific pytest coverage.
- Added a reproducible 200-case ablation runner at `scripts/run_constraint_ablation.py`.

## Real Ablation Results

The ablation was executed against the synthetic legal-risk dataset using a Random Forest model and 200 held-out cases. Three alternatives were generated per case with random search.

| Mode | Test cases | Validity | Proximity | Plausibility | Feasibility rate |
|---|---:|---:|---:|---:|---:|
| none | 200 | 0.918333 | 0.268854 | 1.000000 | 0.634167 |
| ranges | 200 | 0.918333 | 0.281215 | 0.880000 | 0.639167 |
| full | 200 | 0.610000 | 0.199306 | 0.596667 | 0.695000 |

The measured machine-readable report is `reports/ablation_constraints.csv`; the interactive chart is `reports/ablation_constraints.html`.

## Validation

- Run `pytest tests/test_dice.py -q`.
- Run `python scripts/run_constraint_ablation.py`.

Re-run the ablation after changing model seeds, generation methods, dataset sizes, or constraint definitions because the report reflects measured outputs from that configuration.

## Realism and Reporting Update

The legal synthetic scenario now includes program completion, stable housing, substance treatment, and education level as actionable inputs. Models were retrained with these fields; prior offenses may remain unchanged or increase, but explanations reject decreases and changes to immutable fields. The explanation page now reports five plain-language quality metrics and a SHAP top-5 section. The Report page has spinner/error handling, per-decision HTML download, and a compliance mapping marked as a prototype rather than legal advice.

## Plain-language Form Update

The Assess a case form is grouped into About the person and Qualification, with plain labels and field help. Fixed and changeable fields are identified in the form; legal age, previous-offense count, and offense type are fixed. Internal column names are unchanged, while display labels and option text come from dataset config. Models were retrained with the updated synthetic legal data. What could change this? and Report reuse the same wording.

## Directional fairness update

- Retrained the saved models after the latest synthetic legal-risk education and treatment updates.
- Removed the LIME attribution tab and dependency; the explanation page now shows SHAP top 5 only.
- Higher legal education never raises the synthetic risk score, and counterfactual verification rejects improving-to-worse and worsening-to-better contradictions.
- Elevated legal results use recourse toward Lower; Lower results explain changes that may raise the outcome.
- Fairness now supports the selected scenario, with the synthetic legal group or synthetic hiring gender attribute, selection rates, group gaps, and a protected-attribute flip test.
- Report computes held-out fairness automatically, shows a loading spinner, and uses the UI notice “Prototype mapping, not legal advice” for the compliance placeholder table.
- Added weighted-average labels to model precision, recall, and F1 metrics and removed sidebar section heading text.

## Legal form and feature update

- Regenerated the legal CSV with seed 42 while preserving the remaining feature effects, including employment, program completion, treatment, education, offense, age, and prior offenses.
- Retrained all saved legal and hiring models through `python -m src.models.train`.
- Restructured the legal form in the order About the person and Qualification. Work status remains a three-option radio group for Employed, Unemployed, and Student.

## Visual design update

- Added the light Streamlit theme in `.streamlit/config.toml` and shared page styling in `app/style.py`.
- Added navy-teal page banners, cards, primary and secondary buttons, readable helper captions, sidebar navigation styling, and consistent Lower/Elevated result badges.
- Updated the Home page with a gradient hero and four application capability cards.
- Fixed Home route composition so the default page remains available alongside the custom sidebar pages.

## Streamlit warning and layout cleanup

- Replaced deprecated `use_container_width` arguments with the current `width="stretch"` or `width="content"` syntax.
- Converted the Assess a case Input summary's Chosen value column to strings to avoid Arrow serialization warnings.
- Updated `feature_form()` and the DiCE editing flow to initialize widget values through session state without also passing widget defaults, removing the duplicate-value widget warning.
- Added explicit sidebar select-box and control-label contrast, including a visible dark dropdown arrow and white-background selected scenario text.
- Removed approved duplicate outer-folder `src`, `scripts`, and `Makefile` copies; the inner project folder is the single app root.

## Final app check

- Restarted Streamlit on port 8502 and confirmed normal sidebar navigation.
- Opened Home, Assess a case, What could change this?, Fairness check, Report, and Model information without exceptions.
- The fresh terminal remained free of Python warnings, tracebacks, and page exceptions during the validation run.
