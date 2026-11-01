
"""DiCE counterfactual generation for trained sklearn pipelines."""

from __future__ import annotations

import time
import queue
import threading
import itertools
from typing import Any

import dice_ml
import numpy as np
import pandas as pd

from src.data_processing import (
    DATASET_CONFIG,
    HIRING_SENSITIVE,
    HIRING_TARGET,
)


# ==================================================
# CONFIGURATION AND VALIDATION
# ==================================================

class FeasibilityChecker:
    """Validate counterfactuals against dataset feasibility rules."""

    def __init__(
        self,
        feasibility_spec: dict[str, Any] | None = None,
    ) -> None:
        self.spec = feasibility_spec or {}
        self.monotonic_features = self.spec.get(
            "monotonic_features",
            {},
        )
        self.dependency_rules = self.spec.get(
            "dependency_rules",
            [],
        )
        self.cost_weights = self.spec.get(
            "cost_weights",
            {},
        )

    def check(
        self,
        counterfactual: dict[str, Any] | pd.Series,
        original: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Return pass/fail and the names of violated rules."""

        violations: list[str] = []

        if "substance_treatment" in original and "substance_treatment" in counterfactual:
            before = str(original["substance_treatment"]).strip().lower()
            after = str(counterfactual["substance_treatment"]).strip().lower()
            if before == "not_completed":
                if after != "not_completed" and after != "completed":
                    violations.append("substance_treatment_may_only_be_completed")
            elif before != after:
                violations.append("substance_treatment_must_stay_fixed")

        for feature, rule in self.monotonic_features.items():
            if feature not in original or feature not in counterfactual:
                continue

            before = original[feature]
            after = counterfactual[feature]
            if _same_value(before, after):
                continue

            direction = rule.get("direction", "increase")
            order = rule.get("order")

            if order:
                positions = {str(value): index for index, value in enumerate(order)}
                before_rank = positions.get(str(before))
                after_rank = positions.get(str(after))
                violates = (
                    before_rank is None
                    or after_rank is None
                    or (
                        after_rank < before_rank
                        if direction == "increase"
                        else after_rank > before_rank
                    )
                )
            else:
                try:
                    violates = (
                        float(after) < float(before)
                        if direction == "increase"
                        else float(after) > float(before)
                    )
                except (TypeError, ValueError):
                    violates = True

            if violates:
                violations.append(f"{feature}_may_only_{direction}")

        for rule in self.dependency_rules:
            rule_name = rule.get("name", rule.get("type", "unnamed_dependency"))
            rule_type = rule.get("type")

            if rule_type == "work_experience_age":
                experience_feature = rule["experience_feature"]
                age_feature = rule["age_feature"]
                if (
                    experience_feature not in counterfactual
                    or age_feature not in counterfactual
                ):
                    continue
                experience = float(counterfactual[experience_feature])
                age = float(counterfactual[age_feature])
                minimum_age = float(rule.get("minimum_working_age", 16))
                if experience > age - minimum_age:
                    violations.append(rule_name)

            elif rule_type == "feature_pair_bounds":
                feature = rule["feature"]
                reference_feature = rule["reference_feature"]
                if feature not in counterfactual or reference_feature not in counterfactual:
                    continue
                if rule.get("operator") == "nonnegative_when_adult":
                    try:
                        if (
                            float(counterfactual[reference_feature]) >= 18
                            and float(counterfactual[feature]) < 0
                        ):
                            violations.append(rule_name)
                    except (TypeError, ValueError):
                        violations.append(rule_name)

        return not violations, sorted(set(violations))

def _get_dataset_config(target: str) -> dict[str, Any]:
    """Find dataset configuration using its target column."""

    for config in DATASET_CONFIG.values():
        if config["target"] == target:
            return config

    return {}


def _validate_model_result(
    model_result: dict[str, Any],
) -> None:
    """Validate the trained model result."""

    required = [
        "pipeline",
        "features",
        "target",
        "training_data",
    ]

    missing = [
        key for key in required
        if key not in model_result
    ]

    if missing:
        raise ValueError(
            f"Model result is missing: {', '.join(missing)}"
        )

    if not isinstance(model_result["training_data"], pd.DataFrame):
        raise TypeError("training_data must be a pandas DataFrame.")

    if not model_result["features"]:
        raise ValueError("Model features cannot be empty.")


def _validate_record(
    record: dict[str, Any],
    features: list[str],
) -> None:
    """Validate the input record."""

    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary.")

    missing = [
        feature for feature in features
        if feature not in record
    ]

    if missing:
        raise ValueError(
            f"Missing counterfactual inputs: {', '.join(missing)}"
        )

    null_features = [
        feature for feature in features
        if pd.isna(record[feature])
    ]

    if null_features:
        raise ValueError(
            "Input values cannot be missing for: "
            + ", ".join(null_features)
        )


# ==================================================
# PERMITTED RANGE HANDLING
# ==================================================

def _resolve_permitted_range(
    training_data: pd.DataFrame,
    features: list[str],
    configured_ranges: dict[str, Any],
) -> dict[str, list[Any]]:
    """
    Resolve feature constraints into DiCE-compatible ranges.

    Numeric configured ranges are allowed even when their
    boundaries extend beyond the observed training values.
    """

    ranges: dict[str, list[Any]] = {}

    for feature in features:

        if feature not in training_data.columns:
            raise ValueError(
                f"Feature not found in training data: {feature}"
            )

        series = training_data[feature].dropna()

        if series.empty:
            raise ValueError(
                f"No valid training values for feature: {feature}"
            )

        configured = configured_ranges.get(
            feature,
            "observed",
        )

        # Use observed values.
        if isinstance(configured, str):

            if configured != "observed":
                raise ValueError(
                    f"Invalid permitted range for {feature}."
                )

            values = series.unique().tolist()

            try:
                values = sorted(values)
            except TypeError:
                values = sorted(values, key=str)

            ranges[feature] = values

        # Use configured values or numeric boundaries.
        elif isinstance(configured, (list, tuple)):

            if not configured:
                raise ValueError(
                    f"Permitted range for {feature} cannot be empty."
                )

            is_numeric_range = (
                len(configured) == 2
                and pd.api.types.is_numeric_dtype(series)
                and all(
                    isinstance(value, (int, float, np.number))
                    for value in configured
                )
            )

            if is_numeric_range:

                lower = float(configured[0])
                upper = float(configured[1])

                if lower > upper:
                    raise ValueError(
                        f"Invalid numeric range for {feature}: "
                        f"{lower} to {upper}"
                    )

                ranges[feature] = [
                    lower,
                    upper,
                ]

            else:

                observed = set(
                    series.astype(str).unique()
                )

                allowed = [
                    value
                    for value in configured
                    if str(value) in observed
                ]

                if not allowed:
                    raise ValueError(
                        f"No configured values for {feature} "
                        "exist in the training data."
                    )

                ranges[feature] = allowed

        else:
            raise ValueError(
                f"Invalid permitted range configured for {feature}."
            )

    return ranges


def _filter_to_permitted_ranges(
    counterfactuals: pd.DataFrame,
    permitted_range: dict[str, list[Any]],
) -> pd.DataFrame:
    """Remove counterfactuals that violate feature constraints."""

    if counterfactuals.empty:
        return counterfactuals.copy()

    keep = pd.Series(
        True,
        index=counterfactuals.index,
    )

    for feature, allowed in permitted_range.items():

        if feature not in counterfactuals.columns or not allowed:
            keep &= False
            continue

        numeric_range = (
            len(allowed) == 2
            and all(
                isinstance(value, (int, float, np.number))
                for value in allowed
            )
            and pd.api.types.is_numeric_dtype(
                counterfactuals[feature]
            )
        )

        if numeric_range:

            keep &= counterfactuals[feature].between(
                allowed[0],
                allowed[1],
            )

        else:

            keep &= counterfactuals[feature].isin(allowed)

    return (
        counterfactuals.loc[keep]
        .reset_index(drop=True)
    )


# ==================================================
# DICE GENERATION
# ==================================================

def _generate_with_method(
    data: dice_ml.Data,
    dice_model: dice_ml.Model,
    query: pd.DataFrame,
    total_cfs: int,
    features_to_vary: list[str],
    permitted_range: dict[str, list[Any]],
    method: str,
) -> Any:
    """Generate counterfactuals using the selected DiCE method."""

    explainer = dice_ml.Dice(
        data,
        dice_model,
        method=method,
    )

    return explainer.generate_counterfactuals(
        query,
        total_CFs=total_cfs,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        permitted_range=permitted_range,
    )


def _generate_with_time_limit(operation, timeout_seconds: float) -> Any:
    result_queue = queue.Queue(maxsize=1)

    def run_operation() -> None:
        try:
            result_queue.put(("result", operation()))
        except BaseException as error:
            result_queue.put(("error", error))

    worker = threading.Thread(target=run_operation, daemon=True)
    worker.start()
    try:
        status, value = result_queue.get(timeout=max(timeout_seconds, 0.1))
        if status == "error":
            raise value
        return value
    except queue.Empty:
        raise TimeoutError("Counterfactual search exceeded the allowed time.")


def generate_counterfactuals(
    model_result: dict[str, Any],
    record: dict[str, Any],
    total_cfs: int = 3,
    features_to_vary: list[str] | str = "all",
    method: str = "random",
    immutable_features: list[str] | None = None,
    permitted_range: dict[str, Any] | None = None,
    constraint_mode: str = "full",
    max_search_seconds: float = 15.0,
    max_feasibility_retries: int = 2,
) -> pd.DataFrame:
    """
    Generate and verify DiCE counterfactual explanations.

    Only counterfactuals that change the model prediction
    and satisfy the configured constraints are returned.
    """

    _validate_model_result(model_result)

    if not 1 <= total_cfs <= 10:
        raise ValueError(
            "total_cfs must be between 1 and 10."
        )

    if method not in {"random", "genetic", "kdtree"}:
        raise ValueError(
            "method must be random, genetic, or kdtree."
        )

    if constraint_mode not in {"none", "ranges", "full"}:
        raise ValueError(
            "constraint_mode must be none, ranges, or full."
        )

    max_search_seconds = min(max(0.1, float(max_search_seconds)), 15.0)
    max_feasibility_retries = min(max(0, int(max_feasibility_retries)), 2)

    pipeline = model_result["pipeline"]
    features = list(model_result["features"])
    target = model_result["target"]
    training_data = model_result["training_data"]

    if not hasattr(pipeline, "classes_"):
        raise ValueError(
            "The trained pipeline must have classes_. "
            "Train the classifier before generating counterfactuals."
        )

    if len(pipeline.classes_) != 2:
        raise ValueError(
            "Counterfactual generation currently requires "
            "a binary classification model."
        )

    _validate_record(record, features)

    if not set(features).issubset(training_data.columns):
        raise ValueError(
            "Training data does not contain all model features."
        )

    if target not in training_data.columns:
        raise ValueError(
            "Training data does not contain the target column."
        )

    config = _get_dataset_config(target)

    # Prevent sensitive hiring attribute from entering the model.
    if (
        target == HIRING_TARGET
        and HIRING_SENSITIVE in features
    ):
        raise ValueError(
            f"{HIRING_SENSITIVE} must not be a hiring model feature."
        )

    # Combine immutable features.
    immutable = list(
        dict.fromkeys(
            list(config.get("immutable_features", []))
            + (immutable_features or [])
        )
    )
    if (
        target == "risk_category"
        and "substance_treatment" in features
        and str(record.get("substance_treatment", "")).strip().lower()
        in {"completed", "not_needed"}
    ):
        immutable.append("substance_treatment")
    # Dataset metadata may list protected attributes that are intentionally
    # excluded from the model pipeline (for example synthetic_group).
    immutable = [
        feature
        for feature in immutable
        if feature in features
    ]

    unknown_immutable = [
        feature for feature in immutable
        if feature not in features
    ]

    if unknown_immutable:
        raise ValueError(
            "Immutable features are not model features: "
            + ", ".join(unknown_immutable)
        )

    # Dataset configuration takes priority over user ranges.
    if constraint_mode == "none":
        constraints = {}
    else:
        constraints = {
            **(permitted_range or {}),
            **config.get("permitted_range", {}),
        }

    actionable_features = [
        feature
        for feature in features
        if feature not in immutable
    ]

    # Select features that DiCE may change.
    if isinstance(features_to_vary, str):

        if features_to_vary != "all":
            raise ValueError(
                "features_to_vary must be 'all' or a list."
            )

        selected_features = actionable_features

    elif isinstance(features_to_vary, list):

        selected_features = features_to_vary

        invalid = [
            feature for feature in selected_features
            if feature not in features
        ]

        if invalid:
            raise ValueError(
                "Unknown model features: "
                + ", ".join(invalid)
            )

        locked = [
            feature for feature in selected_features
            if feature in immutable
        ]

        if locked:
            raise ValueError(
                "Cannot vary immutable features: "
                + ", ".join(locked)
            )

        if len(set(selected_features)) != len(selected_features):
            raise ValueError(
                "features_to_vary cannot contain duplicates."
            )

    else:
        raise ValueError(
            "features_to_vary must be 'all' or a list."
        )

    if not selected_features:

        empty = pd.DataFrame(columns=features)
        empty.attrs["method_used"] = method
        empty.attrs["method_notice"] = None

        return empty

    # Prepare the training data for DiCE.
    dice_training_data = training_data[
        features + [target]
    ].copy()

    if constraint_mode == "none":
        resolved_ranges = {}
    else:
        resolved_ranges = _resolve_permitted_range(
            dice_training_data,
            selected_features,
            constraints,
        )

    # Prior offenses are not lowered by explanations: they may stay fixed
    # or increase for this synthetic legal-risk scenario.
    if target == "risk_category" and "prior_offenses" in selected_features:
        original_prior = record.get("prior_offenses")
        if original_prior is not None:
            resolved_ranges["prior_offenses"] = [
                float(original_prior),
                float(dice_training_data["prior_offenses"].max()),
            ]

    if target == "risk_category" and "substance_treatment" in selected_features:
        original_treatment = str(record.get("substance_treatment", "")).strip().lower()
        if original_treatment == "not_completed":
            resolved_ranges["substance_treatment"] = ["completed"]
        elif original_treatment in {"completed", "not_needed"}:
            resolved_ranges["substance_treatment"] = [original_treatment]

    continuous_features = [
        feature
        for feature in features
        if pd.api.types.is_numeric_dtype(
            dice_training_data[feature]
        )
    ]

    dice_data = dice_ml.Data(
        dataframe=dice_training_data,
        continuous_features=continuous_features,
        outcome_name=target,
    )

    dice_model = dice_ml.Model(
        model=pipeline,
        backend="sklearn",
        model_type="classifier",
    )

    # Always pass a DataFrame to sklearn.
    query = pd.DataFrame(
        [
            {
                feature: record[feature]
                for feature in features
            }
        ],
        columns=features,
    )

    # Generate counterfactuals.
    method_used = method
    method_notice = None
    started_at = time.perf_counter()

    try:
        def operation():
            return _generate_with_method(
                dice_data,
                dice_model,
                query,
                total_cfs,
                selected_features,
                resolved_ranges,
                method_used,
            )

        remaining = max_search_seconds - (time.perf_counter() - started_at)
        result = _generate_with_time_limit(operation, remaining)

    except TimeoutError:
        empty = pd.DataFrame(columns=features)
        empty.attrs.update({
            "method_used": method_used,
            "method_notice": "The search reached its 15-second time limit.",
            "constraint_mode": constraint_mode,
            "generated_before_feasibility": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejection_rate": 0.0,
            "violated_rules": [],
            "timed_out": True,
        })
        return empty

    except Exception as error:
        if method == "random":
            empty = pd.DataFrame(columns=features)
            empty.attrs.update({
                "method_used": method_used,
                "method_notice": None,
                "constraint_mode": constraint_mode,
                "generated_before_feasibility": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejection_rate": 0.0,
                "violated_rules": [],
                "timed_out": False,
            })
            return empty
        method_used = "random"
        method_notice = (
            f"The {method} method failed. Random generation was attempted instead. "
            f"Reason: {error}"
        )
        remaining = max_search_seconds - (time.perf_counter() - started_at)
        if remaining <= 0:
            empty = pd.DataFrame(columns=features)
            empty.attrs.update({
                "method_used": method_used,
                "method_notice": method_notice,
                "timed_out": True,
                "accepted_count": 0,
                "rejected_count": 0,
                "rejection_rate": 0.0,
                "violated_rules": [],
            })
            return empty
        result = _generate_with_time_limit(
            lambda: _generate_with_method(
                dice_data, dice_model, query, total_cfs,
                selected_features, resolved_ranges, "random",
            ),
            remaining,
        )

    # Extract counterfactual results.
    cf_examples = result.cf_examples_list

    if (
        not cf_examples
        or cf_examples[0].final_cfs_df is None
    ):

        counterfactuals = pd.DataFrame(columns=features)

    else:

        generated = cf_examples[0].final_cfs_df.copy()

        available_features = [
            feature for feature in features
            if feature in generated.columns
        ]

        counterfactuals = generated[
            available_features
        ].copy()

        missing_features = [
            feature for feature in features
            if feature not in counterfactuals.columns
        ]

        if missing_features:
            raise ValueError(
                "DiCE output is missing features: "
                + ", ".join(missing_features)
            )

        counterfactuals = counterfactuals[features]

    # Preserve integer-valued legal features.
    for feature in ("age", "prior_offenses"):

        if feature in counterfactuals.columns:
            counterfactuals[feature] = (
                pd.to_numeric(
                    counterfactuals[feature],
                    errors="coerce",
                )
                .round()
                .astype("Int64")
            )

    # Remove counterfactuals that change immutable features.
    for feature in immutable:

        if feature in counterfactuals.columns:

            unchanged = counterfactuals[feature].map(
                lambda value: _same_value(
                    record[feature],
                    value,
                )
            )

            counterfactuals = counterfactuals.loc[
                unchanged
            ]

    if target == "risk_category" and "prior_offenses" in counterfactuals.columns:
        counterfactuals = counterfactuals.loc[
            pd.to_numeric(
                counterfactuals["prior_offenses"],
                errors="coerce",
            ) >= float(record["prior_offenses"])
        ]

    # Enforce permitted ranges on features being varied.
    if constraint_mode in {"ranges", "full"}:
        counterfactuals = _filter_to_permitted_ranges(
            counterfactuals,
            resolved_ranges,
        )

    generated_before_feasibility = len(counterfactuals)
    feasibility_spec = config.get("feasibility_spec", {})

    if constraint_mode == "full":
        if not counterfactuals.empty:
            for _retry in range(max_feasibility_retries):
                remaining = max_search_seconds - (time.perf_counter() - started_at)
                if remaining <= 0:
                    break
                checker_preview = FeasibilityChecker(feasibility_spec)
                preview_passes = sum(
                    checker_preview.check(candidate, record)[0]
                    for _, candidate in counterfactuals.iterrows()
                )
                if preview_passes >= min(total_cfs, len(counterfactuals)):
                    break
                try:
                    retry_result = _generate_with_time_limit(
                        lambda: _generate_with_method(
                            dice_data, dice_model, query,
                            min(total_cfs * (2 + _retry), 10),
                            selected_features, resolved_ranges, method_used,
                        ),
                        remaining,
                    )
                    retry_examples = retry_result.cf_examples_list
                    if retry_examples and retry_examples[0].final_cfs_df is not None:
                        retry_generated = retry_examples[0].final_cfs_df.copy()
                        available = [feature for feature in features if feature in retry_generated.columns]
                        if set(features).issubset(available):
                            counterfactuals = pd.concat(
                                [counterfactuals, retry_generated[features]],
                                ignore_index=True,
                            ).drop_duplicates()
                except TimeoutError:
                    break

        checker = FeasibilityChecker(feasibility_spec)
        feasible_rows = []
        violation_names = []
        for _, candidate in counterfactuals.iterrows():
            passed, violated = checker.check(candidate, record)
            if passed:
                feasible_rows.append(candidate)
            else:
                violation_names.extend(violated)

        if feasible_rows:
            counterfactuals = pd.DataFrame(
                feasible_rows,
                columns=features,
            ).reset_index(drop=True)
        else:
            counterfactuals = pd.DataFrame(columns=features)
    else:
        violation_names = []

    # Verify that every returned result flips the prediction.
    original_prediction = pipeline.predict(query)[0]

    if counterfactuals.empty:

        verified = pd.DataFrame(columns=features)

    else:

        predictions = pipeline.predict(
            counterfactuals[features]
        )

        changed = [
            (
                not _same_value(original_prediction, prediction)
                and (
                    target != "risk_category"
                    or _outcome_direction_consistent(
                        original_prediction,
                        prediction,
                        record,
                        counterfactuals.iloc[index].to_dict(),
                    )
                )
            )
            for index, prediction in enumerate(predictions)
        ]

        verified = (
            counterfactuals.loc[changed]
            .reset_index(drop=True)
        )

    exhaustive_candidates_found = 0
    exhaustive_rejected = 0
    exhaustive_passed_rules = 0
    exhaustive_added = 0

    if len(verified) < total_cfs:
        exhaustive_features = [
            feature
            for feature in (
                "program_completion",
                "substance_treatment",
                "employment_status",
                "education_level",
            )
            if feature in selected_features
        ]
        if exhaustive_features:
            value_options = []
            for feature in exhaustive_features:
                allowed = sorted(
                    set(
                        training_data[feature]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                )
                if feature == "substance_treatment":
                    original_treatment = str(
                        record.get("substance_treatment", "")
                    ).strip().lower()
                    if original_treatment == "not_completed":
                        allowed = ["completed"]
                    elif original_treatment in {"completed", "not_needed"}:
                        continue
                if not allowed:
                    continue
                value_options.append((feature, allowed))

            if value_options:
                checker = (
                    FeasibilityChecker(feasibility_spec)
                    if constraint_mode == "full"
                    else None
                )
                exhaustive_rows = []
                for values in itertools.product(
                    *(options for _, options in value_options)
                ):
                    candidate = record.copy()
                    for (feature, _), value in zip(value_options, values):
                        original = candidate[feature]
                        candidate[feature] = (
                            type(original)(value)
                            if original is not None and isinstance(original, (int, float))
                            else value
                        )

                    changed_fields = sum(
                        1
                        for feature in exhaustive_features
                        if not _same_value(record[feature], candidate[feature])
                    )
                    if changed_fields == 0:
                        continue
                    candidate["__changed_fields__"] = changed_fields
                    exhaustive_rows.append(candidate)

                if exhaustive_rows:
                    exhaustive_df = pd.DataFrame(exhaustive_rows)
                    predictions = pipeline.predict(
                        exhaustive_df[features]
                    )
                    flipping = []
                    for index, prediction in enumerate(predictions):
                        candidate = exhaustive_df.iloc[index]
                        if not _same_value(original_prediction, prediction) and (
                            target != "risk_category"
                            or _outcome_direction_consistent(
                                original_prediction,
                                prediction,
                                record,
                                candidate.to_dict(),
                            )
                        ):
                            exhaustive_candidates_found += 1
                            if checker is not None:
                                passed, _ = checker.check(candidate.to_dict(), record)
                                if not passed:
                                    exhaustive_rejected += 1
                                    continue
                                exhaustive_passed_rules += 1
                            else:
                                exhaustive_passed_rules += 1
                            flipping.append(candidate)

                    flipping.sort(key=lambda row: row["__changed_fields__"])
                    existing_signatures = {
                        tuple(
                            str(row[feature]) for feature in features
                        )
                        for _, row in verified.iterrows()
                    }
                    added_rows = []
                    for candidate in flipping:
                        signature = tuple(str(candidate[feature]) for feature in features)
                        if signature in existing_signatures:
                            continue
                        added_rows.append(candidate[features].to_dict())
                        existing_signatures.add(signature)
                        if len(verified) + len(added_rows) >= total_cfs:
                            break

                    if added_rows:
                        verified = pd.concat(
                            [
                                verified,
                                pd.DataFrame(added_rows, columns=features),
                            ],
                            ignore_index=True,
                        )
                        exhaustive_added = len(added_rows)

    dice_count = len(verified) - exhaustive_added
    alternative_sources = (
        ["DiCE"] * max(dice_count, 0)
        + ["complete search"] * exhaustive_added
    )
    if exhaustive_added:
        verified.attrs["method_used"] = (
            "DiCE + complete search"
            if dice_count
            else "complete search"
        )
    else:
        verified.attrs["method_used"] = method_used
    verified.attrs["alternative_sources"] = alternative_sources
    verified.attrs["exhaustive_candidates_found"] = exhaustive_candidates_found
    verified.attrs["exhaustive_rejected"] = exhaustive_rejected
    verified.attrs["exhaustive_passed_rules"] = exhaustive_passed_rules
    verified.attrs["method_notice"] = method_notice
    verified.attrs["constraint_mode"] = constraint_mode
    verified.attrs["generated_before_feasibility"] = generated_before_feasibility
    verified.attrs["accepted_count"] = len(verified)
    verified.attrs["rejected_count"] = max(
        generated_before_feasibility - len(verified),
        0,
    )
    verified.attrs["rejection_rate"] = (
        verified.attrs["rejected_count"] / generated_before_feasibility
        if generated_before_feasibility
        else 0.0
    )
    verified.attrs["violated_rules"] = sorted(set(violation_names))

    return verified


# ==================================================
# COUNTERFACTUAL QUALITY METRICS
# ==================================================

def cost_weighted_proximity(
    original: dict[str, Any],
    counterfactual: dict[str, Any] | pd.Series,
    training_data: pd.DataFrame,
    cost_weights: dict[str, float],
) -> float:
    """Return mean normalized feature-change cost in [0, 1]."""

    distances = []
    weighted_total = 0.0
    weight_total = 0.0

    for feature, weight in cost_weights.items():
        if feature not in original or feature not in counterfactual:
            continue
        before = original[feature]
        after = counterfactual[feature]
        if _same_value(before, after):
            continue

        numeric_weight = max(float(weight), 0.0)
        if pd.api.types.is_numeric_dtype(training_data[feature]):
            values = pd.to_numeric(training_data[feature], errors="coerce")
            span = float(values.max() - values.min())
            change = (
                abs(float(after) - float(before)) / span
                if span > 0
                else 1.0
            )
        else:
            change = 1.0

        distances.append(min(change, 1.0) * numeric_weight)
        weighted_total += distances[-1]
        weight_total += numeric_weight

    return float(weighted_total / weight_total) if weight_total else 0.0

def _legal_change_direction(original: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Return whether legal changes are improving, worsening, or mixed."""
    improvement = False
    worsening = False

    employment_improvement = (
        original.get("employment_status") == "unemployed"
        and candidate.get("employment_status") == "employed"
    )
    if employment_improvement:
        improvement = True
    if (
        original.get("employment_status") == "employed"
        and candidate.get("employment_status") == "unemployed"
    ):
        worsening = True

    education_order = ["secondary", "certificate", "college", "graduate"]
    before_education = str(original.get("education_level"))
    after_education = str(candidate.get("education_level"))
    if before_education in education_order and after_education in education_order:
        difference = education_order.index(after_education) - education_order.index(before_education)
        if difference > 0:
            improvement = True
        elif difference < 0:
            worsening = True

    if improvement and worsening:
        return "mixed"
    if improvement:
        return "improvement"
    if worsening:
        return "worsening"
    return "neutral"


def _outcome_direction_consistent(
    starting_prediction: Any,
    candidate_prediction: Any,
    original: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """Reject changes whose direction contradicts the synthetic outcome."""
    direction = _legal_change_direction(original, candidate)
    if direction == "neutral" or direction == "mixed":
        return True
    candidate_lower = str(candidate_prediction).lower() == "lower"
    if direction == "improvement":
        return candidate_lower
    if direction == "worsening":
        return not candidate_lower


def _same_value(
    left: Any,
    right: Any,
) -> bool:
    """Compare values safely, including missing values."""

    left_missing = pd.isna(left)
    right_missing = pd.isna(right)

    if left_missing and right_missing:
        return True

    if left_missing or right_missing:
        return False

    return bool(left == right)


def _normalized_distance(
    left: dict[str, Any] | pd.Series,
    right: dict[str, Any] | pd.Series,
    features: list[str],
    permitted_range: dict[str, list[Any]],
    training_data: pd.DataFrame,
) -> float:
    """Calculate normalized distance between two records."""

    if not features:
        return 0.0

    changes = []

    for feature in features:

        left_value = left[feature]
        right_value = right[feature]

        if _same_value(left_value, right_value):
            changes.append(0.0)
            continue

        if pd.api.types.is_numeric_dtype(
            training_data[feature]
        ):

            allowed = permitted_range.get(feature, [])

            numeric_allowed = (
                len(allowed) == 2
                and all(
                    isinstance(value, (int, float, np.number))
                    for value in allowed
                )
            )

            if numeric_allowed:

                span = float(allowed[1] - allowed[0])

            else:

                observed = training_data[feature]

                span = float(
                    observed.max() - observed.min()
                )

            if span > 0:

                distance = min(
                    abs(
                        float(right_value)
                        - float(left_value)
                    ) / span,
                    1.0,
                )

            else:
                distance = 1.0

            changes.append(distance)

        else:
            changes.append(1.0)

    return float(np.mean(changes))


def counterfactual_quality_metrics(
    model_result: dict[str, Any],
    record: dict[str, Any],
    counterfactuals: pd.DataFrame,
    requested_count: int,
    permitted_range: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Calculate counterfactual quality metrics.

    Validity: proportion of requested counterfactuals that flip prediction.
    Proximity: average normalized distance from the original record.
    Sparsity: average number of changed features.
    Diversity: average pairwise distance between valid counterfactuals.
    Cost-weighted proximity: effort-weighted normalized feature changes.
    """

    _validate_model_result(model_result)

    if requested_count < 1:
        raise ValueError(
            "requested_count must be at least 1."
        )

    features = list(model_result["features"])
    training_data = model_result["training_data"]
    target = model_result["target"]

    _validate_record(record, features)

    missing_columns = [
        feature for feature in features
        if feature not in counterfactuals.columns
    ]

    if not counterfactuals.empty and missing_columns:
        raise ValueError(
            "Counterfactual data is missing columns: "
            + ", ".join(missing_columns)
        )

    config = _get_dataset_config(target)

    constraints = {
        **(permitted_range or {}),
        **config.get("permitted_range", {}),
    }

    ranges = _resolve_permitted_range(
        training_data,
        features,
        constraints,
    )

    pipeline = model_result["pipeline"]

    query = pd.DataFrame(
        [
            {
                feature: record[feature]
                for feature in features
            }
        ],
        columns=features,
    )

    original_prediction = pipeline.predict(query)[0]

    empty_metrics = {
        "validity": 0.0,
        "proximity": 0.0,
        "cost_weighted_proximity": 0.0,
        "sparsity": 0.0,
        "diversity": 0.0,
        "feasibility_rate": 0.0,
    }

    if counterfactuals.empty:
        return empty_metrics

    candidates = counterfactuals[features].copy()

    predictions = pipeline.predict(candidates)

    flips = [
        not _same_value(
            original_prediction,
            prediction,
        )
        for prediction in predictions
    ]

    validity = (
        min(sum(flips), requested_count)
        / requested_count
    )

    verified = (
        candidates.loc[flips]
        .reset_index(drop=True)
    )

    distances = [
        _normalized_distance(
            record,
            row,
            features,
            ranges,
            training_data,
        )
        for _, row in verified.iterrows()
    ]

    sparsities = [
        sum(
            not _same_value(
                record[feature],
                row[feature],
            )
            for feature in features
        )
        for _, row in verified.iterrows()
    ]

    cost_weights = config.get(
        "feasibility_spec",
        {},
    ).get("cost_weights", {})

    cost_distances = [
        cost_weighted_proximity(
            record,
            row,
            training_data,
            cost_weights,
        )
        for _, row in verified.iterrows()
    ]

    feasibility_checker = FeasibilityChecker(
        config.get("feasibility_spec", {})
    )
    feasible_count = sum(
        feasibility_checker.check(row, record)[0]
        for _, row in verified.iterrows()
    )

    pairwise_distances = []

    for left_index in range(len(verified)):

        for right_index in range(
            left_index + 1,
            len(verified),
        ):

            pairwise_distances.append(
                _normalized_distance(
                    verified.iloc[left_index],
                    verified.iloc[right_index],
                    features,
                    ranges,
                    training_data,
                )
            )

    return {
        "validity": float(validity),

        "proximity": (
            float(np.mean(distances))
            if distances else 0.0
        ),

        "cost_weighted_proximity": (
            float(np.mean(cost_distances))
            if cost_distances else 0.0
        ),

        "sparsity": (
            float(np.mean(sparsities))
            if sparsities else 0.0
        ),

        "diversity": (
            float(np.mean(pairwise_distances))
            if pairwise_distances else 0.0
        ),

        "feasibility_rate": (
            float(feasible_count / len(verified))
            if len(verified)
            else 0.0
        ),
    }


# ==================================================
# COMPARE ORIGINAL AND COUNTERFACTUAL VALUES
# ==================================================

def compare_counterfactual(
    record: dict[str, Any],
    counterfactual: pd.Series,
) -> pd.DataFrame:
    """Return a table containing only changed feature values."""

    rows = []

    for feature, original in record.items():

        if feature not in counterfactual.index:
            continue

        changed = not _same_value(
            original,
            counterfactual[feature],
        )

        if changed:

            rows.append(
                {
                    "feature": feature,
                    "original": original,
                    "counterfactual": counterfactual[feature],
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "original",
            "counterfactual",
        ],
    )
