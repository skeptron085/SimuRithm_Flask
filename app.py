from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from matplotlib import category
import pandas as pd
import math
import traceback
import numpy as np


# =========================================================
# Dataset
# =========================================================

from modules.parser import (
    load_dataset,
    dataset_preview
)


# =========================================================
# NLP
# =========================================================

from modules.nlp_analyzer import (
    analyze_headers,
    suggest_target
)


# =========================================================
# Statistics
# =========================================================

from modules.statistics_engine import (
    statistics_summary,
    statistics_table
)


# =========================================================
# Correlation
# =========================================================

from modules.correlation_engine import (
    correlation_matrix,
    strongest_correlations
)


# =========================================================
# Task Detection
# =========================================================

from modules.task_detector import (
    detect_task,
    task_reason
)


# =========================================================
# Classification
# =========================================================

from modules.benchmark import (
    run_benchmark,
    CLASSIFICATION_ALGORITHMS
)


# =========================================================
# Regression
# =========================================================

from modules.regression_benchmark import (
    run_regression_benchmark
)


# =========================================================
# Clustering
# =========================================================

from modules.clustering_benchmark import (
    run_clustering
)


# =========================================================
# Advanced Benchmarks
# =========================================================

from modules.advanced_benchmarks import (
    CATEGORY_ALGORITHMS,
    run_anomaly_benchmarks,
    run_reduction_benchmarks,
    run_sequence_benchmarks,
    run_similarity_benchmarks,
    run_topic_benchmarks
)


# =========================================================
# Feature Importance
# =========================================================

from modules.feature_importance import (
    get_feature_importance
)


# =========================================================
# Tuning
# =========================================================

from modules.tuning import (
    tune_recommended_model
)


# =========================================================
# Recommendation
# =========================================================

from modules.recommender import (
    recommend
)


# =========================================================
# Constants
# =========================================================

SUPERVISED_CATEGORIES = {
    "classification",
    "regression"
}

CATEGORIES_REQUIRING_TARGET = {
    "classification",
    "regression",
    "sequence_modeling"
}

METRIC_DEFAULTS = {
    "classification": ("accuracy", True),
    "regression": ("r2", True),
    "clustering": ("silhouette", True),
    "anomaly_detection": ("primary_score", True),
    "dimensionality_reduction": ("primary_score", True),
    "topic_modeling": ("primary_score", True),
    "text_similarity": ("primary_score", True),
    "sequence_modeling": ("primary_score", True)
}


# =========================================================
# JSON Sanitizer
# =========================================================

def sanitize_json_value(value):
    """
    Convert NumPy values and non-finite floats into
    values that can safely be returned as JSON.
    """

    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_json_value(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())

    if isinstance(value, np.generic):
        return sanitize_json_value(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    # Handle other scalar values when possible.
    if not hasattr(value, "__iter__"):
        try:
            if value != value:
                return None
        except Exception:
            pass

    return value


class SafeJSONProvider(DefaultJSONProvider):

    def dumps(self, obj, **kwargs):
        return super().dumps(
            sanitize_json_value(obj),
            **kwargs
        )


# =========================================================
# Flask Application
# =========================================================

app = Flask(__name__)
app.json = SafeJSONProvider(app)


# =========================================================
# Utility Functions
# =========================================================

def error_response(message, status=400, **extra):
    """
    Return a consistent API error response.
    """

    payload = {
        "success": False,
        "message": message
    }

    payload.update(extra)

    return jsonify(payload), status


def normalize_list(value):
    """
    Normalize optional input values into lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return []


def normalize_results(value):
    """
    Benchmark modules should return a list of dictionaries.
    Normalize a single dictionary as a convenience.
    """

    if isinstance(value, dict):
        return [value]

    if isinstance(value, list):
        return [
            item for item in value
            if isinstance(item, dict)
        ]

    return []


def get_algorithm_name(result):
    """
    Support common algorithm-name field conventions.
    """

    return (
        result.get("algorithm")
        or result.get("algorithm_name")
        or result.get("model")
    )


def get_primary_score(result):
    """
    Read the normalized primary score, with a fallback
    to the named metric.
    """

    score = result.get("primary_score")

    if score is not None:
        try:
            score = float(score)
            return score if math.isfinite(score) else None
        except (TypeError, ValueError):
            return None

    metric = result.get("primary_metric")

    if metric:
        score = result.get(metric)

        if score is not None:
            try:
                score = float(score)
                return score if math.isfinite(score) else None
            except (TypeError, ValueError):
                return None

    return None


def normalize_benchmark_results(results, category):
    """
    Ensure all benchmark results have a consistent
    category, primary metric, primary score, and
    higher_is_better field.
    """

    default_metric, default_direction = METRIC_DEFAULTS.get(
        category,
        ("primary_score", True)
    )

    normalized = []

    for original in results:

        result = dict(original)

        result["category"] = category

        metric = (
            result.get("primary_metric")
            or default_metric
        )

        result["primary_metric"] = metric

        score = result.get("primary_score")

        if score is None:
            score = result.get(metric)

        if score is not None:
            try:
                score = float(score)

                if not math.isfinite(score):
                    score = None

            except (TypeError, ValueError):
                score = None

        result["primary_score"] = score

        result["higher_is_better"] = result.get(
            "higher_is_better",
            default_direction
        )

        if "is_recommended" not in result:
            result["is_recommended"] = False

        normalized.append(result)

    return normalized


def get_successful_results(results):
    """
    Exclude failed benchmarks and results without scores.
    """

    return [
        result
        for result in results
        if not result.get("error")
        and get_primary_score(result) is not None
    ]


def get_results_for_target(results, target):
    """
    Prevent results from different targets from being
    mixed together during supervised recommendation.
    """

    return [
        result
        for result in results
        if result.get("target") == target
    ]


def safe_recommend(results, category):
    """
    Use the existing recommendation module.
    Return a consistent fallback when recommendation
    fails or no valid benchmark results are available.
    """

    successful = get_successful_results(results)

    if not successful:
        return {
            "success": False,
            "best_algorithm": None,
            "message": "No successful benchmark results."
        }

    try:
        output = recommend(successful, category)

        if isinstance(output, dict):
            recommendation = dict(output)
        else:
            recommendation = {
                "best_algorithm": output
            }

        best_algorithm = (
            recommendation.get("best_algorithm")
            or recommendation.get("algorithm")
            or recommendation.get("algorithm_name")
        )

        if best_algorithm:
            recommendation["best_algorithm"] = best_algorithm
            recommendation.setdefault("success", True)
            return recommendation

    except Exception as error:
        recommendation_error = str(error)
    else:
        recommendation_error = (
            "Recommendation module did not return "
            "a best algorithm."
        )

    # Fallback: compare scores only when their metric
    # and optimization direction are compatible.
    first = successful[0]

    reference_metric = first.get("primary_metric")
    reference_direction = first.get("higher_is_better", True)

    comparable = [
        item
        for item in successful
        if item.get("primary_metric") == reference_metric
        and item.get("higher_is_better", True) == reference_direction
    ]

    if comparable:
        best = (
            max(
                comparable,
                key=get_primary_score
            )
            if reference_direction
            else min(
                comparable,
                key=get_primary_score
            )
        )

        return {
            "success": True,
            "best_algorithm": get_algorithm_name(best),
            "primary_metric": reference_metric,
            "primary_score": get_primary_score(best),
            "message": (
                "Selected using a compatible benchmark "
                "metric because the recommendation module "
                "did not return a usable recommendation."
            ),
            "recommendation_warning": recommendation_error
        }

    return {
        "success": False,
        "best_algorithm": None,
        "message": recommendation_error
    }


def mark_recommended_results(results, recommendation):
    """
    Mark the recommended algorithm without marking
    failed benchmark entries.
    """

    best_algorithm = recommendation.get("best_algorithm")

    for result in results:
        result["is_recommended"] = (
            best_algorithm is not None
            and get_algorithm_name(result) == best_algorithm
            and not result.get("error")
            and get_primary_score(result) is not None
        )


def run_one_benchmark(function, *args, **kwargs):
    """
    Execute one algorithm safely so one model failure
    does not stop the remaining algorithms.
    """

    try:
        output = function(*args, **kwargs)
        return normalize_results(output)

    except Exception as error:
        return [{
            "algorithm": kwargs.get(
                "selected_algorithm",
                "Unknown"
            ),
            "error": str(error)
        }]


def validate_column_selection(df, x_columns, y_columns):
    """
    Validate selected X and Y columns against the dataset.
    """

    requested = x_columns + y_columns

    missing = [
        column
        for column in requested
        if column not in df.columns
    ]

    if missing:
        return (
            False,
            "Selected columns were not found in the dataset.",
            {"missing_columns": missing}
        )

    if len(x_columns) != len(set(x_columns)):
        return (
            False,
            "Duplicate X columns are not allowed.",
            {}
        )

    if len(y_columns) != len(set(y_columns)):
        return (
            False,
            "Duplicate Y columns are not allowed.",
            {}
        )

    overlapping = set(x_columns).intersection(y_columns)

    if overlapping:
        return (
            False,
            "A column cannot be selected as both X and Y.",
            {"overlapping_columns": sorted(overlapping)}
        )

    return True, None, {}


def validate_category_compatibility(
    df,
    category,
    x_columns,
    y_columns,
    algorithms
):
    """
    Validate selected columns against the requirements
    of the requested machine-learning category.

    Returns:
        (is_valid, message, details)
    """

    errors = []

    # -----------------------------------------------------
    # 1. Validate selected columns exist
    # -----------------------------------------------------

    missing_x = [
        col for col in x_columns
        if col not in df.columns
    ]

    missing_y = [
        col for col in y_columns
        if col not in df.columns
    ]

    if missing_x:
        errors.append(
            f"Unknown X variable(s): {missing_x}"
        )

    if missing_y:
        errors.append(
            f"Unknown Y variable(s): {missing_y}"
        )

    if errors:
        return False, "Invalid column selection.", {
            "validation_errors": errors,
            "missing_x_columns": missing_x,
            "missing_y_columns": missing_y
        }

    # -----------------------------------------------------
    # 2. Validate duplicate and overlapping selections
    # -----------------------------------------------------

    if len(x_columns) != len(set(x_columns)):
        errors.append(
            "Duplicate X variables were selected."
        )

    if len(y_columns) != len(set(y_columns)):
        errors.append(
            "Duplicate Y variables were selected."
        )

    overlap = sorted(set(x_columns) & set(y_columns))

    if overlap:
        errors.append(
            "A column cannot be both an X variable and "
            f"a Y variable: {overlap}"
        )

    # -----------------------------------------------------
    # 3. Validate required predictors
    # -----------------------------------------------------

    if not x_columns:
        errors.append(
            "Select at least one X variable."
        )

    # -----------------------------------------------------
    # 4. Validate category-specific target requirements
    # -----------------------------------------------------

    target_required = category in {
        "classification",
        "regression",
        "sequence_modeling"
    }

    if target_required and not y_columns:
        errors.append(
            f"{category.replace('_', ' ').title()} "
            "requires at least one Y variable."
        )

    if category not in {
        "classification",
        "regression",
        "sequence_modeling"
    } and y_columns:
        errors.append(
            f"{category.replace('_', ' ').title()} "
            "Does not use Y variables. Remove the selected "
            "Y variables and use X variables as input."
        )

    # -----------------------------------------------------
    # 5. Validate data types for supervised categories
    # -----------------------------------------------------

    for target in y_columns:
        if category == "regression":
            if not pd.api.types.is_numeric_dtype(
                df[target]
            ):
                errors.append(
                    f"Regression target '{target}' must be "
                    "numeric. Choose a numeric Y variable."
                )

        elif category == "classification":
            target_values = df[target].dropna()

            if target_values.nunique() < 2:
                errors.append(
                    f"Classification target '{target}' "
                    "must contain at least two distinct classes."
                )

            elif pd.api.types.is_numeric_dtype(
                target_values
            ):
                # Numeric targets can be valid classes, but
                # a continuous numeric target may be regression.
                try:
                    detected = detect_task(df, target)

                    if detected != "classification":
                        errors.append(
                            f"Target '{target}' was detected as "
                            f"'{detected}', not classification. "
                            "Choose a categorical target or "
                            "verify the target's intended meaning."
                        )
                except Exception as exc:
                    errors.append(
                        f"Could not validate target '{target}': "
                        f"{exc}"
                    )

    # -----------------------------------------------------
    # 6. Validate text-oriented categories
    # -----------------------------------------------------

    if category in {"topic_modeling", "text_similarity"}:
        text_columns = [
            col for col in x_columns
            if (
                pd.api.types.is_string_dtype(df[col])
                or df[col].dtype == object
                or str(df[col].dtype).startswith("category")
            )
        ]

        if not text_columns:
            errors.append(
                f"{category.replace('_', ' ').title()} "
                "Requires at least one text or categorical "
                "input column. Select a suitable text column."
            )

    if errors:
        return False, "The selected variables are incompatible.", {
            "category": category,
            "validation_errors": errors,
            "selected_x_columns": x_columns,
            "selected_y_columns": y_columns,
            "selected_algorithms": algorithms
        }

    return True, "Column selection is valid.", {}

def auto_select_configuration(df, category):
    """
    Infer input and target variables for Automatic mode.

    Returns:
        x_columns, y_columns, selection_info

    Supervised categories require a compatible target.
    Unsupervised categories use X columns only.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    columns = list(df.columns)

    if len(columns) < 1:
        raise ValueError("The dataset has no usable columns.")

    # Find candidate targets using the existing NLP module.
    try:
        candidates = normalize_list(suggest_target(df))
    except Exception:
        candidates = []

    # Only keep candidate names that exist in the dataset.
    candidates = [
        col for col in candidates
        if isinstance(col, str) and col in columns
    ]

    # If NLP did not suggest a target, inspect the columns.
    candidates.extend([
        col for col in columns
        if col not in candidates
    ])

    target = None

    if category in {"classification", "regression"}:
        for candidate in candidates:
            try:
                detected = detect_task(df, candidate)
            except Exception:
                continue

            if detected == category:
                target = candidate
                break

        if target is None:
            raise ValueError(
                f"Automatic mode could not identify a suitable "
                f"target for {category}. Choose a compatible "
                f"target manually or use another category."
            )

    elif category == "sequence_modeling":
        # Sequence-modeling implementations generally need a
        # usable target series. Prefer numeric targets.
        for candidate in candidates:
            if pd.api.types.is_numeric_dtype(df[candidate]):
                target = candidate
                break

        if target is None:
            raise ValueError(
                "Automatic mode could not identify a numeric "
                "target for sequence modeling."
            )

    # Start with all columns, excluding the selected target.
    x_columns = [
        col for col in columns
        if col != target
    ]

    # Text-focused categories need at least one text-like column.
    if category in {"topic_modeling", "text_similarity"}:
        text_columns = [
            col for col in columns
            if (
                pd.api.types.is_string_dtype(df[col])
                or df[col].dtype == object
                or str(df[col].dtype).startswith("category")
            )
        ]

        if text_columns:
            x_columns = [
                col for col in text_columns
                if col != target
            ]

    if not x_columns:
        raise ValueError(
            "Automatic mode could not identify at least one "
            "input variable distinct from the target."
        )

    y_columns = [target] if target is not None else []

    selection_info = {
        "selection_mode": "automatic",
        "selected_x_columns": x_columns,
        "selected_y_columns": y_columns,
        "selection_reason": (
            "Variables were inferred automatically from "
            "dataset columns and target compatibility."
        )
    }

    return x_columns, y_columns, selection_info


# =========================================================
# Home Endpoint
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "system": "SimuRithm",
        "version": "Release 1",
        "status": "Running"
    })


# =========================================================
# Dataset Analysis
#
# Purpose:
# Prepare the frontend Analysis Configuration screen.
# This endpoint does not run benchmarks.
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return error_response(
                "No valid JSON object received."
            )

        file_path = data.get("file_path")

        if not isinstance(file_path, str) or not file_path.strip():
            return error_response(
                "file_path is required."
            )

        df = load_dataset(file_path)

        if df is None or df.empty:
            return error_response(
                "The dataset is empty or could not be loaded."
            )

        # ---------------------------------------------
        # Statistics and preview
        # ---------------------------------------------

        statistics = statistics_summary(df)
        statistics_details = statistics_table(df)
        preview = dataset_preview(df)

        # ---------------------------------------------
        # NLP header analysis
        # ---------------------------------------------

        header_analysis = analyze_headers(df)

        target_candidates = normalize_list(
            suggest_target(df)
        )

        suggested_target = (
            target_candidates[0]
            if target_candidates
            else None
        )

        # ---------------------------------------------
        # Task detection
        # ---------------------------------------------

        suggested_task = detect_task(
            df,
            suggested_target
        )

        explanation = task_reason(
            df,
            suggested_target
        )

        column_names = [
            str(column)
            for column in df.columns
        ]

        suggested_algorithms = CATEGORY_ALGORITHMS.get(
            suggested_task,
            []
        )

        # ---------------------------------------------
        # Response
        # ---------------------------------------------

        return jsonify({

            "success": True,

            "dataset": {
                "target": suggested_target,
                "target_candidates": target_candidates,
                "task": suggested_task,
                "reason": explanation,
                "rows": statistics.get("rows", len(df)),
                "columns": statistics.get("columns", len(df.columns)),
                "column_names": column_names
            },

            "statistics": statistics,

            "statistics_table": statistics_details,

            "preview": preview,

            "header_analysis": header_analysis,

            "nlp_analysis": header_analysis,

            "analysis_configuration": {
                "available_columns": column_names,
                "suggested_target": suggested_target,
                "target_candidates": target_candidates,
                "suggested_task": suggested_task,
                "available_algorithms": suggested_algorithms
            }

        })

    except Exception as error:

        app.logger.exception("Dataset analysis failed.")

        payload = {
            "success": False,
            "message": str(error)
        }

        if app.debug:
            payload["traceback"] = traceback.format_exc()

        return jsonify(payload), 500


# =========================================================
# Shared Benchmark Execution
#
# Used by /run_benchmark and /run_analyze.
# =========================================================

def execute_benchmark(data):

    # -----------------------------------------------------
    # Request validation
    # -----------------------------------------------------

    if not isinstance(data, dict):
        return {
            "success": False,
            "message": "No valid JSON object received."
        }, 400

    file_path = data.get("file_path")
    category = data.get("category")
    selection_mode = str(
        data.get("selection_mode", "manual")
    ).strip().lower()

    x_columns = data.get("x_columns", [])
    y_columns = data.get("y_columns", [])
    algorithms = data.get("algorithms", [])

    if selection_mode not in {"automatic", "manual"}:
        return {
            "success": False,
            "message": (
                "selection_mode must be 'automatic' or 'manual'."
            )
        }, 400

    if selection_mode == "manual" and not x_columns:
        return {
            "success": False,
            "message": "Select at least one X variable in Manual mode."
    }, 400

    if not isinstance(file_path, str) or not file_path.strip():
        return {
            "success": False,
            "message": "file_path is required."
        }, 400

    # -----------------------------------------------------
    # Validate category
    # -----------------------------------------------------

    if category not in CATEGORY_ALGORITHMS:

        return {
            "success": False,
            "message": "Unsupported analysis category.",
            "available_categories": list(CATEGORY_ALGORITHMS.keys())
        }, 400

    # -----------------------------------------------------
    # Validate arrays
    # -----------------------------------------------------

    if not isinstance(x_columns, list):
        return {
            "success": False,
            "message": "x_columns must be an array."
        }, 400

    if not isinstance(y_columns, list):
        return {
            "success": False,
            "message": "y_columns must be an array."
        }, 400

    if not isinstance(algorithms, list):
        return {
            "success": False,
            "message": "algorithms must be an array."
        }, 400

    if not all(isinstance(column, str) for column in x_columns + y_columns):
        return {
            "success": False,
            "message": "Column names must be strings."
        }, 400

    if not all(isinstance(name, str) for name in algorithms):
        return {
            "success": False,
            "message": "Algorithm names must be strings."
        }, 400

    if (selection_mode == "manual" and category in CATEGORIES_REQUIRING_TARGET and not y_columns):
        return {
            "success": False,
            "message": "Select at least one Y variable for this task."
        }, 400

    if category not in CATEGORIES_REQUIRING_TARGET and y_columns:
        # Unsupervised categories may ignore targets, but silently
        # ignoring selected Y columns can confuse users.
        # These categories use X columns as their analysis input.
        pass

    # -----------------------------------------------------
    # Validate algorithms
    # -----------------------------------------------------

    allowed_algorithms = CATEGORY_ALGORITHMS.get(category, [])

    if not algorithms:
        return {
            "success": False,
            "message": "Select at least one algorithm."
        }, 400

    algorithms = list(dict.fromkeys(algorithms))

    invalid_algorithms = [
        algorithm
        for algorithm in algorithms
        if algorithm not in allowed_algorithms
    ]

    if invalid_algorithms:
        return {
            "success": False,
            "message": (
                "One or more selected algorithms are "
                "not available for this task."
            ),
            "invalid_algorithms": invalid_algorithms,
            "allowed_algorithms": allowed_algorithms
        }, 400

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = load_dataset(file_path)

    if df is None or df.empty:
        return {
            "success": False,
            "message": "The dataset is empty or could not be loaded."
        }, 400

    selection_info = {}

    if selection_mode == "automatic":
        try:
            x_columns, y_columns, selection_info = (
                auto_select_configuration(df, category)
            )
        except ValueError as error:
            return {
                "success": False,
                "message": str(error),
                "category": category,
                "selection_mode": selection_mode
            }, 400
        except Exception as error:
            app.logger.exception(
                "Automatic variable selection failed."
            )
            return {
                "success": False,
                "message": "Automatic variable selection failed.",
                "details": str(error)
            }, 400

    
    # -----------------------------------------------------
    # Validate selected columns
    # -----------------------------------------------------

    valid, message, details = validate_column_selection(
        df,
        x_columns,
        y_columns
    )

    if not valid:
        return {
            "success": False,
            "message": message,
            **details
        }, 400

    # -----------------------------------------------------
    # Validate category-specific compatibility
    # -----------------------------------------------------

    is_valid, message, details = validate_category_compatibility(
        df=df,
        category=category,
        x_columns=x_columns,
        y_columns=y_columns,
        algorithms=algorithms
    )

    if not is_valid:
        return {
            "success": False,
            "message": message,
            **details
        }, 400


    # -----------------------------------------------------
    # Initialize results
    # -----------------------------------------------------

    results = []
    target_results = []
    feature_importance_by_target = {}

    recommendation = {}
    tuning = None

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if category == "classification":

        for target in y_columns:

            model_df = df[x_columns + [target]].copy()

            try:
                detected_task = detect_task(model_df, target)
            except Exception as error:
                target_results.append({
                    "target": target,
                    "task": None,
                    "error": f"Task detection failed: {error}"
                })
                continue

            if detected_task != "classification":
                target_results.append({
                    "target": target,
                    "task": detected_task,
                    "error": (
                        f"Target '{target}' was detected as "
                        f"'{detected_task}', not classification. "
                        "Benchmarking was skipped."
                    )
                })
                continue

            target_benchmarks = []

            for algorithm in algorithms:

                algorithm_results = run_one_benchmark(
                    run_benchmark,
                    model_df,
                    target,
                    algorithm
                )

                for result in algorithm_results:
                    result["target"] = target
                    result["category"] = category
                    result.setdefault("algorithm", algorithm)

                    target_benchmarks.append(result)
                    results.append(result)

            try:
                feature_importance_by_target[target] = (
                    get_feature_importance(model_df, target)
                )
            except Exception as error:
                app.logger.warning(
                    "Feature importance failed for target %s: %s",
                    target,
                    error
                )
                feature_importance_by_target[target] = []

            target_results.append({
                "target": target,
                "task": category,
                "benchmarks": target_benchmarks
            })

    # =====================================================
    # REGRESSION
    # =====================================================

    elif category == "regression":

        for target in y_columns:

            model_df = df[x_columns + [target]].copy()

            try:
                detected_task = detect_task(model_df, target)
            except Exception as error:
                target_results.append({
                    "target": target,
                    "task": None,
                    "error": f"Task detection failed: {error}"
                })
                continue

            if detected_task != "regression":
                target_results.append({
                    "target": target,
                    "task": detected_task,
                    "error": (
                        f"Target '{target}' was detected as "
                        f"'{detected_task}', not regression. "
                        "Benchmarking was skipped."
                    )
                })
                continue

            target_benchmarks = []

            for algorithm in algorithms:

                algorithm_results = run_one_benchmark(
                    run_regression_benchmark,
                    model_df,
                    target,
                    algorithm
                )

                for result in algorithm_results:
                    result["target"] = target
                    result["category"] = category
                    result.setdefault("algorithm", algorithm)

                    target_benchmarks.append(result)
                    results.append(result)

            try:
                feature_importance_by_target[target] = (
                    get_feature_importance(model_df, target)
                )
            except Exception as error:
                app.logger.warning(
                    "Feature importance failed for target %s: %s",
                    target,
                    error
                )
                feature_importance_by_target[target] = []

            target_results.append({
                "target": target,
                "task": category,
                "benchmarks": target_benchmarks
            })

    # =====================================================
    # CLUSTERING
    # =====================================================

    elif category == "clustering":

        features = df[x_columns].copy()

        for algorithm in algorithms:

            algorithm_results = run_one_benchmark(
                run_clustering,
                features,
                selected_features=x_columns,
                selected_algorithm=algorithm
            )

            for result in algorithm_results:
                result.setdefault("algorithm", algorithm)
                result["category"] = category
                results.append(result)

        target_results.append({
            "target": None,
            "task": category,
            "benchmarks": results
        })

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    elif category == "anomaly_detection":

        try:
            results = normalize_results(
                run_anomaly_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )
        except Exception as error:
            results = [{
                "algorithm": algorithm,
                "error": str(error)
            } for algorithm in algorithms]

        target_results.append({
            "target": None,
            "task": category,
            "benchmarks": results
        })

    # =====================================================
    # DIMENSIONALITY REDUCTION
    # =====================================================

    elif category == "dimensionality_reduction":

        try:
            results = normalize_results(
                run_reduction_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )
        except Exception as error:
            results = [{
                "algorithm": algorithm,
                "error": str(error)
            } for algorithm in algorithms]

        target_results.append({
            "target": None,
            "task": category,
            "benchmarks": results
        })

    # =====================================================
    # TOPIC MODELING
    # =====================================================

    elif category == "topic_modeling":

        try:
            results = normalize_results(
                run_topic_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )
        except Exception as error:
            results = [{
                "algorithm": algorithm,
                "error": str(error)
            } for algorithm in algorithms]

        target_results.append({
            "target": None,
            "task": category,
            "benchmarks": results
        })

    # =====================================================
    # TEXT SIMILARITY
    # =====================================================

    elif category == "text_similarity":

        try:
            results = normalize_results(
                run_similarity_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )
        except Exception as error:
            results = [{
                "algorithm": algorithm,
                "error": str(error)
            } for algorithm in algorithms]

        target_results.append({
            "target": None,
            "task": category,
            "benchmarks": results
        })

    # =====================================================
    # SEQUENCE MODELING
    # =====================================================

    elif category == "sequence_modeling":

        try:
            results = normalize_results(
                run_sequence_benchmarks(
                    df[x_columns].copy(),
                    df[y_columns].copy(),
                    y_columns,
                    algorithms
                )
            )

            for result in results:
                result.setdefault("target", (
                    y_columns[0] if len(y_columns) == 1 else None
                ))

        except Exception as error:
            results = [{
                "algorithm": algorithm,
                "error": str(error)
            } for algorithm in algorithms]

        target_results.append({
            "target": y_columns[0] if len(y_columns) == 1 else None,
            "task": category,
            "benchmarks": results
        })

    # -----------------------------------------------------
    # Normalize all results
    # -----------------------------------------------------

    results = normalize_benchmark_results(results, category)

    # =====================================================
    # RECOMMENDATIONS
    #
    # Supervised categories get a recommendation per target.
    # Other categories get one recommendation for the task.
    # =====================================================

    recommendation_by_target = {}

    if category in SUPERVISED_CATEGORIES:

        for target in y_columns:

            target_benchmarks = get_results_for_target(
                results,
                target
            )

            target_recommendation = safe_recommend(
                target_benchmarks,
                category
            )

            recommendation_by_target[target] = (
                target_recommendation
            )

            mark_recommended_results(
                target_benchmarks,
                target_recommendation
            )

        if len(y_columns) == 1:
            recommendation = recommendation_by_target[y_columns[0]]
        else:
            recommendation = {
                "success": any(
                    item.get("success")
                    for item in recommendation_by_target.values()
                ),
                "by_target": recommendation_by_target
            }

    else:

        recommendation = safe_recommend(
            results,
            category
        )

        mark_recommended_results(
            results,
            recommendation
        )

    # =====================================================
    # HYPERPARAMETER TUNING
    # =====================================================

    enable_tuning = data.get("enable_tuning", True)

    # Accept only an explicit Boolean to avoid interpreting
    # strings such as "false" as True.
    if not isinstance(enable_tuning, bool):
        enable_tuning = True

    if enable_tuning:

        if category in SUPERVISED_CATEGORIES:

            tuning_by_target = {}

            for target in y_columns:

                target_recommendation = (
                    recommendation_by_target.get(target, {})
                )

                best_algorithm = target_recommendation.get(
                    "best_algorithm"
                )

                target_benchmarks = get_results_for_target(
                    results,
                    target
                )

                if (
                    not best_algorithm
                    or not get_successful_results(target_benchmarks)
                ):
                    tuning_by_target[target] = {
                        "success": False,
                        "target": target,
                        "error": (
                            "No recommended algorithm is available "
                            "for this target."
                        )
                    }
                    continue

                target_frame = df[x_columns + [target]].copy()

                try:
                    tuning_result = tune_recommended_model(
                        target_frame,
                        target,
                        category,
                        best_algorithm
                    )

                    if not isinstance(tuning_result, dict):
                        tuning_result = {
                            "success": True,
                            "result": tuning_result
                        }

                    tuning_result.setdefault("target", target)
                    tuning_result.setdefault("algorithm", best_algorithm)

                    tuning_by_target[target] = tuning_result

                except Exception as error:
                    app.logger.exception(
                        "Tuning failed for target %s.",
                        target
                    )

                    tuning_by_target[target] = {
                        "success": False,
                        "target": target,
                        "algorithm": best_algorithm,
                        "error": str(error)
                    }

            if len(y_columns) == 1:
                tuning = tuning_by_target[y_columns[0]]
            else:
                tuning = {
                    "by_target": tuning_by_target
                }

        elif category == "clustering":

            best_algorithm = recommendation.get("best_algorithm")

            if best_algorithm and get_successful_results(results):
                try:
                    tuning = tune_recommended_model(
                        df[x_columns].copy(),
                        None,
                        category,
                        best_algorithm
                    )
                except Exception as error:
                    tuning = {
                        "success": False,
                        "algorithm": best_algorithm,
                        "category": category,
                        "error": str(error)
                    }

        else:

            tuning = {
                "success": False,
                "category": category,
                "algorithm": recommendation.get("best_algorithm"),
                "message": (
                    "Hyperparameter optimization is not configured "
                    "for this analysis category."
                )
            }

    # =====================================================
    # CORRELATION
    #
    # Supplemental analysis. Failures do not stop results.
    # =====================================================

    selected_columns = list(dict.fromkeys(
        x_columns + y_columns
    ))

    correlation = {}
    strongest = []

    if len(selected_columns) >= 2:

        selected_frame = df[selected_columns].copy()

        try:
            correlation = correlation_matrix(selected_frame)
        except Exception:
            app.logger.exception("Correlation matrix failed.")
            correlation = {}

        try:
            strongest = strongest_correlations(selected_frame)
        except Exception:
            app.logger.exception("Strongest correlations failed.")
            strongest = []

    # =====================================================
    # Feature Importance Compatibility
    # =====================================================

    # Keep the original field for existing frontend code.
    # Add a new per-target field for multi-target analysis.

    if len(y_columns) == 1:
        feature_importance = feature_importance_by_target.get(
            y_columns[0],
            []
        )
    elif len(y_columns) > 1:
        feature_importance = []
    else:
        feature_importance = []

    # =====================================================
    # Final Response
    # =====================================================

    analysis = {

        "dataset": {
            "task": category,
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },

        "selection": {
            "selection_mode": selection_mode,
            "category": category,
            "x_columns": x_columns,
            "y_columns": y_columns,
            "algorithms": algorithms,
            "automatic_selection": selection_info
        },

        "benchmark_results": results,

        "recommendation": recommendation,

        "recommendation_by_target": recommendation_by_target,

        "tuning": tuning,

        "feature_importance": feature_importance,

        "feature_importance_by_target": feature_importance_by_target,

        "target_results": target_results,

        "correlation": correlation,

        "strongest_correlations": strongest

    }

    successful_results = get_successful_results(results)

    return {
        "success": True,
        "message": (
            "SimuRithm analysis completed."
            if successful_results
            else "Analysis completed, but no benchmark produced a valid score."
        ),
        "analysis": analysis
    }, 200


# =========================================================
# Multi-Category Benchmark Runner
# =========================================================

@app.route("/run_benchmark", methods=["POST"])
def run_category_benchmark():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return error_response("No valid JSON object received.")

        file_path = data.get("file_path")

        if not isinstance(file_path, str) or not file_path.strip():
            return error_response("file_path is required.")

        # -------------------------------------------------
        # Backward compatibility: single-category request
        # -------------------------------------------------

        categories = data.get("categories")

        if categories is None:
            response, status = execute_benchmark(data)
            return jsonify(response), status

        if not isinstance(categories, list) or not categories:
            return error_response(
                "categories must be a non-empty array."
            )

        # -------------------------------------------------
        # Execute every selected category independently
        # -------------------------------------------------

        category_results = []
        category_errors = []

        for selection in categories:

            if not isinstance(selection, dict):
                category_errors.append({
                    "category": None,
                    "success": False,
                    "message": "Each category selection must be an object."
                })
                continue

            category = selection.get("category")

            if category not in CATEGORY_ALGORITHMS:
                category_errors.append({
                    "category": category,
                    "success": False,
                    "message": "Unsupported analysis category.",
                    "available_categories": list(
                        CATEGORY_ALGORITHMS.keys()
                    )
                })
                continue

            # Build a request using the existing benchmark engine.
            category_payload = {
                "file_path": file_path,
                "selection_mode": data.get("selection_mode", "manual"),
                "category": category,
                "x_columns": selection.get("x_columns", []),
                "y_columns": selection.get("y_columns", []),
                "algorithms": selection.get("algorithms", []),
                "enable_tuning": data.get("enable_tuning", True)
            }

            response, status = execute_benchmark(category_payload)

            if status != 200 or not response.get("success"):
                category_errors.append({
                    "category": category,
                    "success": False,
                    "message": response.get(
                        "message",
                        "Category benchmarking failed."
                    ),
                    "details": response
                })
                continue

            analysis = response.get("analysis", {})

            # Each category keeps its own metrics,
            # recommendation, tuning, and benchmark results.
            category_results.append({
                "category": category,
                "success": True,
                "selection": analysis.get("selection", {}),
                "benchmark_results": analysis.get(
                    "benchmark_results", []
                ),
                "recommendation": analysis.get(
                    "recommendation", {}
                ),
                "recommendation_by_target": analysis.get(
                    "recommendation_by_target", {}
                ),
                "tuning": analysis.get("tuning"),
                "feature_importance": analysis.get(
                    "feature_importance", []
                ),
                "feature_importance_by_target": analysis.get(
                    "feature_importance_by_target", {}
                ),
                "target_results": analysis.get(
                    "target_results", []
                ),
                "correlation": analysis.get("correlation", {}),
                "strongest_correlations": analysis.get(
                    "strongest_correlations", []
                )
            })

        # -------------------------------------------------
        # Combined response
        # -------------------------------------------------

        successful_count = len(category_results)
        failed_count = len(category_errors)

        return jsonify({
            "success": successful_count > 0,
            "message": (
                "Multi-category analysis completed."
                if failed_count == 0
                else "Multi-category analysis completed with some errors."
                if successful_count > 0
                else "All selected category benchmarks failed."
            ),
            "dataset": {
                "file_path": file_path
            },
            "summary": {
                "requested_categories": len(categories),
                "successful_categories": successful_count,
                "failed_categories": failed_count
            },
            "category_results": category_results,
            "category_errors": category_errors
        }), 200 if successful_count > 0 else 400

    except Exception as error:
        app.logger.exception("Multi-category benchmark execution failed.")

        payload = {
            "success": False,
            "message": str(error)
        }

        if app.debug:
            payload["traceback"] = traceback.format_exc()

        return jsonify(payload), 500


# =========================================================
# Legacy Compatibility Endpoint
#
# /run_analyze now executes a benchmark instead of
# always returning HTTP 410.
# =========================================================

@app.route("/run_analyze", methods=["POST"])
def run_analyze():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return error_response(
                "No valid JSON object received."
            )

        legacy_data = dict(data)

        # ---------------------------------------------
        # Determine category for older frontend payloads.
        # ---------------------------------------------

        category = legacy_data.get("category")

        if not category:
            file_path = legacy_data.get("file_path")

            if not isinstance(file_path, str) or not file_path.strip():
                return error_response(
                    "file_path is required."
                )

            x_columns = legacy_data.get("x_columns", [])
            y_columns = legacy_data.get("y_columns", [])

            if not isinstance(x_columns, list):
                return error_response("x_columns must be an array.")

            if not isinstance(y_columns, list):
                return error_response("y_columns must be an array.")

            # Prefer the selected target for supervised tasks.
            if y_columns:
                df = load_dataset(file_path)

                target = y_columns[0]

                if target not in df.columns:
                    return error_response(
                        "Selected target was not found in the dataset.",
                        400,
                        missing_columns=[target]
                    )

                model_columns = list(dict.fromkeys(
                    x_columns + [target]
                ))

                missing = [
                    column
                    for column in model_columns
                    if column not in df.columns
                ]

                if missing:
                    return error_response(
                        "Selected columns were not found.",
                        400,
                        missing_columns=missing
                    )

                model_df = df[model_columns].copy()

                category = detect_task(
                    model_df,
                    target
                )

            else:
                # Unsupervised tasks cannot be reliably inferred
                # from a missing target. Require the category.
                return error_response(
                    "The legacy request must include 'category' "
                    "when no Y variable is selected."
                )

        # ---------------------------------------------
        # Convert the singular algorithm field.
        # ---------------------------------------------

        algorithms = legacy_data.get("algorithms")

        if not algorithms:
            algorithm = legacy_data.get("algorithm")

            if not algorithm:
                return error_response(
                    "Please select an algorithm."
                )

            algorithms = [algorithm]

        legacy_data["category"] = category
        legacy_data["algorithms"] = algorithms

        response, status = execute_benchmark(legacy_data)

        return jsonify(response), status

    except Exception as error:

        app.logger.exception("Legacy benchmark execution failed.")

        payload = {
            "success": False,
            "message": str(error)
        }

        if app.debug:
            payload["traceback"] = traceback.format_exc()

        return jsonify(payload), 500


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
