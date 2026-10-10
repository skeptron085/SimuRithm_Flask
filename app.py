from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider

import math
import numpy as np
import traceback


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
# JSON Sanitizer
# =========================================================

def sanitize_json_value(value):

    if isinstance(value, dict):

        return {
            key: sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            sanitize_json_value(item)
            for item in value
        ]

    if isinstance(value, np.generic):

        return sanitize_json_value(
            value.item()
        )

    if isinstance(value, float):

        if not math.isfinite(value):

            return None

    return value


class SafeJSONProvider(DefaultJSONProvider):

    def response(
        self,
        *args,
        **kwargs
    ):

        args = tuple(
            sanitize_json_value(value)
            for value in args
        )

        kwargs = {
            key: sanitize_json_value(value)
            for key, value in kwargs.items()
        }

        return super().response(
            *args,
            **kwargs
        )


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)

app.json = SafeJSONProvider(app)


# =========================================================
# Home
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "system":
            "SimuRithm",

        "version":
            "Release 1",

        "status":
            "Running"

    })


# =========================================================
# Dataset Analysis
#
# IMPORTANT:
# This endpoint DOES NOT benchmark algorithms.
#
# Its purpose is to prepare the Analysis Configuration
# screen.
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        data = request.get_json()

        if not isinstance(data, dict):

            return jsonify({

                "success": False,

                "message":
                    "No JSON received."

            }), 400

        file_path = data.get(
            "file_path"
        )

        if not file_path:

            return jsonify({

                "success": False,

                "message":
                    "file_path is required."

            }), 400

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        df = load_dataset(
            file_path
        )

        # -------------------------------------------------
        # Dataset information
        # -------------------------------------------------

        statistics = (
            statistics_summary(df)
        )

        statistics_details = (
            statistics_table(df)
        )

        preview = (
            dataset_preview(df)
        )

        # -------------------------------------------------
        # NLP
        # -------------------------------------------------

        header_analysis = (
            analyze_headers(df)
        )

        target_candidates = (
            suggest_target(df)
        )

        suggested_target = (

            target_candidates[0]
            if target_candidates
            else None

        )

        # -------------------------------------------------
        # Task
        # -------------------------------------------------

        suggested_task = detect_task(

            df,

            suggested_target

        )

        explanation = task_reason(

            df,

            suggested_target

        )

        # -------------------------------------------------
        # Columns
        # -------------------------------------------------

        column_names = [
            str(column)
            for column in df.columns
        ]

        # -------------------------------------------------
        # Algorithm suggestions
        # -------------------------------------------------

        suggested_algorithms = (
            CATEGORY_ALGORITHMS.get(
                suggested_task,
                []
            )
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return jsonify({

            "success":
                True,

            "dataset": {

                "target":
                    suggested_target,

                "target_candidates":
                    target_candidates,

                "task":
                    suggested_task,

                "reason":
                    explanation,

                "rows":
                    statistics.get(
                        "rows",
                        len(df)
                    ),

                "columns":
                    statistics.get(
                        "columns",
                        len(df.columns)
                    ),

                "column_names":
                    column_names

            },

            "statistics":
                statistics,

            "statistics_table":
                statistics_details,

            "preview":
                preview,

            "header_analysis":
                header_analysis,

            "nlp_analysis":
                header_analysis,

            # -------------------------------------------------
            # THIS OBJECT FEEDS THE FRONTEND CONFIGURATION
            # -------------------------------------------------

            "analysis_configuration": {

                "available_columns":
                    column_names,

                "suggested_target":
                    suggested_target,

                "target_candidates":
                    target_candidates,

                "suggested_task":
                    suggested_task,

                "available_algorithms":
                    suggested_algorithms

            }

        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "message":
                str(error),

            "traceback":
                traceback.format_exc()

        }), 500


# =========================================================
# Benchmark Runner
# =========================================================

@app.route(
    "/run_benchmark",
    methods=["POST"]
)
def run_category_benchmark():

    try:

        data = request.get_json()

        if not isinstance(data, dict):

            return jsonify({

                "success":
                    False,

                "message":
                    "No JSON received."

            }), 400

        # -------------------------------------------------
        # Request
        # -------------------------------------------------

        file_path = data.get(
            "file_path"
        )

        category = data.get(
            "category"
        )

        x_columns = data.get(
            "x_columns",
            []
        )

        y_columns = data.get(
            "y_columns",
            []
        )

        algorithms = data.get(
            "algorithms",
            []
        )

        if not file_path:

            return jsonify({

                "success":
                    False,

                "message":
                    "file_path is required."

            }), 400

        if category not in CATEGORY_ALGORITHMS:

            return jsonify({

                "success":
                    False,

                "message":
                    "Unsupported analysis category.",

                "available_categories":
                    list(
                        CATEGORY_ALGORITHMS.keys()
                    )

            }), 400

        # -------------------------------------------------
        # Validate arrays
        # -------------------------------------------------

        if not isinstance(
            x_columns,
            list
        ):

            return jsonify({

                "success":
                    False,

                "message":
                    "x_columns must be an array."

            }), 400

        if not isinstance(
            y_columns,
            list
        ):

            return jsonify({

                "success":
                    False,

                "message":
                    "y_columns must be an array."

            }), 400

        if not isinstance(
            algorithms,
            list
        ):

            return jsonify({

                "success":
                    False,

                "message":
                    "algorithms must be an array."

            }), 400

        if not x_columns:

            return jsonify({

                "success":
                    False,

                "message":
                    "Select at least one X variable."

            }), 400

        # -------------------------------------------------
        # Supervised targets
        # -------------------------------------------------

        supervised_categories = {

            "classification",
            "regression"

        }

        if (
            category in supervised_categories
            and
            not y_columns
        ):

            return jsonify({

                "success":
                    False,

                "message":
                    "Select at least one Y variable."

            }), 400

        # -------------------------------------------------
        # Algorithms
        # -------------------------------------------------

        allowed_algorithms = (
            CATEGORY_ALGORITHMS[
                category
            ]
        )

        if not algorithms:

            return jsonify({

                "success":
                    False,

                "message":
                    "Select at least one algorithm."

            }), 400

        algorithms = list(
            dict.fromkeys(
                algorithms
            )
        )

        invalid_algorithms = [

            algorithm

            for algorithm in algorithms

            if algorithm
            not in allowed_algorithms

        ]

        if invalid_algorithms:

            return jsonify({

                "success":
                    False,

                "message":
                    "One or more selected algorithms "
                    "are not available for this task.",

                "invalid_algorithms":
                    invalid_algorithms,

                "allowed_algorithms":
                    allowed_algorithms

            }), 400

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        df = load_dataset(
            file_path
        )

        # -------------------------------------------------
        # Validate columns
        # -------------------------------------------------

        requested_columns = (
            x_columns +
            y_columns
        )

        missing_columns = [

            column

            for column
            in requested_columns

            if column not in df.columns

        ]

        if missing_columns:

            return jsonify({

                "success":
                    False,

                "message":
                    "Selected columns were not found.",

                "missing_columns":
                    missing_columns

            }), 400

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        results = []

        feature_importance = []

        target_results = []

        # =================================================
        # CLASSIFICATION
        # =================================================

        if category == "classification":

            for target in y_columns:

                model_df = df[
                    x_columns + [target]
                ].copy()

                task = detect_task(
                    model_df,
                    target
                )

                if task != "classification":

                    target_results.append({

                        "target":
                            target,

                        "task":
                            task,

                        "error":
                            (
                                f"Target '{target}' "
                                f"was detected as "
                                f"'{task}', so classification "
                                f"benchmarking was skipped."
                            )

                    })

                    continue

                target_benchmarks = []

                for algorithm in algorithms:

                    algorithm_result = (
                        run_benchmark(
                            model_df,
                            target,
                            algorithm
                        )
                    )

                    for result in algorithm_result:

                        result["target"] = target

                        result["category"] = (
                            category
                        )

                        target_benchmarks.append(
                            result
                        )

                        results.append(
                            result
                        )

                if not feature_importance:

                    try:

                        feature_importance = (
                            get_feature_importance(
                                model_df,
                                target
                            )
                        )

                    except Exception:

                        feature_importance = []

                target_results.append({

                    "target":
                        target,

                    "task":
                        category,

                    "benchmarks":
                        target_benchmarks

                })

        # =================================================
        # REGRESSION
        # =================================================

        elif category == "regression":

            for target in y_columns:

                model_df = df[
                    x_columns + [target]
                ].copy()

                task = detect_task(
                    model_df,
                    target
                )

                if task != "regression":

                    target_results.append({

                        "target":
                            target,

                        "task":
                            task,

                        "error":
                            (
                                f"Target '{target}' "
                                f"was detected as "
                                f"'{task}', so regression "
                                f"benchmarking was skipped."
                            )

                    })

                    continue

                target_benchmarks = []

                for algorithm in algorithms:

                    algorithm_result = (
                        run_regression_benchmark(
                            model_df,
                            target,
                            algorithm
                        )
                    )

                    for result in algorithm_result:

                        result["target"] = target

                        result["category"] = (
                            category
                        )

                        target_benchmarks.append(
                            result
                        )

                        results.append(
                            result
                        )

                if not feature_importance:

                    try:

                        feature_importance = (
                            get_feature_importance(
                                model_df,
                                target
                            )
                        )

                    except Exception:

                        feature_importance = []

                target_results.append({

                    "target":
                        target,

                    "task":
                        category,

                    "benchmarks":
                        target_benchmarks

                })

        # =================================================
        # CLUSTERING
        # =================================================

        elif category == "clustering":

            features = df[
                x_columns
            ].copy()

            for algorithm in algorithms:

                algorithm_results = (
                    run_clustering(

                        features,

                        selected_features=
                            x_columns,

                        selected_algorithm=
                            algorithm

                    )
                )

                for result in algorithm_results:

                    result["category"] = (
                        category
                    )

                    results.append(
                        result
                    )

        # =================================================
        # ANOMALY DETECTION
        # =================================================

        elif category == "anomaly_detection":

            results = (
                run_anomaly_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )

        # =================================================
        # DIMENSIONALITY REDUCTION
        # =================================================

        elif category == "dimensionality_reduction":

            results = (
                run_reduction_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )

        # =================================================
        # TOPIC MODELING
        # =================================================

        elif category == "topic_modeling":

            results = (
                run_topic_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )

        # =================================================
        # TEXT SIMILARITY
        # =================================================

        elif category == "text_similarity":

            results = (
                run_similarity_benchmarks(
                    df[x_columns].copy(),
                    algorithms
                )
            )

        # =================================================
        # SEQUENCE MODELING
        # =================================================

        elif category == "sequence_modeling":

            results = (
                run_sequence_benchmarks(

                    df[x_columns].copy(),

                    df[y_columns].copy(),

                    y_columns,

                    algorithms

                )
            )

        # -------------------------------------------------
        # Normalize result metrics
        # -------------------------------------------------

        metric_defaults = {

            "classification":
                ("accuracy", True),

            "regression":
                ("r2", True),

            "clustering":
                ("silhouette", True),

            "anomaly_detection":
                ("primary_score", True),

            "dimensionality_reduction":
                ("primary_score", True),

            "topic_modeling":
                ("primary_score", True),

            "text_similarity":
                ("primary_score", True),

            "sequence_modeling":
                ("primary_score", True)

        }

        default_metric, higher_is_better = (
            metric_defaults.get(
                category,
                ("primary_score", True)
            )
        )

        for result in results:

            result["category"] = (
                category
            )

            metric = result.get(
                "primary_metric"
            ) or default_metric

            score = result.get(
                "primary_score"
            )

            if score is None:

                score = result.get(
                    metric
                )

            result["primary_metric"] = (
                metric
            )

            result["primary_score"] = (
                score
            )

            result["higher_is_better"] = (
                result.get(
                    "higher_is_better",
                    higher_is_better
                )
            )

        # -------------------------------------------------
        # Recommendation
        # -------------------------------------------------

        successful_results = [

            result

            for result in results

            if (
                result.get("error")
                in (None, "")
            )
            and
            result.get("primary_score")
            is not None

        ]

        recommendation = recommend(

            successful_results,

            category

        )

        # -------------------------------------------------
        # Mark recommended algorithm
        # -------------------------------------------------

        best_algorithm = (
            recommendation.get(
                "best_algorithm"
            )
        )

        for result in results:

            result["is_recommended"] = (

                result.get("algorithm")
                == best_algorithm

                and

                result.get("error")
                in (None, "")

            )

        # =================================================
        # TUNING
        # =================================================

        tuning = None

        enable_tuning = bool(
            data.get(
                "enable_tuning",
                True
            )
        )

        if (
            enable_tuning
            and
            best_algorithm
            and
            successful_results
        ):

            if category in {
                "classification",
                "regression"
            }:

                tuning_results = []

                for target in y_columns:

                    target_frame = df[
                        x_columns + [target]
                    ].copy()

                    tuning_result = (
                        tune_recommended_model(

                            target_frame,

                            target,

                            category,

                            best_algorithm

                        )
                    )

                    tuning_results.append(
                        tuning_result
                    )

                tuning = (

                    tuning_results[0]
                    if len(tuning_results) == 1
                    else
                    {
                        "by_target":
                            tuning_results
                    }

                )

            elif category == "clustering":

                tuning = (
                    tune_recommended_model(

                        df[x_columns].copy(),

                        None,

                        category,

                        best_algorithm

                    )
                )

            else:

                tuning = {

                    "success":
                        False,

                    "algorithm":
                        best_algorithm,

                    "category":
                        category,

                    "error":
                        (
                            "Hyperparameter optimization "
                            "is not configured for this "
                            "analysis category."
                        )

                }

        # =================================================
        # Correlation
        #
        # IMPORTANT:
        # Correlation is supplemental only.
        # Failure here must NEVER stop benchmarking.
        # =================================================

        selected_columns = list(
            dict.fromkeys(
                x_columns + y_columns
            )
        )

        selected_frame = df[
            selected_columns
        ].copy()

        correlation = {}
        strongest = []

        if len(selected_columns) >= 2:

            try:

                correlation = (
                    correlation_matrix(
                        selected_frame
                    )
                )

            except Exception:

                correlation = {}

            try:

                strongest = (
                    strongest_correlations(
                        selected_frame
                    )
                )

            except Exception:

                strongest = []

        # =================================================
        # Final Analysis
        # =================================================

        analysis = {

            "dataset": {

                "task":
                    category,

                "rows":
                    int(len(df)),

                "columns":
                    int(len(df.columns))

            },

            "selection": {

                "category":
                    category,

                "x_columns":
                    x_columns,

                "y_columns":
                    y_columns,

                "algorithms":
                    algorithms

            },

            "benchmark_results":
                results,

            "recommendation":
                recommendation,

            "tuning":
                tuning,

            "feature_importance":
                feature_importance,

            "target_results":
                target_results,

            "correlation":
                correlation,

            "strongest_correlations":
                strongest

        }

        return jsonify({

            "success":
                True,

            "message":
                "SimuRithm analysis completed.",

            "analysis":
                analysis

        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "message":
                str(error),

            "traceback":
                traceback.format_exc()

        }), 500


# =========================================================
# Legacy Compatibility Endpoint
#
# Existing frontend can continue using /run_analyze
# while we transition it to /run_benchmark.
# =========================================================

@app.route(
    "/run_analyze",
    methods=["POST"]
)
def run_analyze():

    data = request.get_json()

    if not isinstance(data, dict):

        return jsonify({

            "success":
                False,

            "message":
                "No JSON received."

        }), 400

    # -----------------------------------------------------
    # Convert old request format to new benchmark format
    # -----------------------------------------------------

    file_path = data.get(
        "file_path"
    )

    x_columns = data.get(
        "x_columns",
        []
    )

    y_columns = data.get(
        "y_columns",
        []
    )

    algorithm = data.get(
        "algorithm"
    )

    if not algorithm:

        return jsonify({

            "success":
                False,

            "message":
                "Please select an algorithm."

        }), 400

    df = load_dataset(
        file_path
    )

    target = (
        y_columns[0]
        if y_columns
        else None
    )

    task = detect_task(
        df,
        target
    )

    # -----------------------------------------------------
    # Forward to benchmark logic by internally recreating
    # the request payload.
    # -----------------------------------------------------

    category = task

    if category not in CATEGORY_ALGORITHMS:

        return jsonify({

            "success":
                False,

            "message":
                f"Detected task '{category}' "
                "cannot be executed by the legacy "
                "/run_analyze endpoint."

        }), 400

    request_payload = {

        "file_path":
            file_path,

        "category":
            category,

        "x_columns":
            x_columns,

        "y_columns":
            y_columns,

        "algorithms":
            [algorithm],

        "enable_tuning":
            True

    }

    # Directly reproduce the endpoint call is undesirable;
    # instead return a clear migration instruction while
    # preserving the endpoint.
    return jsonify({

        "success":
            False,

        "message":
            (
                "The legacy /run_analyze endpoint now "
                "requires the frontend to use /run_benchmark."
            ),

        "migration": {

            "endpoint":
                "/run_benchmark",

            "payload":
                request_payload

        }

    }), 410


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )