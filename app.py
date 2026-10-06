from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
import math
import numpy as np
import traceback

from modules.parser import (
    load_dataset,
    dataset_preview
)

from modules.nlp_analyzer import (
    analyze_headers,
    suggest_target
)

from modules.statistics_engine import (
    statistics_summary,
    statistics_table
)

from modules.correlation_engine import (
    correlation_matrix,
    strongest_correlations
)

from modules.task_detector import (
    detect_task,
    task_reason
)

from modules.benchmark import (
    run_benchmark
)

from modules.regression_benchmark import (
    run_regression_benchmark
)

from modules.clustering_benchmark import (
    run_clustering
)

from modules.advanced_benchmarks import (
    CATEGORY_ALGORITHMS,
    run_anomaly_benchmarks,
    run_reduction_benchmarks,
    run_sequence_benchmarks,
    run_similarity_benchmarks,
    run_topic_benchmarks
)

from modules.feature_importance import (
    get_feature_importance
)

from modules.tuning import (
    tune_random_forest,
    tune_recommended_model
)

from modules.recommender import (
    recommend
)

def sanitize_json_value(value):

    if isinstance(value, dict):

        return {
            key: sanitize_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [sanitize_json_value(item) for item in value]

    if isinstance(value, np.generic):

        return sanitize_json_value(value.item())

    if isinstance(value, float) and not math.isfinite(value):

        return None

    return value


class SafeJSONProvider(DefaultJSONProvider):

    def response(self, *args, **kwargs):

        args = tuple(sanitize_json_value(value) for value in args)

        kwargs = {
            key: sanitize_json_value(value)
            for key, value in kwargs.items()
        }

        return super().response(*args, **kwargs)


app = Flask(__name__)
app.json = SafeJSONProvider(app)


@app.route("/")
def home():

    return jsonify({

        "system": "SimuRithm",

        "version": "Release 1",

        "status": "Running"

    })


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({

                "success": False,

                "message": "No JSON received."

            }), 400


        # -------------------------------------------------
        # File Path
        # -------------------------------------------------

        file_path = data.get("file_path")

        if not file_path:

            return jsonify({

                "success": False,

                "message": "file_path is required."

            }), 400


        # -------------------------------------------------
        # Load Dataset
        # -------------------------------------------------

        df = load_dataset(file_path)


        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        statistics = statistics_summary(df)

        statistics_details = statistics_table(df)


        # -------------------------------------------------
        # Dataset Preview
        # -------------------------------------------------

        preview = dataset_preview(df)


        # -------------------------------------------------
        # NLP Header Analysis
        # -------------------------------------------------

        header_analysis = analyze_headers(df)


        # -------------------------------------------------
        # Detect Suggested Target
        # -------------------------------------------------

        targets = suggest_target(df)

        if len(targets) > 0:

            suggested_target = targets[0]

        else:

            suggested_target = None


        # -------------------------------------------------
        # Detect Suggested ML Task
        # -------------------------------------------------

        task = detect_task(

            df,

            suggested_target

        )


        # -------------------------------------------------
        # Task Explanation
        # -------------------------------------------------

        explanation = task_reason(

            df,

            suggested_target

        )


        # -------------------------------------------------
        # Dataset Column Names
        # -------------------------------------------------

        column_names = df.columns.tolist()


        # -------------------------------------------------
        # Return Preparation Data
        # -------------------------------------------------

        return jsonify({

            "success": True,


            # ---------------------------------------------
            # Dataset Information
            # ---------------------------------------------

            "dataset": {

                "target": suggested_target,

                "task": task,

                "reason": explanation,

                "rows": statistics["rows"],

                "columns": statistics["columns"],

                "column_names": column_names

            },


            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            "statistics": statistics,

            "statistics_table": statistics_details,


            # ---------------------------------------------
            # Dataset Preview
            # ---------------------------------------------

            "preview": preview,


            # ---------------------------------------------
            # NLP Header Analysis
            # ---------------------------------------------

            "header_analysis": header_analysis,

            "nlp_analysis": header_analysis,


            # ---------------------------------------------
            # Analysis Configuration
            # ---------------------------------------------

            "analysis_configuration": {

                "available_columns":
                    column_names,

                "suggested_target":
                    suggested_target,

                "suggested_task":
                    task

            }

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e),

            "traceback": traceback.format_exc()

        }), 500

@app.route("/run_analyze", methods=["POST"])
def run_analyze():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({

                "success": False,

                "message": "No JSON received."

            }), 400

        # -------------------------------------------------
        # Required File Path
        # -------------------------------------------------

        file_path = data.get("file_path")

        if not file_path:

            return jsonify({

                "success": False,

                "message": "file_path is required."

            }), 400

        # -------------------------------------------------
        # User Selected X Variables
        # -------------------------------------------------

        x_columns = data.get("x_columns", [])

        # -------------------------------------------------
        # User Selected Y Variables
        # -------------------------------------------------

        y_columns = data.get("y_columns", [])

        # -------------------------------------------------
        # User Selected Algorithm
        # -------------------------------------------------

        algorithm = data.get("algorithm")

        # -------------------------------------------------
        # Validate X Variables
        # -------------------------------------------------

        if not isinstance(x_columns, list):

            return jsonify({

                "success": False,

                "message": "x_columns must be an array."

            }), 400

        if len(x_columns) == 0:

            return jsonify({

                "success": False,

                "message": "Please select at least one X variable."

            }), 400

        # -------------------------------------------------
        # Validate Y Variables
        # -------------------------------------------------

        if not isinstance(y_columns, list):

            return jsonify({

                "success": False,

                "message": "y_columns must be an array."

            }), 400

        if len(y_columns) == 0:

            return jsonify({

                "success": False,

                "message": "Please select at least one Y variable."

            }), 400

        # -------------------------------------------------
        # Validate Algorithm
        # -------------------------------------------------

        if not algorithm:

            return jsonify({

                "success": False,

                "message": "Please select an algorithm."

            }), 400

        # -------------------------------------------------
        # Load Dataset
        # -------------------------------------------------

        df = load_dataset(file_path)

        # -------------------------------------------------
        # Validate Columns Against Dataset
        # -------------------------------------------------

        missing_x = [

            column

            for column in x_columns

            if column not in df.columns

        ]

        missing_y = [

            column

            for column in y_columns

            if column not in df.columns

        ]

        if missing_x:

            return jsonify({

                "success": False,

                "message": "Selected X column(s) not found.",

                "missing_x_columns": missing_x

            }), 400

        if missing_y:

            return jsonify({

                "success": False,

                "message": "Selected Y column(s) not found.",

                "missing_y_columns": missing_y

            }), 400

        # -------------------------------------------------
        # Prevent Same Column From Being X and Y
        # -------------------------------------------------

        overlapping = list(

            set(x_columns).intersection(

                set(y_columns)

            )

        )

        if overlapping:

            return jsonify({

                "success": False,

                "message":
                    "A column cannot be both X and Y.",

                "overlapping_columns":
                    overlapping

            }), 400

        # -------------------------------------------------
        # Validate Algorithm Names
        # -------------------------------------------------

        allowed_algorithms = [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "KNN",

            "Naive Bayes",

            "SVM",

            "Linear Regression",

            "Support Vector Regression",

            "K-Means",

            "Agglomerative",

            "DBSCAN"

        ]

        if algorithm not in allowed_algorithms:

            return jsonify({

                "success": False,

                "message":
                    "Invalid algorithm selected.",

                "allowed_algorithms":
                    allowed_algorithms

            }), 400

        # -------------------------------------------------
        # Selected Columns
        # -------------------------------------------------

        selected_columns = (

            x_columns +

            y_columns

        )

        selected_df = df[selected_columns].copy()

        # -------------------------------------------------
        # Correlation Analysis
        #
        # Correlation is calculated only from the
        # variables selected by the user.
        # -------------------------------------------------

        correlation = correlation_matrix(

            selected_df

        )

        strongest = strongest_correlations(

            selected_df

        )

        # -------------------------------------------------
        # Results Containers
        # -------------------------------------------------

        benchmark_results = []

        feature_importance = []

        tuning_results = {}

        target_results = []

        # -------------------------------------------------
        # Process Each Y Variable
        # -------------------------------------------------

        for target in y_columns:

            # ---------------------------------------------
            # Create dataset for this target
            #
            # Only X variables + current Y are used.
            # Other Y variables are NOT used as predictors.
            # ---------------------------------------------

            model_columns = (

                x_columns +

                [target]

            )

            model_df = df[model_columns].copy()

            # ---------------------------------------------
            # Detect Task
            # ---------------------------------------------

            task = detect_task(

                model_df,

                target

            )

            explanation = task_reason(

                model_df,

                target

            )

            # ---------------------------------------------
            # Benchmark
            # ---------------------------------------------

            current_benchmark = []

            current_feature_importance = []

            current_tuning = {}

            if task == "classification":

                classification_algorithms = {

                    "Logistic Regression",

                    "Decision Tree",

                    "Random Forest",

                    "KNN",

                    "Naive Bayes",

                    "SVM"

                }

                if algorithm not in classification_algorithms:

                    return jsonify({

                        "success": False,

                        "message":
                            f"{algorithm} is not available for classification targets.",

                        "task": task,

                        "allowed_algorithms":
                            sorted(classification_algorithms)

                    }), 400

                all_results = run_benchmark(

                    model_df,

                    target,

                    algorithm

                )

                # -----------------------------------------
                # Keep only the algorithm selected by user
                # -----------------------------------------

                current_benchmark = [

                    result

                    for result in all_results

                    if result.get("algorithm")
                    == algorithm

                ]

                current_feature_importance = (

                    get_feature_importance(

                        model_df,

                        target

                    )

                )

                # -----------------------------------------
                # Tuning only when Random Forest is selected
                # -----------------------------------------

                if algorithm == "Random Forest":

                    current_tuning = (

                        tune_random_forest(

                            model_df,

                            target

                        )

                    )

            elif task == "regression":

                regression_algorithms = {

                    "Linear Regression",

                    "Decision Tree",

                    "Random Forest",

                    "KNN",

                    "Support Vector Regression"

                }

                if algorithm not in regression_algorithms:

                    return jsonify({

                        "success": False,

                        "message":
                            f"{algorithm} is not available for regression targets.",

                        "task": task,

                        "allowed_algorithms":
                            sorted(regression_algorithms)

                    }), 400

                all_results = run_regression_benchmark(

                    model_df,

                    target,

                    algorithm

                )

                current_benchmark = [

                    result

                    for result in all_results

                    if result.get("algorithm")
                    == algorithm

                ]

                current_feature_importance = (

                    get_feature_importance(

                        model_df,

                        target

                    )

                )

                if algorithm == "Random Forest":

                    current_tuning = (

                        tune_random_forest(

                            model_df,

                            target

                        )

                    )

            else:

                # -----------------------------------------
                # Clustering does not use a Y target
                # -----------------------------------------

                current_benchmark = run_clustering(

                    model_df,

                    selected_features=x_columns,

                    selected_algorithm=algorithm

                )

            # ---------------------------------------------
            # Recommendation
            # ---------------------------------------------

            current_recommendation = recommend(

                current_benchmark,

                task

            )

            # ---------------------------------------------
            # Store Target Result
            # ---------------------------------------------

            target_result = {

                "target": target,

                "task": task,

                "reason": explanation,

                "algorithm": algorithm,

                "benchmark_results":
                    current_benchmark,

                "feature_importance":
                    current_feature_importance,

                "tuning":
                    current_tuning,

                "recommendation":
                    current_recommendation

            }

            target_results.append(

                target_result

            )

            # ---------------------------------------------
            # Keep combined benchmark results
            # ---------------------------------------------

            benchmark_results.extend(

                current_benchmark

            )

            # ---------------------------------------------
            # Keep feature importance from last target
            # ---------------------------------------------

            if current_feature_importance:

                feature_importance = (

                    current_feature_importance

                )

            # ---------------------------------------------
            # Keep tuning from last target
            # ---------------------------------------------

            if current_tuning:

                tuning_results = current_tuning

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Selected variables analyzed successfully.",

            # ---------------------------------------------
            # User Selection
            # ---------------------------------------------

            "selection": {

                "x_columns": x_columns,

                "y_columns": y_columns,

                "algorithm": algorithm

            },

            # ---------------------------------------------
            # Dataset
            # ---------------------------------------------

            "dataset": {

                "rows": len(df),

                "columns": len(df.columns),

                "selected_columns":
                    selected_columns,

                "selected_rows":
                    len(selected_df),

                "selected_columns_count":
                    len(selected_columns)

            },

            # ---------------------------------------------
            # Correlation
            # ---------------------------------------------

            "correlation":

                correlation,

            "correlation_matrix":

                correlation,

            # ---------------------------------------------
            # Strongest Correlations
            # ---------------------------------------------

            "strongest_correlations":

                strongest,

            # ---------------------------------------------
            # Results Per Y
            # ---------------------------------------------

            "target_results":

                target_results,

            # ---------------------------------------------
            # Benchmark
            # ---------------------------------------------

            "benchmark_results":

                benchmark_results,

            # ---------------------------------------------
            # Feature Importance
            # ---------------------------------------------

            "feature_importance":

                feature_importance,

            # ---------------------------------------------
            # Tuning
            # ---------------------------------------------

            "tuning":

                tuning_results

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e),

            "traceback":
                traceback.format_exc()

        }), 500


@app.route("/run_benchmark", methods=["POST"])

def run_category_benchmark():

    try:

        data = request.get_json()

        if not isinstance(data, dict):

            return jsonify(
                
                {"success": False, 
                 
                 "message": "No JSON received."
                 
                 }
                 
                 ), 400

        file_path = data.get(
           
            "file_path"
            
            )
    
        category = data.get(
            
            "category"
            )

        x_columns = data.get("x_columns", [])

        y_columns = data.get("y_columns", [])

        algorithms = data.get("algorithms", [])

        selected_algorithms = data.get("selected_algorithms", algorithms)

        if not file_path:

            return jsonify(

                {"success": False, 
                 
                 "message": "file_path is required."
                 
                 }

                 ), 400
        
        if category not in CATEGORY_ALGORITHMS:

            return jsonify(
                
                {"success": False, 
                            
                    "message": "Unsupported analysis category."
                            
                 }
                
                 ), 400
        
        if (not isinstance
            
            (x_columns, list
             
             ) or not isinstance
             
             (y_columns, list
              
              )
           
            or not isinstance(algorithms, list) or not isinstance(selected_algorithms, list)):
            
            return jsonify({"success": False, "message": "Features, targets, and algorithms must be arrays."}), 400

        allowed_algorithms = CATEGORY_ALGORITHMS[category]
        
        algorithms = list(dict.fromkeys(algorithms))
        
        minimum = min(3, len(allowed_algorithms))
        
        maximum = min(5, len(allowed_algorithms))

        if not minimum <= len(algorithms) <= maximum:
          
            return jsonify({
             
                "success": False,
               
                "message": f"Select between {minimum} and {maximum} algorithms for this category."}), 400
        if any(algorithm not in allowed_algorithms for algorithm in algorithms):
            return jsonify({"success": False, "message": "One or more algorithms do not belong to this category."}), 400
        selected_algorithms = list(dict.fromkeys(selected_algorithms))
        if not selected_algorithms or any(algorithm not in algorithms for algorithm in selected_algorithms):
            return jsonify({"success": False, "message": "Selected algorithms must be included in the benchmark request."}), 400

        unsupervised = category in {
            "clustering", "anomaly_detection", "dimensionality_reduction",
            "topic_modeling", "text_similarity"
        }
        if not x_columns:
            return jsonify({"success": False, "message": "Select at least one feature."}), 400
        if category in {"classification", "regression", "sequence_modeling"} and not y_columns:
            return jsonify({"success": False, "message": "Select at least one target."}), 400

        df = load_dataset(file_path)
        missing_columns = [column for column in x_columns + y_columns if column not in df.columns]
        if missing_columns:
            return jsonify({"success": False, "message": "Selected columns were not found.",
                            "missing_columns": missing_columns}), 400

        results = []
        feature_importance = []
        if category in {"classification", "regression"}:
            runner = run_benchmark if category == "classification" else run_regression_benchmark
            for target in y_columns:
                model_df = df[x_columns + [target]].copy()
                for algorithm in algorithms:
                    for result in runner(model_df, target, algorithm):
                        result["target"] = target
                        results.append(result)
                if not feature_importance:
                    feature_importance = get_feature_importance(model_df, target)
        elif category == "clustering":
            features = df[x_columns].copy()
            for algorithm in algorithms:
                results.extend(run_clustering(features, selected_features=x_columns,
                                              selected_algorithm=algorithm))
        elif category == "anomaly_detection":
            results = run_anomaly_benchmarks(df[x_columns].copy(), algorithms)
        elif category == "dimensionality_reduction":
            results = run_reduction_benchmarks(df[x_columns].copy(), algorithms)
        elif category == "topic_modeling":
            results = run_topic_benchmarks(df[x_columns].copy(), algorithms)
        elif category == "text_similarity":
            results = run_similarity_benchmarks(df[x_columns].copy(), algorithms)
        else:
            results = run_sequence_benchmarks(
                df[x_columns].copy(),
                df[y_columns].copy(),
                y_columns,
                algorithms
            )

        score_fields = {
            "classification": ("accuracy", True),
            "regression": ("r2", True),
            "clustering": ("silhouette", True)
        }
        default_metric, higher_is_better = score_fields.get(category, (None, True))
        for result in results:
            result["category"] = category
            metric = result.get("primary_metric", default_metric)
            score = result.get("primary_score")
            if metric is not None and score is None:
                score = result.get(metric)
            result["primary_metric"] = metric or "Score"
            result["primary_score"] = score
            result["higher_is_better"] = result.get("higher_is_better", higher_is_better)

        selected_results = [
            result for result in results
            if result.get("algorithm") in selected_algorithms
        ]
        recommendation = recommend(selected_results, category)
        recommendation_ranks = {
            item["algorithm"]: rank
            for rank, item in enumerate(recommendation.get("ranking", []), start=1)
        }
        for result in results:
            result["category_rank"] = recommendation_ranks.get(result.get("algorithm"))
        best_algorithm = recommendation.get("best_algorithm")
        best_rows = [
            result for result in selected_results
            if result.get("algorithm") == best_algorithm and not result.get("error")
        ]
        best = None
        if best_algorithm in selected_algorithms and best_rows:
            best = {
                "algorithm": best_algorithm,
                "score": recommendation.get("score"),
                "metric": recommendation.get("metric", best_rows[0].get("primary_metric", "Score")),
                "recommendation_score": recommendation.get("recommendation_score")
            }
        if best:
            for result in results:
                result["is_recommended"] = (
                    result.get("algorithm") == best["algorithm"]
                    and result.get("algorithm") in selected_algorithms
                    and not result.get("error")
                )
            recommendation["task"] = category.replace("_", " ").title()
        else:
            for result in results:
                result["is_recommended"] = False
            recommendation = {
                "best_algorithm": "No successful model",
                "task": category.replace("_", " ").title(),
                "reason": "The selected models did not produce comparable scores. Review benchmark errors."
            }

        tuning = None
        if data.get("enable_tuning"):
            if best and category in {"classification", "regression"}:
                target_tunings = []
                for target in y_columns:
                    target_frame = df[x_columns + [target]].copy()
                    target_tunings.append(
                        tune_recommended_model(target_frame, target, category, best["algorithm"])
                    )
                tuning = target_tunings[0] if len(target_tunings) == 1 else {"by_target": target_tunings}
            elif best and category == "clustering":
                tuning = tune_recommended_model(
                    df[x_columns].copy(), None, category, best["algorithm"]
                )
            elif best:
                tuning = {
                    "success": False,
                    "algorithm": best["algorithm"],
                    "category": category,
                    "error": f"Hyperparameter optimization is not configured for {category}."
                }
            else:
                tuning = {
                    "success": False,
                    "category": category,
                    "error": "Tuning was skipped because no selected model produced a usable benchmark score."
                }

        selected_columns = x_columns + y_columns
        selected_frame = df[list(dict.fromkeys(selected_columns))].copy()
        analysis = {
            "dataset": {"task": category, "rows": len(df), "columns": len(df.columns)},
            "selection": {"category": category, "x_columns": x_columns,
                          "y_columns": y_columns, "algorithms": selected_algorithms},
            "correlation": correlation_matrix(selected_frame) if len(selected_columns) > 1 else {},
            "strongest_correlations": strongest_correlations(selected_frame) if len(selected_columns) > 1 else [],
            "benchmark_results": results,
            "feature_importance": feature_importance,
            "recommendation": recommendation,
            "tuning": tuning,
            "target_results": []
        }
        return jsonify({"success": True, "analysis": analysis})
    except Exception as error:
        return jsonify({"success": False, "message": str(error),
                        "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )