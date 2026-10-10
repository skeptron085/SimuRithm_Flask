
import math
from collections import defaultdict


# =========================================================
# RECOMMENDATION CONFIGURATION
# =========================================================

TASK_METRICS = {
    "classification": [
        (("accuracy",), True, 0.25),
        (("precision", "precision_score"), True, 0.15),
        (("recall", "recall_score"), True, 0.15),
        (("f1", "f1_score"), True, 0.25),
        (
            ("cv_score", "cross_validation_score", "cross_val_score"),
            True,
            0.20
        )
    ],

    "regression": [
        (("r2", "r2_score"), True, 0.35),
        (
            ("cv_score", "cross_validation_score", "cross_val_score"),
            True,
            0.30
        ),
        (("rmse", "root_mean_squared_error"), False, 0.20),
        (("mae", "mean_absolute_error"), False, 0.15)
    ],

    "clustering": [
        (("silhouette",), True, 0.55),
        (("calinski_harabasz",), True, 0.20),
        (("davies_bouldin",), False, 0.25)
    ],

    "anomaly_detection": [
        (("primary_score",), True, 1.0)
    ],

    "dimensionality_reduction": [
        (("primary_score",), True, 1.0)
    ],

    "topic_modeling": [
        (("primary_score",), True, 1.0)
    ],

    "text_similarity": [
        (("primary_score",), True, 1.0)
    ],

    "sequence_modeling": [
        (("primary_score", "r2", "r2_score"), True, 0.60),
        (("rmse", "root_mean_squared_error"), False, 0.25),
        (("mae", "mean_absolute_error"), False, 0.15)
    ]
}


TASK_NAMES = {
    "classification": "Classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "anomaly_detection": "Anomaly Detection",
    "dimensionality_reduction": "Dimensionality Reduction",
    "topic_modeling": "Topic Modeling",
    "text_similarity": "Text Similarity",
    "sequence_modeling": "Sequence Modeling"
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _finite_number(value):
    """Convert a value to a finite float, or return None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _get_metric_value(item, aliases):
    """Return the first valid metric value matching an alias."""

    for alias in aliases:
        if alias not in item:
            continue

        value = _finite_number(item.get(alias))

        if value is not None:
            return value

    return None


def _get_task_metrics(task):
    """
    Return configured metrics for a task.

    Unknown tasks use the primary_score field and respect
    the direction provided by each benchmark result.
    """

    return TASK_METRICS.get(
        task,
        [("primary_score",)]
    )


def _valid_results(results):
    """Keep successful results with a valid algorithm name."""

    valid = []

    if not isinstance(results, (list, tuple)):
        return valid

    for item in results:
        if not isinstance(item, dict):
            continue

        if item.get("error"):
            continue

        algorithm = item.get("algorithm")

        if not isinstance(algorithm, str) or not algorithm.strip():
            continue

        valid.append(item)

    return valid


# =========================================================
# MAIN RECOMMENDER
# =========================================================

def recommend(results, task):
    """
    Recommend the best-performing algorithm for a task.

    Supports:
        classification
        regression
        clustering
        anomaly_detection
        dimensionality_reduction
        topic_modeling
        text_similarity
        sequence_modeling

    Legacy results use task-specific metric fields.
    Advanced results use primary_score, primary_metric,
    and higher_is_better.

    Returns a dictionary suitable for Flask JSON responses.
    """

    task = str(task or "").strip().lower()

    valid_results = _valid_results(results)

    if not valid_results:
        return {
            "best_algorithm": "No successful model",
            "task": task,
            "reason": (
                "No selected model returned usable benchmark results."
            ),
            "score": None,
            "metric": None,
            "recommendation_score": None,
            "ranking": []
        }

    # -----------------------------------------------------
    # Configure metric weights
    # -----------------------------------------------------

    if task in TASK_METRICS:
        metrics = TASK_METRICS[task]
    else:
        metrics = [
            (("primary_score",), True, 1.0)
        ]

    # -----------------------------------------------------
    # Collect values by algorithm and metric
    # -----------------------------------------------------

    # Each algorithm may have multiple result rows, such as
    # sequence models evaluated against multiple targets.
    observations = defaultdict(
        lambda: defaultdict(list)
    )

    metric_directions = {}
    metric_weights = {}

    algorithm_rows = defaultdict(list)

    for item in valid_results:
        algorithm = item["algorithm"]
        algorithm_rows[algorithm].append(item)

        for metric_spec in metrics:
            aliases, configured_direction, weight = (
                metric_spec
                if len(metric_spec) == 3
                else (
                    metric_spec[0],
                    True,
                    1.0
                )
            )

            aliases = (
                (aliases,)
                if isinstance(aliases, str)
                else aliases
            )

            value = _get_metric_value(item, aliases)

            if value is None:
                continue

            # Advanced benchmark scores are identified by their
            # primary metric name so unrelated scores are never
            # normalized against each other.
            if aliases == ("primary_score",):
                metric_name = str(
                    item.get("primary_metric") or "Primary Score"
                )

                metric_key = "primary_score:" + metric_name

                direction = item.get(
                    "higher_is_better",
                    configured_direction
                )

                if not isinstance(direction, bool):
                    direction = configured_direction
            else:
                metric_key = aliases[0]
                direction = configured_direction

            observations[algorithm][metric_key].append(value)

            metric_directions[metric_key] = direction
            metric_weights[metric_key] = weight

    # -----------------------------------------------------
    # Aggregate repeated observations per algorithm
    # -----------------------------------------------------

    algorithm_metrics = defaultdict(dict)

    for algorithm, grouped_metrics in observations.items():
        for metric_key, values in grouped_metrics.items():
            if values:
                algorithm_metrics[algorithm][metric_key] = (
                    sum(values) / len(values)
                )

    if not algorithm_metrics:
        return {
            "best_algorithm": "No successful model",
            "task": task,
            "reason": (
                "Selected models did not return usable comparable metrics."
            ),
            "score": None,
            "metric": None,
            "recommendation_score": None,
            "ranking": []
        }

    # -----------------------------------------------------
    # Calculate metric bounds across algorithms
    # -----------------------------------------------------

    metric_bounds = {}

    all_metric_keys = {
        metric_key
        for metrics_by_algorithm in algorithm_metrics.values()
        for metric_key in metrics_by_algorithm
    }

    for metric_key in all_metric_keys:
        values = [
            metrics_by_algorithm[metric_key]
            for metrics_by_algorithm in algorithm_metrics.values()
            if metric_key in metrics_by_algorithm
        ]

        if values:
            metric_bounds[metric_key] = (
                min(values),
                max(values)
            )

    # -----------------------------------------------------
    # Normalize and calculate weighted scores
    # -----------------------------------------------------

    ranked = []

    for algorithm, values in algorithm_metrics.items():
        weighted_total = 0.0
        available_weight = 0.0

        for metric_key, value in values.items():
            if metric_key not in metric_bounds:
                continue

            low, high = metric_bounds[metric_key]

            if high == low:
                normalized_score = 0.5
            else:
                normalized_score = (
                    (value - low) / (high - low)
                )

            if not metric_directions.get(metric_key, True):
                normalized_score = 1.0 - normalized_score

            weight = metric_weights.get(metric_key, 1.0)

            weighted_total += normalized_score * weight
            available_weight += weight

        if available_weight <= 0:
            continue

        recommendation_score = (
            weighted_total / available_weight
        )

        rows = algorithm_rows[algorithm]

        primary_scores = [
            _finite_number(row.get("primary_score"))
            for row in rows
        ]

        primary_scores = [
            score for score in primary_scores
            if score is not None
        ]

        primary_metrics = [
            str(row["primary_metric"])
            for row in rows
            if row.get("primary_metric")
        ]

        # Prefer the reported primary score for advanced
        # benchmarks; otherwise summarize legacy metrics.
        if primary_scores:
            display_score = sum(primary_scores) / len(primary_scores)
        else:
            usable_values = list(values.values())

            display_score = (
                sum(usable_values) / len(usable_values)
                if usable_values
                else None
            )

        ranked.append({
            "algorithm": algorithm,
            "recommendation_score": round(
                recommendation_score, 4
            ),
            "score": (
                round(display_score, 4)
                if display_score is not None
                else None
            ),
            "metric": (
                primary_metrics[0]
                if primary_metrics
                else task
            ),
            "evaluated_metrics": sorted(values.keys()),
            "rows": rows
        })

    # -----------------------------------------------------
    # Handle missing comparable scores
    # -----------------------------------------------------

    if not ranked:
        return {
            "best_algorithm": "No successful model",
            "task": task,
            "reason": (
                "Selected models did not return comparable scores."
            ),
            "score": None,
            "metric": None,
            "recommendation_score": None,
            "ranking": []
        }

    # Stable tie-break: keep original insertion order when
    # recommendation scores are equal.
    ranked.sort(
        key=lambda candidate: candidate["recommendation_score"],
        reverse=True
    )

    winner = ranked[0]

    # -----------------------------------------------------
    # Explain recommendation
    # -----------------------------------------------------

    task_name = TASK_NAMES.get(
        task,
        task.replace("_", " ").title()
    )

    if task == "classification":
        reason = (
            "Highest normalized classification score based on "
            "available accuracy, precision, recall, F1, and "
            "cross-validation metrics."
        )

    elif task == "regression":
        reason = (
            "Highest normalized regression score based on "
            "available R², cross-validation, RMSE, and MAE metrics."
        )

    elif task == "clustering":
        reason = (
            "Highest normalized clustering score based on "
            "available silhouette, Calinski-Harabasz, and "
            "Davies-Bouldin metrics."
        )

    elif task == "anomaly_detection":
        reason = (
            "Highest normalized score among successful anomaly "
            "detection benchmarks. Silhouette is descriptive and "
            "does not establish ground-truth anomaly accuracy."
        )

    elif task == "dimensionality_reduction":
        reason = (
            "Highest trustworthiness score among successful "
            "dimensionality reduction benchmarks."
        )

    elif task == "topic_modeling":
        reason = (
            "Highest topic diversity score among successful topic "
            "modeling benchmarks. Topic coherence should also be "
            "considered when validating the result."
        )

    elif task == "text_similarity":
        reason = (
            "Highest normalized mean nearest-similarity score. "
            "This measures similarity, not retrieval relevance "
            "or overall system quality."
        )

    elif task == "sequence_modeling":
        reason = (
            "Highest normalized sequence-model score based on "
            "available R², RMSE, and MAE results."
        )

    else:
        reason = (
            f"Highest normalized score among successful "
            f"{task_name} benchmark results."
        )

    # Do not expose internal result rows in the ranking summary.
    ranking = [
        {
            key: value
            for key, value in candidate.items()
            if key != "rows"
        }
        for candidate in ranked
    ]

    return {
        "best_algorithm": winner["algorithm"],
        "score": winner["score"],
        "metric": winner["metric"],
        "recommendation_score": winner["recommendation_score"],
        "task": task,
        "reason": reason,
        "ranking": ranking
    }
