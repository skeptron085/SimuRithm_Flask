def recommend(results, task):
    valid_results = [
        item for item in results
        if isinstance(item, dict)
        and not item.get("error")
        and item.get("algorithm")
    ]
    if not valid_results:
        return {
            "best_algorithm": "No successful model",
            "task": task,
            "reason": "No selected model returned usable benchmark metrics."
        }

    metric_weights = {
        "classification": [
            (("accuracy",), True, 0.25),
            (("precision", "precision_score"), True, 0.15),
            (("recall", "recall_score"), True, 0.15),
            (("f1", "f1_score"), True, 0.25),
            (("cv_score", "cross_validation_score", "cross_val_score"), True, 0.20)
        ],
        "regression": [
            (("r2",), True, 0.35),
            (("cv_score", "cross_validation_score", "cross_val_score"), True, 0.30),
            (("rmse",), False, 0.20),
            (("mae",), False, 0.15)
        ],
        "clustering": [
            (("silhouette",), True, 0.55),
            (("calinski_harabasz",), True, 0.20),
            (("davies_bouldin",), False, 0.25)
        ]
    }
    metrics = metric_weights.get(task, [(("primary_score",), True, 1.0)])
    metric_values = {}
    for aliases, _, _ in metrics:
        values = []
        for item in valid_results:
            value = next((item.get(key) for key in aliases if item.get(key) is not None), None)
            if value is not None:
                try:
                    numeric_value = float(value)
                    if numeric_value == numeric_value and abs(numeric_value) != float("inf"):
                        values.append(numeric_value)
                except (TypeError, ValueError):
                    continue
        metric_values[aliases[0]] = values

    bounds = {
        key: (min(values), max(values))
        for key, values in metric_values.items()
        if values
    }
    normalized = {}
    for item in valid_results:
        algorithm = item["algorithm"]
        candidate_metrics = normalized.setdefault(algorithm, {"weighted": 0.0, "weight": 0.0})
        for aliases, higher_is_better, weight in metrics:
            key = aliases[0]
            value = next((item.get(alias) for alias in aliases if item.get(alias) is not None), None)
            if value is None or key not in bounds:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            low, high = bounds[key]
            scaled = 0.5 if high == low else (value - low) / (high - low)
            if not higher_is_better:
                scaled = 1.0 - scaled
            candidate_metrics["weighted"] += scaled * weight
            candidate_metrics["weight"] += weight

    ranked = []
    for algorithm, scores in normalized.items():
        if scores["weight"] == 0:
            continue
        rows = [item for item in valid_results if item["algorithm"] == algorithm]
        primary_scores = [
            float(item["primary_score"])
            for item in rows
            if item.get("primary_score") is not None
        ]
        metric_names = [item.get("primary_metric") for item in rows if item.get("primary_metric")]
        ranked.append({
            "algorithm": algorithm,
            "recommendation_score": round(scores["weighted"] / scores["weight"], 4),
            "score": round(sum(primary_scores) / len(primary_scores), 4) if primary_scores else None,
            "metric": metric_names[0] if metric_names else task,
            "rows": rows
        })

    if not ranked:
        return {
            "best_algorithm": "No successful model",
            "task": task,
            "reason": "Selected models did not return comparable scores."
        }

    ranked.sort(key=lambda item: item["recommendation_score"], reverse=True)
    winner = ranked[0]
    if task == "classification":
        reason = "Highest normalized score across accuracy, precision, recall, F1, and available cross-validation results."
    elif task == "regression":
        reason = "Highest normalized score across R², available cross-validation, RMSE, and MAE."
    elif task == "clustering":
        reason = "Highest normalized clustering-quality score across silhouette, Calinski-Harabasz, and Davies-Bouldin metrics."
    else:
        reason = f"Highest average {winner['metric']} among user-selected algorithms."

    return {
        "best_algorithm": winner["algorithm"],
        "score": winner["score"],
        "metric": winner["metric"],
        "recommendation_score": winner["recommendation_score"],
        "task": task,
        "reason": reason,
        "ranking": [
            {key: value for key, value in candidate.items() if key != "rows"}
            for candidate in ranked
        ]
    }