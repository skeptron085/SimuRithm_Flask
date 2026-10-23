
"""
SimuRithm - Hyperparameter Tuning Engine
File: tuning.py

Supports:
    - Classification
    - Regression
    - Clustering

Public functions:
    - tune_random_forest(df, target)
    - tune_recommended_model(df, target, category, algorithm)

The module returns JSON-serializable dictionaries suitable for
integration with the SimuRithm Flask API.
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)

from sklearn.metrics import silhouette_score


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

MAX_DENSE_CELLS = 5_000_000
MAX_CLUSTER_ROWS = 10_000
MAX_SILHOUETTE_SAMPLES = 5_000

MAX_GRID_SEARCH_ROWS = 10_000_000

SUPPORTED_SUPERVISED_TASKS = {
    "classification",
    "regression",
}

TASK_NAMES = {
    "classification": "Classification",
    "regression": "Regression",
    "clustering": "Clustering",
}


# =========================================================
# OPTIONAL TASK DETECTOR
# =========================================================

try:
    from task_detector import detect_task
except ImportError:
    detect_task = None


# =========================================================
# GENERAL UTILITIES
# =========================================================

def _error_result(
    algorithm,
    error,
    target=None,
    category=None,
    cv_folds=0,
):
    """Create a consistent failure response."""

    result = {
        "success": False,
        "algorithm": algorithm,
        "error": str(error),
    }

    if target is not None:
        result["target"] = str(target)

    if category is not None:
        result["category"] = str(category)

    result["cv_folds"] = int(cv_folds)

    return result


def _safe_float(value):
    """Convert a numeric value to a finite Python float."""

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError, OverflowError):
        pass

    return None


def _normalise_category(category):
    """Normalize task labels received from other modules."""

    if category is None:
        return ""

    value = str(category).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    aliases = {
        "classification": "classification",
        "classify": "classification",
        "regression": "regression",
        "clustering": "clustering",
        "cluster": "clustering",
    }

    return aliases.get(value, value)


def _normalise_algorithm(algorithm):
    """Normalize common algorithm naming variations."""

    if algorithm is None:
        return ""

    value = " ".join(str(algorithm).strip().lower().split())

    aliases = {
        "logistic regression": "Logistic Regression",
        "decision tree": "Decision Tree",
        "random forest": "Random Forest",
        "svm": "SVM",
        "support vector machine": "SVM",
        "knn": "KNN",
        "k-nearest neighbors": "KNN",
        "naive bayes": "Naive Bayes",

        "linear regression": "Linear Regression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "decision tree regressor": "Decision Tree Regressor",
        "random forest regressor": "Random Forest Regressor",
        "svr": "SVR",

        "k-means": "K-Means",
        "kmeans": "K-Means",
        "hierarchical": "Hierarchical Clustering",
        "hierarchical clustering": "Hierarchical Clustering",
        "agglomerative": "Hierarchical Clustering",
        "agglomerative clustering": "Hierarchical Clustering",
        "dbscan": "DBSCAN",
    }

    return aliases.get(value, str(algorithm).strip())


def _prepare_frame(frame):
    """
    Return a safe copy of a dataset.

    Infinite numeric values become missing values. Empty and
    constant columns are removed to reduce preprocessing issues.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("The input dataset must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("The dataset contains no rows.")

    if len(frame.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    result = frame.copy()

    # Prevent duplicate column names from causing ambiguous selection.
    if result.columns.duplicated().any():
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename the duplicate columns before tuning."
        )

    # Replace positive/negative infinity in numeric columns.
    numeric_columns = result.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numeric_columns:
        result[column] = result[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Convert boolean features to numeric values.
    boolean_columns = result.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in boolean_columns:
        result[column] = result[column].astype("float64")

    # Remove columns that contain no usable values.
    all_missing = [
        column
        for column in result.columns
        if result[column].isna().all()
    ]

    if all_missing:
        result = result.drop(columns=all_missing)

    # Remove constant columns.
    constant_columns = [
        column
        for column in result.columns
        if result[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        result = result.drop(columns=constant_columns)

    return result


def _dense_features(values):
    """
    Convert sparse features to dense only when the matrix is small
    enough to reduce the risk of excessive memory usage.
    """

    if hasattr(values, "toarray"):
        rows, columns = values.shape

        if rows * columns > MAX_DENSE_CELLS:
            raise ValueError(
                "The encoded feature matrix is too large to convert "
                "to dense format. Reduce the number of rows or "
                "categorical feature values."
            )

        return values.toarray()

    return np.asarray(values)


def _make_one_hot_encoder():
    """Build an encoder compatible with different sklearn versions."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def _build_preprocessor(frame):
    """Create numeric and categorical preprocessing pipelines."""

    numeric_features = frame.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = frame.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    _make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical features remain "
            "after dataset preprocessing."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _clean_supervised_data(df, target):
    """Validate target and prepare X/y for supervised tuning."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input dataset must be a pandas DataFrame.")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' was not found in the dataset."
        )

    if len(df.columns) < 2:
        raise ValueError(
            "At least one feature column and one target column "
            "are required."
        )

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Remove rows with missing target values.
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Infinite target values are invalid for model training.
    if pd.api.types.is_numeric_dtype(y):
        finite_mask = np.isfinite(
            pd.to_numeric(y, errors="coerce")
        )

        X = X.loc[finite_mask].copy()
        y = y.loc[finite_mask].copy()

    if len(y) < 3:
        raise ValueError(
            "At least three labeled rows are required for tuning."
        )

    X = _prepare_frame(X)

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after preprocessing."
        )

    # Align rows after preprocessing and column removal.
    X = X.loc[y.index]

    if X.isna().all(axis=1).all():
        raise ValueError(
            "No usable feature values remain in the dataset."
        )

    return X, y


# =========================================================
# CLASSIFICATION CROSS-VALIDATION
# =========================================================

def _get_classification_cv(y):
    """
    Configure stratified cross-validation safely.

    Class counts are checked before label encoding.
    """

    if len(y) < 2:
        return (
            None,
            0,
            "At least two labeled rows are required for classification.",
        )

    counts = pd.Series(y).value_counts(dropna=True)

    if len(counts) < 2:
        return (
            None,
            0,
            "Classification requires at least two distinct classes.",
        )

    minimum_class_count = int(counts.min())

    if minimum_class_count < 2:
        return (
            None,
            0,
            (
                "Stratified cross-validation is unavailable because "
                "at least one class contains only one sample. "
                "Add more examples for the rare class or select a "
                "different target column."
            ),
        )

    folds = min(5, minimum_class_count)

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cv, folds, None


# =========================================================
# REGRESSION CROSS-VALIDATION
# =========================================================

def _get_regression_cv(y):
    """Configure KFold and scoring for regression tuning."""

    sample_count = len(y)

    if sample_count < 3:
        return (
            None,
            0,
            None,
            "At least three labeled rows are required for regression tuning.",
        )

    folds = min(5, sample_count)

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # R² is undefined when a validation fold contains one sample.
    # MAE remains defined for small validation folds.
    scoring = (
        "r2"
        if sample_count >= 5
        else "neg_mean_absolute_error"
    )

    return cv, folds, scoring, None


# =========================================================
# CLUSTERING OPTIMIZATION
# =========================================================

def _tune_clustering_model(frame, algorithm):
    """Find the best tested clustering parameters using silhouette."""

    algorithm = _normalise_algorithm(algorithm)

    try:
        X = _prepare_frame(frame)

        if len(X) < 4:
            return _error_result(
                algorithm,
                "At least four rows are required for clustering optimization.",
                category="clustering",
            )

        # Clustering can be expensive. Use a reproducible sample when
        # the input dataset is larger than the configured limit.
        if len(X) > MAX_CLUSTER_ROWS:
            X = X.sample(
                n=MAX_CLUSTER_ROWS,
                random_state=RANDOM_STATE,
            )

        preprocessor = _build_preprocessor(X)
        matrix = preprocessor.fit_transform(X)

        if hasattr(matrix, "toarray"):
            if matrix.shape[0] * matrix.shape[1] > MAX_DENSE_CELLS:
                return _error_result(
                    algorithm,
                    (
                        "The encoded clustering matrix is too large "
                        "for safe optimization. Reduce the number of "
                        "features or categorical values."
                    ),
                    category="clustering",
                )

            matrix = matrix.toarray()

        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2 or matrix.shape[1] == 0:
            return _error_result(
                algorithm,
                "No usable features remain for clustering.",
                category="clustering",
            )

        if not np.isfinite(matrix).all():
            return _error_result(
                algorithm,
                "The preprocessed clustering matrix contains invalid values.",
                category="clustering",
            )

        maximum_clusters = min(8, len(matrix) - 1)

        if maximum_clusters < 2:
            return _error_result(
                algorithm,
                "Not enough rows are available to evaluate clusters.",
                category="clustering",
            )

        candidates = []

        # -----------------------------------------------------
        # K-Means
        # -----------------------------------------------------

        if algorithm == "K-Means":
            for cluster_count in range(2, maximum_clusters + 1):
                candidates.append(
                    (
                        KMeans(
                            n_clusters=cluster_count,
                            n_init=10,
                            random_state=RANDOM_STATE,
                        ),
                        {
                            "n_clusters": cluster_count,
                            "n_init": 10,
                        },
                    )
                )

        # -----------------------------------------------------
        # Hierarchical / Agglomerative Clustering
        # -----------------------------------------------------

        elif algorithm == "Hierarchical Clustering":
            for cluster_count in range(2, maximum_clusters + 1):
                for linkage in ("ward", "complete", "average"):
                    candidates.append(
                        (
                            AgglomerativeClustering(
                                n_clusters=cluster_count,
                                linkage=linkage,
                            ),
                            {
                                "n_clusters": cluster_count,
                                "linkage": linkage,
                            },
                        )
                    )

        # -----------------------------------------------------
        # DBSCAN
        # -----------------------------------------------------

        elif algorithm == "DBSCAN":
            for epsilon in (0.3, 0.5, 0.7, 1.0):
                for minimum_samples in (3, 5, 10):
                    if minimum_samples < len(matrix):
                        candidates.append(
                            (
                                DBSCAN(
                                    eps=epsilon,
                                    min_samples=minimum_samples,
                                ),
                                {
                                    "eps": epsilon,
                                    "min_samples": minimum_samples,
                                },
                            )
                        )

        else:
            return _error_result(
                algorithm,
                f"No clustering optimizer is configured for '{algorithm}'.",
                category="clustering",
            )

        best = None

        for model, parameters in candidates:
            try:
                labels = model.fit_predict(matrix)

                # DBSCAN label -1 represents noise, not a cluster.
                valid_rows = (
                    labels != -1
                    if algorithm == "DBSCAN"
                    else np.ones(len(labels), dtype=bool)
                )

                valid_labels = labels[valid_rows]
                valid_matrix = matrix[valid_rows]

                unique_labels = np.unique(valid_labels)

                if len(unique_labels) < 2:
                    continue

                if len(unique_labels) >= len(valid_labels):
                    continue

                # Avoid excessive silhouette computation on large samples.
                if len(valid_labels) > MAX_SILHOUETTE_SAMPLES:
                    rng = np.random.default_rng(RANDOM_STATE)
                    sample_indices = rng.choice(
                        len(valid_labels),
                        size=MAX_SILHOUETTE_SAMPLES,
                        replace=False,
                    )

                    score_matrix = valid_matrix[sample_indices]
                    score_labels = valid_labels[sample_indices]

                    # A random sample can omit a cluster or contain
                    # only one point per cluster. Skip such samples.
                    sampled_unique = np.unique(score_labels)

                    if (
                        len(sampled_unique) < 2
                        or len(sampled_unique) >= len(score_labels)
                    ):
                        continue

                else:
                    score_matrix = valid_matrix
                    score_labels = valid_labels

                score = silhouette_score(
                    score_matrix,
                    score_labels,
                )

                if not np.isfinite(score):
                    continue

                if best is None or score > best["best_score"]:
                    best = {
                        "best_score": float(score),
                        "best_parameters": parameters,
                        "clusters": int(len(unique_labels)),
                        "noise_points": int(np.sum(labels == -1)),
                    }

            except Exception:
                # A failed candidate should not prevent other
                # parameter combinations from being evaluated.
                continue

        if best is None:
            return _error_result(
                algorithm,
                (
                    "No tested parameter set produced a valid silhouette "
                    "score. Try another clustering algorithm or review "
                    "the dataset's feature values."
                ),
                category="clustering",
            )

        return {
            "success": True,
            "algorithm": algorithm,
            "category": "clustering",
            "scoring": "silhouette",
            "best_score": round(best["best_score"], 4),
            "best_parameters": best["best_parameters"],
            "clusters": best["clusters"],
            "noise_points": best["noise_points"],
            "rows_evaluated": int(len(X)),
        }

    except Exception as error:
        return _error_result(
            algorithm,
            error,
            category="clustering",
        )


# =========================================================
# SUPERVISED ALGORITHM CONFIGURATION
# =========================================================

def _classification_models():
    """Return classification estimators and parameter grids."""

    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Decision Tree": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2],
        },
        "SVM": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
        },
        "KNN": {
            # The neighbor grid is adjusted based on training size.
            "model__n_neighbors": [1, 3, 5],
            "model__weights": ["uniform", "distance"],
        },
        "Naive Bayes": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7],
        },
    }

    return estimators, grids


def _regression_models():
    """Return regression estimators and parameter grids."""

    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "SVR": SVR(),
    }

    grids = {
        "Linear Regression": {
            "model__fit_intercept": [True, False],
        },
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
        },
        "Decision Tree Regressor": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "Random Forest Regressor": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2],
        },
        "SVR": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
        },
    }

    return estimators, grids


# =========================================================
# RANDOM FOREST TUNING - LEGACY COMPATIBILITY
# =========================================================

def tune_random_forest(df, target):
    """
    Tune a Random Forest model.

    Task detection uses task_detector.detect_task when available.
    Otherwise, a basic target-type fallback is used.
    """

    try:
        if target not in df.columns:
            return _error_result(
                "Random Forest",
                f"Target column '{target}' was not found.",
                target=target,
            )

        task = None

        if detect_task is not None:
            try:
                detected = detect_task(df, target)

                if isinstance(detected, str):
                    task = _normalise_category(detected)

                elif isinstance(detected, dict):
                    task = _normalise_category(
                        detected.get("task")
                        or detected.get("category")
                        or detected.get("type")
                    )

            except Exception:
                task = None

        if task not in SUPPORTED_SUPERVISED_TASKS:
            y = df[target].dropna()

            if (
                pd.api.types.is_object_dtype(y)
                or isinstance(y.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(y)
            ):
                task = "classification"
            else:
                task = "regression"

        algorithm = (
            "Random Forest"
            if task == "classification"
            else "Random Forest Regressor"
        )

        return tune_recommended_model(
            df=df,
            target=target,
            category=task,
            algorithm=algorithm,
        )

    except Exception as error:
        return _error_result(
            "Random Forest",
            error,
            target=target,
        )


# =========================================================
# RECOMMENDED MODEL TUNING
# =========================================================

def tune_recommended_model(
    df,
    target,
    category,
    algorithm,
):
    """
    Tune the algorithm recommended by SimuRithm.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    target : str
        Target column for classification or regression.
        Clustering does not require a target.

    category : str
        classification, regression, or clustering.

    algorithm : str
        Algorithm name returned by the recommender/benchmark module.
    """

    started_at = time.perf_counter()

    category = _normalise_category(category)
    algorithm = _normalise_algorithm(algorithm)

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        return _error_result(
            algorithm,
            "The input dataset must be a pandas DataFrame.",
            target=target,
            category=category,
        )

    if df.empty:
        return _error_result(
            algorithm,
            "The dataset contains no rows.",
            target=target,
            category=category,
        )

    # -----------------------------------------------------
    # CLUSTERING
    # -----------------------------------------------------

    if category == "clustering":
        result = _tune_clustering_model(
            df,
            algorithm,
        )

        result["category"] = "clustering"
        result["target"] = None
        result["execution_time"] = round(
            time.perf_counter() - started_at,
            4,
        )

        return result

    # -----------------------------------------------------
    # Validate supported supervised task
    # -----------------------------------------------------

    if category not in SUPPORTED_SUPERVISED_TASKS:
        return _error_result(
            algorithm,
            (
                "Hyperparameter tuning is currently supported for "
                "classification, regression, and clustering. "
                f"The received category was '{category}'."
            ),
            target=target,
            category=category,
        )

    # -----------------------------------------------------
    # Prepare supervised data
    # -----------------------------------------------------

    try:
        X, y = _clean_supervised_data(
            df,
            target,
        )

    except Exception as error:
        return _error_result(
            algorithm,
            error,
            target=target,
            category=category,
        )

    # -----------------------------------------------------
    # Configure models and cross-validation
    # -----------------------------------------------------

    if category == "classification":
        cv, cv_folds, cv_error = _get_classification_cv(y)

        if cv is None:
            return _error_result(
                algorithm,
                cv_error,
                target=target,
                category=category,
                cv_folds=cv_folds,
            )

        # Encode the target only after validating class counts.
        label_encoder = LabelEncoder()

        try:
            y = label_encoder.fit_transform(y)
        except Exception as error:
            return _error_result(
                algorithm,
                f"Unable to encode the classification target: {error}",
                target=target,
                category=category,
                cv_folds=cv_folds,
            )

        scoring = "accuracy"

        estimators, grids = _classification_models()

        # Find the smallest training fold size to avoid asking KNN
        # for more neighbors than a training fold contains.
        smallest_training_fold = (
            len(y) - int(np.ceil(len(y) / cv_folds))
        )

        if smallest_training_fold < 1:
            return _error_result(
                algorithm,
                "Not enough training rows are available for cross-validation.",
                target=target,
                category=category,
                cv_folds=cv_folds,
            )

        valid_neighbors = [
            neighbors
            for neighbors in (1, 3, 5, 7)
            if neighbors <= smallest_training_fold
        ]

        if "KNN" in grids:
            grids["KNN"]["model__n_neighbors"] = (
                valid_neighbors or [1]
            )

    else:
        cv, cv_folds, scoring, cv_error = _get_regression_cv(y)

        if cv is None:
            return _error_result(
                algorithm,
                cv_error,
                target=target,
                category=category,
                cv_folds=cv_folds,
            )

        estimators, grids = _regression_models()

    # -----------------------------------------------------
    # Validate selected algorithm
    # -----------------------------------------------------

    if algorithm not in estimators:
        return _error_result(
            algorithm,
            (
                f"No tuning search is configured for '{algorithm}' "
                f"in the {category} category."
            ),
            target=target,
            category=category,
            cv_folds=cv_folds,
        )

    # -----------------------------------------------------
    # Limit extremely large grid searches
    # -----------------------------------------------------

    if len(X) > MAX_GRID_SEARCH_ROWS:
        return _error_result(
            algorithm,
            (
                f"The dataset contains {len(X):,} labeled rows, which "
                f"exceeds the tuning limit of {MAX_GRID_SEARCH_ROWS:,}. "
                "Use a smaller dataset or sample rows before tuning."
            ),
            target=target,
            category=category,
            cv_folds=cv_folds,
        )

    # -----------------------------------------------------
    # Build preprocessing and model pipeline
    # -----------------------------------------------------

    try:
        preprocessor = _build_preprocessor(X)

        steps = [
            (
                "preprocessor",
                preprocessor,
            ),
        ]

        if algorithm == "Naive Bayes":
            steps.append(
                (
                    "dense",
                    FunctionTransformer(
                        _dense_features,
                        accept_sparse=True,
                        validate=False,
                    ),
                )
            )

        steps.append(
            (
                "model",
                estimators[algorithm],
            )
        )

        pipeline = Pipeline(steps=steps)

    except Exception as error:
        return _error_result(
            algorithm,
            f"Unable to construct the tuning pipeline: {error}",
            target=target,
            category=category,
            cv_folds=cv_folds,
        )

    # -----------------------------------------------------
    # Grid search
    # -----------------------------------------------------

    try:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grids[algorithm],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score="raise",
            refit=True,
            return_train_score=False,
        )

        # Suppress expected warnings from individual candidate models.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        best_score = _safe_float(search.best_score_)

        if best_score is None:
            return _error_result(
                algorithm,
                "Grid search did not produce a finite validation score.",
                target=target,
                category=category,
                cv_folds=cv_folds,
            )

        best_parameters = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }

        result = {
            "success": True,
            "algorithm": algorithm,
            "target": str(target),
            "category": category,
            "scoring": scoring,
            "cv_folds": int(cv_folds),
            "best_score": round(best_score, 4),
            "best_parameters": best_parameters,
            "rows_used": int(len(X)),
            "features_used": int(X.shape[1]),
            "execution_time": round(
                time.perf_counter() - started_at,
                4,
            ),
        }

        if category == "classification":
            result["classes"] = [
                str(value)
                for value in label_encoder.classes_
            ]

        if category == "regression" and scoring == "neg_mean_absolute_error":
            # sklearn negates loss values so that larger scores are better.
            # Provide a positive MAE alongside the original CV score.
            result["best_mae"] = round(-best_score, 4)

        return result

    except Exception as error:
        return _error_result(
            algorithm,
            error,
            target=target,
            category=category,
            cv_folds=cv_folds,
        )
