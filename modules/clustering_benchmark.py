import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# Prevent excessive memory use during dense conversion.
MAX_DENSE_CELLS = 5_000_000

# Agglomerative clustering can become expensive on large datasets.
MAX_AGGLOMERATIVE_ROWS = 10_000

# Limit the number of samples used for silhouette scoring.
MAX_SILHOUETTE_SAMPLES = 10_000


# =========================================================
# CLUSTERING ALGORITHMS
# =========================================================

def _get_clustering_algorithms():
    """
    Return fresh clustering model instances.

    Hierarchical Clustering is supported as an alias for
    Agglomerative clustering. It is not benchmarked twice
    when all algorithms are selected.
    """

    return {
        "K-Means": KMeans(
            n_clusters=3,
            random_state=RANDOM_STATE,
            n_init=10
        ),

        "Agglomerative": AgglomerativeClustering(
            n_clusters=3
        ),

        "DBSCAN": DBSCAN(
            eps=0.5,
            min_samples=5
        )
    }


# =========================================================
# PREPROCESSING
# =========================================================

def _build_preprocessor(X):
    """
    Build a preprocessing pipeline for numeric and
    categorical clustering features.
    """

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError(
            "No supported numeric or categorical features "
            "are available for clustering."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# METRIC CALCULATION
# =========================================================

def _calculate_metrics(X_processed, labels):
    """
    Calculate internal clustering metrics safely.

    DBSCAN noise points (-1) are excluded from the
    cluster-quality metrics.

    Silhouette scoring uses a reproducible sample when
    the dataset exceeds MAX_SILHOUETTE_SAMPLES.
    """

    labels = np.asarray(labels)

    if len(labels) != X_processed.shape[0]:
        raise ValueError(
            "The number of cluster labels does not match "
            "the number of dataset rows."
        )

    noise_mask = labels == -1

    X_evaluated = X_processed[~noise_mask]
    labels_evaluated = labels[~noise_mask]

    cluster_count = len(np.unique(labels_evaluated))
    noise_count = int(np.sum(noise_mask))
    evaluated_count = len(labels_evaluated)

    silhouette = None
    calinski = None
    davies = None

    if (
        cluster_count >= 2
        and evaluated_count > cluster_count
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if evaluated_count > MAX_SILHOUETTE_SAMPLES:
                    silhouette = silhouette_score(
                        X_evaluated,
                        labels_evaluated,
                        sample_size=MAX_SILHOUETTE_SAMPLES,
                        random_state=RANDOM_STATE
                    )
                else:
                    silhouette = silhouette_score(
                        X_evaluated,
                        labels_evaluated
                    )

            silhouette = round(float(silhouette), 4)

        except Exception:
            silhouette = None

        try:
            calinski = round(
                float(
                    calinski_harabasz_score(
                        X_evaluated,
                        labels_evaluated
                    )
                ),
                4
            )
        except Exception:
            calinski = None

        try:
            davies = round(
                float(
                    davies_bouldin_score(
                        X_evaluated,
                        labels_evaluated
                    )
                ),
                4
            )
        except Exception:
            davies = None

    return {
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "clusters": int(cluster_count),
        "noise_points": noise_count
    }


# =========================================================
# FEATURE VALIDATION
# =========================================================

def _prepare_features(df, selected_features):
    """
    Validate and clean selected features before preprocessing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Clustering requires a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if len(df) < 3:
        raise ValueError(
            "At least 3 rows are required for clustering."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Rename duplicate columns before clustering."
        )

    if selected_features is None:
        selected_features = df.columns.tolist()
    else:
        selected_features = list(selected_features)

    if not selected_features:
        raise ValueError(
            "Select at least one feature for clustering."
        )

    if len(selected_features) != len(set(selected_features)):
        raise ValueError(
            "Selected features contain duplicate column names."
        )

    missing_features = [
        column
        for column in selected_features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Selected feature(s) not found: "
            + ", ".join(map(str, missing_features))
        )

    X = df.loc[:, selected_features].copy()

    # Remove columns that contain only missing values.
    X = X.dropna(axis=1, how="all")

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No usable features remain after removing "
            "all-missing columns."
        )

    unsupported_columns = []

    for column in X.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(X[column])
            or pd.api.types.is_timedelta64_dtype(X[column])
        ):
            unsupported_columns.append(str(column))

    if unsupported_columns:
        raise ValueError(
            "Convert date/time columns into suitable numeric "
            "features before clustering: "
            + ", ".join(unsupported_columns)
        )

    # Convert infinite numeric values into missing values
    # so the numeric imputer can handle them.
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_columns):
        X[numeric_columns] = X[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Normalize categorical representations to avoid
    # mixed-type category errors during one-hot encoding.
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns

    for column in categorical_columns:
        X[column] = X[column].map(
            lambda value: (
                np.nan if pd.isna(value) else str(value)
            )
        )

    # Remove columns that are constant after cleaning.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        X = X.drop(columns=constant_columns)

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No informative features remain after removing "
            "constant columns."
        )

    return X


# =========================================================
# MAIN CLUSTERING BENCHMARK
# =========================================================

def run_clustering(
    df,
    selected_features=None,
    selected_algorithm=None
):
    """
    Benchmark clustering algorithms for SimuRithm.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to analyze.

    selected_features : list, optional
        Columns to use for clustering.

    selected_algorithm : str, optional
        Supported values:
        - K-Means
        - Agglomerative
        - Hierarchical Clustering
        - DBSCAN

        If omitted, all unique supported algorithms run.

    Returns
    -------
    list[dict]
        Benchmark results sorted by silhouette score,
        with valid scores ranked ahead of invalid scores.
    """

    # -----------------------------------------------------
    # Validate and prepare features
    # -----------------------------------------------------

    X = _prepare_features(df, selected_features)

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    try:
        preprocessor = _build_preprocessor(X)

        X_processed = preprocessor.fit_transform(X)

    except Exception as exc:
        raise ValueError(
            "Failed to preprocess clustering features: "
            + str(exc)
        ) from exc

    # Check memory before converting sparse matrices to dense.
    if hasattr(X_processed, "toarray"):
        rows, columns = X_processed.shape

        if rows * columns > MAX_DENSE_CELLS:
            raise ValueError(
                "The encoded dataset is too large for safe "
                "clustering with the current configuration. "
                "Reduce the number of rows or categorical "
                "features and try again."
            )

        X_processed = X_processed.toarray()

    X_processed = np.asarray(
        X_processed,
        dtype=np.float64
    )

    if (
        X_processed.ndim != 2
        or X_processed.shape[0] < 3
        or X_processed.shape[1] == 0
    ):
        raise ValueError(
            "Preprocessing did not produce enough usable "
            "rows and features for clustering."
        )

    if not np.isfinite(X_processed).all():
        raise ValueError(
            "Preprocessing produced non-finite values."
        )

    # -----------------------------------------------------
    # Configure algorithms and aliases
    # -----------------------------------------------------

    algorithms = _get_clustering_algorithms()

    aliases = {
        "Hierarchical Clustering": "Agglomerative"
    }

    if selected_algorithm:
        canonical_name = aliases.get(
            selected_algorithm,
            selected_algorithm
        )

        if canonical_name not in algorithms:
            supported = list(algorithms.keys()) + list(aliases.keys())

            raise ValueError(
                "Unsupported clustering algorithm: "
                + str(selected_algorithm)
                + ". Supported algorithms: "
                + ", ".join(supported)
            )

        algorithms = {
            canonical_name: algorithms[canonical_name]
        }

    # -----------------------------------------------------
    # Run benchmarks
    # -----------------------------------------------------

    results = []

    for name, model in algorithms.items():
        start_time = time.perf_counter()
        result_name = (
            selected_algorithm
            if selected_algorithm == "Hierarchical Clustering"
            else name
        )

        try:
            if (
                isinstance(model, AgglomerativeClustering)
                and len(X_processed) > MAX_AGGLOMERATIVE_ROWS
            ):
                raise ValueError(
                    "Agglomerative clustering is limited to "
                    f"{MAX_AGGLOMERATIVE_ROWS} rows for this "
                    "configuration. Reduce the dataset size "
                    "or use another clustering algorithm."
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                labels = model.fit_predict(X_processed)

            metrics = _calculate_metrics(
                X_processed,
                labels
            )

            results.append({
                "algorithm": result_name,
                "selected_features": X.columns.tolist(),
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
                "execution_time": round(
                    time.perf_counter() - start_time,
                    4
                ),
                "clusters": metrics["clusters"],
                "noise_points": metrics["noise_points"]
            })

        except Exception as exc:
            results.append({
                "algorithm": result_name,
                "selected_features": X.columns.tolist(),
                "silhouette": None,
                "calinski_harabasz": None,
                "davies_bouldin": None,
                "execution_time": round(
                    time.perf_counter() - start_time,
                    4
                ),
                "clusters": None,
                "noise_points": None,
                "error": str(exc)
            })

    # -----------------------------------------------------
    # Sort results
    # -----------------------------------------------------

    results.sort(
        key=lambda result: (
            result["silhouette"] is not None,
            (
                result["silhouette"]
                if result["silhouette"] is not None
                else float("-inf")
            )
        ),
        reverse=True
    )

    return results
