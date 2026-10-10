
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
    PCA
)
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

MAX_ENCODED_CELLS = 5_000_000
MAX_TEXT_DOCUMENTS = 1_000
MAX_TOPIC_DOCUMENTS = 2_000
MAX_TEXT_FEATURES = 3_000
MAX_SEQUENCE_ROWS = 20_000

CATEGORY_ALGORITHMS = {
    "classification": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN",
        "Naive Bayes"
    ],
    "regression": [
        "Linear Regression",
        "Ridge",
        "Lasso",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "SVR"
    ],
    "clustering": [
        "K-Means",
        "DBSCAN",
        "Hierarchical Clustering"
    ],
    "anomaly_detection": [
        "Isolation Forest",
        "One-Class SVM",
        "Local Outlier Factor (LOF)"
    ],
    "dimensionality_reduction": [
        "PCA",
        "t-SNE",
        "UMAP"
    ],
    "topic_modeling": [
        "LDA",
        "NMF"
    ],
    "text_similarity": [
        "TF-IDF + Cosine Similarity",
        "Jaccard Similarity"
    ],
    "sequence_modeling": [
        "LSTM",
        "GRU",
        "Transformer"
    ]
}


# =========================================================
# COMMON RESULT FORMAT
# =========================================================

def _metric_result(
    algorithm,
    metric,
    score,
    elapsed,
    higher_is_better=True,
    **details
):
    """
    Return a consistent benchmark result.

    primary_score is None if the metric could not be
    calculated or returned a non-finite value.
    """

    if score is not None:
        try:
            score = float(score)
            if not np.isfinite(score):
                score = None
            else:
                score = round(score, 4)
        except (TypeError, ValueError):
            score = None

    return {
        "algorithm": algorithm,
        "primary_metric": metric,
        "primary_score": score,
        "higher_is_better": bool(higher_is_better),
        "execution_time": round(float(elapsed), 4),
        **details
    }


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("The dataset contains no rows.")

    if not frame.columns.is_unique:
        raise ValueError(
            "Duplicate column names must be resolved before analysis."
        )


def _validate_algorithms(algorithms, supported, task_name):
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    if algorithms is None:
        raise ValueError(
            f"Select at least one {task_name} algorithm."
        )

    algorithms = list(algorithms)

    if not algorithms:
        raise ValueError(
            f"Select at least one {task_name} algorithm."
        )

    if len(algorithms) != len(set(algorithms)):
        raise ValueError(
            f"Duplicate {task_name} algorithms were selected."
        )

    unsupported = [
        name for name in algorithms
        if name not in supported
    ]

    if unsupported:
        raise ValueError(
            f"Unsupported {task_name} algorithm(s): "
            + ", ".join(unsupported)
            + ". Supported algorithms: "
            + ", ".join(supported)
        )

    return algorithms


# =========================================================
# NUMERIC FEATURE PREPROCESSING
# =========================================================

def _numeric_features(frame):
    """
    Impute, scale and encode dataset features.

    Returns a dense float32 matrix after enforcing a
    maximum encoded matrix size.
    """

    _validate_frame(frame)

    X = frame.copy()

    # Replace invalid numeric values before imputation.
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Convert Boolean columns to numeric.
    boolean_columns = X.select_dtypes(
        include=["bool"]
    ).columns

    for column in boolean_columns:
        X[column] = X[column].astype("float64")

    # Normalize nonnumeric values for safe categorical encoding.
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns

    for column in categorical_columns:
        X[column] = X[column].map(
            lambda value: (
                np.nan if pd.isna(value) else str(value)
            )
        )

    # Remove columns without useful information.
    X = X.dropna(axis=1, how="all")

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        X = X.drop(columns=constant_columns)

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No informative features are available."
        )

    numeric = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical = [
        column for column in X.columns
        if column not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical
        ))

    if not transformers:
        raise ValueError(
            "No numeric or categorical features are available."
        )

    processor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    matrix = processor.fit_transform(X)

    rows, columns = matrix.shape

    if rows * columns > MAX_ENCODED_CELLS:
        raise ValueError(
            "The encoded feature matrix exceeds the configured "
            "memory limit. Reduce the number of rows or features."
        )

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    matrix = np.asarray(matrix, dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature preprocessing produced invalid numeric values."
        )

    return matrix


# =========================================================
# ANOMALY DETECTION
# =========================================================

def run_anomaly_benchmarks(frame, algorithms):
    """
    Benchmark anomaly detection algorithms.

    Outlier labels:
        -1 = anomaly
         1 = normal

    Silhouette is reported as a secondary descriptive metric,
    not as ground-truth anomaly-detection accuracy.
    """

    algorithms = _validate_algorithms(
        algorithms,
        CATEGORY_ALGORITHMS["anomaly_detection"],
        "anomaly detection"
    )

    matrix = _numeric_features(frame)

    if len(matrix) < 4:
        raise ValueError(
            "Anomaly detection requires at least four usable rows."
        )

    models = {
        "Isolation Forest": lambda: IsolationForest(
            contamination="auto",
            random_state=RANDOM_STATE,
            n_estimators=100
        ),
        "One-Class SVM": lambda: OneClassSVM(
            nu=0.1,
            gamma="scale"
        ),
        "Local Outlier Factor (LOF)": lambda: LocalOutlierFactor(
            n_neighbors=min(20, len(matrix) - 1),
            contamination="auto"
        )
    }

    results = []

    for name in algorithms:
        started = time.perf_counter()

        try:
            model = models[name]()
            labels = model.fit_predict(matrix)

            anomaly_count = int(np.sum(labels == -1))
            anomaly_rate = float(np.mean(labels == -1))

            # Subsample large matrices for the descriptive metric.
            if len(matrix) > 10_000:
                rng = np.random.default_rng(RANDOM_STATE)
                indexes = rng.choice(
                    len(matrix), 10_000, replace=False
                )
                metric_matrix = matrix[indexes]
                metric_labels = labels[indexes]
            else:
                metric_matrix = matrix
                metric_labels = labels

            score = None

            if (
                2 <= len(np.unique(metric_labels))
                < len(metric_labels)
            ):
                score = silhouette_score(
                    metric_matrix,
                    metric_labels
                )

            results.append(_metric_result(
                name,
                "Silhouette",
                score,
                time.perf_counter() - started,
                anomaly_rate=round(anomaly_rate, 4),
                outliers=anomaly_count,
                evaluated_rows=len(matrix)
            ))

        except Exception as error:
            results.append(_metric_result(
                name,
                "Silhouette",
                None,
                time.perf_counter() - started,
                error=str(error)
            ))

    return results


# =========================================================
# DIMENSIONALITY REDUCTION
# =========================================================

def run_reduction_benchmarks(frame, algorithms):
    """
    Benchmark PCA, t-SNE and UMAP.

    Trustworthiness is the primary metric; higher is better.
    Large datasets are sampled reproducibly.
    """

    algorithms = _validate_algorithms(
        algorithms,
        CATEGORY_ALGORITHMS["dimensionality_reduction"],
        "dimensionality reduction"
    )

    matrix = _numeric_features(frame)

    if len(matrix) < 4:
        raise ValueError(
            "Dimensionality reduction requires at least four rows."
        )

    if len(matrix) > 1_000:
        rng = np.random.default_rng(RANDOM_STATE)
        indexes = rng.choice(
            len(matrix), 1_000, replace=False
        )
        matrix = matrix[indexes]

    if matrix.shape[1] < 2 and "PCA" in algorithms:
        # PCA can still represent one-dimensional data, but
        # a two-dimensional embedding would be impossible.
        pass

    neighbors = max(
        1,
        min(5, (len(matrix) - 1) // 2)
    )

    results = []

    for name in algorithms:
        started = time.perf_counter()

        try:
            if name == "PCA":
                if matrix.shape[1] < 2:
                    raise ValueError(
                        "Two-dimensional PCA requires at least "
                        "two input features."
                    )

                embedding = PCA(
                    n_components=2,
                    random_state=RANDOM_STATE
                ).fit_transform(matrix)

            elif name == "t-SNE":
                perplexity = min(30, len(matrix) - 1)

                embedding = TSNE(
                    n_components=2,
                    random_state=RANDOM_STATE,
                    init="pca",
                    learning_rate="auto",
                    perplexity=perplexity
                ).fit_transform(matrix)

            else:
                try:
                    import umap
                except ImportError as error:
                    raise RuntimeError(
                        "UMAP requires umap-learn. Install it "
                        "in the active Flask environment."
                    ) from error

                embedding = umap.UMAP(
                    n_components=2,
                    random_state=RANDOM_STATE,
                    n_neighbors=min(15, len(matrix) - 1)
                ).fit_transform(matrix)

            score = trustworthiness(
                matrix,
                embedding,
                n_neighbors=neighbors
            )

            results.append(_metric_result(
                name,
                "Trustworthiness",
                score,
                time.perf_counter() - started,
                embedding_dimensions=2,
                evaluated_rows=len(matrix)
            ))

        except Exception as error:
            results.append(_metric_result(
                name,
                "Trustworthiness",
                None,
                time.perf_counter() - started,
                error=str(error)
            ))

    return results


# =========================================================
# TEXT DOCUMENT EXTRACTION
# =========================================================

def _documents(frame):
    """
    Convert dataframe rows into text documents.

    Missing values are ignored. Empty documents are removed.
    """

    _validate_frame(frame)

    text_frame = frame.copy()

    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(
            lambda value: (
                "" if pd.isna(value) else str(value)
            )
        )

    documents = (
        text_frame
        .agg(" ".join, axis=1)
        .str.strip()
    )

    documents = documents[documents.ne("")]

    if len(documents) < 2:
        raise ValueError(
            "At least two non-empty text documents are required."
        )

    return documents.tolist()


def _sample_documents(documents, limit):
    if len(documents) <= limit:
        return documents

    rng = np.random.default_rng(RANDOM_STATE)

    indexes = np.sort(
        rng.choice(len(documents), limit, replace=False)
    )

    return [documents[index] for index in indexes]


# =========================================================
# TOPIC MODELING
# =========================================================

def _topic_diversity(components, terms):
    """
    Measure the proportion of unique words among the
    top ten terms for each topic.
    """

    top_terms = []

    for component in components:
        count = min(10, len(component))

        indexes = np.argsort(component)[-count:]

        top_terms.extend(terms[indexes])

    return len(set(top_terms)) / max(1, len(top_terms))


def run_topic_benchmarks(frame, algorithms):
    """
    Benchmark LDA and NMF using topic diversity.

    Topic diversity measures vocabulary variety; it does
    not by itself establish topic coherence or quality.
    """

    algorithms = _validate_algorithms(
        algorithms,
        CATEGORY_ALGORITHMS["topic_modeling"],
        "topic modeling"
    )

    documents = _sample_documents(
        _documents(frame),
        MAX_TOPIC_DOCUMENTS
    )

    results = []

    for name in algorithms:
        started = time.perf_counter()

        try:
            if name == "LDA":
                vectorizer = CountVectorizer(
                    max_features=MAX_TEXT_FEATURES,
                    stop_words="english",
                    min_df=1
                )

                matrix = vectorizer.fit_transform(documents)

                topic_count = min(
                    10,
                    matrix.shape[0],
                    matrix.shape[1]
                )

                if topic_count < 2:
                    raise ValueError(
                        "Topic modeling requires at least two "
                        "documents and two distinct terms."
                    )

                model = LatentDirichletAllocation(
                    n_components=topic_count,
                    random_state=RANDOM_STATE,
                    max_iter=10
                )

            else:
                vectorizer = TfidfVectorizer(
                    max_features=MAX_TEXT_FEATURES,
                    stop_words="english",
                    min_df=1
                )

                matrix = vectorizer.fit_transform(documents)

                topic_count = min(
                    10,
                    matrix.shape[0],
                    matrix.shape[1]
                )

                if topic_count < 2:
                    raise ValueError(
                        "Topic modeling requires at least two "
                        "documents and two distinct terms."
                    )

                model = NMF(
                    n_components=topic_count,
                    random_state=RANDOM_STATE,
                    max_iter=400,
                    init="nndsvda"
                )

            model.fit(matrix)

            diversity = _topic_diversity(
                model.components_,
                vectorizer.get_feature_names_out()
            )

            reconstruction_error = getattr(
                model, "reconstruction_err_", None
            )

            results.append(_metric_result(
                name,
                "Topic Diversity",
                diversity,
                time.perf_counter() - started,
                topics=int(model.components_.shape[0]),
                reconstruction_error=(
                    round(float(reconstruction_error), 4)
                    if reconstruction_error is not None
                    else None
                ),
                document_count=len(documents)
            ))

        except Exception as error:
            results.append(_metric_result(
                name,
                "Topic Diversity",
                None,
                time.perf_counter() - started,
                error=str(error)
            ))

    return results


# =========================================================
# TEXT SIMILARITY
# =========================================================

def run_similarity_benchmarks(frame, algorithms):
    """
    Benchmark text similarity algorithms.

    Score = mean nearest-neighbor similarity, excluding
    each document's self-similarity.

    Higher scores indicate greater average similarity,
    not necessarily better retrieval quality.
    """

    algorithms = _validate_algorithms(
        algorithms,
        CATEGORY_ALGORITHMS["text_similarity"],
        "text similarity"
    )

    documents = _sample_documents(
        _documents(frame),
        MAX_TEXT_DOCUMENTS
    )

    results = []

    for name in algorithms:
        started = time.perf_counter()

        try:
            if name == "TF-IDF + Cosine Similarity":
                matrix = TfidfVectorizer(
                    max_features=MAX_TEXT_FEATURES,
                    stop_words="english"
                ).fit_transform(documents)

                # Sparse matrix multiplication avoids allocating
                # a dense intermediate feature matrix.
                similarities = matrix @ matrix.T
                similarities.setdiag(0)
                similarities = similarities.tocsr()

                nearest = np.zeros(
                    similarities.shape[0],
                    dtype=np.float64
                )

                for row in range(similarities.shape[0]):
                    start = similarities.indptr[row]
                    end = similarities.indptr[row + 1]

                    values = similarities.data[start:end]

                    nearest[row] = (
                        float(np.max(values))
                        if len(values)
                        else 0.0
                    )

                score = float(np.mean(nearest))

            else:
                token_sets = [
                    set(document.lower().split())
                    for document in documents
                ]

                nearest = []

                for row, left in enumerate(token_sets):
                    best = 0.0

                    for column, right in enumerate(token_sets):
                        if row == column:
                            continue

                        union = left | right

                        similarity = (
                            len(left & right) / len(union)
                            if union
                            else 0.0
                        )

                        best = max(best, similarity)

                    nearest.append(best)

                score = float(np.mean(nearest))

            results.append(_metric_result(
                name,
                "Mean Nearest Similarity",
                score,
                time.perf_counter() - started,
                document_count=len(documents)
            ))

        except Exception as error:
            results.append(_metric_result(
                name,
                "Mean Nearest Similarity",
                None,
                time.perf_counter() - started,
                error=str(error)
            ))

    return results


# =========================================================
# SEQUENCE MODELING
# =========================================================

def run_sequence_benchmarks(
    feature_frame,
    target_frame,
    targets,
    algorithms
):
    """
    Benchmark LSTM, GRU and Transformer-style attention
    models for numeric sequence prediction.

    The rows must already be in chronological order.

    A temporal holdout is used instead of a random split.
    The target is excluded from the input features.
    """

    algorithms = _validate_algorithms(
        algorithms,
        CATEGORY_ALGORITHMS["sequence_modeling"],
        "sequence modeling"
    )

    _validate_frame(feature_frame)
    _validate_frame(target_frame)

    if len(feature_frame) != len(target_frame):
        raise ValueError(
            "Feature and target frames must contain the same "
            "number of rows in the same order."
        )

    if not feature_frame.index.equals(target_frame.index):
        raise ValueError(
            "Feature and target indexes must match. Align the "
            "frames before running sequence modeling."
        )

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Sequence models require TensorFlow in the active "
            "Flask environment."
        ) from error

    if isinstance(targets, str):
        targets = [targets]

    if not targets:
        raise ValueError(
            "Select at least one numeric sequence target."
        )

    if len(feature_frame) > MAX_SEQUENCE_ROWS:
        raise ValueError(
            f"Sequence modeling is limited to "
            f"{MAX_SEQUENCE_ROWS} rows in this configuration."
        )

    results = []

    for target in targets:
        if target not in target_frame.columns:
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="The selected target column does not exist."
                ))
            continue

        if not pd.api.types.is_numeric_dtype(
            target_frame[target]
        ):
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="Sequence targets must be numeric."
                ))
            continue

        # Prevent target leakage if the target also appears
        # among the supplied feature columns.
        features = feature_frame.drop(
            columns=[target],
            errors="ignore"
        ).select_dtypes(
            include=["number", "bool"]
        ).copy()

        if features.empty:
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="No numeric input features remain."
                ))
            continue

        numeric_target = pd.to_numeric(
            target_frame[target],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        numeric_features = features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Keep chronological order while removing unusable rows.
        combined = numeric_features.copy()
        combined["__target__"] = numeric_target

        combined = combined.dropna()

        if len(combined) < 10:
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="At least 10 valid chronological rows "
                          "are required for sequence modeling."
                ))
            continue

        x_values = combined.drop(
            columns=["__target__"]
        ).to_numpy(dtype=np.float32)

        y_values = combined["__target__"].to_numpy(
            dtype=np.float32
        )

        # Build windows using only preceding input rows.
        window = min(5, max(2, len(combined) // 5))

        if len(combined) <= window + 3:
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="Not enough rows to construct sequence windows."
                ))
            continue

        # Chronological 80/20 split. Fit normalization only
        # on training-period rows to reduce leakage.
        split_row = max(
            window + 2,
            int(len(combined) * 0.8)
        )

        if split_row >= len(combined):
            split_row = len(combined) - 1

        x_mean = x_values[:split_row].mean(axis=0)
        x_std = x_values[:split_row].std(axis=0)
        x_std = np.where(x_std == 0, 1, x_std)

        y_mean = float(y_values[:split_row].mean())
        y_std = float(y_values[:split_row].std())

        if y_std == 0:
            y_std = 1.0

        x_scaled = (x_values - x_mean) / x_std
        y_scaled = (y_values - y_mean) / y_std

        sequences = np.asarray([
            x_scaled[index - window:index]
            for index in range(window, len(combined))
        ])

        labels = y_scaled[window:]

        # Sequence label at index i predicts the target at i,
        # using only features from rows before i.
        train_count = split_row - window

        x_train = sequences[:train_count]
        y_train = labels[:train_count]

        x_test = sequences[train_count:]
        y_test = labels[train_count:]

        if not len(x_train) or not len(x_test):
            for name in algorithms:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    0,
                    target=target,
                    error="The chronological split produced an "
                          "empty training or test set."
                ))
            continue

        for name in algorithms:
            started = time.perf_counter()

            try:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(RANDOM_STATE)

                inputs = tf.keras.Input(
                    shape=(window, x_scaled.shape[1])
                )

                if name == "LSTM":
                    encoded = tf.keras.layers.LSTM(24)(inputs)

                elif name == "GRU":
                    encoded = tf.keras.layers.GRU(24)(inputs)

                else:
                    attention = (
                        tf.keras.layers.MultiHeadAttention(
                            num_heads=2,
                            key_dim=8
                        )(inputs, inputs)
                    )

                    normalized = (
                        tf.keras.layers.LayerNormalization()(
                            inputs + attention
                        )
                    )

                    encoded = (
                        tf.keras.layers.GlobalAveragePooling1D()(
                            normalized
                        )
                    )

                outputs = tf.keras.layers.Dense(1)(encoded)

                model = tf.keras.Model(
                    inputs=inputs,
                    outputs=outputs
                )

                model.compile(
                    optimizer="adam",
                    loss="mse"
                )

                model.fit(
                    x_train,
                    y_train,
                    epochs=8,
                    batch_size=min(32, len(x_train)),
                    verbose=0
                )

                predictions_scaled = model.predict(
                    x_test,
                    verbose=0
                ).reshape(-1)

                predictions = (
                    predictions_scaled * y_std + y_mean
                )

                actual = y_test * y_std + y_mean

                # R² is undefined for a single test observation.
                score = (
                    r2_score(actual, predictions)
                    if len(actual) >= 2
                    else None
                )

                rmse = float(
                    np.sqrt(
                        mean_squared_error(actual, predictions)
                    )
                )

                mae = float(
                    mean_absolute_error(actual, predictions)
                )

                results.append(_metric_result(
                    name,
                    "R²",
                    score,
                    time.perf_counter() - started,
                    target=target,
                    rmse=round(rmse, 4),
                    mae=round(mae, 4),
                    train_rows=len(x_train),
                    test_rows=len(x_test),
                    window_size=window
                ))

            except Exception as error:
                results.append(_metric_result(
                    name,
                    "R²",
                    None,
                    time.perf_counter() - started,
                    target=target,
                    error=str(error)
                ))

            finally:
                tf.keras.backend.clear_session()

    return results
