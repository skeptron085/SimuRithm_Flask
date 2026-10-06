import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


CATEGORY_ALGORITHMS = {
    "classification": [
        "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes"
    ],
    "regression": [
        "Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor",
        "Random Forest Regressor", "SVR"
    ],
    "clustering": ["K-Means", "DBSCAN", "Hierarchical Clustering"],
    "anomaly_detection": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor (LOF)"],
    "dimensionality_reduction": ["PCA", "t-SNE", "UMAP"],
    "topic_modeling": ["LDA", "NMF"],
    "text_similarity": ["TF-IDF + Cosine Similarity", "Jaccard Similarity"],
    "sequence_modeling": ["LSTM", "GRU", "Transformer"]
}


def _numeric_features(frame):
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", max_categories=50))
        ]), categorical))
    if not transformers:
        raise ValueError("Select at least one usable feature.")
    matrix = ColumnTransformer(transformers).fit_transform(frame)
    if hasattr(matrix, "toarray"):
        if matrix.shape[0] * matrix.shape[1] > 20_000_000:
            raise ValueError("Encoded feature matrix is too large for this analysis.")
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _metric_result(algorithm, metric, score, elapsed, **details):
    return {
        "algorithm": algorithm,
        "primary_metric": metric,
        "primary_score": None if score is None or not np.isfinite(score) else round(float(score), 4),
        "higher_is_better": True,
        "execution_time": round(float(elapsed), 4),
        **details
    }


def run_anomaly_benchmarks(frame, algorithms):
    matrix = _numeric_features(frame)
    if len(matrix) < 4:
        raise ValueError("Anomaly detection needs at least four usable rows.")
    models = {
        "Isolation Forest": IsolationForest(contamination="auto", random_state=42),
        "One-Class SVM": OneClassSVM(nu=0.1, gamma="scale"),
        "Local Outlier Factor (LOF)": LocalOutlierFactor(n_neighbors=min(20, len(matrix) - 1), contamination="auto")
    }
    results = []
    for name in algorithms:
        started = time.perf_counter()
        try:
            labels = models[name].fit_predict(matrix)
            anomaly_rate = float(np.mean(labels == -1))
            valid_labels = labels
            score = None
            if 2 <= len(np.unique(valid_labels)) < len(valid_labels):
                score = silhouette_score(matrix, valid_labels)
            results.append(_metric_result(name, "Silhouette", score, time.perf_counter() - started,
                                          anomaly_rate=round(anomaly_rate, 4), outliers=int(np.sum(labels == -1))))
        except Exception as error:
            results.append(_metric_result(name, "Silhouette", None, time.perf_counter() - started, error=str(error)))
    return results


def run_reduction_benchmarks(frame, algorithms):
    matrix = _numeric_features(frame)
    if len(matrix) < 4:
        raise ValueError("Dimensionality reduction needs at least four usable rows.")
    if len(matrix) > 1000:
        matrix = matrix[np.random.default_rng(42).choice(len(matrix), 1000, replace=False)]
    neighbors = max(1, min(5, (len(matrix) - 1) // 2))
    results = []
    for name in algorithms:
        started = time.perf_counter()
        try:
            if name == "PCA":
                embedding = PCA(n_components=2, random_state=42).fit_transform(matrix)
            elif name == "t-SNE":
                embedding = TSNE(n_components=2, random_state=42, init="pca",
                                 learning_rate="auto", perplexity=min(30, len(matrix) - 1)).fit_transform(matrix)
            else:
                try:
                    import umap
                except ImportError as error:
                    raise RuntimeError("UMAP requires the umap-learn package; install it from requirement.txt.") from error
                embedding = umap.UMAP(n_components=2, random_state=42,
                                      n_neighbors=min(15, len(matrix) - 1)).fit_transform(matrix)
            score = trustworthiness(matrix, embedding, n_neighbors=neighbors)
            results.append(_metric_result(name, "Trustworthiness", score, time.perf_counter() - started,
                                          embedding_dimensions=2))
        except Exception as error:
            results.append(_metric_result(name, "Trustworthiness", None, time.perf_counter() - started, error=str(error)))
    return results


def _documents(frame):
    documents = frame.fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    documents = documents[documents.ne("")]
    if len(documents) < 2:
        raise ValueError("Select text columns containing at least two non-empty documents.")
    return documents.tolist()


def _topic_diversity(components, terms):
    top_terms = []
    for component in components:
        indexes = np.argsort(component)[-10:]
        top_terms.extend(terms[indexes])
    return len(set(top_terms)) / max(1, len(top_terms))


def run_topic_benchmarks(frame, algorithms):
    documents = _documents(frame)
    if len(documents) > 2000:
        indexes = np.random.default_rng(42).choice(len(documents), 2000, replace=False)
        documents = [documents[index] for index in indexes]
    results = []
    for name in algorithms:
        started = time.perf_counter()
        try:
            if name == "LDA":
                vectorizer = CountVectorizer(max_features=3000, stop_words="english", min_df=1)
                matrix = vectorizer.fit_transform(documents)
                topic_count = min(10, matrix.shape[0], matrix.shape[1])
                if topic_count < 2:
                    raise ValueError("Topic modeling needs at least two documents and two distinct terms.")
                model = LatentDirichletAllocation(n_components=topic_count,
                                                  random_state=42, max_iter=10)
            else:
                vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", min_df=1)
                matrix = vectorizer.fit_transform(documents)
                topic_count = min(10, matrix.shape[0], matrix.shape[1])
                if topic_count < 2:
                    raise ValueError("Topic modeling needs at least two documents and two distinct terms.")
                model = NMF(n_components=topic_count,
                            random_state=42, max_iter=400, init="nndsvda")
            model.fit(matrix)
            score = _topic_diversity(model.components_, vectorizer.get_feature_names_out())
            results.append(_metric_result(name, "Topic Diversity", score, time.perf_counter() - started,
                                          topics=int(model.components_.shape[0]),
                                          reconstruction_error=round(float(getattr(model, "reconstruction_err_", 0)), 4)))
        except Exception as error:
            results.append(_metric_result(name, "Topic Diversity", None, time.perf_counter() - started, error=str(error)))
    return results


def run_similarity_benchmarks(frame, algorithms):
    documents = _documents(frame)
    if len(documents) > 1000:
        indexes = np.random.default_rng(42).choice(len(documents), 1000, replace=False)
        documents = [documents[index] for index in indexes]
    results = []
    for name in algorithms:
        started = time.perf_counter()
        try:
            if name == "TF-IDF + Cosine Similarity":
                matrix = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform(documents)
                similarities = (matrix @ matrix.T).toarray()
            else:
                token_sets = [set(document.lower().split()) for document in documents]
                similarities = np.zeros((len(token_sets), len(token_sets)), dtype=float)
                for row, left in enumerate(token_sets):
                    for column in range(row + 1, len(token_sets)):
                        right = token_sets[column]
                        union = left | right
                        value = len(left & right) / len(union) if union else 0.0
                        similarities[row, column] = value
                        similarities[column, row] = value
            np.fill_diagonal(similarities, 0)
            score = float(np.mean(np.max(similarities, axis=1)))
            results.append(_metric_result(name, "Mean Nearest Similarity", score, time.perf_counter() - started,
                                          document_count=len(documents)))
        except Exception as error:
            results.append(_metric_result(name, "Mean Nearest Similarity", None,
                                          time.perf_counter() - started, error=str(error)))
    return results


def run_sequence_benchmarks(feature_frame, target_frame, targets, algorithms):
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Sequence models require TensorFlow; install the configured Flask requirements.") from error

    features = feature_frame.select_dtypes(include=["number", "bool"]).copy()
    if features.empty:
        raise ValueError("Sequence modeling requires numeric X variables.")
    results = []
    for target in targets:
        if target not in target_frame.columns or not pd.api.types.is_numeric_dtype(target_frame[target]):
            for name in algorithms:
                results.append(_metric_result(name, "R²", None, 0, target=target,
                                              error="Sequence targets must be numeric."))
            continue
        combined = features.join(target_frame[target].rename("__target__")).dropna()
        values = combined.to_numpy(dtype=np.float32)
        window = min(5, max(2, len(values) // 5))
        if len(values) <= window + 2:
            for name in algorithms:
                results.append(_metric_result(name, "R²", None, 0, target=target,
                                              error="Not enough rows to build sequence windows."))
            continue
        x_values = values[:, :-1]
        y_values = values[:, -1]
        x_mean = x_values.mean(axis=0)
        x_std = np.where(x_values.std(axis=0) == 0, 1, x_values.std(axis=0))
        y_mean = y_values.mean()
        y_std = y_values.std() or 1
        x_values = (x_values - x_mean) / x_std
        y_values = (y_values - y_mean) / y_std
        sequences = np.array([x_values[index - window:index] for index in range(window, len(values))])
        labels = y_values[window:]
        split = max(1, int(len(sequences) * 0.8))
        x_train, x_test = sequences[:split], sequences[split:]
        y_train, y_test = labels[:split], labels[split:]
        if not len(x_test):
            x_train, x_test = sequences[:-1], sequences[-1:]
            y_train, y_test = labels[:-1], labels[-1:]

        for name in algorithms:
            started = time.perf_counter()
            try:
                tf.keras.utils.set_random_seed(42)
                inputs = tf.keras.Input(shape=(window, x_values.shape[1]))
                if name == "LSTM":
                    encoded = tf.keras.layers.LSTM(24)(inputs)
                elif name == "GRU":
                    encoded = tf.keras.layers.GRU(24)(inputs)
                else:
                    attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)(inputs, inputs)
                    encoded = tf.keras.layers.GlobalAveragePooling1D()(tf.keras.layers.LayerNormalization()(inputs + attention))
                outputs = tf.keras.layers.Dense(1)(encoded)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer="adam", loss="mse")
                model.fit(x_train, y_train, epochs=8, batch_size=min(32, len(x_train)), verbose=0)
                predictions = model.predict(x_test, verbose=0).reshape(-1) * y_std + y_mean
                actual = y_test * y_std + y_mean
                score = r2_score(actual, predictions) if len(actual) > 1 else None
                results.append(_metric_result(name, "R²", score, time.perf_counter() - started,
                                              target=target,
                                              rmse=round(float(mean_squared_error(actual, predictions) ** 0.5), 4),
                                              mae=round(float(mean_absolute_error(actual, predictions)), 4)))
            except Exception as error:
                results.append(_metric_result(name, "R²", None, time.perf_counter() - started,
                                              target=target, error=str(error)))
    return results
