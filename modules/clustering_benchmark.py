import time
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

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


def run_clustering(
    df,
    selected_features=None,
    selected_algorithm=None
):

    # ---------------------------------------------------------
    # Validate selected features
    # ---------------------------------------------------------

    if selected_features is None:

        selected_features = df.columns.tolist()

    if not selected_features:

        raise ValueError(
            "At least one feature must be selected for clustering."
        )

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    missing_features = [

        column

        for column in selected_features

        if column not in df.columns

    ]

    if missing_features:

        raise ValueError(

            "Selected feature(s) not found: "

            + ", ".join(missing_features)

        )

    # ---------------------------------------------------------
    # Select X variables
    # ---------------------------------------------------------

    X = df[selected_features].copy()

    # ---------------------------------------------------------
    # Numeric / Categorical Features
    # ---------------------------------------------------------

    numeric_features = X.select_dtypes(

        include=["number"]

    ).columns.tolist()

    categorical_features = X.select_dtypes(

        exclude=["number"]

    ).columns.tolist()

    # ---------------------------------------------------------
    # Numeric Preprocessing
    # ---------------------------------------------------------

    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(

                    strategy="median"

                )

            ),

            (
                "scaler",

                StandardScaler()

            )

        ]

    )

    # ---------------------------------------------------------
    # Categorical Preprocessing
    # ---------------------------------------------------------

    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(

                    strategy="most_frequent"

                )

            ),

            (
                "encoder",

                OneHotEncoder(

                    handle_unknown="ignore"

                )

            )

        ]

    )

    # ---------------------------------------------------------
    # Column Transformer
    # ---------------------------------------------------------

    transformers = []

    if numeric_features:

        transformers.append(

            (
                "num",

                numeric_transformer,

                numeric_features

            )

        )

    if categorical_features:

        transformers.append(

            (
                "cat",

                categorical_transformer,

                categorical_features

            )

        )

    if not transformers:

        raise ValueError(

            "No valid numeric or categorical features available."

        )

    preprocessor = ColumnTransformer(

        transformers=transformers

    )

    # ---------------------------------------------------------
    # Transform Dataset
    # ---------------------------------------------------------

    X_processed = preprocessor.fit_transform(X)

    # ---------------------------------------------------------
    # Algorithms
    # ---------------------------------------------------------

    algorithms = {

        "K-Means":

            KMeans(

                n_clusters=3,

                random_state=42,

                n_init=10

            ),

        "Agglomerative":

            AgglomerativeClustering(

                n_clusters=3

            ),

        "Hierarchical Clustering":

            AgglomerativeClustering(

                n_clusters=3

            ),

        "DBSCAN":

            DBSCAN(

                eps=0.5,

                min_samples=5

            )

    }

    # ---------------------------------------------------------
    # Validate Selected Algorithm
    # ---------------------------------------------------------

    if selected_algorithm:

        if selected_algorithm not in algorithms:

            raise ValueError(

                "Unsupported clustering algorithm: "

                + str(selected_algorithm)

            )

        algorithms = {

            selected_algorithm:

            algorithms[selected_algorithm]

        }

    # ---------------------------------------------------------
    # Run Benchmark
    # ---------------------------------------------------------

    results = []

    for name, model in algorithms.items():

        try:

            start = time.time()

            labels = model.fit_predict(

                X_processed

            )

            end = time.time()

            unique = np.unique(labels)

            # -------------------------------------------------
            # Validate Number of Clusters
            # -------------------------------------------------

            if len(unique) < 2:

                silhouette = None

                calinski = None

                davies = None

            else:

                silhouette = round(

                    silhouette_score(

                        X_processed,

                        labels

                    ),

                    4

                )

                calinski = round(

                    calinski_harabasz_score(

                        X_processed,

                        labels

                    ),

                    4

                )

                davies = round(

                    davies_bouldin_score(

                        X_processed,

                        labels

                    ),

                    4

                )

            # -------------------------------------------------
            # Result
            # -------------------------------------------------

            results.append({

                "algorithm": name,

                "selected_features": selected_features,

                "silhouette": silhouette,

                "calinski_harabasz": calinski,

                "davies_bouldin": davies,

                "execution_time": round(

                    end - start,

                    4

                ),

                "clusters": int(

                    len(unique)

                )

            })

        except Exception as e:

            results.append({

                "algorithm": name,

                "selected_features": selected_features,

                "silhouette": None,

                "calinski_harabasz": None,

                "davies_bouldin": None,

                "execution_time": None,

                "clusters": None,

                "error": str(e)

            })

    # ---------------------------------------------------------
    # Sort Results
    # ---------------------------------------------------------

    results.sort(

        key=lambda x:

            x["silhouette"]

            if x["silhouette"] is not None

            else -999,

        reverse=True

    )

    return results