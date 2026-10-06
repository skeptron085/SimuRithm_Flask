"""
==========================================================
SimuRithm
Anomaly Detection Module
==========================================================

Supported Algorithms:
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor (LOF)
==========================================================
"""

import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from sklearn.neighbors import LocalOutlierFactor


# ==========================================================
# Feature Preparation
# ==========================================================

def _prepare_features(df):

    if df is None or df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    numeric_features = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        df.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    if not transformers:

        raise ValueError(
            "No usable features were found "
            "for anomaly detection."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    matrix = preprocessor.fit_transform(
        df
    )

    if hasattr(matrix, "toarray"):

        if (
            matrix.shape[0]
            *
            matrix.shape[1]
            >
            20_000_000
        ):

            raise ValueError(
                "The encoded feature matrix is "
                "too large for anomaly detection."
            )

        matrix = matrix.toarray()

    return np.asarray(
        matrix,
        dtype=float
    )


# ==========================================================
# Anomaly Detection
# ==========================================================

def run_anomaly_detection(
    df,
    selected_algorithm,
    contamination=0.10
):
    """
    Run the selected anomaly detection algorithm.
    """

    allowed_algorithms = {

        "Isolation Forest",

        "One-Class SVM",

        "Local Outlier Factor (LOF)"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported anomaly detection algorithm: "
            f"'{selected_algorithm}'."
        )

    matrix = _prepare_features(
        df
    )

    if len(matrix) < 5:

        raise ValueError(
            "At least five rows are recommended "
            "for anomaly detection."
        )

    try:

        start = time.time()

        # --------------------------------------------------
        # Isolation Forest
        # --------------------------------------------------

        if selected_algorithm == "Isolation Forest":

            model = IsolationForest(
                contamination=contamination,
                random_state=42
            )

            predictions = model.fit_predict(
                matrix
            )

            scores = model.decision_function(
                matrix
            )

        # --------------------------------------------------
        # One-Class SVM
        # --------------------------------------------------

        elif selected_algorithm == "One-Class SVM":

            model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=contamination
            )

            predictions = model.fit_predict(
                matrix
            )

            scores = model.decision_function(
                matrix
            )

        # --------------------------------------------------
        # LOF
        # --------------------------------------------------

        else:

            model = LocalOutlierFactor(
                n_neighbors=min(
                    20,
                    len(matrix) - 1
                ),
                contamination=contamination
            )

            predictions = model.fit_predict(
                matrix
            )

            scores = model.negative_outlier_factor_

        end = time.time()

        # --------------------------------------------------
        # Convert:
        #
        # -1 = anomaly
        #  1 = normal
        # --------------------------------------------------

        anomaly_mask = (
            predictions == -1
        )

        anomaly_count = int(
            np.sum(
                anomaly_mask
            )
        )

        normal_count = int(
            np.sum(
                predictions == 1
            )
        )

        total_rows = len(
            predictions
        )

        anomaly_rate = (
            anomaly_count / total_rows
        )

        # --------------------------------------------------
        # Score Information
        # --------------------------------------------------

        score_min = float(
            np.min(scores)
        )

        score_max = float(
            np.max(scores)
        )

        score_mean = float(
            np.mean(scores)
        )

        return [
            {
                "success": True,

                "category":
                    "anomaly_detection",

                "algorithm":
                    selected_algorithm,

                "total_rows":
                    int(total_rows),

                "anomaly_count":
                    anomaly_count,

                "normal_count":
                    normal_count,

                "anomaly_rate":
                    round(
                        anomaly_rate,
                        4
                    ),

                "score_min":
                    round(
                        score_min,
                        4
                    ),

                "score_max":
                    round(
                        score_max,
                        4
                    ),

                "score_mean":
                    round(
                        score_mean,
                        4
                    ),

                "execution_time":
                    round(
                        end - start,
                        4
                    )
            }
        ]

    except Exception as error:

        return [
            {
                "success": False,

                "category":
                    "anomaly_detection",

                "algorithm":
                    selected_algorithm,

                "anomaly_count":
                    None,

                "normal_count":
                    None,

                "anomaly_rate":
                    None,

                "execution_time":
                    0,

                "error":
                    str(error)
            }
        ]