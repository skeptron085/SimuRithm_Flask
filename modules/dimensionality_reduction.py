"""
==========================================================
SimuRithm
Dimensionality Reduction Module
==========================================================

Supported Algorithms:
    - PCA
    - t-SNE
    - UMAP
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

from sklearn.decomposition import PCA

from sklearn.manifold import (
    TSNE,
    trustworthiness
)


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

        transformers.append(
            (
                "numeric",
                Pipeline(
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
                ),
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                Pipeline(
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
                ),
                categorical_features
            )
        )

    if not transformers:

        raise ValueError(
            "No usable features were found."
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
                "too large for dimensionality reduction."
            )

        matrix = matrix.toarray()

    return np.asarray(
        matrix,
        dtype=float
    )


# ==========================================================
# Dimensionality Reduction
# ==========================================================

def run_dimensionality_reduction(
    df,
    selected_algorithm,
    n_components=2
):
    """
    Run the selected dimensionality reduction algorithm.
    """

    allowed_algorithms = {

        "PCA",

        "t-SNE",

        "UMAP"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported dimensionality reduction "
            f"algorithm: '{selected_algorithm}'."
        )

    matrix = _prepare_features(
        df
    )

    if len(matrix) < 3:

        raise ValueError(
            "At least three rows are required."
        )

    try:

        start = time.time()

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        if selected_algorithm == "PCA":

            maximum_components = min(
                matrix.shape[0],
                matrix.shape[1]
            )

            components = min(
                n_components,
                maximum_components
            )

            if components < 1:

                raise ValueError(
                    "Unable to determine PCA components."
                )

            model = PCA(
                n_components=components,
                random_state=42
            )

            reduced = model.fit_transform(
                matrix
            )

            explained_variance = (
                model.explained_variance_ratio_
            )

            variance_sum = float(
                explained_variance.sum()
            )

        # --------------------------------------------------
        # t-SNE
        # --------------------------------------------------

        elif selected_algorithm == "t-SNE":

            components = min(
                n_components,
                3
            )

            # t-SNE requires perplexity < number of samples.
            perplexity = min(
                30,
                max(
                    2,
                    (len(matrix) - 1) // 3
                )
            )

            model = TSNE(
                n_components=components,
                perplexity=perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto"
            )

            reduced = model.fit_transform(
                matrix
            )

            explained_variance = None

            variance_sum = None

        # --------------------------------------------------
        # UMAP
        # --------------------------------------------------

        else:

            try:

                import umap

            except ImportError:

                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install it using: pip install umap-learn"
                )

            components = min(
                n_components,
                3
            )

            neighbors = min(
                15,
                max(
                    2,
                    len(matrix) - 1
                )
            )

            model = umap.UMAP(
                n_components=components,
                n_neighbors=neighbors,
                random_state=42
            )

            reduced = model.fit_transform(
                matrix
            )

            explained_variance = None

            variance_sum = None

        end = time.time()

        # --------------------------------------------------
        # Trustworthiness
        # --------------------------------------------------

        trust_score = None

        try:

            if (
                len(matrix) >= 10
                and
                reduced.shape[1] >= 2
            ):

                neighbors_for_score = min(
                    5,
                    len(matrix) - 1
                )

                trust_score = round(
                    float(
                        trustworthiness(
                            matrix,
                            reduced,
                            n_neighbors=neighbors_for_score
                        )
                    ),
                    4
                )

        except Exception:

            trust_score = None

        # --------------------------------------------------
        # Preview
        # --------------------------------------------------

        preview_rows = []

        preview_count = min(
            10,
            len(reduced)
        )

        for index in range(
            preview_count
        ):

            row = {
                "row": int(index + 1)
            }

            for component_index in range(
                reduced.shape[1]
            ):

                row[
                    f"component_{component_index + 1}"
                ] = round(
                    float(
                        reduced[
                            index,
                            component_index
                        ]
                    ),
                    6
                )

            preview_rows.append(
                row
            )

        return [
            {
                "success": True,

                "category":
                    "dimensionality_reduction",

                "algorithm":
                    selected_algorithm,

                "original_dimensions":
                    int(matrix.shape[1]),

                "reduced_dimensions":
                    int(reduced.shape[1]),

                "explained_variance_ratio":
                    (
                        explained_variance.tolist()
                        if explained_variance is not None
                        else None
                    ),

                "explained_variance_total":
                    (
                        round(
                            variance_sum,
                            4
                        )
                        if variance_sum is not None
                        else None
                    ),

                "trustworthiness":
                    trust_score,

                "preview":
                    preview_rows,

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
                    "dimensionality_reduction",

                "algorithm":
                    selected_algorithm,

                "execution_time":
                    0,

                "error":
                    str(error)
            }
        ]