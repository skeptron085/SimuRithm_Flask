"""
==========================================================
SimuRithm
Text Similarity Module
==========================================================

Supported Algorithms:
    - TF-IDF + Cosine Similarity
    - Jaccard Similarity
==========================================================
"""

import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# Text Column Detection
# ==========================================================

def _find_text_column(
    df,
    text_column=None
):

    if text_column:

        if text_column not in df.columns:

            raise ValueError(
                f"Text column '{text_column}' "
                f"was not found."
            )

        return text_column

    keywords = [
        "text",
        "document",
        "content",
        "description",
        "message",
        "comment",
        "review",
        "article",
        "body",
        "title"
    ]

    candidates = []

    for column in df.columns:

        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or
            pd.api.types.is_string_dtype(series)
        ):

            continue

        values = series.dropna()

        if len(values) == 0:

            continue

        average_length = (
            values.astype(str)
            .str.len()
            .mean()
        )

        name = str(
            column
        ).lower()

        if (
            any(
                keyword in name
                for keyword in keywords
            )
            or
            average_length >= 15
        ):

            candidates.append(
                column
            )

    if not candidates:

        raise ValueError(
            "No suitable text column was found."
        )

    return candidates[0]


# ==========================================================
# Jaccard Similarity
# ==========================================================

def _jaccard_similarity(
    text_a,
    text_b
):

    tokens_a = set(
        str(text_a)
        .lower()
        .split()
    )

    tokens_b = set(
        str(text_b)
        .lower()
        .split()
    )

    if not tokens_a and not tokens_b:

        return 1.0

    union = (
        tokens_a
        |
        tokens_b
    )

    if not union:

        return 0.0

    intersection = (
        tokens_a
        &
        tokens_b
    )

    return (
        len(intersection)
        /
        len(union)
    )


# ==========================================================
# Text Similarity
# ==========================================================

def run_text_similarity(
    df,
    selected_algorithm,
    text_column=None
):
    """
    Compare documents using the selected algorithm.
    """

    allowed_algorithms = {

        "TF-IDF + Cosine Similarity",

        "Jaccard Similarity"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported text similarity algorithm: "
            f"'{selected_algorithm}'."
        )

    try:

        start = time.time()

        column = _find_text_column(
            df,
            text_column
        )

        documents = (
            df[column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if len(documents) < 2:

            raise ValueError(
                "At least two documents are required."
            )

        # --------------------------------------------------
        # TF-IDF + Cosine
        # --------------------------------------------------

        if (
            selected_algorithm
            ==
            "TF-IDF + Cosine Similarity"
        ):

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            matrix = vectorizer.fit_transform(
                documents
            )

            similarity_matrix = (
                cosine_similarity(
                    matrix
                )
            )

            # ----------------------------------------------
            # Find Most Similar Pair
            # ----------------------------------------------

            similarity_copy = (
                similarity_matrix.copy()
            )

            np.fill_diagonal(
                similarity_copy,
                -1
            )

            best_index = np.unravel_index(
                np.argmax(
                    similarity_copy
                ),
                similarity_copy.shape
            )

            first_index = int(
                best_index[0]
            )

            second_index = int(
                best_index[1]
            )

            best_score = float(
                similarity_matrix[
                    first_index,
                    second_index
                ]
            )

        # --------------------------------------------------
        # Jaccard
        # --------------------------------------------------

        else:

            best_score = -1.0

            first_index = 0

            second_index = 0

            # Limit pairwise processing for
            # extremely large text datasets.
            maximum_documents = min(
                len(documents),
                1000
            )

            for i in range(
                maximum_documents
            ):

                for j in range(
                    i + 1,
                    maximum_documents
                ):

                    score = (
                        _jaccard_similarity(
                            documents[i],
                            documents[j]
                        )
                    )

                    if score > best_score:

                        best_score = score

                        first_index = i

                        second_index = j

            similarity_matrix = None

        # --------------------------------------------------
        # Similarity Preview
        # --------------------------------------------------

        preview = {

            "document_a":
                int(first_index + 1),

            "document_b":
                int(second_index + 1),

            "similarity":
                round(
                    best_score,
                    4
                )
        }

        end = time.time()

        return [
            {
                "success": True,

                "category":
                    "text_similarity",

                "algorithm":
                    selected_algorithm,

                "text_column":
                    str(column),

                "documents":
                    int(len(documents)),

                "highest_similarity":
                    round(
                        best_score,
                        4
                    ),

                "most_similar_pair":
                    preview,

                "similarity_matrix":
                    (
                        similarity_matrix.round(4).tolist()
                        if similarity_matrix is not None
                        and len(documents) <= 500
                        else None
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
                    "text_similarity",

                "algorithm":
                    selected_algorithm,

                "execution_time":
                    0,

                "error":
                    str(error)
            }
        ]