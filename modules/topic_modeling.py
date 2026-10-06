"""
==========================================================
SimuRithm
Topic Modeling Module
==========================================================

Supported Algorithms:
    - LDA
    - NMF
==========================================================
"""

import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF
)


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

    candidates = []

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

        column_name = str(
            column
        ).lower()

        named_text = any(
            keyword in column_name
            for keyword in keywords
        )

        if (
            named_text
            or
            average_length >= 30
        ):

            candidates.append(
                column
            )

    if not candidates:

        raise ValueError(
            "No suitable text column was found "
            "for topic modeling."
        )

    return candidates[0]


# ==========================================================
# Topic Modeling
# ==========================================================

def run_topic_modeling(
    df,
    selected_algorithm,
    text_column=None,
    n_topics=5,
    top_words=10
):
    """
    Run LDA or NMF on a text column.
    """

    allowed_algorithms = {

        "LDA",

        "NMF"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported topic modeling algorithm: "
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

        # --------------------------------------------------
        # Remove empty documents
        # --------------------------------------------------

        documents = [
            document
            for document in documents
            if document.strip()
        ]

        if len(documents) < 2:

            raise ValueError(
                "At least two non-empty documents "
                "are required."
            )

        # --------------------------------------------------
        # LDA
        # --------------------------------------------------

        if selected_algorithm == "LDA":

            vectorizer = CountVectorizer(
                stop_words="english",
                max_features=5000
            )

            matrix = vectorizer.fit_transform(
                documents
            )

            if matrix.shape[1] == 0:

                raise ValueError(
                    "No usable words were found "
                    "after text preprocessing."
                )

            topics = min(
                n_topics,
                matrix.shape[1]
            )

            model = LatentDirichletAllocation(
                n_components=topics,
                random_state=42,
                learning_method="batch"
            )

            document_topic_matrix = (
                model.fit_transform(
                    matrix
                )
            )

            topic_word_matrix = (
                model.components_
            )

            feature_names = (
                vectorizer.get_feature_names_out()
            )

        # --------------------------------------------------
        # NMF
        # --------------------------------------------------

        else:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000
            )

            matrix = vectorizer.fit_transform(
                documents
            )

            if matrix.shape[1] == 0:

                raise ValueError(
                    "No usable words were found "
                    "after text preprocessing."
                )

            topics = min(
                n_topics,
                matrix.shape[1]
            )

            model = NMF(
                n_components=topics,
                random_state=42,
                init="nndsvda",
                max_iter=500
            )

            document_topic_matrix = (
                model.fit_transform(
                    matrix
                )
            )

            topic_word_matrix = (
                model.components_
            )

            feature_names = (
                vectorizer.get_feature_names_out()
            )

        # --------------------------------------------------
        # Extract Topics
        # --------------------------------------------------

        topic_results = []

        for topic_index, topic_values in enumerate(
            topic_word_matrix
        ):

            top_indices = (
                topic_values
                .argsort()[
                    ::-1
                ][
                    :top_words
                ]
            )

            words = [
                str(
                    feature_names[index]
                )
                for index in top_indices
            ]

            topic_results.append(
                {
                    "topic":
                        int(topic_index + 1),

                    "keywords":
                        words
                }
            )

        # --------------------------------------------------
        # Document Topic Preview
        # --------------------------------------------------

        document_preview = []

        preview_count = min(
            10,
            len(document_topic_matrix)
        )

        for index in range(
            preview_count
        ):

            dominant_topic = int(
                np.argmax(
                    document_topic_matrix[
                        index
                    ]
                )
                + 1
            )

            document_preview.append(
                {
                    "document":
                        int(index + 1),

                    "dominant_topic":
                        dominant_topic,

                    "topic_distribution":
                        [
                            round(
                                float(value),
                                4
                            )
                            for value
                            in document_topic_matrix[
                                index
                            ]
                        ]
                }
            )

        end = time.time()

        return [
            {
                "success": True,

                "category":
                    "topic_modeling",

                "algorithm":
                    selected_algorithm,

                "text_column":
                    str(column),

                "documents":
                    int(len(documents)),

                "vocabulary_size":
                    int(len(feature_names)),

                "topics":
                    int(topics),

                "topic_keywords":
                    topic_results,

                "document_preview":
                    document_preview,

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
                    "topic_modeling",

                "algorithm":
                    selected_algorithm,

                "execution_time":
                    0,

                "error":
                    str(error)
            }
        ]