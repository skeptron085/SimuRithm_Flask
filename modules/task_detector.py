import pandas as pd
import numpy as np


# =========================================================
# SimuRithm Task Categories
# =========================================================

CLASSIFICATION = "classification"
REGRESSION = "regression"
CLUSTERING = "clustering"
ANOMALY_DETECTION = "anomaly_detection"
DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
TOPIC_MODELING = "topic_modeling"
TEXT_SIMILARITY = "text_similarity"
SEQUENCE_MODELING = "sequence_modeling"


# =========================================================
# Keywords
# =========================================================

ANOMALY_KEYWORDS = {
    "anomaly",
    "anomalies",
    "outlier",
    "outliers",
    "fraud",
    "intrusion",
    "attack",
    "abnormal",
    "unusual",
    "defect",
    "fault"
}


SEQUENCE_KEYWORDS = {
    "timestamp",
    "datetime",
    "date",
    "sequence",
    "sequence_id",
    "session",
    "session_id",
    "timestep",
    "time_step",
    "event_time",
    "event_date"
}


TEXT_KEYWORDS = {
    "text",
    "document",
    "content",
    "description",
    "message",
    "comment",
    "review",
    "article",
    "paragraph",
    "sentence",
    "abstract",
    "body",
    "news",
    "tweet",
    "post"
}


REGRESSION_KEYWORDS = {
    "salary",
    "income",
    "price",
    "amount",
    "weight",
    "height",
    "temperature",
    "distance",
    "duration",
    "revenue",
    "cost",
    "balance",
    "percentage",
    "sales",
    "profit",
    "expense",
    "measurement"
}


# =========================================================
# Basic Helpers
# =========================================================

def _normalized_columns(df):

    return [
        str(column).strip().lower()
        for column in df.columns
    ]


def _column_contains_keyword(column, keywords):

    name = str(column).strip().lower()

    tokens = (
        name
        .replace("-", "_")
        .replace(" ", "_")
        .split("_")
    )

    for keyword in keywords:

        if keyword in tokens:
            return True

        if keyword in name:
            return True

    return False


def _numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def _categorical_columns(df):

    return df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()


def _datetime_columns(df):

    columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):

            columns.append(column)

    return columns


def _text_columns(df):

    result = []

    for column in df.columns:

        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or
            pd.api.types.is_string_dtype(series)
        ):
            continue

        values = (
            series
            .dropna()
            .astype(str)
            .str.strip()
        )

        if len(values) == 0:
            continue

        average_length = float(
            values.str.len().mean()
        )

        unique_ratio = (
            values.nunique() / len(values)
        )

        named_text = _column_contains_keyword(
            column,
            TEXT_KEYWORDS
        )

        long_text = average_length >= 30

        descriptive_text = (
            average_length >= 20
            and
            unique_ratio >= 0.30
        )

        if (
            named_text
            or
            long_text
            or
            descriptive_text
        ):

            result.append(column)

    return result


# =========================================================
# Target Information
# =========================================================

def target_cardinality(df, target_column):

    if (
        df is None
        or
        df.empty
        or
        target_column not in df.columns
    ):

        return {
            "valid": False,
            "sample_count": 0,
            "unique_count": 0,
            "unique_ratio": 1.0
        }

    y = df[target_column].dropna()

    sample_count = len(y)
    unique_count = y.nunique()

    unique_ratio = (
        unique_count / sample_count
        if sample_count > 0
        else 1.0
    )

    return {
        "valid": sample_count > 0,
        "sample_count": int(sample_count),
        "unique_count": int(unique_count),
        "unique_ratio": float(unique_ratio)
    }


def is_high_cardinality_target(
    df,
    target_column,
    threshold=0.50
):

    info = target_cardinality(
        df,
        target_column
    )

    return (
        info["valid"]
        and
        info["unique_ratio"] > threshold
    )


# =========================================================
# Target Task Detection
# =========================================================

def detect_supervised_task(df, target_column):

    if (
        df is None
        or
        df.empty
        or
        target_column not in df.columns
    ):

        return CLUSTERING

    y = df[target_column].dropna()

    if len(y) < 2:

        return CLUSTERING

    target_name = (
        str(target_column)
        .strip()
        .lower()
    )

    unique_count = int(y.nunique())
    sample_count = int(len(y))

    unique_ratio = (
        unique_count / sample_count
    )

    dtype = y.dtype

    # -----------------------------------------------------
    # Boolean
    # -----------------------------------------------------

    if pd.api.types.is_bool_dtype(dtype):

        return CLASSIFICATION

    # -----------------------------------------------------
    # Categorical
    # -----------------------------------------------------

    if isinstance(
        dtype,
        pd.CategoricalDtype
    ):

        return CLASSIFICATION

    # -----------------------------------------------------
    # Text / String
    # -----------------------------------------------------

    if (
        pd.api.types.is_object_dtype(dtype)
        or
        pd.api.types.is_string_dtype(dtype)
    ):

        values = y.astype(str)

        average_length = float(
            values.str.len().mean()
        )

        if average_length >= 30:

            return TOPIC_MODELING

        return CLASSIFICATION

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(dtype):

        # Binary numeric target
        if unique_count == 2:

            return CLASSIFICATION

        # Small repeated numeric classes
        if (
            unique_count <= 10
            and
            unique_ratio <= 0.20
        ):

            return CLASSIFICATION

        # Moderate repeated classes
        if (
            unique_count <= 20
            and
            unique_ratio <= 0.10
        ):

            return CLASSIFICATION

        # Explicit continuous measurement indicators
        explicit_regression_names = {
            "salary",
            "income",
            "price",
            "amount",
            "weight",
            "height",
            "temperature",
            "distance",
            "duration",
            "revenue",
            "cost",
            "balance",
            "sales",
            "profit",
            "expense",
            "measurement"
        }

        target_tokens = set(
            target_name
            .replace("-", "_")
            .replace(" ", "_")
            .split("_")
        )

        if target_tokens.intersection(
            explicit_regression_names
        ):

            return REGRESSION

        # High-cardinality numeric target
        if unique_ratio > 0.50:

            return REGRESSION

        # Default numeric target
        return REGRESSION

    return CLASSIFICATION


# =========================================================
# Unsupervised Task Detection
# =========================================================

def detect_unsupervised_task(df):

    if df is None or df.empty:

        return CLUSTERING

    numeric = _numeric_columns(df)
    text = _text_columns(df)
    datetime = _datetime_columns(df)

    # -----------------------------------------------------
    # Text
    # -----------------------------------------------------

    if len(text) >= 2:

        return TEXT_SIMILARITY

    if len(text) == 1:

        return TOPIC_MODELING

    # -----------------------------------------------------
    # Explicit anomaly dataset
    # -----------------------------------------------------

    if any(
        _column_contains_keyword(
            column,
            ANOMALY_KEYWORDS
        )
        for column in df.columns
    ):

        return ANOMALY_DETECTION

    # -----------------------------------------------------
    # Sequence / time-series structure
    # -----------------------------------------------------

    if datetime:

        return SEQUENCE_MODELING

    sequence_columns = [

        column
        for column in df.columns

        if _column_contains_keyword(
            column,
            SEQUENCE_KEYWORDS
        )

    ]

    if sequence_columns:

        return SEQUENCE_MODELING

    # -----------------------------------------------------
    # High-dimensional data
    # -----------------------------------------------------

    if len(numeric) >= 10:

        return DIMENSIONALITY_REDUCTION

    if (
        len(df.columns) >= 10
        and
        len(numeric) >= 5
    ):

        return DIMENSIONALITY_REDUCTION

    # -----------------------------------------------------
    # Default
    # -----------------------------------------------------

    return CLUSTERING


# =========================================================
# Main Detector
# =========================================================

def detect_task(
    df,
    target_column=None
):

    if df is None or df.empty:

        return CLUSTERING

    # -----------------------------------------------------
    # No target = unsupervised/specialized task
    # -----------------------------------------------------

    if target_column is None:

        return detect_unsupervised_task(df)

    # -----------------------------------------------------
    # Target unavailable
    # -----------------------------------------------------

    if target_column not in df.columns:

        return detect_unsupervised_task(df)

    # -----------------------------------------------------
    # Supervised target
    # -----------------------------------------------------

    return detect_supervised_task(
        df,
        target_column
    )


# =========================================================
# Dataset Information
# =========================================================

def dataset_information(df):

    return {

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "numeric_columns":
            int(len(_numeric_columns(df))),

        "categorical_columns":
            int(len(_categorical_columns(df))),

        "text_columns":
            int(len(_text_columns(df))),

        "datetime_columns":
            int(len(_datetime_columns(df)))

    }


# =========================================================
# Task Explanation
# =========================================================

def task_reason(
    df,
    target_column=None
):

    task = detect_task(
        df,
        target_column
    )

    if task == CLASSIFICATION:

        return (
            "The selected target contains categorical, "
            "binary, boolean, or repeated discrete values "
            "that are suitable for classification."
        )

    if task == REGRESSION:

        return (
            "The selected target contains continuous or "
            "high-cardinality numeric values suitable for "
            "regression."
        )

    if task == CLUSTERING:

        return (
            "No supervised target was selected, so the "
            "dataset can be analyzed by grouping similar "
            "observations."
        )

    if task == ANOMALY_DETECTION:

        return (
            "The dataset contains characteristics associated "
            "with identifying unusual or abnormal observations."
        )

    if task == DIMENSIONALITY_REDUCTION:

        return (
            "The dataset contains many numeric features, "
            "making dimensionality reduction appropriate."
        )

    if task == TOPIC_MODELING:

        return (
            "The dataset contains document-like text that "
            "can be analyzed for underlying topics."
        )

    if task == TEXT_SIMILARITY:

        return (
            "The dataset contains multiple text fields that "
            "can be compared for similarity."
        )

    if task == SEQUENCE_MODELING:

        return (
            "The dataset contains temporal or ordered "
            "observations suitable for sequence modeling."
        )

    return (
        "The system could not determine a suitable task."
    )