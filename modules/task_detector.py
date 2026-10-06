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
# Keyword Definitions
# =========================================================

REGRESSION_KEYWORDS = [
    "score",
    "grade",
    "salary",
    "income",
    "price",
    "amount",
    "weight",
    "height",
    "temperature",
    "distance",
    "time",
    "duration",
    "rate",
    "value",
    "revenue",
    "cost",
    "balance",
    "percentage",
    "percent",
    "age",
    "sales",
    "profit",
    "expense",
    "measurement"
]


ANOMALY_KEYWORDS = [
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
    "fault",
    "error"
]


SEQUENCE_KEYWORDS = [
    "sequence",
    "sequential",
    "timestamp",
    "datetime",
    "date",
    "time",
    "step",
    "timestep",
    "time_step",
    "event",
    "event_id",
    "session",
    "session_id",
    "sequence_id",
    "order",
    "position"
]


TEXT_KEYWORDS = [
    "text",
    "document",
    "documents",
    "content",
    "description",
    "message",
    "comment",
    "comments",
    "review",
    "reviews",
    "article",
    "articles",
    "paragraph",
    "sentence",
    "abstract",
    "body",
    "title",
    "news",
    "tweet",
    "post"
]


# =========================================================
# Helper Functions
# =========================================================

def _normalized_column_names(df):
    """
    Return normalized dataframe column names.
    """

    return [
        str(column).strip().lower()
        for column in df.columns
    ]


def _contains_keyword(column_names, keywords):
    """
    Check whether any column name contains
    one of the supplied keywords.
    """

    for column in column_names:

        for keyword in keywords:

            if keyword in column:
                return True

    return False


def _get_text_columns(df):
    """
    Identify possible text columns.
    """

    text_columns = []

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

        sample = values.astype(str)

        average_length = sample.str.len().mean()

        unique_ratio = (
            sample.nunique() / len(sample)
            if len(sample) > 0
            else 0
        )

        column_name = (
            str(column)
            .strip()
            .lower()
        )

        named_text = any(
            keyword in column_name
            for keyword in TEXT_KEYWORDS
        )

        long_text = average_length >= 30

        high_unique_text = (
            average_length >= 15
            and
            unique_ratio >= 0.30
        )

        if (
            named_text
            or
            long_text
            or
            high_unique_text
        ):

            text_columns.append(column)

    return text_columns


def _get_numeric_columns(df):
    """
    Return numeric dataframe columns.
    """

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def _get_datetime_columns(df):
    """
    Identify datetime columns.
    """

    datetime_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):

            datetime_columns.append(column)

    return datetime_columns


def _has_sequence_structure(df):
    """
    Determine whether the dataset has characteristics
    associated with sequential/time-series data.
    """

    columns = _normalized_column_names(df)

    if _contains_keyword(
        columns,
        SEQUENCE_KEYWORDS
    ):
        return True

    datetime_columns = _get_datetime_columns(df)

    if datetime_columns:
        return True

    return False


def _has_anomaly_indicators(df):
    """
    Determine whether the dataset contains explicit
    anomaly/outlier-related indicators.
    """

    columns = _normalized_column_names(df)

    return _contains_keyword(
        columns,
        ANOMALY_KEYWORDS
    )


# =========================================================
# Target Cardinality
# =========================================================

def target_cardinality(df, target_column):
    """
    Return target cardinality information.

    This is used by the benchmark modules to determine
    whether a target is appropriate for classification.
    """

    if (
        df is None
        or df.empty
        or target_column not in df.columns
    ):

        return {
            "valid": False,
            "sample_count": 0,
            "unique_count": 0,
            "unique_ratio": 1.0
        }

    y = df[target_column].dropna()

    sample_count = int(len(y))
    unique_count = int(y.nunique())

    unique_ratio = (
        unique_count / sample_count
        if sample_count > 0
        else 1.0
    )

    return {
        "valid": sample_count > 0,
        "sample_count": sample_count,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio
    }


def is_high_cardinality_target(
    df,
    target_column,
    threshold=0.50
):
    """
    Determine whether a target has an unusually high
    number of unique values.

    Example:

        950 unique values / 1000 samples = 95%

    Such a target should not be passed to a normal
    classification benchmark.
    """

    information = target_cardinality(
        df,
        target_column
    )

    return (
        information["valid"]
        and
        information["unique_ratio"] > threshold
    )


# =========================================================
# Detect Unsupervised / Specialized Task
# =========================================================

def detect_unsupervised_task(df):
    """
    Detect an appropriate task when a supervised target
    is not selected.
    """

    if df is None or df.empty:

        return CLUSTERING

    columns = _normalized_column_names(df)

    numeric_columns = _get_numeric_columns(df)

    text_columns = _get_text_columns(df)

    # =====================================================
    # 1. Sequence Modeling
    # =====================================================

    if _has_sequence_structure(df):

        return SEQUENCE_MODELING

    # =====================================================
    # 2. Text-based Tasks
    # =====================================================

    if len(text_columns) >= 2:

        return TEXT_SIMILARITY

    if len(text_columns) >= 1:

        return TOPIC_MODELING

    # =====================================================
    # 3. Anomaly Detection
    # =====================================================

    if _has_anomaly_indicators(df):

        return ANOMALY_DETECTION

    # =====================================================
    # 4. Dimensionality Reduction
    # =====================================================

    numeric_count = len(numeric_columns)

    total_columns = len(df.columns)

    if numeric_count >= 10:

        return DIMENSIONALITY_REDUCTION

    if (
        total_columns >= 10
        and
        numeric_count >= 5
    ):

        return DIMENSIONALITY_REDUCTION

    # =====================================================
    # 5. Default
    # =====================================================

    return CLUSTERING


# =========================================================
# Main Task Detector
# =========================================================

def detect_task(df, target_column=None):
    """
    Detect the Machine Learning task.

    Returns one of:

        classification
        regression
        clustering
        anomaly_detection
        dimensionality_reduction
        topic_modeling
        text_similarity
        sequence_modeling
    """

    if df is None or df.empty:

        return CLUSTERING

    if target_column is None:

        return detect_unsupervised_task(df)

    if target_column not in df.columns:

        return detect_unsupervised_task(df)

    y = df[target_column].dropna()

    if len(y) == 0:

        return detect_unsupervised_task(df)

    unique_count = int(y.nunique())

    sample_count = int(len(y))

    dtype = y.dtype

    unique_ratio = (
        unique_count / sample_count
        if sample_count > 0
        else 1.0
    )

    target_name = (
        str(target_column)
        .strip()
        .lower()
    )

    # =====================================================
    # Explicit Anomaly Detection
    # =====================================================

    if any(
        keyword in target_name
        for keyword in ANOMALY_KEYWORDS
    ):

        return ANOMALY_DETECTION

    # =====================================================
    # Explicit Sequence Modeling
    # =====================================================

    if any(
        keyword in target_name
        for keyword in SEQUENCE_KEYWORDS
    ):

        return SEQUENCE_MODELING

    # =====================================================
    # Text Target
    # =====================================================

    if (
        pd.api.types.is_object_dtype(dtype)
        or
        pd.api.types.is_string_dtype(dtype)
    ):

        values = y.astype(str)

        average_length = (
            values.str.len().mean()
        )

        if average_length >= 30:

            return TOPIC_MODELING

        return CLASSIFICATION

    # =====================================================
    # Boolean Target
    # =====================================================

    if pd.api.types.is_bool_dtype(dtype):

        return CLASSIFICATION

    # =====================================================
    # Categorical Target
    # =====================================================

    if isinstance(
        dtype,
        pd.CategoricalDtype
    ):

        return CLASSIFICATION

    # =====================================================
    # Numeric Target
    # =====================================================

    if pd.api.types.is_numeric_dtype(dtype):

        # -----------------------------------------------
        # Explicit regression names
        # -----------------------------------------------

        if any(
            keyword in target_name
            for keyword in REGRESSION_KEYWORDS
        ):

            return REGRESSION

        # -----------------------------------------------
        # Small numeric class target
        #
        # Example:
        # 0 / 1
        # 1 / 2 / 3
        # 1 / 2 / 3 / 4 / 5
        # -----------------------------------------------

        if (
            unique_count <= 10
            and
            unique_ratio <= 0.20
        ):

            return CLASSIFICATION

        # -----------------------------------------------
        # Moderate numeric class target
        # -----------------------------------------------

        if (
            unique_count <= 20
            and
            unique_ratio <= 0.10
        ):

            return CLASSIFICATION

        # -----------------------------------------------
        # High-cardinality numeric target
        #
        # More than 50% unique values means that the
        # target should not be treated as classification.
        # -----------------------------------------------

        if unique_ratio > 0.50:

            return REGRESSION

        # -----------------------------------------------
        # Remaining numeric targets
        # -----------------------------------------------

        return REGRESSION

    # =====================================================
    # Fallback
    # =====================================================

    return CLASSIFICATION


# =========================================================
# Dataset Information
# =========================================================

def dataset_information(df):
    """
    Return basic dataset information.
    """

    return {

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "numeric_columns":
            int(
                len(
                    df.select_dtypes(
                        include=np.number
                    ).columns
                )
            ),

        "categorical_columns":
            int(
                len(
                    df.select_dtypes(
                        exclude=np.number
                    ).columns
                )
            ),

        "text_columns":
            int(
                len(
                    _get_text_columns(df)
                )
            ),

        "datetime_columns":
            int(
                len(
                    _get_datetime_columns(df)
                )
            )

    }


# =========================================================
# Task Reason
# =========================================================

def task_reason(df, target_column=None):
    """
    Explain why the task was selected.
    """

    task = detect_task(
        df,
        target_column
    )

    if task == CLASSIFICATION:

        information = target_cardinality(
            df,
            target_column
        )

        if (
            information["valid"]
            and
            information["unique_ratio"] <= 0.20
        ):

            return (
                "The target column contains a relatively "
                "small number of repeated classes."
            )

        return (
            "The target column contains categorical "
            "values suitable for classification."
        )

    if task == REGRESSION:

        information = target_cardinality(
            df,
            target_column
        )

        if (
            information["valid"]
            and
            information["unique_ratio"] > 0.50
        ):

            return (
                "The target column contains numeric values "
                "with a high number of unique observations "
                "and is therefore treated as a regression target."
            )

        return (
            "The target column contains numeric values "
            "that are treated as continuous measurements."
        )

    if task == CLUSTERING:

        return (
            "No supervised target was selected and the "
            "dataset is suitable for grouping similar records."
        )

    if task == ANOMALY_DETECTION:

        return (
            "The dataset contains characteristics associated "
            "with detecting unusual, abnormal, or outlier records."
        )

    if task == DIMENSIONALITY_REDUCTION:

        return (
            "The dataset contains multiple numeric features "
            "and can be transformed into a lower-dimensional representation."
        )

    if task == TOPIC_MODELING:

        return (
            "The dataset contains text or document-like data "
            "that can be analyzed for underlying topics."
        )

    if task == TEXT_SIMILARITY:

        return (
            "The dataset contains multiple text fields that "
            "can be compared using text similarity techniques."
        )

    if task == SEQUENCE_MODELING:

        return (
            "The dataset contains temporal, sequential, or "
            "ordered observations suitable for sequence modeling."
        )

    return (
        "No suitable task could be determined."
    )