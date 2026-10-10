import re
import nltk
import spacy
import pandas as pd

# Download tokenizer only once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise Exception(
        "spaCy model not found.\n"
        "Run:\n"
        "python -m spacy download en_core_web_sm"
    )


# =========================================================
# SimuRithm NLP Analyzer
# =========================================================

TARGET_KEYWORDS = {

    "target",
    "label",
    "class",
    "category",
    "output",
    "result",
    "outcome",
    "status",
    "grade",
    "score",
    "passed",
    "prediction",
    "response"
}


IDENTIFIER_KEYWORDS = {

    "id",
    "identifier",
    "studentid",
    "student_id",
    "userid",
    "user_id",
    "recordid",
    "record_id"
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


# =========================================================
# Tokenization
# =========================================================

def tokenize_header(header):

    return re.findall(
        r"[a-zA-Z0-9]+",
        str(header)
        .strip()
        .lower()
    )


# =========================================================
# Value Type
# =========================================================

def detect_value_type(series):

    if pd.api.types.is_bool_dtype(series):

        return "Boolean"

    if pd.api.types.is_numeric_dtype(series):

        return "Numeric"

    if pd.api.types.is_datetime64_any_dtype(series):

        return "Datetime"

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    if len(values) == 0:

        return "Unknown"

    has_alpha = values.str.contains(
        r"[A-Za-z]",
        regex=True
    ).any()

    has_digit = values.str.contains(
        r"\d",
        regex=True
    ).any()

    if has_alpha and has_digit:

        return "Alphanumeric"

    if has_alpha:

        return "Text"

    if has_digit:

        return "Numeric"

    return "Unknown"


# =========================================================
# Role Detection
# =========================================================

def detect_role(
    column,
    series,
    value_type
):

    name = str(column).strip().lower()

    tokens = tokenize_header(column)

    # -----------------------------------------------------
    # Identifier
    # -----------------------------------------------------

    if (
        any(
            keyword in tokens
            for keyword in IDENTIFIER_KEYWORDS
        )
        or
        name.endswith("_id")
    ):

        return "Identifier"

    # -----------------------------------------------------
    # Target Candidate
    # -----------------------------------------------------

    if any(
        keyword in tokens
        for keyword in TARGET_KEYWORDS
    ):

        return "Target Candidate"

    # -----------------------------------------------------
    # Text
    # -----------------------------------------------------

    if value_type == "Text":

        return "Text"

    if any(
        keyword in tokens
        for keyword in TEXT_KEYWORDS
    ):

        return "Text"

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    if value_type == "Numeric":

        return "Numeric"

    # -----------------------------------------------------
    # Boolean
    # -----------------------------------------------------

    if value_type == "Boolean":

        return "Categorical"

    # -----------------------------------------------------
    # Categorical
    # -----------------------------------------------------

    if value_type == "Alphanumeric":

        return "Categorical"

    return "Unknown"


# =========================================================
# Header Analysis
# =========================================================

def analyze_headers(df):

    results = []

    for column in df.columns:

        series = df[column]

        value_type = detect_value_type(
            series
        )

        role = detect_role(
            column,
            series,
            value_type
        )

        results.append({

            "column":
                str(column),

            "tokens":
                tokenize_header(column),

            "data_type":
                value_type,

            "role":
                role,

            "missing_values":
                int(series.isna().sum()),

            "unique_values":
                int(series.nunique(dropna=True))

        })

    return results


# =========================================================
# Target Suggestion
# =========================================================

def suggest_target(df):

    candidates = []

    # -----------------------------------------------------
    # First priority:
    # semantic target candidates
    # -----------------------------------------------------

    for column in df.columns:

        tokens = tokenize_header(
            column
        )

        if any(
            keyword in tokens
            for keyword in TARGET_KEYWORDS
        ):

            candidates.append(column)

    if candidates:

        return candidates

    # -----------------------------------------------------
    # Second priority:
    # non-identifier categorical column
    # -----------------------------------------------------

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_bool_dtype(series):

            return [column]

        if (
            pd.api.types.is_object_dtype(series)
            or
            pd.api.types.is_categorical_dtype(series)
        ):

            unique_count = (
                series.nunique(
                    dropna=True
                )
            )

            if (
                unique_count >= 2
                and
                unique_count <= 20
            ):

                return [column]

    # -----------------------------------------------------
    # Final fallback:
    # last non-ID column
    # -----------------------------------------------------

    usable_columns = []

    for column in df.columns:

        role = detect_role(
            column,
            df[column],
            detect_value_type(df[column])
        )

        if role != "Identifier":

            usable_columns.append(
                column
            )

    if usable_columns:

        return [
            usable_columns[-1]
        ]

    return []