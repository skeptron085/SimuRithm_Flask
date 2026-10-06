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


# ==========================================================
# Target Keywords
# ==========================================================

TARGET_KEYWORDS = [

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
    "pass",
    "prediction",
    "response",
    "gender"

]


# ==========================================================
# Tokenize Header
# ==========================================================

def tokenize_header(header):

    header = str(header).strip().lower()

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        header
    )

    return tokens


# ==========================================================
# Detect Value Type
# ==========================================================

def detect_value_type(series):

    # ------------------------------------------------------
    # Numeric
    # ------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        return "Numeric"

    # ------------------------------------------------------
    # Remove missing values
    # ------------------------------------------------------

    values = (

        series
        .dropna()
        .astype(str)
        .str.strip()

    )

    if len(values) == 0:

        return "Unknown"

    # ------------------------------------------------------
    # Check values
    # ------------------------------------------------------

    has_alpha = values.str.contains(
        r"[A-Za-z]",
        regex=True
    ).any()

    has_digit = values.str.contains(
        r"\d",
        regex=True
    ).any()

    # ------------------------------------------------------
    # Alphabetic
    # ------------------------------------------------------

    if has_alpha and not has_digit:

        return "Alphabetic"

    # ------------------------------------------------------
    # Alphanumeric
    # ------------------------------------------------------

    if has_alpha and has_digit:

        return "Alphanumeric"

    # ------------------------------------------------------
    # Numeric stored as text
    # ------------------------------------------------------

    if not has_alpha and has_digit:

        return "Numeric"

    return "Unknown"


# ==========================================================
# Detect Semantic Role
# ==========================================================

def detect_role(
    column,
    series,
    value_type
):

    column_lower = str(
        column
    ).strip().lower()

    tokens = tokenize_header(
        column_lower
    )

    # ------------------------------------------------------
    # Target Candidate
    # ------------------------------------------------------

    for keyword in TARGET_KEYWORDS:

        if keyword in tokens:

            return "Target Candidate"

    # ------------------------------------------------------
    # Identifier
    # ------------------------------------------------------

    identifier_keywords = [

        "id",
        "identifier",
        "studentid",
        "student_id",
        "userid",
        "user_id",
        "recordid",
        "record_id"

    ]

    for keyword in identifier_keywords:

        if keyword in column_lower:

            return "Identifier"

    # ------------------------------------------------------
    # Numeric
    # ------------------------------------------------------

    if value_type == "Numeric":

        return "Numeric"

    # ------------------------------------------------------
    # Alphabetic
    # ------------------------------------------------------

    if value_type == "Alphabetic":

        return "Text"

    # ------------------------------------------------------
    # Alphanumeric
    # ------------------------------------------------------

    if value_type == "Alphanumeric":

        return "Categorical"

    # ------------------------------------------------------
    # Unknown
    # ------------------------------------------------------

    return "Unknown"


# ==========================================================
# Analyze Headers
# ==========================================================

def analyze_headers(df):

    results = []

    for column in df.columns:

        series = df[column]

        tokens = tokenize_header(
            column
        )

        value_type = detect_value_type(
            series
        )

        role = detect_role(

            column,

            series,

            value_type

        )

        results.append({

            "column": str(column),

            "tokens": tokens,

            "data_type": value_type,

            "role": role

        })

    return results


# ==========================================================
# Suggest Target
# ==========================================================

def suggest_target(df):

    candidates = []

    # ------------------------------------------------------
    # Check semantic header keywords
    # ------------------------------------------------------

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        tokens = tokenize_header(
            column_lower
        )

        for keyword in TARGET_KEYWORDS:

            if keyword in tokens:

                candidates.append(
                    column
                )

                break

    # ------------------------------------------------------
    # Return semantic candidates first
    # ------------------------------------------------------

    if candidates:

        return candidates

    # ------------------------------------------------------
    # Fallback:
    # Last column
    # ------------------------------------------------------

    if len(df.columns) > 0:

        return [
            df.columns[-1]
        ]

    return []