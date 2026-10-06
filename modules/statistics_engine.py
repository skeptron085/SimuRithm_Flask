import pandas as pd
import numpy as np


def statistics_summary(df):
    """
    Generate dataset statistics.

    Returns
    -------
    dict
    """

    rows = int(df.shape[0])
    columns = int(df.shape[1])

    missing_values = int(df.isnull().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    numeric_count = len(numeric_columns)

    categorical_count = len(categorical_columns)

    memory_usage = round(
        df.memory_usage(deep=True).sum()
        / 1024,
        2
    )

    missing_percent = round(
        (missing_values / (rows * columns)) * 100,
        2
    ) if rows > 0 and columns > 0 else 0

    duplicate_percent = round(
        (duplicate_rows / rows) * 100,
        2
    ) if rows > 0 else 0

    return {

        "rows": rows,

        "columns": columns,

        "missing_values": missing_values,

        "duplicate_rows": duplicate_rows,

        "missing_percent": missing_percent,

        "duplicate_percent": duplicate_percent,

        "numeric_columns": numeric_count,

        "categorical_columns": categorical_count,

        "memory_usage_kb": memory_usage

    }


def statistics_table(df):
    """
    Returns per-column statistics.
    """

    table = []

    for column in df.columns:

        dtype = str(df[column].dtype)

        missing = int(
            df[column].isnull().sum()
        )

        unique = int(
            df[column].nunique()
        )

        sample = ""

        try:
            sample = str(
                df[column]
                .dropna()
                .iloc[0]
            )

        except:
            sample = ""

        record = {

            "column": column,

            "datatype": dtype,

            "missing": missing,

            "unique": unique,

            "sample": sample

        }

        # Numeric statistics

        if pd.api.types.is_numeric_dtype(df[column]):

            record["mean"] = round(
                float(df[column].mean()),
                4
            )

            record["median"] = round(
                float(df[column].median()),
                4
            )

            record["std"] = round(
                float(df[column].std()),
                4
            )

            record["min"] = round(
                float(df[column].min()),
                4
            )

            record["max"] = round(
                float(df[column].max()),
                4
            )

        else:

            record["mean"] = None
            record["median"] = None
            record["std"] = None
            record["min"] = None
            record["max"] = None

        table.append(record)

    return table