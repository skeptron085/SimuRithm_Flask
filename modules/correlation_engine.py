import numpy as np
import pandas as pd


def correlation_matrix(df):
    """
    Returns Pearson Correlation Matrix
    as JSON-friendly dictionary.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:

        return {
            "columns": [],
            "matrix": []
        }

    corr = numeric_df.corr(
        method="pearson"
    )

    corr = corr.round(4)

    return {

        "columns": corr.columns.tolist(),

        "matrix": corr.values.tolist()

    }


def strongest_correlations(df, top=10):
    """
    Returns the strongest positive
    and negative correlations.

    The original correlation sign is preserved.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:

        return []

    corr = numeric_df.corr(
        method="pearson"
    )

    pairs = []

    cols = corr.columns

    for i in range(len(cols)):

        for j in range(i + 1, len(cols)):

            value = corr.iloc[i, j]

            if pd.isna(value):

                continue

            pairs.append({

                "feature_1": cols[i],

                "feature_2": cols[j],

                "correlation": round(
                    float(value),
                    4
                )

            })

    pairs = sorted(

        pairs,

        key=lambda x: abs(
            x["correlation"]
        ),

        reverse=True

    )

    return pairs[:top] 