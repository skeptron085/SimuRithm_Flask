import os
import pandas as pd


SUPPORTED_EXTENSIONS = [
    ".csv",
    ".xlsx",
    ".xls"
]


def load_dataset(file_path):
    """
    Load a dataset and return a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    if extension == ".csv":

        # Automatically detect encoding
        encodings = [
            "utf-8",
            "utf-8-sig",
            "latin1",
            "cp1252"
        ]

        last_error = None

        for enc in encodings:

            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc
                )
                break

            except Exception as e:
                last_error = e

        else:
            raise last_error

    else:

        engine = "xlrd" if extension == ".xls" else "openpyxl"

        df = pd.read_excel(
            file_path,
            engine=engine
        )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Trim column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    return df


def dataset_preview(df, rows=10):
    """
    Return the first rows of the dataset.
    """

    return df.head(rows).to_dict(
        orient="records"
    )


def dataset_columns(df):
    """
    Return all column names.
    """

    return list(df.columns)