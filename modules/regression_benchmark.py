
"""
SimuRithm — Regression Benchmark
Release 1.3

Responsibilities:
- Benchmark suitable regression algorithms.
- Handle missing values and mixed feature types.
- Evaluate models with RMSE, MAE, R², and cross-validation.
- Return consistent results for recommender.py.

Main interface:
    run_regression_benchmark(df, target, selected_algorithm=None)
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor


RANDOM_STATE = 42
MIN_ROWS = 5
MAX_CV_FOLDS = 5

REGRESSION_ALGORITHMS = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(),

    "Lasso": Lasso(
        max_iter=5000,
        random_state=RANDOM_STATE,
    ),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=RANDOM_STATE,
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "SVR": SVR(),
}


# =========================================================
# Helpers
# =========================================================

def _result(
    algorithm,
    rmse=None,
    mae=None,
    r2=None,
    cv_score=None,
    execution_time=None,
    error=None,
):
    """Build a consistent benchmark result."""

    result = {
        "algorithm": algorithm,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_score": cv_score,
        "execution_time": execution_time,
    }

    if error:
        result["error"] = str(error)

    return result


def _round_metric(value, decimals=4):
    """Return a finite rounded metric or None."""

    if value is None:
        return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return round(value, decimals)

    except (TypeError, ValueError, OverflowError):
        return None


def _make_preprocessor(X):
    """Create preprocessing fitted only inside the model pipeline."""

    numeric_features = X.select_dtypes(
        include=["number"],
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"],
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical input features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _make_cv(y):
    """Choose valid shuffled K-fold cross-validation."""

    sample_count = len(y)

    if sample_count < 3:
        return None

    folds = min(MAX_CV_FOLDS, sample_count)

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# =========================================================
# Main Regression Benchmark
# =========================================================

def run_regression_benchmark(
    df,
    target,
    selected_algorithm=None,
):
    """
    Benchmark regression algorithms.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing input features and the target.

    target : str
        Name of the numeric target column.

    selected_algorithm : str or None
        If None, benchmark every configured algorithm.
        Otherwise, benchmark only the requested algorithm.

    Returns
    -------
    list[dict]
        Benchmark results sorted by R², then RMSE.
        Failed algorithms remain in the result with an error.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Regression benchmarking requires a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The uploaded dataset contains no rows."
        )

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' was not found."
        )

    if selected_algorithm is not None:
        if selected_algorithm not in REGRESSION_ALGORITHMS:
            raise ValueError(
                f"Unsupported regression algorithm: "
                f"{selected_algorithm}. Supported algorithms: "
                f"{', '.join(REGRESSION_ALGORITHMS)}"
            )

        algorithms = {
            selected_algorithm: REGRESSION_ALGORITHMS[
                selected_algorithm
            ]
        }
    else:
        algorithms = REGRESSION_ALGORITHMS

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Treat infinity as missing for numeric columns.
    numeric_X = X.select_dtypes(
        include=["number"],
    ).columns

    if len(numeric_X):
        X.loc[:, numeric_X] = X.loc[:, numeric_X].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    if pd.api.types.is_numeric_dtype(y):
        y = y.replace([np.inf, -np.inf], np.nan)
    else:
        raise ValueError(
            f"Target '{target}' must be numeric for regression."
        )

    # Drop rows without a usable target.
    valid_target = y.notna()

    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    # Do not treat a target column containing only one
    # distinct value as a meaningful regression problem.
    if y.nunique(dropna=True) < 2:
        raise ValueError(
            "The regression target must contain at least "
            "two distinct numeric values."
        )

    if len(y) < MIN_ROWS:
        raise ValueError(
            f"At least {MIN_ROWS} valid target rows are required "
            "for regression benchmarking."
        )

    # Remove columns that are completely missing.
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError(
            "No usable input features remain after removing "
            "the target and empty columns."
        )

    # Remove non-finite values from the numeric target.
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(
            "The target contains invalid numeric values."
        )

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    cv = _make_cv(y)

    results = []

    # -----------------------------------------------------
    # Benchmark each candidate
    # -----------------------------------------------------

    for algorithm_name, estimator in algorithms.items():

        started = time.perf_counter()

        try:
            preprocessor = _make_preprocessor(X)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            # Fit and evaluate on the held-out test set.
            pipeline.fit(
                X_train,
                y_train,
            )

            predictions = pipeline.predict(X_test)

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            # -------------------------------------------------
            # Cross-validation
            # -------------------------------------------------

            cv_score = None

            if cv is not None:
                cv_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", _make_preprocessor(X)),
                        ("model", estimator),
                    ]
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    cv_values = cross_val_score(
                        cv_pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="r2",
                        n_jobs=1,
                        error_score=np.nan,
                    )

                finite_scores = cv_values[
                    np.isfinite(cv_values)
                ]

                if len(finite_scores):
                    cv_score = _round_metric(
                        np.mean(finite_scores)
                    )

            elapsed = time.perf_counter() - started

            results.append(
                _result(
                    algorithm=algorithm_name,
                    rmse=_round_metric(rmse),
                    mae=_round_metric(mae),
                    r2=_round_metric(r2),
                    cv_score=cv_score,
                    execution_time=_round_metric(elapsed),
                )
            )

        except Exception as exc:
            elapsed = time.perf_counter() - started

            results.append(
                _result(
                    algorithm=algorithm_name,
                    execution_time=_round_metric(elapsed),
                    error=str(exc),
                )
            )

    # -----------------------------------------------------
    # Rank successful models
    # -----------------------------------------------------

    # Higher R² is better. RMSE breaks ties.
    # Failed models are retained at the end for transparency.
    results.sort(
        key=lambda item: (
            item.get("error") is None,
            (
                item.get("r2")
                if item.get("r2") is not None
                else -float("inf")
            ),
            -(
                item.get("rmse")
                if item.get("rmse") is not None
                else float("inf")
            ),
        ),
        reverse=True,
    )

    return results
