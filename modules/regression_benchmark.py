import time

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.neighbors import (
    KNeighborsRegressor
)

from sklearn.svm import (
    SVR
)

from modules.task_detector import (
    detect_task,
    REGRESSION
)


# =========================================================
# Regression Benchmark
# =========================================================

def run_regression_benchmark(
    df,
    target,
    selected_algorithm
):
    """
    Run one selected regression algorithm.
    """

    # --------------------------------------------------
    # Validate Target
    # --------------------------------------------------

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found."
        )

    # --------------------------------------------------
    # Allowed Regression Algorithms
    # --------------------------------------------------

    allowed_algorithms = {

        "Linear Regression",

        "Ridge",

        "Lasso",

        "Decision Tree Regressor",

        "Random Forest Regressor",

        "SVR",

        "Decision Tree",

        "Random Forest",

        "KNN",

        "Support Vector Regression"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported regression algorithm: "
            f"'{selected_algorithm}'. "
            f"Allowed algorithms are: "
            f"{', '.join(sorted(allowed_algorithms))}"
        )

    # --------------------------------------------------
    # Prepare X and Y
    # --------------------------------------------------

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    # --------------------------------------------------
    # Remove Missing Target Values
    # --------------------------------------------------

    mask = y.notna()

    X = X.loc[mask]

    y = y.loc[mask]

    # --------------------------------------------------
    # Validate Dataset
    # --------------------------------------------------

    if len(y) < 2:

        raise ValueError(
            "Not enough valid rows to perform "
            "regression benchmarking."
        )

    # --------------------------------------------------
    # Validate Numeric Regression Target
    # --------------------------------------------------

    if not pd.api.types.is_numeric_dtype(y):

        raise ValueError(
            f"Target '{target}' is not numeric and "
            "cannot be processed by the regression benchmark."
        )

    # --------------------------------------------------
    # Verify Task Detection
    # --------------------------------------------------

    detected_task = detect_task(
        pd.DataFrame({
            target: y
        }),
        target
    )

    if detected_task != REGRESSION:

        raise ValueError(
            f"Target '{target}' was detected as "
            f"'{detected_task}', not regression."
        )

    # --------------------------------------------------
    # Numeric / Categorical Features
    # --------------------------------------------------

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------
    # Numeric Preprocessing
    # --------------------------------------------------

    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )

        ]
    )

    # --------------------------------------------------
    # Categorical Preprocessing
    # --------------------------------------------------

    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]
    )

    # --------------------------------------------------
    # Column Transformer
    # --------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",

                numeric_transformer,

                numeric_features
            ),

            (
                "cat",

                categorical_transformer,

                categorical_features
            )

        ]
    )

    # --------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42

        )
    )

    # --------------------------------------------------
    # Regression Algorithms
    # --------------------------------------------------

    algorithms = {

        "Linear Regression":

            LinearRegression(),

        "Ridge":

            Ridge(),

        "Lasso":

            Lasso(
                max_iter=5000
            ),

        "Decision Tree":

            DecisionTreeRegressor(
                random_state=42
            ),

        "Decision Tree Regressor":

            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":

            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

        "Random Forest Regressor":

            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

        "KNN":

            KNeighborsRegressor(),

        "Support Vector Regression":

            SVR(),

        "SVR":

            SVR()

    }

    # --------------------------------------------------
    # Select Requested Algorithm
    # --------------------------------------------------

    model = algorithms[
        selected_algorithm
    ]

    # --------------------------------------------------
    # Build Pipeline
    # --------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                model
            )

        ]
    )

    # --------------------------------------------------
    # Execute Benchmark
    # --------------------------------------------------

    try:

        start = time.time()

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        end = time.time()

        # --------------------------------------------------
        # Cross Validation
        # --------------------------------------------------

        sample_count = len(y)

        cv_score = None

        if sample_count >= 2:

            cv_folds = min(
                5,
                sample_count
            )

            if cv_folds >= 2:

                cv = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv_folds,
                    scoring="r2"
                )

                cv_score = round(
                    float(cv.mean()),
                    4
                )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        rmse = (
            mean_squared_error(
                y_test,
                predictions
            ) ** 0.5
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        result = {

            "algorithm":
                selected_algorithm,

            "rmse":
                round(
                    float(rmse),
                    4
                ),

            "mae":
                round(
                    float(mae),
                    4
                ),

            "r2":
                round(
                    float(r2),
                    4
                ),

            "cv_score":
                cv_score,

            "execution_time":
                round(
                    end - start,
                    4
                )

        }

        return [result]

    # --------------------------------------------------
    # Error Handling
    # --------------------------------------------------

    except Exception as e:

        return [{

            "algorithm":
                selected_algorithm,

            "rmse":
                None,

            "mae":
                None,

            "r2":
                None,

            "cv_score":
                None,

            "execution_time":
                None,

            "error":
                str(e)

        }]