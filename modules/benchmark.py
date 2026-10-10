import time

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from modules.task_detector import (
    detect_task,
    target_cardinality,
    CLASSIFICATION
)


# =========================================================
# Supported Algorithms
# =========================================================

CLASSIFICATION_ALGORITHMS = [

    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "Naive Bayes",
    "SVM"

]


# =========================================================
# Safe OneHotEncoder
# =========================================================

def _make_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# =========================================================
# Build Preprocessor
# =========================================================

def _build_preprocessor(X):

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

    transformers = []

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    if numeric_features:

        numeric_pipeline = Pipeline(

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

        transformers.append(

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )

        )

    # -----------------------------------------------------
    # Categorical
    # -----------------------------------------------------

    if categorical_features:

        categorical_pipeline = Pipeline(

            steps=[

                (
                    "imputer",

                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",

                    _make_encoder()
                )

            ]
        )

        transformers.append(

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        )

    if not transformers:

        raise ValueError(
            "No usable predictor columns were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# Build Algorithm
# =========================================================

def _build_algorithm(name):

    algorithms = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "KNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "SVM":
            SVC()

    }

    if name not in algorithms:

        raise ValueError(
            f"Unsupported classification algorithm: "
            f"{name}"
        )

    return algorithms[name]


# =========================================================
# Classification Benchmark
# =========================================================

def run_benchmark(
    df,
    target,
    selected_algorithm
):

    result = {

        "algorithm":
            selected_algorithm,

        "accuracy":
            None,

        "precision":
            None,

        "recall":
            None,

        "f1_score":
            None,

        "cv_score":
            None,

        "execution_time":
            None,

        "primary_metric":
            "accuracy",

        "primary_score":
            None,

        "higher_is_better":
            True,

        "error":
            None

    }

    start_time = time.time()

    try:

        # -------------------------------------------------
        # Validate target
        # -------------------------------------------------

        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' not found."
            )

        if (
            selected_algorithm
            not in CLASSIFICATION_ALGORITHMS
        ):

            raise ValueError(
                f"Unsupported classification "
                f"algorithm: {selected_algorithm}"
            )

        # -------------------------------------------------
        # Prepare data
        # -------------------------------------------------

        data = df.copy()

        data = data.dropna(
            subset=[target]
        )

        if len(data) < 4:

            raise ValueError(
                "Not enough valid rows for classification."
            )

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        # -------------------------------------------------
        # Validate task
        # -------------------------------------------------

        detected_task = detect_task(
            data,
            target
        )

        if detected_task != CLASSIFICATION:

            raise ValueError(
                f"Target '{target}' was detected as "
                f"'{detected_task}', not classification."
            )

        # -------------------------------------------------
        # Encode target
        # -------------------------------------------------

        if (
            y.dtype == object
            or
            pd.api.types.is_string_dtype(y)
            or
            str(y.dtype).startswith("category")
        ):

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y.astype(str)
            )

        else:

            y = y.to_numpy()

        # -------------------------------------------------
        # Class validation
        # -------------------------------------------------

        class_counts = (
            pd.Series(y)
            .value_counts()
        )

        if len(class_counts) < 2:

            raise ValueError(
                "Classification requires at least "
                "two target classes."
            )

        # -------------------------------------------------
        # Preprocessor
        # -------------------------------------------------

        preprocessor = _build_preprocessor(
            X
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model = _build_algorithm(
            selected_algorithm
        )

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

        # -------------------------------------------------
        # Stratification
        # -------------------------------------------------

        minimum_class_count = int(
            class_counts.min()
        )

        can_stratify = (
            minimum_class_count >= 2
        )

        # -------------------------------------------------
        # Train / Test Split
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,
                y,

                test_size=0.20,

                random_state=42,

                stratify=(
                    y
                    if can_stratify
                    else None
                )

            )
        )

        # -------------------------------------------------
        # Fit
        # -------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # -------------------------------------------------
        # Cross Validation
        # -------------------------------------------------

        cv_score = None

        cv_folds = min(
            5,
            minimum_class_count
        )

        if cv_folds >= 2:

            cv = StratifiedKFold(

                n_splits=cv_folds,

                shuffle=True,

                random_state=42

            )

            scores = cross_val_score(

                pipeline,

                X,

                y,

                cv=cv,

                scoring="accuracy"

            )

            cv_score = float(
                scores.mean()
            )

        # -------------------------------------------------
        # Select primary score
        #
        # Cross-validation is preferred because it
        # provides a more stable benchmark.
        # -------------------------------------------------

        primary_score = (

            cv_score
            if cv_score is not None
            else float(accuracy)

        )

        # -------------------------------------------------
        # Final result
        # -------------------------------------------------

        result.update({

            "accuracy":
                round(float(accuracy), 4),

            "precision":
                round(float(precision), 4),

            "recall":
                round(float(recall), 4),

            "f1_score":
                round(float(f1), 4),

            "cv_score":
                (
                    round(float(cv_score), 4)
                    if cv_score is not None
                    else None
                ),

            "primary_score":
                round(
                    float(primary_score),
                    4
                ),

            "execution_time":
                round(
                    time.time() - start_time,
                    4
                )

        })

        return [result]

    except Exception as error:

        result.update({

            "execution_time":
                round(
                    time.time() - start_time,
                    4
                ),

            "error":
                str(error)

        })

        return [result]