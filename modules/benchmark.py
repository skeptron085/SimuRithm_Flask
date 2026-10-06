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
# Classification Benchmark
# =========================================================

def run_benchmark(
    df,
    target,
    selected_algorithm
):
    """
    Run one selected classification algorithm.

    The function validates the target before allowing
    sklearn classification models to process it.
    """

    # --------------------------------------------------
    # Validate Target Column
    # --------------------------------------------------

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found."
        )

    # --------------------------------------------------
    # Allowed Classification Algorithms
    # --------------------------------------------------

    allowed_algorithms = {

        "Logistic Regression",

        "Decision Tree",

        "Random Forest",

        "KNN",

        "Naive Bayes",

        "SVM"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported classification algorithm: "
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
    # Validate Sample Count
    # --------------------------------------------------

    if len(y) < 2:

        raise ValueError(
            "Not enough valid rows to perform "
            "classification benchmarking."
        )

    # --------------------------------------------------
    # Target Cardinality Check
    # --------------------------------------------------

    cardinality = target_cardinality(
        pd.DataFrame({
            target: y
        }),
        target
    )

    unique_count = cardinality["unique_count"]

    unique_ratio = cardinality["unique_ratio"]

    # --------------------------------------------------
    # Prevent Regression Target From Entering
    # Classification Benchmark
    # --------------------------------------------------

    detected_task = detect_task(
        pd.DataFrame({
            target: y
        }),
        target
    )

    if detected_task != CLASSIFICATION:

        raise ValueError(
            f"Target '{target}' was detected as "
            f"'{detected_task}', not classification. "
            f"The classification benchmark was stopped "
            f"to prevent an inappropriate sklearn model."
        )

    # --------------------------------------------------
    # Explicit High-Cardinality Protection
    # --------------------------------------------------

    if unique_ratio > 0.50:

        raise ValueError(
            f"Target '{target}' has "
            f"{unique_count} unique values out of "
            f"{len(y)} samples "
            f"({unique_ratio:.1%} unique). "
            "This target is too high-cardinality "
            "for the classification benchmark."
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
    # Encode Categorical Target
    # --------------------------------------------------

    label_encoder = None

    if (
        y.dtype == object
        or
        str(y.dtype).startswith("category")
        or
        pd.api.types.is_string_dtype(y)
    ):

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

    # --------------------------------------------------
    # Validate Class Distribution
    # --------------------------------------------------

    class_counts = pd.Series(
        y
    ).value_counts()

    if len(class_counts) < 2:

        raise ValueError(
            "Classification requires at least "
            "two different target classes."
        )

    # --------------------------------------------------
    # Determine Whether Stratification Is Safe
    # --------------------------------------------------

    minimum_class_count = int(
        class_counts.min()
    )

    can_stratify = (
        len(class_counts) >= 2
        and
        minimum_class_count >= 2
    )

    # --------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Classification Algorithms
    # --------------------------------------------------

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
                random_state=42
            ),

        "KNN":

            KNeighborsClassifier(),

        "Naive Bayes":

            GaussianNB(),

        "SVM":

            SVC()

    }

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
        # Classification Metrics
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Cross Validation
        # --------------------------------------------------

        sample_count = len(y)

        cv_score = None

        if sample_count >= 2:

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

                cv_score = round(
                    float(scores.mean()),
                    4
                )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        result = {

            "algorithm":
                selected_algorithm,

            "accuracy":
                round(
                    float(accuracy),
                    4
                ),

            "precision":
                round(
                    float(precision),
                    4
                ),

            "recall":
                round(
                    float(recall),
                    4
                ),

            "f1_score":
                round(
                    float(f1),
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

            "error":
                str(e)

        }]