from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold
)

from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans
)

from sklearn.metrics import silhouette_score

import numpy as np


# ---------------------------------------------------------
# Task detector
# ---------------------------------------------------------

try:
    from task_detector import detect_task
except ImportError:
    detect_task = None


# =========================================================
# Utility
# =========================================================

def _dense_features(values):

    if hasattr(values, "toarray"):

        if values.shape[0] * values.shape[1] > 20_000_000:

            raise ValueError(
                "Naive Bayes feature matrix is too large to densify."
            )

        return values.toarray()

    return values


# =========================================================
# Classification CV
# =========================================================

def _get_classification_cv(y):
    """
    Determine whether stratified cross-validation is possible.

    Returns
    -------
    cv : StratifiedKFold or None
    cv_folds : int
    error : str or None
    """

    if len(y) < 2:

        return (
            None,
            0,
            "At least two labeled rows are required for classification."
        )

    # -----------------------------------------------------
    # Always calculate class counts BEFORE LabelEncoder
    # -----------------------------------------------------

    if hasattr(y, "value_counts"):

        class_counts = y.value_counts()

    else:

        _, counts = np.unique(
            np.asarray(y),
            return_counts=True
        )

        class_counts = counts

    if len(class_counts) < 2:

        return (
            None,
            0,
            "Classification requires at least two classes."
        )

    minimum_class_count = int(
        np.min(class_counts)
    )

    # -----------------------------------------------------
    # Stratified CV requires at least two examples
    # per class.
    # -----------------------------------------------------

    if minimum_class_count < 2:

        return (
            None,
            0,
            (
                "Stratified cross-validation is unavailable "
                "because the rarest class contains only "
                f"{minimum_class_count} sample. "
                "At least two samples are required in every class."
            )
        )

    # -----------------------------------------------------
    # Use at most 5 folds.
    # -----------------------------------------------------

    cv_folds = min(
        5,
        minimum_class_count
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    return (
        cv,
        cv_folds,
        None
    )


# =========================================================
# Regression CV
# =========================================================

def _get_regression_cv(y):
    """
    Determine appropriate KFold configuration.
    """

    sample_count = len(y)

    if sample_count < 3:

        return (
            None,
            0,
            "At least three labeled rows are required for regression tuning."
        )

    cv_folds = min(
        5,
        sample_count
    )

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    return (
        cv,
        cv_folds,
        None
    )


# =========================================================
# Clustering Optimization
# =========================================================

def _tune_clustering_model(frame, algorithm):

    if len(frame) < 4:

        return {
            "success": False,
            "algorithm": algorithm,
            "error": (
                "At least four rows are required "
                "for clustering optimization."
            )
        }

    numeric_features = frame.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = frame.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "num",

                Pipeline(
                    [
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
                ),

                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "cat",

                Pipeline(
                    [
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
                ),

                categorical_features
            )
        )

    if not transformers:

        return {
            "success": False,
            "algorithm": algorithm,
            "error": (
                "No usable features were available "
                "for clustering optimization."
            )
        }

    matrix = ColumnTransformer(
        transformers
    ).fit_transform(frame)

    if hasattr(matrix, "toarray"):

        if matrix.shape[0] * matrix.shape[1] > 20_000_000:

            return {
                "success": False,
                "algorithm": algorithm,
                "error": (
                    "Encoded feature matrix is too large "
                    "for clustering optimization."
                )
            }

        matrix = matrix.toarray()

    matrix = np.asarray(
        matrix,
        dtype=float
    )

    maximum_clusters = min(
        8,
        len(matrix) - 1
    )

    candidates = []

    # -----------------------------------------------------
    # K-Means
    # -----------------------------------------------------

    if algorithm == "K-Means":

        for cluster_count in range(
            2,
            maximum_clusters + 1
        ):

            candidates.append(
                (
                    KMeans(
                        n_clusters=cluster_count,
                        n_init=10,
                        random_state=42
                    ),

                    {
                        "n_clusters": cluster_count,
                        "n_init": 10
                    }
                )
            )

    # -----------------------------------------------------
    # Hierarchical Clustering
    # -----------------------------------------------------

    elif algorithm == "Hierarchical Clustering":

        for cluster_count in range(
            2,
            maximum_clusters + 1
        ):

            for linkage in (
                "ward",
                "complete",
                "average"
            ):

                candidates.append(
                    (
                        AgglomerativeClustering(
                            n_clusters=cluster_count,
                            linkage=linkage
                        ),

                        {
                            "n_clusters": cluster_count,
                            "linkage": linkage
                        }
                    )
                )

    # -----------------------------------------------------
    # DBSCAN
    # -----------------------------------------------------

    elif algorithm == "DBSCAN":

        for epsilon in (
            0.3,
            0.5,
            0.7,
            1.0
        ):

            for minimum_samples in (
                3,
                5,
                10
            ):

                if minimum_samples < len(matrix):

                    candidates.append(
                        (
                            DBSCAN(
                                eps=epsilon,
                                min_samples=minimum_samples
                            ),

                            {
                                "eps": epsilon,
                                "min_samples": minimum_samples
                            }
                        )
                    )

    else:

        return {
            "success": False,
            "algorithm": algorithm,
            "error": (
                f"No clustering optimizer is configured "
                f"for {algorithm}."
            )
        }

    best = None

    for model, parameters in candidates:

        try:

            labels = model.fit_predict(
                matrix
            )

            valid_rows = (
                labels != -1
                if algorithm == "DBSCAN"
                else np.ones(
                    len(labels),
                    dtype=bool
                )
            )

            valid_labels = labels[
                valid_rows
            ]

            unique_labels = np.unique(
                valid_labels
            )

            if (
                len(unique_labels) < 2
                or
                len(unique_labels) >= len(valid_labels)
            ):

                continue

            score = float(
                silhouette_score(
                    matrix[valid_rows],
                    valid_labels
                )
            )

            if (
                best is None
                or
                score > best["best_score"]
            ):

                best = {
                    "best_score": score,
                    "best_parameters": parameters,
                    "clusters": int(
                        len(unique_labels)
                    )
                }

        except Exception:

            continue

    if best is None:

        return {
            "success": False,
            "algorithm": algorithm,
            "scoring": "silhouette",
            "error": (
                "No tested parameter set produced "
                "a valid silhouette score."
            )
        }

    return {

        "success": True,

        "algorithm": algorithm,

        "scoring": "silhouette",

        "best_score": round(
            best["best_score"],
            4
        ),

        "best_parameters":
            best["best_parameters"],

        "clusters":
            best["clusters"]

    }


# =========================================================
# Legacy Random Forest Tuning
# =========================================================

def tune_random_forest(df, target):

    if target not in df.columns:

        return {
            "success": False,
            "error": f"Target column '{target}' was not found."
        }

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    mask = y.notna()

    X = X.loc[mask]

    y = y.loc[mask]

    if len(y) < 3:

        return {
            "success": False,
            "error": (
                "At least three labeled rows "
                "are required for tuning."
            )
        }

    # -----------------------------------------------------
    # Detect task consistently
    # -----------------------------------------------------

    if detect_task is not None:

        task = detect_task(
            df,
            target
        )

    else:

        if (
            y.dtype == object
            or
            str(y.dtype).startswith("category")
        ):

            task = "classification"

        else:

            task = "regression"

    classification = (
        task == "classification"
    )

    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if classification:

        cv, cv_folds, cv_error = (
            _get_classification_cv(y)
        )

        if cv is None:

            return {
                "success": False,
                "algorithm": "Random Forest",
                "target": target,
                "category": "classification",
                "cv_folds": 0,
                "error": cv_error
            }

        encoder = LabelEncoder()

        y = encoder.fit_transform(
            y
        )

        model = RandomForestClassifier(
            random_state=42
        )

        scoring = "accuracy"

    # -----------------------------------------------------
    # Regression
    # -----------------------------------------------------

    else:

        cv, cv_folds, cv_error = (
            _get_regression_cv(y)
        )

        if cv is None:

            return {
                "success": False,
                "algorithm": "Random Forest Regressor",
                "target": target,
                "category": "regression",
                "cv_folds": 0,
                "error": cv_error
            }

        model = RandomForestRegressor(
            random_state=42
        )

        scoring = "r2"

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

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

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "num",
                numeric_transformer,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers
    )

    pipeline = Pipeline(
        [
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

    # -----------------------------------------------------
    # Parameter Grid
    # -----------------------------------------------------

    param_grid = {

        "model__n_estimators": [
            100,
            200,
            300
        ],

        "model__max_depth": [
            None,
            10,
            20
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ]

    }

    # -----------------------------------------------------
    # Grid Search
    # -----------------------------------------------------

    grid = GridSearchCV(

        pipeline,

        param_grid,

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        error_score="raise"

    )

    try:

        grid.fit(
            X,
            y
        )

    except Exception as error:

        return {
            "success": False,
            "algorithm": (
                "Random Forest"
                if classification
                else
                "Random Forest Regressor"
            ),
            "target": target,
            "category": task,
            "cv_folds": cv_folds,
            "error": str(error)
        }

    return {

        "success": True,

        "algorithm": (
            "Random Forest"
            if classification
            else
            "Random Forest Regressor"
        ),

        "target": target,

        "category": task,

        "scoring": scoring,

        "cv_folds": cv_folds,

        "best_score": round(
            float(
                grid.best_score_
            ),
            4
        ),

        "best_parameters": {

            key.replace(
                "model__",
                ""
            ): value

            for key, value
            in grid.best_params_.items()

        }

    }


# =========================================================
# Recommended Model Tuning
# =========================================================

def tune_recommended_model(
    df,
    target,
    category,
    algorithm
):

    # -----------------------------------------------------
    # Clustering
    # -----------------------------------------------------

    if category == "clustering":

        result = _tune_clustering_model(
            df,
            algorithm
        )

        result["category"] = category

        return result

    # -----------------------------------------------------
    # Supported supervised categories
    # -----------------------------------------------------

    if category not in {
        "classification",
        "regression"
    }:

        return {

            "success": False,

            "algorithm": algorithm,

            "target": target,

            "error": (
                "Hyperparameter optimization "
                f"is not supported for {category}."
            )

        }

    # -----------------------------------------------------
    # Target validation
    # -----------------------------------------------------

    if target not in df.columns:

        return {

            "success": False,

            "algorithm": algorithm,

            "target": target,

            "error": (
                f"Target column '{target}' "
                "was not found."
            )

        }

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]

    if len(y) < 3:

        return {

            "success": False,

            "algorithm": algorithm,

            "target": target,

            "error": (
                "At least three labeled rows "
                "are required for tuning."
            )

        }

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if category == "classification":

        # -------------------------------------------------
        # IMPORTANT:
        # Check class counts BEFORE LabelEncoder
        # -------------------------------------------------

        cv, cv_folds, cv_error = (
            _get_classification_cv(y)
        )

        if cv is None:

            return {

                "success": False,

                "algorithm": algorithm,

                "target": target,

                "category": category,

                "cv_folds": 0,

                "error": cv_error

            }

        # -------------------------------------------------
        # Encode labels AFTER checking class distribution
        # -------------------------------------------------

        if (
            y.dtype == object
            or
            str(y.dtype).startswith("category")
        ):

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y
            )

        scoring = "accuracy"

        estimators = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42
                ),

            "SVM":
                SVC(),

            "KNN":
                KNeighborsClassifier(),

            "Naive Bayes":
                GaussianNB()

        }

        # -------------------------------------------------
        # Safe KNN neighbors
        # -------------------------------------------------

        training_size = (
            len(y)
            -
            int(
                np.ceil(
                    len(y) / cv_folds
                )
            )
        )

        valid_neighbors = [
            value
            for value in (
                3,
                5,
                7
            )
            if value <= training_size
        ]

        if not valid_neighbors:

            valid_neighbors = [1]

        grids = {

            "Logistic Regression": {

                "model__C": [
                    0.1,
                    1.0,
                    10.0
                ]

            },

            "Decision Tree": {

                "model__max_depth": [
                    None,
                    5,
                    10
                ],

                "model__min_samples_split": [
                    2,
                    5
                ]

            },

            "Random Forest": {

                "model__n_estimators": [
                    50,
                    100
                ],

                "model__max_depth": [
                    None,
                    10
                ],

                "model__min_samples_leaf": [
                    1,
                    2
                ]

            },

            "SVM": {

                "model__C": [
                    0.1,
                    1.0,
                    10.0
                ],

                "model__kernel": [
                    "rbf",
                    "linear"
                ]

            },

            "KNN": {

                "model__n_neighbors":
                    valid_neighbors,

                "model__weights": [
                    "uniform",
                    "distance"
                ]

            },

            "Naive Bayes": {

                "model__var_smoothing": [
                    1e-9,
                    1e-8,
                    1e-7
                ]

            }

        }

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        cv, cv_folds, cv_error = (
            _get_regression_cv(y)
        )

        if cv is None:

            return {

                "success": False,

                "algorithm": algorithm,

                "target": target,

                "category": category,

                "cv_folds": 0,

                "error": cv_error

            }

        scoring = "r2"

        estimators = {

            "Linear Regression":
                LinearRegression(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(
                    max_iter=5000
                ),

            "Decision Tree Regressor":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest Regressor":
                RandomForestRegressor(
                    random_state=42
                ),

            "SVR":
                SVR()

        }

        grids = {

            "Linear Regression": {

                "model__fit_intercept": [
                    True,
                    False
                ]

            },

            "Ridge": {

                "model__alpha": [
                    0.1,
                    1.0,
                    10.0,
                    100.0
                ]

            },

            "Lasso": {

                "model__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1.0
                ]

            },

            "Decision Tree Regressor": {

                "model__max_depth": [
                    None,
                    5,
                    10
                ],

                "model__min_samples_split": [
                    2,
                    5
                ]

            },

            "Random Forest Regressor": {

                "model__n_estimators": [
                    50,
                    100
                ],

                "model__max_depth": [
                    None,
                    10
                ],

                "model__min_samples_leaf": [
                    1,
                    2
                ]

            },

            "SVR": {

                "model__C": [
                    0.1,
                    1.0,
                    10.0
                ],

                "model__kernel": [
                    "rbf",
                    "linear"
                ]

            }

        }

    # =====================================================
    # ALGORITHM VALIDATION
    # =====================================================

    if algorithm not in estimators:

        return {

            "success": False,

            "algorithm": algorithm,

            "target": target,

            "error": (
                f"No tuning search is configured "
                f"for {algorithm} in {category}."
            )

        }

    # =====================================================
    # PREPROCESSING
    # =====================================================

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        [
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

    categorical_transformer = Pipeline(
        [
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

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "num",
                numeric_transformer,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers
    )

    # =====================================================
    # PIPELINE
    # =====================================================

    steps = [

        (
            "preprocessor",
            preprocessor
        )

    ]

    if algorithm == "Naive Bayes":

        steps.append(
            (
                "dense",

                FunctionTransformer(
                    _dense_features,
                    accept_sparse=True,
                    validate=False
                )
            )
        )

    steps.append(
        (
            "model",
            estimators[algorithm]
        )
    )

    # =====================================================
    # GRID SEARCH
    # =====================================================

    search = GridSearchCV(

        Pipeline(steps),

        grids[algorithm],

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        error_score="raise"

    )

    # =====================================================
    # EXECUTE
    # =====================================================

    try:

        search.fit(
            X,
            y
        )

        return {

            "success": True,

            "algorithm": algorithm,

            "target": target,

            "category": category,

            "scoring": scoring,

            "cv_folds": cv_folds,

            "best_score": round(
                float(
                    search.best_score_
                ),
                4
            ),

            "best_parameters": {

                key.removeprefix(
                    "model__"
                ): value

                for key, value
                in search.best_params_.items()

            }

        }

    except Exception as error:

        return {

            "success": False,

            "algorithm": algorithm,

            "target": target,

            "category": category,

            "cv_folds": cv_folds,

            "error": str(error)

        }