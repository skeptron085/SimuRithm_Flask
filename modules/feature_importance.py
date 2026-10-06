import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def get_feature_importance(df, target):

    if target not in df.columns:
        return []

    X = df.drop(columns=[target]).copy()

    y = df[target].copy()

    # ----------------------------------------
    # Remove rows with missing target
    # ----------------------------------------

    mask = y.notna()

    X = X.loc[mask]

    y = y.loc[mask]

    # ----------------------------------------
    # Detect task
    # ----------------------------------------

    classification = False

    if y.dtype == object or y.nunique() <= 20:

        classification = True

        encoder = LabelEncoder()

        y = encoder.fit_transform(y)

    # ----------------------------------------
    # Column groups
    # ----------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # ----------------------------------------
    # Preprocessing
    # ----------------------------------------

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

    X_processed = preprocessor.fit_transform(X)

    # ----------------------------------------
    # Feature Names
    # ----------------------------------------

    feature_names = numeric_features.copy()

    if len(categorical_features) > 0:

        encoder = preprocessor.named_transformers_[
            "cat"
        ].named_steps["encoder"]

        encoded_names = encoder.get_feature_names_out(
            categorical_features
        )

        feature_names.extend(encoded_names.tolist())

    # ----------------------------------------
    # Train Model
    # ----------------------------------------

    if classification:

        model = RandomForestClassifier(

            n_estimators=200,

            random_state=42

        )

    else:

        model = RandomForestRegressor(

            n_estimators=200,

            random_state=42

        )

    model.fit(

        X_processed,

        y

    )

    importance = model.feature_importances_

    result = []

    for feature, score in zip(

        feature_names,

        importance

    ):

        result.append({

            "feature": feature,

            "importance": round(

                float(score),

                6

            )

        })

    result.sort(

        key=lambda x: x["importance"],

        reverse=True

    )

    return result