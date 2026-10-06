"""
==========================================================
SimuRithm
Sequence Modeling Module
==========================================================

Supported Algorithms:
    - LSTM
    - GRU
    - Transformer

Requires:
    tensorflow

Install:
    pip install tensorflow
==========================================================
"""

import time

import numpy as np
import pandas as pd


# ==========================================================
# TensorFlow Loader
# ==========================================================

def _load_tensorflow():

    try:

        import tensorflow as tf

        return tf

    except ImportError:

        raise ImportError(
            "Sequence modeling requires TensorFlow. "
            "Install it using: pip install tensorflow"
        )


# ==========================================================
# Prepare Sequence Data
# ==========================================================

def _prepare_sequence_data(
    df,
    target=None,
    sequence_length=10
):

    if df is None or df.empty:

        raise ValueError(
            "The dataset is empty."
        )

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    if not numeric_columns:

        raise ValueError(
            "Sequence modeling requires "
            "at least one numeric column."
        )

    # ------------------------------------------------------
    # Select Target
    # ------------------------------------------------------

    if target is not None:

        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' "
                f"was not found."
            )

        if target not in numeric_columns:

            raise ValueError(
                "Sequence target must be numeric."
            )

        feature_columns = [
            column
            for column in numeric_columns
            if column != target
        ]

        if not feature_columns:

            feature_columns = [
                target
            ]

    else:

        # Use the last numeric column as
        # the prediction target.
        target = numeric_columns[-1]

        feature_columns = [
            column
            for column in numeric_columns
            if column != target
        ]

        if not feature_columns:

            feature_columns = [
                target
            ]

    # ------------------------------------------------------
    # Fill Missing Values
    # ------------------------------------------------------

    data = (
        df[
            feature_columns
            +
            [target]
        ]
        .copy()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.interpolate(
        method="linear"
    )

    data = data.fillna(
        data.median(
            numeric_only=True
        )
    )

    data = data.dropna()

    if len(data) <= sequence_length:

        raise ValueError(
            "The dataset does not contain enough rows "
            "for the selected sequence length."
        )

    X_values = (
        data[
            feature_columns
        ]
        .values
        .astype("float32")
    )

    y_values = (
        data[
            target
        ]
        .values
        .astype("float32")
    )

    X = []

    y = []

    for index in range(
        sequence_length,
        len(data)
    ):

        X.append(
            X_values[
                index - sequence_length:index
            ]
        )

        y.append(
            y_values[index]
        )

    X = np.asarray(
        X,
        dtype="float32"
    )

    y = np.asarray(
        y,
        dtype="float32"
    )

    return (
        X,
        y,
        feature_columns,
        target
    )


# ==========================================================
# Build Sequence Model
# ==========================================================

def _build_model(
    tf,
    algorithm,
    input_shape
):

    inputs = tf.keras.Input(
        shape=input_shape
    )

    # ------------------------------------------------------
    # LSTM
    # ------------------------------------------------------

    if algorithm == "LSTM":

        x = tf.keras.layers.LSTM(
            64
        )(inputs)

        x = tf.keras.layers.Dropout(
            0.2
        )(x)

    # ------------------------------------------------------
    # GRU
    # ------------------------------------------------------

    elif algorithm == "GRU":

        x = tf.keras.layers.GRU(
            64
        )(inputs)

        x = tf.keras.layers.Dropout(
            0.2
        )(x)

    # ------------------------------------------------------
    # Transformer
    # ------------------------------------------------------

    elif algorithm == "Transformer":

        attention = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=2,
                key_dim=32
            )
        )

        attention_output = attention(
            inputs,
            inputs
        )

        x = tf.keras.layers.LayerNormalization()(
            inputs + attention_output
        )

        feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    64,
                    activation="relu"
                ),

                tf.keras.layers.Dense(
                    input_shape[-1]
                )
            ]
        )

        feed_output = feed_forward(
            x
        )

        x = tf.keras.layers.LayerNormalization()(
            x + feed_output
        )

        x = tf.keras.layers.GlobalAveragePooling1D()(
            x
        )

    else:

        raise ValueError(
            f"Unsupported sequence modeling "
            f"algorithm: '{algorithm}'."
        )

    outputs = tf.keras.layers.Dense(
        1
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ==========================================================
# Sequence Modeling
# ==========================================================

def run_sequence_modeling(
    df,
    selected_algorithm,
    target=None,
    sequence_length=10,
    epochs=10
):
    """
    Run LSTM, GRU, or Transformer.
    """

    allowed_algorithms = {

        "LSTM",

        "GRU",

        "Transformer"

    }

    if selected_algorithm not in allowed_algorithms:

        raise ValueError(
            f"Unsupported sequence modeling "
            f"algorithm: '{selected_algorithm}'."
        )

    try:

        tf = _load_tensorflow()

        start = time.time()

        (
            X,
            y,
            feature_columns,
            target_column
        ) = _prepare_sequence_data(
            df,
            target=target,
            sequence_length=sequence_length
        )

        if len(X) < 10:

            raise ValueError(
                "At least 10 generated sequences "
                "are recommended."
            )

        # --------------------------------------------------
        # Train/Test Split
        # --------------------------------------------------

        split_index = int(
            len(X) * 0.80
        )

        if split_index <= 0 or split_index >= len(X):

            raise ValueError(
                "Unable to create a valid training "
                "and testing split."
            )

        X_train = X[
            :split_index
        ]

        X_test = X[
            split_index:
        ]

        y_train = y[
            :split_index
        ]

        y_test = y[
            split_index:
        ]

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model = _build_model(
            tf,
            selected_algorithm,
            (
                X_train.shape[1],
                X_train.shape[2]
            )
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_test,
                y_test
            ),
            epochs=epochs,
            batch_size=min(
                32,
                len(X_train)
            ),
            verbose=0
        )

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        evaluation = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

        test_loss = float(
            evaluation[0]
        )

        test_mae = float(
            evaluation[1]
        )

        predictions = model.predict(
            X_test,
            verbose=0
        ).reshape(-1)

        # --------------------------------------------------
        # RMSE
        # --------------------------------------------------

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        predictions
                        -
                        y_test
                    )
                    ** 2
                )
            )
        )

        end = time.time()

        # --------------------------------------------------
        # Training History
        # --------------------------------------------------

        loss_history = [
            round(
                float(value),
                6
            )
            for value
            in history.history.get(
                "loss",
                []
            )
        ]

        validation_loss_history = [
            round(
                float(value),
                6
            )
            for value
            in history.history.get(
                "val_loss",
                []
            )
        ]

        return [
            {
                "success": True,

                "category":
                    "sequence_modeling",

                "algorithm":
                    selected_algorithm,

                "target":
                    str(target_column),

                "feature_columns":
                    feature_columns,

                "sequence_length":
                    int(sequence_length),

                "training_sequences":
                    int(len(X_train)),

                "testing_sequences":
                    int(len(X_test)),

                "epochs":
                    int(epochs),

                "mse":
                    round(
                        test_loss,
                        6
                    ),

                "mae":
                    round(
                        test_mae,
                        6
                    ),

                "rmse":
                    round(
                        rmse,
                        6
                    ),

                "loss_history":
                    loss_history,

                "validation_loss_history":
                    validation_loss_history,

                "execution_time":
                    round(
                        end - start,
                        4
                    )
            }
        ]

    except Exception as error:

        return [
            {
                "success": False,

                "category":
                    "sequence_modeling",

                "algorithm":
                    selected_algorithm,

                "execution_time":
                    0,

                "error":
                    str(error)
            }
        ]