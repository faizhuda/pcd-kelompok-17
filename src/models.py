"""Model builders: SVM, Random Forest, MobileNetV2."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight


def build_svm_pipeline(random_state: int = 42, cv: int = 3, n_jobs: int = -1, verbose: int = 0) -> GridSearchCV:
    """StandardScaler + SVC RBF with a small GridSearchCV (f1_weighted).

    The grid is intentionally compact (4 candidates x 3 folds = 12 fits). An
    exhaustive grid over C and gamma adds hours of compute on a large dataset
    for negligible accuracy gain - not worth it for this project, where the
    focus is the pipeline and analysis, not squeezing out the last F1 point.
    Adding gamma=0.01 alongside 'scale' ensures a fairer comparison vs CNN.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=random_state,
                    probability=False,
                ),
            ),
        ]
    )
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=n_jobs,
        verbose=verbose,
    )


def build_rf_classifier(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights for Keras training."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_mobilenetv2(num_classes: int = 2, input_shape: tuple[int, int, int] = (224, 224, 3)) -> Any:
    """
    MobileNetV2 transfer learning model with frozen base.
    Output: num_classes with softmax (default 2: fresh/rotten).
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def compile_mobilenet(model: Any, learning_rate: float = 1e-4) -> Any:
    """Compile MobileNetV2 for categorical crossentropy."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_last_layers(model: Any, n: int = 20) -> Any:
    """Unfreeze last n layers of MobileNetV2 base for fine-tuning."""
    import tensorflow as tf

    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_layer = layer
            break
    if base_layer is None:
        base_layer = model.layers[1]

    base_layer.trainable = True
    for layer in base_layer.layers[:-n]:
        layer.trainable = False
    return model


def get_mobilenet_callbacks(
    checkpoint_path: str,
    monitor: str = "val_loss",
) -> list:
    """EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor=monitor,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor=monitor,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor=monitor,
        ),
    ]
