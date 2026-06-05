"""Tests for evaluation utilities: metrics, McNemar, CSV writers, Grad-CAM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    append_significance_test,
    build_gradcam_model,
    compute_metrics,
    make_gradcam_heatmap,
    mcnemar_test,
    save_scenario_metrics,
)


def test_compute_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_weighted"] == 1.0


def test_compute_metrics_keys_and_value():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert set(m) == {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"}
    assert m["accuracy"] == 0.75


def test_mcnemar_identical_predictions_non_significant():
    # b + c == 0 -> test is undefined -> treated as non-significant.
    y_true = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 0, 1])
    stat, pval, conclusion = mcnemar_test(y_true, pred, pred)
    assert pval == 1.0
    assert conclusion == "tidak signifikan"


def test_mcnemar_strong_disagreement_significant():
    # Model A correct on all, Model B wrong on all -> maximal discordance.
    n = 40
    y_true = np.zeros(n, dtype=int)
    pred_a = np.zeros(n, dtype=int)
    pred_b = np.ones(n, dtype=int)
    stat, pval, conclusion = mcnemar_test(y_true, pred_a, pred_b)
    assert pval < 0.05
    assert conclusion == "signifikan"


def test_save_scenario_metrics_writes_csv(tmp_path):
    metrics = {
        "accuracy": 0.9,
        "precision_weighted": 0.9,
        "recall_weighted": 0.9,
        "f1_weighted": 0.9,
    }
    path = save_scenario_metrics(
        scenario_id=1,
        enhancement="none",
        segmentation=False,
        features="all",
        model="SVM",
        metrics=metrics,
        inference_time_ms=1.2,
        n_test_samples=100,
        metrics_dir=tmp_path,
    )
    assert path.exists()
    df = pd.read_csv(path)
    assert df.loc[0, "scenario_id"] == 1
    assert df.loc[0, "f1_weighted"] == 0.9
    assert "inference_time_ms_per_image" in df.columns


def test_append_significance_test_appends(tmp_path):
    append_significance_test("A vs B", "A", "B", 1.0, 0.04, "signifikan", tmp_path)
    append_significance_test("C vs D", "C", "D", 2.0, 0.50, "tidak signifikan", tmp_path)
    df = pd.read_csv(tmp_path / "significance_tests.csv")
    assert len(df) == 2
    assert list(df["comparison"]) == ["A vs B", "C vs D"]


def test_build_gradcam_model_returns_two_outputs():
    # Skip when TensorFlow is unavailable (e.g. CI installs only requirements.txt).
    tf = pytest.importorskip("tensorflow")

    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="last_conv")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    grad_model = build_gradcam_model(model)
    # Must expose exactly two outputs: the conv feature map and the predictions.
    assert len(grad_model.outputs) == 2

    conv_out, preds = grad_model(np.zeros((1, 8, 8, 3), dtype=np.float32))
    assert len(conv_out.shape) == 4
    assert preds.shape[-1] == 2


def test_make_gradcam_heatmap_flat_model():
    tf = pytest.importorskip("tensorflow")

    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    heatmap = make_gradcam_heatmap(model, np.random.rand(1, 8, 8, 3).astype(np.float32))
    assert heatmap.ndim == 2
    assert heatmap.shape == (8, 8)


def test_make_gradcam_heatmap_nested_model():
    # Regression: MobileNetV2 is wrapped as a nested model layer. Rebuilding a
    # functional sub-model for this case raises KeyError in Keras 3; the
    # GradientTape replay must handle it.
    tf = pytest.importorskip("tensorflow")

    base_in = tf.keras.Input(shape=(8, 8, 3))
    base_out = tf.keras.layers.Conv2D(4, 3, padding="same")(base_in)
    base = tf.keras.Model(base_in, base_out, name="base")

    inp = tf.keras.Input(shape=(8, 8, 3))
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inp, out)

    heatmap = make_gradcam_heatmap(model, np.random.rand(1, 8, 8, 3).astype(np.float32))
    assert heatmap.ndim == 2
    assert heatmap.shape == (8, 8)
