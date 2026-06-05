"""Evaluation: metrics, McNemar, confusion matrix, Grad-CAM, summary table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from statsmodels.stats.contingency_tables import mcnemar


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision/recall/f1 (weighted)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, str]:
    """
    McNemar's test comparing two models on the same test set.
    Returns: statistic, p_value, conclusion ('signifikan' / 'tidak signifikan').
    """
    a = int(np.sum((y_pred_a == y_true) & (y_pred_b == y_true)))
    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    d = int(np.sum((y_pred_a != y_true) & (y_pred_b != y_true)))

    if b + c == 0:
        # Both models agree on every sample - test is undefined; treat as non-significant.
        return 0.0, 1.0, "tidak signifikan"

    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=(b + c) < 25)
    conclusion = "signifikan" if result.pvalue < alpha else "tidak signifikan"
    return float(result.statistic), float(result.pvalue), conclusion


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot and optionally save confusion matrix heatmap."""
    if labels is None:
        labels = ["fresh", "rotten"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_scenario_metrics(
    scenario_id: int,
    enhancement: str,
    segmentation: bool,
    features: str,
    model: str,
    metrics: dict[str, float],
    inference_time_ms: float,
    n_test_samples: int,
    metrics_dir: str | Path,
    restoration: str = "ssr",
) -> Path:
    """Save per-scenario metrics CSV."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "scenario_id": scenario_id,
        "restoration": restoration,
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
        "model": model,
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "inference_time_ms_per_image": inference_time_ms,
        "n_test_samples": n_test_samples,
    }
    path = out_dir / f"scenario_{scenario_id:02d}.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def append_significance_test(
    comparison: str,
    model_a: str,
    model_b: str,
    statistic: float,
    p_value: float,
    conclusion: str,
    metrics_dir: str | Path,
) -> None:
    """Append one McNemar result to significance_tests.csv."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "significance_tests.csv"
    row = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "model_a": model_a,
                "model_b": model_b,
                "statistic": statistic,
                "p_value": p_value,
                "conclusion": conclusion,
            }
        ]
    )
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)


def print_summary_table(metrics_dir: str | Path = "results/metrics/") -> pd.DataFrame:
    """Load all scenario_XX.csv files and print summary table."""
    metrics_path = Path(metrics_dir)
    files = sorted(metrics_path.glob("scenario_*.csv"))
    if not files:
        print("No scenario metrics found.")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    summary = pd.concat(dfs, ignore_index=True)
    summary = summary.sort_values("scenario_id")
    cols = [
        "scenario_id",
        "restoration",
        "enhancement",
        "segmentation",
        "features",
        "model",
        "f1_weighted",
        "accuracy",
        "inference_time_ms_per_image",
    ]
    display_cols = [c for c in cols if c in summary.columns]
    print(summary[display_cols].to_string(index=False))
    return summary


def plot_feature_importance(
    importances: np.ndarray,
    group_labels: list[str],
    title: str = "Feature Importance by Group",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar plot of aggregated feature importance per group."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(group_labels, importances)
    ax.set_xlabel("Mean importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def aggregate_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], np.ndarray]:
    """Aggregate RF importances into color, texture, shape groups."""
    groups = {"color": [], "texture": [], "shape": []}
    for name, imp in zip(feature_names, importances):
        if name.startswith("hist_") or name.startswith("mean_") or name.startswith("std_") or name.startswith("skew_"):
            groups["color"].append(imp)
        elif name.startswith("glcm_") or name.startswith("lbp_"):
            groups["texture"].append(imp)
        else:
            groups["shape"].append(imp)

    labels = []
    values = []
    for g in ("color", "texture", "shape"):
        if groups[g]:
            labels.append(g)
            values.append(float(np.mean(groups[g])))
    return labels, np.array(values)


def build_gradcam_model(model: Any, last_conv_layer_name: str | None = None) -> Any:
    """Build the Grad-CAM sub-model once and reuse across many images.

    Constructing a new tf.keras.Model on every call (as the old make_gradcam_heatmap
    did) re-allocates the TF graph each time - expensive for loops over hundreds of
    images.  Build this once, then pass it to make_gradcam_heatmap.

    Usage::

        grad_model = build_gradcam_model(model)
        for img in images:
            heatmap = make_gradcam_heatmap(grad_model, img_array)
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            # Keras 3 (TF 2.16+) removed Layer.output_shape; fall back to output.shape.
            try:
                shape = layer.output.shape
            except AttributeError:
                shape = getattr(layer, "output_shape", None)
            if shape is not None and len(shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv layer with 4-D output in the model.")

    return tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )


def make_gradcam_heatmap(
    model: Any,
    img_array: np.ndarray,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image (batch of 1).

    Pass the FULL classifier model (e.g. the MobileNetV2 transfer model). The
    heatmap is computed by replaying the network eagerly under a GradientTape
    rather than rebuilding a functional sub-model.

    Why not a functional sub-model: when the target feature map lives inside a
    *nested* model (MobileNetV2 wrapped as a single layer), Keras 3 cannot
    traverse `Model(model.inputs, [nested.output, model.output])` - calling it
    raises ``KeyError`` in ``_run_through_graph``. Replaying layer-by-layer and
    watching the intermediate tensor avoids that entirely and works on both
    Keras 2 and Keras 3.

    Args:
        model:      Trained Keras classifier. Assumed to be a linear stack of
                    layers (Input -> feature extractor -> head), which is the case
                    for ``build_mobilenetv2``.
        img_array:  Preprocessed image, shape (1, H, W, C).
        last_conv_layer_name: Optional explicit feature-map layer name. When
                    omitted, the last layer with a 4-D output is used.
    """
    import tensorflow as tf

    x_in = tf.convert_to_tensor(img_array, dtype=tf.float32)
    layers = list(model.layers)

    # Locate the feature-map producer (last 4-D output layer, or named layer).
    conv_index = None
    for i, layer in enumerate(layers):
        if last_conv_layer_name is not None and layer.name == last_conv_layer_name:
            conv_index = i
            break
    if conv_index is None:
        for i in range(len(layers) - 1, -1, -1):
            try:
                shape = layers[i].output.shape
            except AttributeError:
                shape = getattr(layers[i], "output_shape", None)
            if shape is not None and len(shape) == 4:
                conv_index = i
                break
    if conv_index is None:
        raise ValueError("Could not find a 4-D feature-map layer for Grad-CAM.")

    with tf.GradientTape() as tape:
        x = x_in
        # Skip layers[0] (the InputLayer); replay up to the feature map.
        for layer in layers[1:conv_index + 1]:
            x = layer(x, training=False)
        conv_outputs = x
        tape.watch(conv_outputs)
        for layer in layers[conv_index + 1:]:
            x = layer(x, training=False)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(
    img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    title: str = "Grad-CAM",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay Grad-CAM heatmap on original BGR image."""
    import cv2

    from src.preprocessing import to_uint8

    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_u8 = to_uint8(img_bgr)
    overlay = cv2.addWeighted(img_u8, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay_rgb)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
