"""Experiment runners for classical ML scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import SCENARIO_CONFIG
from src.evaluate import (
    append_significance_test,
    compute_metrics,
    mcnemar_test,
    plot_confusion_matrix,
    save_scenario_metrics,
)
from src.features import get_feature_group_names
from src.models import build_rf_classifier, build_svm_pipeline
from src.pipeline import batch_extract_features
from src.utils import label_encode, timeit


def extract_split_matrix(
    df: pd.DataFrame,
    enhancement: str,
    do_segment: bool,
    feature_groups: str,
    cache_path: Path | None = None,
    split_name: str = "train",
    restoration: str = "ssr",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract features for all rows in df; optional npz cache."""
    cache_key = f"{split_name}_{restoration}_{enhancement}_{do_segment}_{feature_groups}"
    # Filenames identify the rows this matrix belongs to, independent of the
    # absolute mount path. Used to detect a stale cache built for a different
    # split (which would otherwise raise IndexError or, worse, silently
    # misalign features with labels).
    current_files = [Path(p).name for p in df["filepath"].tolist()]

    if cache_path is not None:
        npz_file = cache_path / f"features_{cache_key}.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            cached_files = data["df_files"].tolist() if "df_files" in data else None
            valid_idx = data["valid_idx"].tolist()
            cache_ok = (
                cached_files == current_files
                and (not valid_idx or max(valid_idx) < len(df))
            )
            if cache_ok:
                valid_df = df.iloc[valid_idx].reset_index(drop=True)
                return data["X"], data["y"], valid_df
            # Stale/mismatched cache (different split or old format) -> recompute.

    from src.utils import get_project_paths

    failure_log = get_project_paths()["metrics"] / "segmentation_failures.csv"
    X, valid_idx = batch_extract_features(
        df["filepath"].tolist(),
        enhancement=enhancement,
        do_segment=do_segment,
        feature_groups=feature_groups,
        metadata=df,
        failure_log_path=failure_log if do_segment else None,
        restoration=restoration,
    )
    y, _ = label_encode(df.iloc[valid_idx]["label"])
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path / f"features_{cache_key}.npz",
            X=X,
            y=y,
            valid_idx=np.array(valid_idx),
            df_files=np.array(current_files),
        )

    return X, y, valid_df


def run_classical_scenario(
    scenario_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate one classical scenario (scenario_id 1-8).

    Raises:
        KeyError: if scenario_id is not in SCENARIO_CONFIG (classical scenarios).
            CNN scenarios (9-10, MobileNetV2) are handled in notebooks/03_experiments_cnn.ipynb.
    """
    from src.utils import read_best_enhancement

    if scenario_id not in SCENARIO_CONFIG:
        valid = sorted(SCENARIO_CONFIG.keys())
        raise KeyError(
            f"scenario_id={scenario_id!r} not found. "
            f"Classical scenarios are {valid[0]}-{valid[-1]}. "
            "CNN scenarios are run directly in notebooks/03_experiments_cnn.ipynb."
        )

    cfg = SCENARIO_CONFIG[scenario_id].copy()
    enhancement = cfg["enhancement"]
    if enhancement == "best":
        enhancement = read_best_enhancement(metrics_dir)

    restoration = cfg.get("restoration", "ssr")
    do_segment = cfg["segment"]
    feature_groups = cfg["features"]
    model_type = cfg["model"]

    X_train, y_train, train_v = extract_split_matrix(
        train_df, enhancement, do_segment, feature_groups, cache_dir, "train", restoration
    )
    X_val, y_val, val_v = extract_split_matrix(
        val_df, enhancement, do_segment, feature_groups, cache_dir, "val", restoration
    )
    X_test, y_test, test_v = extract_split_matrix(
        test_df, enhancement, do_segment, feature_groups, cache_dir, "test", restoration
    )

    if model_type == "svm":
        model = build_svm_pipeline(verbose=0)
    else:
        model = build_rf_classifier()

    model.fit(X_train, y_train)

    @timeit
    def predict_fn(X):
        return model.predict(X)

    y_pred, infer_ms = predict_fn(X_test)
    val_pred = model.predict(X_val)

    metrics = compute_metrics(y_test, y_pred)
    infer_per_img = infer_ms / max(len(y_test), 1)

    feat_label = feature_groups if do_segment else "color+texture"
    save_scenario_metrics(
        scenario_id=scenario_id,
        enhancement=enhancement,
        segmentation=do_segment,
        features=feat_label,
        model=model_type.upper(),
        metrics=metrics,
        inference_time_ms=infer_per_img,
        n_test_samples=len(y_test),
        metrics_dir=metrics_dir,
        restoration=restoration,
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        title=f"Scenario {scenario_id}",
        save_path=figures_dir / f"scenario_{scenario_id:02d}.png",
    )
    plt_close()

    # Persist the full-pipeline SVM (S5, McNemar anchor vs CNN) and the RF (S9,
    # feature-importance analysis) so downstream notebooks can reload them.
    if scenario_id in (5, 9):
        joblib.dump(model, models_dir / f"{'svm' if model_type == 'svm' else 'rf'}_scenario_{scenario_id:02d}.pkl")
    if scenario_id == 5:
        # test_v is already the filtered valid-rows DataFrame (aligned with y_pred);
        # do NOT re-index test_df with it. Saved for per-commodity analysis in nb04.
        pred_df = test_v.reset_index(drop=True).copy()
        pred_df["pred"] = y_pred
        pred_df.to_csv(metrics_dir / "predictions_s5.csv", index=False)

    val_f1 = compute_metrics(y_val, val_pred)["f1_weighted"]

    result = {
        "scenario_id": scenario_id,
        "enhancement": enhancement,
        "restoration": restoration,
        "val_f1": val_f1,
        "test_metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "model": model,
    }

    if model_type == "rf":
        segmented = do_segment
        names = get_feature_group_names("all" if feature_groups == "all" else feature_groups, segmented)
        if len(names) != len(model.feature_importances_):
            names = [f"f{i}" for i in range(len(model.feature_importances_))]
        result["feature_importances"] = model.feature_importances_
        result["feature_names"] = names

    return result


def plt_close() -> None:
    """Close all open matplotlib figures to free memory."""
    import matplotlib.pyplot as plt

    plt.close("all")


def select_best_enhancement(val_f1_map: dict[str, float], metrics_dir: Path) -> str:
    """Pick enhancement with highest validation F1 and save to best_enhancement.txt."""
    best = max(val_f1_map, key=val_f1_map.get)
    path = metrics_dir / "best_enhancement.txt"
    path.write_text(best, encoding="utf-8")
    return best


def run_mcnemar_pair(
    name: str,
    model_a: str,
    model_b: str,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metrics_dir: Path,
) -> None:
    stat, pval, conclusion = mcnemar_test(y_true, pred_a, pred_b)
    append_significance_test(name, model_a, model_b, stat, pval, conclusion, metrics_dir)
