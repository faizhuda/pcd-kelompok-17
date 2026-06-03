"""Utilities: reproducibility, paths, dataset indexing, splits, timing."""

from __future__ import annotations

import functools
import json
import random
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FRESH_KEYWORDS = {"fresh", "segar", "good", "healthy", "normal"}
ROTTEN_KEYWORDS = {"rotten", "busuk", "bad", "spoiled", "spoilt", "rot", "disease"}

KAGGLE_DATASET_SLUG = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"

T = TypeVar("T")


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def get_project_paths() -> dict[str, Path]:
    """Return common project paths."""
    root = get_project_root()
    return {
        "root": root,
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "splits": root / "data" / "splits.json",
        "results": root / "results",
        "metrics": root / "results" / "metrics",
        "models": root / "results" / "models",
        "figures": root / "results" / "figures",
        "figures_enhancement": root / "results" / "figures" / "enhancement",
        "figures_segmentation": root / "results" / "figures" / "segmentation",
        "figures_confusion": root / "results" / "figures" / "confusion_matrix",
        "figures_gradcam": root / "results" / "figures" / "gradcam",
    }


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def _normalize_label(token: str) -> str | None:
    """Parse 'fresh' or 'rotten' from a path component token. Returns None if unrecognised."""
    t = token.lower().strip().replace("-", " ").replace("_", " ")
    words = t.split()
    for word in words:
        if word in FRESH_KEYWORDS:
            return "fresh"
        if word in ROTTEN_KEYWORDS:
            return "rotten"
    if "fresh" in t:
        return "fresh"
    if "rotten" in t:
        return "rotten"
    return None


def _parse_filepath_parts(rel_parts: tuple[str, ...]) -> tuple[str, str] | None:
    """Infer (commodity, label) from path components."""
    IGNORE_PARTS = {
        "fruit and vegetable diseases dataset",
        "fruit and vegetable disease healthy vs rotten",
        "archive",
        "dataset",
        "1.archive",
        "versions",
        "1"
    }
    
    filtered_parts = [p for p in rel_parts if p.lower().strip() not in IGNORE_PARTS]
    if not filtered_parts:
        return None

    # 1. Try checking for "__" in any part, from right to left (most specific folder first)
    for part in reversed(filtered_parts):
        if "__" in part:
            subparts = part.split("__")
            if len(subparts) == 2:
                comm, lbl = subparts[0], subparts[1]
                norm_lbl = _normalize_label(lbl)
                if norm_lbl is not None:
                    commodity = comm.replace("_", " ").replace("-", " ").title().strip()
                    return commodity, norm_lbl

    # 2. Fallback: Parse from right to left
    label = None
    commodity_parts: list[str] = []
    
    for part in reversed(filtered_parts):
        parsed = _normalize_label(part)
        if parsed is not None and label is None:
            label = parsed
        else:
            commodity_parts.insert(0, part)
            
    if label is None:
        return None
        
    commodity = " ".join(commodity_parts).strip() or filtered_parts[0]
    commodity = commodity.replace("_", " ").replace("-", " ").title()
    return commodity, label


def build_dataset_index(raw_dir: str | Path) -> pd.DataFrame:
    """
    Walk raw_dir and build DataFrame with columns:
    filepath, label (fresh/rotten), commodity.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    records: list[dict[str, str]] = []

    for img_path in raw_path.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel_parts = img_path.relative_to(raw_path).parts[:-1]
        if not rel_parts:
            continue

        parsed = _parse_filepath_parts(rel_parts)
        if parsed is None:
            continue

        commodity, label = parsed
        records.append(
            {
                "filepath": str(img_path.resolve()),
                "label": label,
                "commodity": commodity,
            }
        )

    if not records:
        raise ValueError(
            f"No images indexed under {raw_path}. "
            "Expected folder structure containing 'fresh' or 'rotten' in path."
        )

    df = pd.DataFrame(records)
    df["strat_key"] = df["commodity"] + "_" + df["label"]
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70% train / 15% val / 15% test by commodity × label.
    """
    work = df.copy()
    if "strat_key" not in work.columns:
        work["strat_key"] = work["commodity"] + "_" + work["label"]

    train, temp = train_test_split(
        work,
        test_size=0.30,
        stratify=work["strat_key"],
        random_state=seed,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["strat_key"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Save train/val/test splits to JSON using paths relative to project root.

    Storing relative paths makes the file portable across machines and OS.
    """
    if path is None:
        path = get_project_paths()["splits"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = get_project_root()

    def _to_relative(records: list[dict]) -> list[dict]:
        result = []
        for rec in records:
            rec = rec.copy()
            try:
                rec["filepath"] = str(Path(rec["filepath"]).relative_to(root).as_posix())
            except ValueError:
                # filepath is already relative or outside project root — keep as-is
                rec["filepath"] = Path(rec["filepath"]).as_posix()
            result.append(rec)
        return result

    payload = {
        "train": _to_relative(train[["filepath", "label", "commodity"]].to_dict(orient="records")),
        "val": _to_relative(val[["filepath", "label", "commodity"]].to_dict(orient="records")),
        "test": _to_relative(test[["filepath", "label", "commodity"]].to_dict(orient="records")),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def load_splits(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from JSON.

    Filepaths stored as POSIX-relative strings are resolved to absolute paths
    based on the current project root, making the splits portable across machines.
    """
    if path is None:
        path = get_project_paths()["splits"]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    root = get_project_root()

    def _to_absolute(records: list[dict]) -> list[dict]:
        result = []
        for rec in records:
            rec = rec.copy()
            p = Path(rec["filepath"])
            if not p.is_absolute():
                rec["filepath"] = str((root / p).resolve())
            result.append(rec)
        return result

    train = pd.DataFrame(_to_absolute(payload["train"]))
    val = pd.DataFrame(_to_absolute(payload["val"]))
    test = pd.DataFrame(_to_absolute(payload["test"]))
    for df in (train, val, test):
        df["strat_key"] = df["commodity"] + "_" + df["label"]
    return train, val, test


def timeit(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator returning (result, elapsed_ms)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms

    return wrapper


def label_encode(y: pd.Series | np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Encode fresh=0, rotten=1.

    Raises ValueError with an informative message for any label not in
    {"fresh", "rotten"}, including pre-existing NaN / missing values.
    """
    mapping = {"fresh": 0, "rotten": 1}
    valid_labels = set(mapping.keys())

    if isinstance(y, pd.Series):
        # Guard against pre-existing NaN (missing data) before attempting map.
        pre_nan = y[y.isna()]
        if not pre_nan.empty:
            raise ValueError(
                f"Labels column contains {len(pre_nan)} missing value(s) (NaN). "
                "Fix missing data before encoding."
            )
        encoded = y.map(mapping)
        unknown_mask = encoded.isna()
        if unknown_mask.any():
            bad = y[unknown_mask].unique().tolist()
            raise ValueError(
                f"Unknown label(s) {bad} found. Expected one of {sorted(valid_labels)}."
            )
        return encoded.values.astype(np.int64), mapping
    else:
        bad = [str(v) for v in y if str(v) not in valid_labels]
        if bad:
            raise ValueError(
                f"Unknown label(s) {list(dict.fromkeys(bad))} found. "
                f"Expected one of {sorted(valid_labels)}."
            )
        return np.array([mapping[str(v)] for v in y], dtype=np.int64), mapping


def download_kaggle_dataset(
    slug: str = KAGGLE_DATASET_SLUG,
    target_dir: str | Path | None = None,
) -> Path:
    """Download dataset via kagglehub and copy image files into data/raw/.

    Returns path to data/raw.

    Raises:
        RuntimeError: if kagglehub fails or returns an empty / invalid path.
    """
    import shutil

    from tqdm import tqdm

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    if target_dir is None:
        target_dir = get_project_paths()["data_raw"]
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{slug}' via kagglehub …")
    try:
        downloaded = Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        raise RuntimeError(
            f"kagglehub failed to download '{slug}'. "
            "Check your Kaggle API key (~/.kaggle/kaggle.json) and internet connection.\n"
            f"Original error: {exc}"
        ) from exc

    if not downloaded.exists() or not downloaded.is_dir():
        raise RuntimeError(
            f"kagglehub returned an invalid path: {downloaded}. "
            "Expected a directory containing the dataset files."
        )

    image_files = [
        item for item in downloaded.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise RuntimeError(
            f"No image files found in the downloaded dataset at {downloaded}. "
            "The dataset layout may have changed."
        )

    copied = 0
    for item in tqdm(image_files, desc="Copying images to data/raw"):
        rel = item.relative_to(downloaded)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(item, dest)
            copied += 1

    print(f"Done — {copied} new files copied to {target} ({len(image_files) - copied} already present).")
    return target


def read_best_enhancement(metrics_dir: str | Path | None = None) -> str:
    """Read best enhancement method from results/metrics/best_enhancement.txt."""
    if metrics_dir is None:
        metrics_dir = get_project_paths()["metrics"]
    path = Path(metrics_dir) / "best_enhancement.txt"
    if not path.exists():
        return "none"
    return path.read_text(encoding="utf-8").strip()
