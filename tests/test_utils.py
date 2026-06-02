"""Unit tests for src/utils.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import label_encode, load_splits, save_splits

# ---------------------------------------------------------------------------
# label_encode
# ---------------------------------------------------------------------------

def test_label_encode_series_fresh_rotten():
    y = pd.Series(["fresh", "rotten", "fresh"])
    encoded, mapping = label_encode(y)
    assert list(encoded) == [0, 1, 0]
    assert mapping == {"fresh": 0, "rotten": 1}


def test_label_encode_ndarray():
    y = np.array(["rotten", "fresh"])
    encoded, _ = label_encode(y)
    assert list(encoded) == [1, 0]


def test_label_encode_series_unknown_raises():
    """Unknown label in pd.Series must raise ValueError, not silently produce NaN."""
    y = pd.Series(["fresh", "Fresh", "rotten"])  # 'Fresh' is unknown
    with pytest.raises(ValueError, match="Unknown label"):
        label_encode(y)


def test_label_encode_ndarray_unknown_raises():
    """Unknown label in ndarray must raise ValueError (same type as pd.Series path)."""
    y = np.array(["fresh", "spoiled"])
    with pytest.raises(ValueError, match="Unknown label"):
        label_encode(y)


def test_label_encode_series_preexisting_nan_raises():
    """Pre-existing NaN in pd.Series must raise ValueError about missing data."""
    y = pd.Series(["fresh", None, "rotten"])
    with pytest.raises(ValueError, match="missing value"):
        label_encode(y)


def test_label_encode_ndarray_case_sensitive_raises():
    """'Fresh' (uppercase) is not a valid label — must raise ValueError."""
    y = np.array(["fresh", "Fresh"])
    with pytest.raises(ValueError, match="Unknown label"):
        label_encode(y)


def test_label_encode_returns_int64_dtype():
    y = pd.Series(["fresh", "rotten"])
    encoded, _ = label_encode(y)
    assert encoded.dtype == np.int64


# ---------------------------------------------------------------------------
# save_splits / load_splits round-trip (relative path portability)
# ---------------------------------------------------------------------------

def _make_dummy_df(root: Path) -> pd.DataFrame:
    """Create a tiny DataFrame with real-looking relative filepaths."""
    files = [root / "data" / "raw" / "Apple" / "fresh" / f"img_{i:03d}.jpg" for i in range(4)]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")  # empty placeholder
    return pd.DataFrame({
        "filepath": [str(f) for f in files],
        "label": ["fresh"] * 4,
        "commodity": ["Apple"] * 4,
    })


def test_save_splits_stores_relative_paths(tmp_path):
    df = _make_dummy_df(tmp_path)
    splits_file = tmp_path / "data" / "splits.json"

    # Monkey-patch get_project_root so save_splits uses tmp_path as root
    import src.utils as utils_mod
    original = utils_mod.get_project_root
    utils_mod.get_project_root = lambda: tmp_path
    try:
        save_splits(df.iloc[:2], df.iloc[2:3], df.iloc[3:], path=splits_file)
    finally:
        utils_mod.get_project_root = original

    data = json.loads(splits_file.read_text())
    for split in ("train", "val", "test"):
        for rec in data[split]:
            assert not Path(rec["filepath"]).is_absolute(), (
                f"Expected relative path, got: {rec['filepath']}"
            )
            assert "\\" not in rec["filepath"], "Path separator must be POSIX forward-slash"


def test_load_splits_resolves_to_absolute(tmp_path):
    df = _make_dummy_df(tmp_path)
    splits_file = tmp_path / "data" / "splits.json"

    import src.utils as utils_mod
    original = utils_mod.get_project_root
    utils_mod.get_project_root = lambda: tmp_path
    try:
        save_splits(df.iloc[:2], df.iloc[2:3], df.iloc[3:], path=splits_file)
        train, val, test = load_splits(path=splits_file)
    finally:
        utils_mod.get_project_root = original

    for fp in train["filepath"]:
        assert Path(fp).is_absolute(), f"Expected absolute path after load, got: {fp}"
