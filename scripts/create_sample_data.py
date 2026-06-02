"""Generate minimal sample images under data/raw/ for local testing."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import build_dataset_index, get_project_paths, make_splits, save_splits, set_seed

COMMODITIES = ["Apple", "Tomato"]
LABELS = ["fresh", "rotten"]
IMAGES_PER_CLASS = 20


def main() -> None:
    set_seed(42)
    raw = get_project_paths()["data_raw"]
    rng = np.random.default_rng(42)

    for commodity in COMMODITIES:
        for label in LABELS:
            out_dir = raw / commodity / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(IMAGES_PER_CLASS):
                color = (0, 180, 0) if label == "fresh" else (0, 0, 180)
                img = np.full((256, 256, 3), color, dtype=np.uint8)
                noise = rng.integers(0, 50, img.shape, dtype=np.uint8)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / f"{commodity}_{label}_{i:03d}.jpg"), img)

    df = build_dataset_index(raw)
    train, val, test = make_splits(df)
    save_splits(train, val, test)
    print(f"Created {len(df)} sample images.")
    print(f"Splits saved to {get_project_paths()['splits']}")


if __name__ == "__main__":
    main()
