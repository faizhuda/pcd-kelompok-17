"""Experiment configuration: scenario definitions."""

from __future__ import annotations

from typing import Any

# Shared threshold: segmentation is considered failed if foreground < this fraction.
# Must be kept in sync with segmentation.py and features.py imports.
SEGMENTATION_FALLBACK_RATIO: float = 0.05

# Each entry maps scenario_id → pipeline parameters (classical ML only, scenarios 1–10).
# Scenarios 11+ (CNN/MobileNetV2) are managed directly in notebooks/03_experiments_cnn.ipynb.
# "enhancement": "best" is resolved at runtime from results/metrics/best_enhancement.txt.
SCENARIO_CONFIG: dict[int, dict[str, Any]] = {
    1: {"enhancement": "none",  "segment": False, "features": "all", "model": "svm"},
    2: {"enhancement": "clahe", "segment": False, "features": "all", "model": "svm"},
    3: {"enhancement": "histeq","segment": False, "features": "all", "model": "svm"},
    4: {"enhancement": "gamma", "segment": False, "features": "all", "model": "svm"},
    5: {"enhancement": "none",  "segment": True,  "features": "all", "model": "svm"},
    6: {"enhancement": "best",  "segment": True,  "features": "all", "model": "svm"},
    7: {"enhancement": "best",  "segment": True,  "features": "color",   "model": "svm"},
    8: {"enhancement": "best",  "segment": True,  "features": "texture", "model": "svm"},
    9: {"enhancement": "best",  "segment": True,  "features": "shape",   "model": "svm"},
    10: {"enhancement": "best", "segment": True,  "features": "all", "model": "rf"},
}
