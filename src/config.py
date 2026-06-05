"""Experiment configuration: scenario definitions."""

from __future__ import annotations

from typing import Any

# Shared threshold: segmentation is considered failed if foreground < this fraction.
# Must be kept in sync with segmentation.py and features.py imports.
SEGMENTATION_FALLBACK_RATIO: float = 0.05

# Classical ML scenarios (1-9). Each row changes ONE variable vs a sibling so
# every comparison in the report isolates a single factor:
#   - restoration:  S1 (none, raw baseline)  vs  S2 (ssr)          → value of SSR
#   - enhancement:  S2 (none) vs S3 (clahe) vs S4 (gamma)          → pick E*
#   - segmentation: S5 (E*+seg) vs the E* no-seg sibling (S2/3/4)  → value of segmentation
#   - features:     S5 (all) vs S6 (color) vs S7 (texture) vs S8 (shape) → which features matter
#   - classifier:   S5 (SVM) vs S9 (RF)                            → model choice
# "enhancement": "best" is resolved at runtime from results/metrics/best_enhancement.txt.
# CNN scenarios (10 segmented+E*, 11 raw) are handled in notebooks/03_experiments_cnn.ipynb.
SCENARIO_CONFIG: dict[int, dict[str, Any]] = {
    1: {"restoration": "none", "enhancement": "none",  "segment": False, "features": "all", "model": "svm"},
    2: {"restoration": "ssr",  "enhancement": "none",  "segment": False, "features": "all", "model": "svm"},
    3: {"restoration": "ssr",  "enhancement": "clahe", "segment": False, "features": "all", "model": "svm"},
    4: {"restoration": "ssr",  "enhancement": "gamma", "segment": False, "features": "all", "model": "svm"},
    5: {"restoration": "ssr",  "enhancement": "best",  "segment": True,  "features": "all",     "model": "svm"},
    6: {"restoration": "ssr",  "enhancement": "best",  "segment": True,  "features": "color",   "model": "svm"},
    7: {"restoration": "ssr",  "enhancement": "best",  "segment": True,  "features": "texture", "model": "svm"},
    8: {"restoration": "ssr",  "enhancement": "best",  "segment": True,  "features": "shape",   "model": "svm"},
    9: {"restoration": "ssr",  "enhancement": "best",  "segment": True,  "features": "all",     "model": "rf"},
}
