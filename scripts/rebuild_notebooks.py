"""Rebuild all 4 notebooks with Kaggle Notebooks setup cells."""
import json
import uuid
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def make_id():
    return uuid.uuid4().hex[:8]


def code_cell(source, cell_id=None):
    if isinstance(source, str):
        source = list(source.splitlines(True))
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or make_id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source, cell_id=None):
    if isinstance(source, str):
        source = list(source.splitlines(True))
    return {
        "cell_type": "markdown",
        "id": cell_id or make_id(),
        "metadata": {},
        "source": source,
    }


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── Sel setup Kaggle (sama untuk semua notebook) ─────────────────────────
KAGGLE_SETUP = code_cell(
    "# ============================================================\n"
    "# Setup cell - Kaggle Notebooks (Kaggle-only). Jalankan PALING ATAS.\n"
    "# Cara attach dataset: panel kanan > + Add Data > cari\n"
    "#   'fruit and vegetable disease healthy vs rotten' > Add.\n"
    "# ============================================================\n"
    "import os\n"
    "import sys\n"
    "import shutil\n"
    "import subprocess\n"
    "from pathlib import Path\n"
    "\n"
    "# 1. Clone repo dari GitHub (atau pull jika sudah ada di sesi ini)\n"
    'REPO_URL = "https://github.com/faizhuda/pcd-kelompok-17.git"\n'
    'PROJECT_DIR = Path("/kaggle/working/pcd-kelompok-17")\n'
    "if not PROJECT_DIR.exists():\n"
    '    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)], check=True)\n'
    "else:\n"
    '    subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=False)\n'
    "\n"
    "# 2. Working directory ke root project + tambah ke sys.path\n"
    "os.chdir(PROJECT_DIR)\n"
    "if str(PROJECT_DIR) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_DIR))\n"
    "\n"
    "# 3. Dependency inti SUDAH pre-installed di Kaggle -> tidak ada pip install.\n"
    "\n"
    "# 4. Dataset gambar (read-only, hasil + Add Data)\n"
    "# Auto-detect: Kaggle bisa mount di /kaggle/input/<slug> atau\n"
    "# /kaggle/input/datasets/<user>/<slug> tergantung cara attach.\n"
    "_DATASET_SLUG = 'fruit-and-vegetable-disease-healthy-vs-rotten'\n"
    "_candidates = [\n"
    "    Path('/kaggle/input') / _DATASET_SLUG,\n"
    "    Path('/kaggle/input/datasets/muhammad0subhan') / _DATASET_SLUG,\n"
    "]\n"
    "RAW_DATA_DIR = next((p for p in _candidates if p.exists()), None)\n"
    "if RAW_DATA_DIR is None:\n"
    "    # Fallback: cari folder mana saja di /kaggle/input yang berisi gambar dataset\n"
    "    for _p in Path('/kaggle/input').rglob(_DATASET_SLUG):\n"
    "        if _p.is_dir():\n"
    "            RAW_DATA_DIR = _p\n"
    "            break\n"
    'assert RAW_DATA_DIR is not None, "Dataset belum di-attach. + Add Data > cari dataset > Add."\n'
    "\n"
    "# 5. Auto-restore hasil notebook sebelumnya (untuk notebook 03 & 04).\n"
    "#    Attach output run lama via: + Add Data > Your Work / Dataset bersama.\n"
    "def restore_previous_outputs():\n"
    "    # Kaggle mounts notebook outputs di /kaggle/input/notebooks/<user>/<notebook>/\n"
    "    # sehingga perlu rglob, bukan glob satu level.\n"
    "    restored = []\n"
    '    for repo in Path("/kaggle/input").rglob("pcd-kelompok-17"):\n'
    "        if not repo.is_dir():\n"
    "            continue\n"
    '        for sub in ("results", "data/processed"):\n'
    "            src_dir = repo / sub\n"
    "            if src_dir.exists():\n"
    "                shutil.copytree(src_dir, PROJECT_DIR / sub, dirs_exist_ok=True)\n"
    "                restored.append(str(src_dir))\n"
    "    return restored\n"
    "\n"
    "restored = restore_previous_outputs()\n"
    'print("Project :", PROJECT_DIR)\n'
    'print("Dataset :", RAW_DATA_DIR)\n'
    'print("Restore :", restored or "(mulai dari nol)")\n'
)

NEW_ROOT = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "# Setup cell sudah chdir ke PROJECT_DIR & menambah sys.path (Kaggle-only).\n"
    'ROOT = Path("/kaggle/working/pcd-kelompok-17")\n'
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n"
    "\n"
)


# ══════════════════════════════════════════════════════════════════════════
# 01_eda.ipynb
# ══════════════════════════════════════════════════════════════════════════
nb01 = make_nb(
    [
        md_cell(
            "# 01 — EDA & Data Preparation\n"
            "\n"
            "Distribusi dataset, visualisasi sampel, pemeriksaan imbalance,"
            " dan pembuatan split stratified."
        ),
        KAGGLE_SETUP,
        code_cell(
            NEW_ROOT
            + "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "\n"
            "from src.preprocessing import check_integrity\n"
            "from src.utils import (\n"
            "    build_dataset_index,\n"
            "    get_project_paths,\n"
            "    make_splits,\n"
            "    save_splits,\n"
            "    set_seed,\n"
            ")\n"
            "\n"
            "set_seed(42)\n"
            "paths = get_project_paths()\n"
            "paths\n"
        ),
        code_cell(
            "# RAW_DATA_DIR sudah di-set di setup cell (/kaggle/input/...).\n"
            "# Tidak ada download: dataset Kaggle di-attach langsung (read-only).\n"
            'print("Sumber dataset:", RAW_DATA_DIR)\n'
        ),
        code_cell(
            "df = build_dataset_index(RAW_DATA_DIR)\n"
            'print(f"Total citra: {len(df)}")\n'
            "print(f\"Komoditas: {df['commodity'].nunique()}\")\n"
            "print(f\"Label: {df['label'].value_counts().to_dict()}\")\n"
            "df.head()\n"
        ),
        code_cell(
            'fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n'
            'df["label"].value_counts().plot(kind="bar", ax=axes[0], title="Distribusi Kelas")\n'
            'df.groupby(["commodity", "label"]).size().unstack(fill_value=0).plot(\n'
            '    kind="bar", stacked=True, ax=axes[1], title="Komoditas x Kelas"\n'
            ")\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        code_cell(
            "import cv2\n"
            "\n"
            'sample = df.groupby(["commodity", "label"]).first().reset_index().head(6)\n'
            "fig, axes = plt.subplots(2, 3, figsize=(12, 8))\n"
            "for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):\n"
            '    img = cv2.imread(row["filepath"])\n'
            "    if img is not None:\n"
            "        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n"
            "    ax.set_title(f\"{row['commodity']} -- {row['label']}\")\n"
            '    ax.axis("off")\n'
            'plt.suptitle("Contoh Citra")\n'
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        code_cell(
            'corrupt = [fp for fp in df["filepath"] if not check_integrity(fp)]\n'
            'print(f"Citra corrupt / tidak terbaca: {len(corrupt)}")\n'
        ),
        code_cell(
            "train, val, test = make_splits(df)\n"
            'print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")\n'
            'split_path = save_splits(train, val, test, paths["splits"])\n'
            'print(f"Split disimpan ke: {split_path}")\n'
        ),
    ]
)


# ══════════════════════════════════════════════════════════════════════════
# 02_experiments_classical.ipynb
# ══════════════════════════════════════════════════════════════════════════
nb02 = make_nb(
    [
        md_cell(
            "# 02 — Eksperimen Klasik (Skenario 1–8)\n"
            "\n"
            "SVM & Random Forest dengan feature engineering manual. Tiap skenario\n"
            "mengubah **satu variabel** untuk isolasi efek (restorasi, enhancement,\n"
            "segmentasi, fitur, classifier). Jalankan **berurutan** setelah `01_eda.ipynb`."
        ),
        KAGGLE_SETUP,
        code_cell(
            NEW_ROOT
            + "from src.experiments import (\n"
            "    run_classical_scenario,\n"
            "    run_mcnemar_pair,\n"
            "    select_best_enhancement,\n"
            ")\n"
            "from src.utils import build_dataset_index, get_project_paths, make_splits, set_seed\n"
            "\n"
            "set_seed(42)\n"
            "paths = get_project_paths()\n"
            "# Split di-regenerate dari dataset (deterministik, SEED=42) - tidak baca splits.json\n"
            "# RAW_DATA_DIR sudah di-set setup cell (auto-detect path Kaggle)\n"
            "train, val, test = make_splits(build_dataset_index(RAW_DATA_DIR))\n"
            'cache_dir = paths["data_processed"]\n'
            'metrics_dir = paths["metrics"]\n'
            'figures_dir = paths["figures_confusion"]\n'
            'models_dir = paths["models"]\n'
            "print(len(train), len(val), len(test))\n"
        ),
        md_cell(
            "## Skenario 1–4: Baseline, Restorasi (SSR), Enhancement\n"
            "\n"
            "- **S1** = baseline mentah (tanpa restorasi, tanpa enhancement)\n"
            "- **S2** = + restorasi SSR (isolasi efek koreksi pencahayaan vs S1)\n"
            "- **S3/S4** = SSR + CLAHE / gamma. E* dipilih dari S2–S4 (val F1)."
        ),
        code_cell(
            "val_f1_map = {}\n"
            "scenario_results = {}\n"
            "\n"
            "for sid in range(1, 5):\n"
            '    print(f"\\n=== Skenario {sid} ===")\n'
            "    res = run_classical_scenario(\n"
            "        sid, train, val, test,\n"
            "        metrics_dir, figures_dir, models_dir, cache_dir,\n"
            "    )\n"
            "    scenario_results[sid] = res\n"
            "    # E* dipilih di antara skenario ber-SSR (S2 none, S3 clahe, S4 gamma).\n"
            "    # S1 = baseline mentah (tanpa restorasi) -> tidak ikut pemilihan enhancement.\n"
            "    if sid >= 2:\n"
            '        val_f1_map[res["enhancement"]] = res["val_f1"]\n'
            "    print(f\"Val F1: {res['val_f1']:.4f} | Test F1: {res['test_metrics']['f1_weighted']:.4f}\")\n"
            "\n"
            "best_enh = select_best_enhancement(val_f1_map, metrics_dir)\n"
            'print(f"\\nE* (enhancement terbaik): {best_enh}")\n'
        ),
        md_cell(
            "## Skenario 5–8: Segmentasi, Ablasi Fitur, Random Forest\n"
            "\n"
            "- **S5** = E* + segmentasi, semua fitur, SVM (pipeline klasik penuh)\n"
            "- **S6/S7** = ablasi fitur (warna saja / tekstur saja)\n"
            "- **S8** = S5 dengan Random Forest (perbandingan classifier + feature importance)"
        ),
        code_cell(
            "for sid in range(5, 9):\n"
            '    print(f"\\n=== Skenario {sid} ===")\n'
            "    res = run_classical_scenario(\n"
            "        sid, train, val, test,\n"
            "        metrics_dir, figures_dir, models_dir, cache_dir,\n"
            "    )\n"
            "    scenario_results[sid] = res\n"
            "    print(f\"Test F1: {res['test_metrics']['f1_weighted']:.4f}\")\n"
        ),
        md_cell("## Uji Signifikansi McNemar (isolasi tiap tahap)"),
        code_cell(
            "from src.utils import read_best_enhancement\n"
            "\n"
            "best_enh = read_best_enhancement(metrics_dir)\n"
            "# Skenario no-seg yang memakai E* (untuk isolasi efek segmentasi):\n"
            '# none -> S2, clahe -> S3, gamma -> S4.\n'
            'enh_noseg_sid = {"none": 2, "clahe": 3, "gamma": 4}[best_enh]\n'
            'y_true = scenario_results[1]["y_test"]\n'
            "\n"
            "# 1. Efek restorasi SSR: S2 (ssr) vs S1 (mentah)\n"
            "run_mcnemar_pair(\n"
            '    "S2 vs S1 (SSR)", "S2", "S1",\n'
            '    y_true, scenario_results[2]["y_pred"], scenario_results[1]["y_pred"], metrics_dir,\n'
            ")\n"
            "\n"
            "# 2. Efek enhancement E*: S{E*} vs S2 (hanya bermakna bila E* != none)\n"
            "if enh_noseg_sid != 2:\n"
            "    run_mcnemar_pair(\n"
            '        f"E*({best_enh}) vs S2", f"S{enh_noseg_sid}", "S2",\n'
            '        y_true, scenario_results[enh_noseg_sid]["y_pred"], scenario_results[2]["y_pred"], metrics_dir,\n'
            "    )\n"
            "\n"
            "# 3. Efek segmentasi: S5 (E*+seg) vs E* tanpa seg\n"
            "run_mcnemar_pair(\n"
            '    "S5 vs E*-noseg (segmentasi)", "S5", f"S{enh_noseg_sid}",\n'
            '    y_true, scenario_results[5]["y_pred"], scenario_results[enh_noseg_sid]["y_pred"], metrics_dir,\n'
            ")\n"
            'print("McNemar CNN (S9/S10 vs S5) dijalankan di notebook 03.")\n'
        ),
        code_cell(
            "import pandas as pd\n"
            'sig_path = metrics_dir / "significance_tests.csv"\n'
            "if sig_path.exists():\n"
            "    display(pd.read_csv(sig_path))\n"
        ),
    ]
)


# ══════════════════════════════════════════════════════════════════════════
# 03_experiments_cnn.ipynb
# ══════════════════════════════════════════════════════════════════════════
nb03 = make_nb(
    [
        md_cell(
            "# 03 — Eksperimen CNN (Skenario 9–10)\n"
            "\n"
            "MobileNetV2 two-stage fine-tuning, Grad-CAM, McNemar vs S5.\n"
            "- **S9** = SSR + E* + segmentasi (full pipeline klasik, diganti CNN)\n"
            "- **S10** = tanpa restorasi, tanpa enhancement (baseline murni CNN vs S1 klasik)"
        ),
        KAGGLE_SETUP,
        code_cell(
            NEW_ROOT
            + "import numpy as np\n"
            "import tensorflow as tf\n"
            "from sklearn.utils.class_weight import compute_class_weight\n"
            "\n"
            "from src.evaluate import (\n"
            "    append_significance_test,\n"
            "    compute_metrics,\n"
            "    make_gradcam_heatmap,\n"
            "    mcnemar_test,\n"
            "    plot_confusion_matrix,\n"
            "    plot_gradcam,\n"
            "    save_scenario_metrics,\n"
            ")\n"
            "from src.models import (\n"
            "    build_mobilenetv2,\n"
            "    compile_mobilenet,\n"
            "    get_mobilenet_callbacks,\n"
            "    unfreeze_last_layers,\n"
            ")\n"
            "from src.pipeline import image_to_cnn_input, process_image\n"
            "from src.utils import build_dataset_index, get_project_paths, make_splits, read_best_enhancement, set_seed\n"
            "\n"
            "set_seed(42)\n"
            "paths = get_project_paths()\n"
            "# Split di-regenerate dari dataset (deterministik) - identik dengan notebook 01/02\n"
            "# RAW_DATA_DIR sudah di-set setup cell (auto-detect path Kaggle)\n"
            "train_df, val_df, test_df = make_splits(build_dataset_index(RAW_DATA_DIR))\n"
            'enhancement = read_best_enhancement(paths["metrics"])\n'
            'print(f"Menggunakan enhancement E*: {enhancement}")\n'
        ),
        code_cell(
            "# Cache preprocessing ke disk (SSR + segmentasi dihitung sekali, bukan per epoch).\n"
            "CACHE_DIR = Path('/kaggle/temp/tfcache')\n"
            "CACHE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "def make_dataset(df, batch_size=32, shuffle=False,\n"
            "                 restoration='ssr', do_segment=True,\n"
            "                 enhancement_method=None, cache_name=None):\n"
            "    if enhancement_method is None:\n"
            "        enhancement_method = enhancement\n"
            "    def generator():\n"
            '        label_map = {"fresh": 0, "rotten": 1}\n'
            "        for _, row in df.iterrows():\n"
            "            out = process_image(\n"
            '                path=row["filepath"],\n'
            "                restoration=restoration,\n"
            "                enhancement=enhancement_method,\n"
            "                do_segment=do_segment,\n"
            "            )\n"
            '            if out["img"] is None:\n'
            "                continue\n"
            '            x = image_to_cnn_input(out["img"])[0]\n'
            "            y = np.zeros(2, dtype=np.float32)\n"
            "            y[label_map[row['label']]] = 1.0\n"
            "            yield x, y\n"
            "\n"
            "    dataset = tf.data.Dataset.from_generator(\n"
            "        generator,\n"
            "        output_signature=(\n"
            "            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),\n"
            "            tf.TensorSpec(shape=(2,), dtype=tf.float32),\n"
            "        )\n"
            "    )\n"
            "    if cache_name is not None:\n"
            "        dataset = dataset.cache(str(CACHE_DIR / cache_name))\n"
            "    if shuffle:\n"
            "        dataset = dataset.shuffle(buffer_size=2048, seed=42, reshuffle_each_iteration=True)\n"
            "    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)\n"
            "\n"
            "# S9: SSR + E* + segmentasi (identik dengan S5 klasik, tapi CNN)\n"
            "train_ds_s9 = make_dataset(train_df, shuffle=True, restoration='ssr', do_segment=True, cache_name='train_s9')\n"
            "val_ds_s9   = make_dataset(val_df,   restoration='ssr', do_segment=True, cache_name='val_s9')\n"
            "test_ds_s9  = make_dataset(test_df,  restoration='ssr', do_segment=True, cache_name='test_s9')\n"
            "\n"
            "# S10: tanpa restorasi, tanpa enhancement, tanpa segmentasi (baseline murni CNN)\n"
            "train_ds_s10 = make_dataset(train_df, shuffle=True, restoration='none', do_segment=False, enhancement_method='none', cache_name='train_s10')\n"
            "val_ds_s10   = make_dataset(val_df,   restoration='none', do_segment=False, enhancement_method='none', cache_name='val_s10')\n"
            "test_ds_s10  = make_dataset(test_df,  restoration='none', do_segment=False, enhancement_method='none', cache_name='test_s10')\n"
        ),
        code_cell(
            'y_train_labels = train_df["label"].map({"fresh": 0, "rotten": 1}).values\n'
            "classes = np.unique(y_train_labels)\n"
            'weights = compute_class_weight("balanced", classes=classes, y=y_train_labels)\n'
            "class_weight = {int(c): float(w) for c, w in zip(classes, weights)}\n"
            "class_weight\n"
        ),
        md_cell("## Skenario 9: CNN — SSR + E* + Segmentasi (mirror S5)"),
        md_cell("### Stage 1 — Base frozen (20 epoch)"),
        code_cell(
            "model_s9 = build_mobilenetv2(num_classes=2)\n"
            "model_s9 = compile_mobilenet(model_s9, learning_rate=1e-4)\n"
            'cb_s9 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s09_stage1.h5"))\n'
            "\n"
            "history1_s9 = model_s9.fit(\n"
            "    train_ds_s9, validation_data=val_ds_s9, epochs=20,\n"
            "    class_weight=class_weight, callbacks=cb_s9,\n"
            ")\n"
        ),
        md_cell("### Stage 2 — Fine-tune 20 lapisan terakhir (50 epoch)"),
        code_cell(
            "model_s9 = unfreeze_last_layers(model_s9, n=20)\n"
            "model_s9 = compile_mobilenet(model_s9, learning_rate=1e-5)\n"
            'cb2_s9 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s09_stage2.h5"))\n'
            "\n"
            "history2_s9 = model_s9.fit(\n"
            "    train_ds_s9, validation_data=val_ds_s9, epochs=50,\n"
            "    class_weight=class_weight, callbacks=cb2_s9,\n"
            ")\n"
        ),
        md_cell("### Evaluasi Skenario 9"),
        code_cell(
            "import time\n"
            "\n"
            "y_true_list, y_pred_list = [], []\n"
            "t0 = time.perf_counter()\n"
            "n = 0\n"
            "for x_batch, y_batch in test_ds_s9:\n"
            "    preds = model_s9.predict_on_batch(x_batch)\n"
            "    y_pred_list.extend(np.argmax(preds, axis=1))\n"
            "    y_true_list.extend(np.argmax(y_batch.numpy(), axis=1))\n"
            "    n += len(y_batch)\n"
            "\n"
            "infer_ms = (time.perf_counter() - t0) * 1000 / max(n, 1)\n"
            "y_true_s9 = np.array(y_true_list)\n"
            "y_pred_s9 = np.array(y_pred_list)\n"
            "metrics_s9 = compute_metrics(y_true_s9, y_pred_s9)\n"
            "save_scenario_metrics(\n"
            '    9, enhancement, True, "cnn", "MobileNetV2",\n'
            '    metrics_s9, infer_ms, len(y_true_s9), paths["metrics"], restoration="ssr",\n'
            ")\n"
            'plot_confusion_matrix(y_true_s9, y_pred_s9, title="Skenario 9 CNN (SSR+E*+Seg)",\n'
            '                      save_path=paths["figures_confusion"] / "scenario_09.png")\n'
            "metrics_s9\n"
        ),
        md_cell("## Skenario 10: CNN — Tanpa Restorasi, Tanpa Enhancement (mirror S1)"),
        md_cell("### Stage 1 — Base frozen (20 epoch)"),
        code_cell(
            "model_s10 = build_mobilenetv2(num_classes=2)\n"
            "model_s10 = compile_mobilenet(model_s10, learning_rate=1e-4)\n"
            'cb_s10 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s10_stage1.h5"))\n'
            "\n"
            "history1_s10 = model_s10.fit(\n"
            "    train_ds_s10, validation_data=val_ds_s10, epochs=20,\n"
            "    class_weight=class_weight, callbacks=cb_s10,\n"
            ")\n"
        ),
        md_cell("### Stage 2 — Fine-tune 20 lapisan terakhir (50 epoch)"),
        code_cell(
            "model_s10 = unfreeze_last_layers(model_s10, n=20)\n"
            "model_s10 = compile_mobilenet(model_s10, learning_rate=1e-5)\n"
            'cb2_s10 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s10_stage2.h5"))\n'
            "\n"
            "history2_s10 = model_s10.fit(\n"
            "    train_ds_s10, validation_data=val_ds_s10, epochs=50,\n"
            "    class_weight=class_weight, callbacks=cb2_s10,\n"
            ")\n"
        ),
        md_cell("### Evaluasi Skenario 10"),
        code_cell(
            "y_true_list, y_pred_list = [], []\n"
            "t0 = time.perf_counter()\n"
            "n = 0\n"
            "for x_batch, y_batch in test_ds_s10:\n"
            "    preds = model_s10.predict_on_batch(x_batch)\n"
            "    y_pred_list.extend(np.argmax(preds, axis=1))\n"
            "    y_true_list.extend(np.argmax(y_batch.numpy(), axis=1))\n"
            "    n += len(y_batch)\n"
            "\n"
            "infer_ms = (time.perf_counter() - t0) * 1000 / max(n, 1)\n"
            "y_true_s10 = np.array(y_true_list)\n"
            "y_pred_s10 = np.array(y_pred_list)\n"
            "metrics_s10 = compute_metrics(y_true_s10, y_pred_s10)\n"
            "save_scenario_metrics(\n"
            '    10, "none", False, "cnn", "MobileNetV2",\n'
            '    metrics_s10, infer_ms, len(y_true_s10), paths["metrics"], restoration="none",\n'
            ")\n"
            'plot_confusion_matrix(y_true_s10, y_pred_s10, title="Skenario 10 CNN (Raw)",\n'
            '                      save_path=paths["figures_confusion"] / "scenario_10.png")\n'
            "metrics_s10\n"
        ),
        md_cell("## McNemar Significance Tests (dijalankan sebelum Grad-CAM)"),
        code_cell(
            "# S5 (SVM, full pipeline) adalah anchor klasik untuk perbandingan vs CNN.\n"
            "import joblib\n"
            "\n"
            's5_path = paths["models"] / "svm_scenario_05.pkl"\n'
            "if s5_path.exists():\n"
            "    from src.experiments import extract_split_matrix\n"
            "    s5_model = joblib.load(s5_path)\n"
            "    X_test_s5, _, _ = extract_split_matrix(\n"
            '        test_df, read_best_enhancement(paths["metrics"]), True, "all",\n'
            '        paths["data_processed"], restoration="ssr",\n'
            "    )\n"
            "    y_pred_s5 = s5_model.predict(X_test_s5)\n"
            "\n"
            "    # 1. S9 (CNN full) vs S5 (SVM full) — apakah CNN signifikan lebih baik?\n"
            "    stat, pval, concl = mcnemar_test(y_true_s9, y_pred_s9, y_pred_s5)\n"
            '    append_significance_test("S9 vs S5 (CNN vs SVM)", "S9", "S5", stat, pval, concl, paths["metrics"])\n'
            "    print('S9 vs S5:', stat, pval, concl)\n"
            "\n"
            "    # 2. S10 (CNN raw) vs S1 (SVM raw) — perbandingan baseline raw\n"
            "    from src.experiments import extract_split_matrix as esm\n"
            '    X_test_s1, _, _ = esm(test_df, "none", False, "all", paths["data_processed"], restoration="none")\n'
            "    # S1 tidak di-save modelnya karena bukan model utama; re-train cepat (hanya 6 fit).\n"
            "    from src.config import SCENARIO_CONFIG\n"
            "    from src.models import build_svm_pipeline\n"
            '    X_train_s1, _, _ = esm(train_df, "none", False, "all", paths["data_processed"], restoration="none")\n'
            "    from src.utils import label_encode\n"
            "    y_train_s1, _ = label_encode(train_df['label'])\n"
            "    y_test_s1, _ = label_encode(test_df.iloc[:len(X_test_s1)]['label'])\n"
            "    s1_model = build_svm_pipeline()\n"
            "    s1_model.fit(X_train_s1, y_train_s1)\n"
            "    y_pred_s1 = s1_model.predict(X_test_s1)\n"
            "    stat2, pval2, concl2 = mcnemar_test(y_true_s10, y_pred_s10, y_pred_s1)\n"
            '    append_significance_test("S10 vs S1 (CNN-raw vs SVM-raw)", "S10", "S1", stat2, pval2, concl2, paths["metrics"])\n'
            "    print('S10 vs S1:', stat2, pval2, concl2)\n"
            "\n"
            "    # 3. S9 vs S10 — apakah full pipeline CNN > raw CNN?\n"
            "    stat3, pval3, concl3 = mcnemar_test(y_true_s9, y_pred_s9, y_pred_s10)\n"
            '    append_significance_test("S9 vs S10 (full vs raw CNN)", "S9", "S10", stat3, pval3, concl3, paths["metrics"])\n'
            "    print('S9 vs S10:', stat3, pval3, concl3)\n"
            "else:\n"
            '    print("Model S5 tidak ditemukan. Pastikan notebook 02 sudah dijalankan.")\n'
        ),
        md_cell("## Grad-CAM (Skenario 9 — CNN full pipeline)"),
        code_cell(
            "import matplotlib.pyplot as plt\n"
            "\n"
            'gradcam_dir = paths["figures_gradcam"]\n'
            "gradcam_dir.mkdir(parents=True, exist_ok=True)\n"
            'representative = ["Apple", "Banana", "Tomato"]\n'
            "\n"
            "for commodity in representative:\n"
            '    for label in ["fresh", "rotten"]:\n'
            "        subset = test_df[\n"
            "            (test_df[\"commodity\"].str.contains(commodity, case=False, na=False)) &\n"
            '            (test_df["label"] == label)\n'
            "        ]\n"
            "        if subset.empty:\n"
            '            subset = test_df[test_df["label"] == label].head(3)\n'
            "        for _, row in subset.head(3).iterrows():\n"
            "            out = process_image(\n"
            '                path=row["filepath"], restoration="ssr",\n'
            "                enhancement=enhancement, do_segment=True,\n"
            "            )\n"
            '            if out["img"] is None:\n'
            "                continue\n"
            '            x = image_to_cnn_input(out["img"])\n'
            "            heatmap = make_gradcam_heatmap(model_s9, x)\n"
            '            fname = Path(row["filepath"]).stem\n'
            "            save = gradcam_dir / f\"{commodity}_{label}_{fname}.png\"\n"
            "            plot_gradcam(out[\"img\"], heatmap, save_path=save)\n"
            "            plt.close(\"all\")\n"
        ),
    ]
)


# ══════════════════════════════════════════════════════════════════════════
# 04_results_summary.ipynb
# ══════════════════════════════════════════════════════════════════════════
nb04 = make_nb(
    [
        md_cell(
            "# 04 — Ringkasan Hasil\n"
            "\n"
            "Tabel 3, plot komparatif F1, analisis segmentation failures, feature importance S10."
        ),
        KAGGLE_SETUP,
        code_cell(
            NEW_ROOT
            + "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "\n"
            "from src.evaluate import aggregate_feature_importance, plot_feature_importance, print_summary_table\n"
            "from src.utils import get_project_paths, read_best_enhancement\n"
            "\n"
            "paths = get_project_paths()\n"
            'metrics_dir = paths["metrics"]\n'
        ),
        md_cell("## Tabel 3 — Ringkasan Semua Skenario"),
        code_cell("summary = print_summary_table(metrics_dir)\nsummary\n"),
        md_cell("## Plot Komparatif F1-Score"),
        code_cell(
            "if not summary.empty:\n"
            "    fig, ax = plt.subplots(figsize=(12, 5))\n"
            '    sns.barplot(data=summary, x="scenario_id", y="f1_weighted", hue="model", ax=ax)\n'
            '    ax.set_title("F1-Score (weighted) per Skenario")\n'
            '    ax.set_xlabel("Skenario")\n'
            "    plt.tight_layout()\n"
            '    plt.savefig(paths["figures"] / "f1_comparison.png", dpi=150)\n'
            "    plt.show()\n"
        ),
        md_cell("## Enhancement Terpilih (E*)"),
        code_cell('print(f"Best enhancement: {read_best_enhancement(metrics_dir)}")\n'),
        md_cell("## Segmentation Failures"),
        code_cell(
            'fail_path = metrics_dir / "segmentation_failures.csv"\n'
            "if fail_path.exists():\n"
            "    fails = pd.read_csv(fail_path)\n"
            '    display(fails.groupby("commodity").size().sort_values(ascending=False).head(10))\n'
            "else:\n"
            '    print("Belum ada log segmentasi. Jalankan skenario dengan segmentasi aktif.")\n'
        ),
        md_cell("## Uji Signifikansi McNemar"),
        code_cell(
            'sig_path = metrics_dir / "significance_tests.csv"\n'
            "if sig_path.exists():\n"
            "    display(pd.read_csv(sig_path))\n"
            "else:\n"
            '    print("Jalankan notebook 02 dan 03 untuk uji McNemar.")\n'
        ),
        md_cell("## Feature Importance (Skenario 8 — RF)"),
        code_cell(
            "import joblib\n"
            "\n"
            'rf_path = paths["models"] / "rf_scenario_08.pkl"\n'
            "if rf_path.exists():\n"
            "    rf = joblib.load(rf_path)\n"
            "    from src.features import get_feature_group_names\n"
            '    names = get_feature_group_names("all", segmented=True)\n'
            "    if len(names) != len(rf.feature_importances_):\n"
            '        names = [f"f{i}" for i in range(len(rf.feature_importances_))]\n'
            "    labels, vals = aggregate_feature_importance(rf.feature_importances_, names)\n"
            '    plot_feature_importance(vals, labels, save_path=paths["figures"] / "feature_importance_s08.png")\n'
            "    plt.show()\n"
            "else:\n"
            '    print("Model RF S8 belum tersedia. Jalankan notebook 02 terlebih dahulu.")\n'
        ),
        md_cell("## Inference Time Comparison"),
        code_cell(
            'if not summary.empty and "inference_time_ms_per_image" in summary.columns:\n'
            "    fig, ax = plt.subplots(figsize=(10, 4))\n"
            '    sns.barplot(data=summary, x="scenario_id", y="inference_time_ms_per_image", ax=ax)\n'
            '    ax.set_title("Inference Time (ms/image)")\n'
            "    plt.tight_layout()\n"
            "    plt.show()\n"
        ),
    ]
)


if __name__ == "__main__":
    for nb_data, fname in [
        (nb01, "01_eda.ipynb"),
        (nb02, "02_experiments_classical.ipynb"),
        (nb03, "03_experiments_cnn.ipynb"),
        (nb04, "04_results_summary.ipynb"),
    ]:
        out_path = NOTEBOOKS_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        n = len(nb_data["cells"])
        print(f"Written: {fname} ({n} cells)")
        for i, c in enumerate(nb_data["cells"]):
            src_preview = "".join(c["source"])[:60].split("\n")[0]
            print(f"  [{i}] {c['cell_type']}: {src_preview}")
        print()
    print("Semua notebook selesai di-rebuild!")
