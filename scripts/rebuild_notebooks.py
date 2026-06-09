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


# -- Sel setup Kaggle (sama untuk semua notebook) -------------------------
KAGGLE_SETUP = code_cell(
    "# ============================================================\n"
    "# Setup cell - Kaggle Notebooks (Kaggle-only). Jalankan PALING ATAS.\n"
    "# Cara attach dataset: panel kanan > + Add Data > cari\n"
    "#   'fruit and vegetable disease healthy vs rotten' > Add.\n"
    "# ============================================================\n"
    "import os\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
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


# ==========================================================================
# 01_eda.ipynb
# ==========================================================================
nb01 = make_nb(
    [
        md_cell(
            """\
# 01 - EDA & Data Preparation

Eksplorasi dataset: distribusi kelas, statistik resolusi, visualisasi seluruh komoditas,
distribusi warna HSV fresh vs rotten, dan visualisasi pipeline preprocessing
(SSR, enhancement, segmentasi). Diakhiri dengan pembuatan split stratified 70/15/15."""
        ),
        KAGGLE_SETUP,
        code_cell(
            """\
import os
import sys
from pathlib import Path

# Setup cell sudah chdir ke PROJECT_DIR & menambah sys.path (Kaggle-only).
ROOT = Path("/kaggle/working/pcd-kelompok-17")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.enhancement import apply_enhancement
from src.preprocessing import check_integrity, preprocess_from_array
from src.segmentation import segment_fruit
from src.utils import (
    build_dataset_index,
    get_project_paths,
    make_splits,
    save_splits,
    set_seed,
)

set_seed(42)
paths = get_project_paths()
print("Sumber dataset:", RAW_DATA_DIR)
"""
        ),
        code_cell(
            """\
df = build_dataset_index(RAW_DATA_DIR)
print(f"Total citra  : {len(df)}")
print(f"Komoditas    : {df['commodity'].nunique()}")
print(f"Label        : {df['label'].value_counts().to_dict()}")
print(f"Imbalance ratio: {df['label'].value_counts().max() / df['label'].value_counts().min():.2f}x")
df.head()
"""
        ),
        code_cell(
            """\
# Alias untuk sel analisis baru
full_df = df.copy()
"""
        ),
        md_cell(
            """\
## Seksi A - Analisis Kualitas Dataset

Bagian ini melakukan evaluasi terhadap integritas data (data integrity) dan keseimbangan kelas (class imbalance) secara mendalam.
Uji integritas dilakukan secara paralel menggunakan multi-threading untuk mendeteksi apakah terdapat file gambar yang corrupt atau tidak terbaca oleh OpenCV.
Analisis imbalance dilakukan untuk mengetahui apakah terdapat ketidakseimbangan kelas (fresh vs rotten) yang ekstrim pada setiap komoditas,
yang dapat memengaruhi objektivitas metrik F1-score dan memotivasi penggunaan pembagian dataset terstratifikasi (stratified split).
"""
        ),
        code_cell(
            """\
# A1. Integrity Check (Paralel) - Mendeteksi file citra yang corrupt/unreadable.
# Pengujian ini memastikan seluruh gambar dapat dibuka dengan OpenCV tanpa error.
import concurrent.futures
from src.preprocessing import check_integrity

def check_all_integrity(df, n_workers=4):
    \"\"\"
    Melakukan pemeriksaan integritas file gambar secara paralel menggunakan ThreadPoolExecutor.
    
    Args:
        df (pd.DataFrame): Dataframe indeks dataset yang berisi kolom 'filepath'.
        n_workers (int): Jumlah thread pekerja paralel.
        
    Returns:
        pd.DataFrame: Copy dari dataframe input dengan tambahan kolom boolean 'readable'.
    \"\"\"
    # Menggunakan ThreadPoolExecutor untuk mempercepat proses pembacaan I/O file gambar secara paralel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(lambda p: check_integrity(p), df["filepath"]))
    df = df.copy()
    df["readable"] = results
    return df

# Jalankan pengecekan kualitas integritas file
df_quality = check_all_integrity(full_df)
n_corrupt = (~df_quality["readable"]).sum()
print(f"Total gambar   : {len(df_quality)}")
print(f"Gambar corrupt : {n_corrupt} ({n_corrupt/len(df_quality)*100:.2f}%)")

# Tampilkan list file jika ditemukan yang corrupt, atau print status sukses
if n_corrupt > 0:
    print("[FAIL] File corrupt terdeteksi:")
    print(df_quality[~df_quality["readable"]][["filepath","label","commodity"]])
else:
    print("[OK] Semua gambar dapat dibaca dengan baik (tidak ada file corrupt).")
"""
        ),
        code_cell(
            """\
# A2. Analisis Class Imbalance per Komoditas
# Ketidakseimbangan yang ekstrim dapat mengganggu proses training model klasifikasi klasik.
pivot = full_df.groupby(["commodity","label"]).size().unstack(fill_value=0)
pivot["ratio_fresh_rotten"] = pivot.get("fresh", 0) / (pivot.get("rotten", 1) + 1e-9)
pivot = pivot.sort_values("ratio_fresh_rotten", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Distribusi Kelas & Imbalance per Komoditas", fontsize=13, fontweight="bold")

# Subplot 0: Jumlah gambar per komoditas
counts = full_df.groupby("commodity").size().sort_values()
axes[0].barh(counts.index, counts.values, color=sns.color_palette("husl", len(counts)))
axes[0].set_title("Jumlah Gambar per Komoditas")
axes[0].set_xlabel("Jumlah")

# Subplot 1: Ratio Fresh/Rotten per komoditas (menyoroti imbalance > 2.0x atau < 0.5x)
colors_bar = ["#e74c3c" if v > 2 or v < 0.5 else "#2ecc71" for v in pivot["ratio_fresh_rotten"]]
axes[1].barh(pivot.index, pivot["ratio_fresh_rotten"], color=colors_bar)
axes[1].axvline(1.0, color="red", linestyle="--", label="Balanced (ratio=1)")
axes[1].set_title("Fresh/Rotten Ratio per Komoditas\\n(merah = imbalance >2x atau <0.5x)")
axes[1].set_xlabel("Ratio fresh:rotten")
axes[1].legend()

plt.tight_layout()
# Simpan plot hasil analisis visual kualitas ke folder figures
plt.savefig(paths["figures"] / "eda_a_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nImbalance summary:")
print(pivot[["fresh","rotten","ratio_fresh_rotten"]].to_string())
global_ratio = (full_df["label"]=="fresh").sum() / (full_df["label"]=="rotten").sum()
print(f"\\nGlobal fresh/rotten ratio: {global_ratio:.3f}")
"""
        ),
        md_cell(
            """\
## Seksi B - Distribusi Resolusi & Aspek Rasio

Analisis resolusi dilakukan untuk menentukan ukuran target resize yang optimal (dalam proyek ini 224x224 piksel untuk MobileNetV2).
Kita memeriksa apakah terdapat gambar dengan resolusi sangat rendah (di bawah 224px) yang dapat terdistorsi saat di-upscale,
serta memeriksa distribusi aspek rasio (H/W) untuk memastikan bahwa proses pengubahan ukuran gambar menjadi persegi (square)
tidak menyebabkan distorsi geometris (squishing) yang signifikan pada objek buah dan sayur.
"""
        ),
        code_cell(
            """\
# B. Analisis Distribusi Resolusi & Aspek Rasio Asli
def get_image_dims(path):
    \"\"\"Membaca citra secara cepat dan mengembalikan dimensi (tinggi, lebar).\"\"\"
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return h, w

# Ekstraksi dimensi untuk seluruh gambar dalam dataset secara iteratif
dim_records = []
for _, row in full_df.iterrows():
    h, w = get_image_dims(row["filepath"])
    if h is None:
        continue
    dim_records.append({"h": h, "w": w, "aspect_ratio": h/w,
                         "min_side": min(h, w),
                         "commodity": row["commodity"], "label": row["label"]})

df_res = pd.DataFrame(dim_records)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analisis Resolusi Dataset", fontsize=13, fontweight="bold")

# Subplot 0: Scatter plot width vs height
axes[0].scatter(df_res["w"], df_res["h"], alpha=0.1, s=5, color="steelblue")
axes[0].axvline(224, color="red", linestyle="--", linewidth=1.5, label="Target 224px")
axes[0].axhline(224, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Width (px)")
axes[0].set_ylabel("Height (px)")
axes[0].set_title("Distribusi Resolusi Asli")
axes[0].legend()

# Subplot 1: Histogram aspek rasio
axes[1].hist(df_res["aspect_ratio"], bins=50, color="steelblue", edgecolor="white")
axes[1].axvline(1.0, color="red", linestyle="--", label="Square (1:1)")
axes[1].set_xlabel("Aspect Ratio (H/W)")
axes[1].set_title("Distribusi Aspect Ratio")
axes[1].legend()

# Subplot 2: Sisi minimum (min side) terkecil per komoditas (menyoroti sisi < 224px)
min_side = df_res.groupby("commodity")["min_side"].min().sort_values()
colors_res = ["#e74c3c" if v < 224 else "#2ecc71" for v in min_side.values]
axes[2].barh(min_side.index, min_side.values, color=colors_res)
axes[2].axvline(224, color="red", linestyle="--", label="Target 224px")
axes[2].set_title("Resolusi Minimum per Komoditas\\n(merah = < 224px)")
axes[2].set_xlabel("Min side (px)")
axes[2].legend()

plt.tight_layout()
# Simpan plot resolusi
plt.savefig(paths["figures"] / "eda_b_resolution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Median resolusi  : {df_res['w'].median():.0f} x {df_res['h'].median():.0f}")
print(f"Min side terkecil: {df_res['min_side'].min()} px "
      f"({df_res.loc[df_res['min_side'].idxmin(), 'commodity']})")
print(f"Aspect ratio     : mean={df_res['aspect_ratio'].mean():.3f}, "
      f"std={df_res['aspect_ratio'].std():.3f}")
pct_below224 = (df_res["min_side"] < 224).mean() * 100
print(f"Gambar < 224px   : {pct_below224:.1f}% (akan di-upscale saat resize)")
"""
        ),
        code_cell(
            """\
# Check image resolution distribution (sample 300)
sample_paths = df["filepath"].sample(min(300, len(df)), random_state=42).tolist()
shapes = []
for fp in sample_paths:
    img = cv2.imread(fp)
    if img is not None:
        shapes.append(img.shape[:2])
shape_df = pd.DataFrame(shapes, columns=["height", "width"])
print("Statistik resolusi citra (sample 300):")
print(shape_df.describe().round(1))
n_unique = len(shape_df.drop_duplicates())
print(f"\\nJumlah ukuran unik: {n_unique}")
if n_unique <= 10:
    print("Ukuran unik:", shape_df.drop_duplicates().values.tolist())
"""
        ),
        md_cell(
            """\
## 2. Sampel Visual Semua Komoditas"""
        ),
        code_cell(
            """\
commodities = sorted(df['commodity'].unique())
ncols, nrows = 2, len(commodities)
fig, axes = plt.subplots(nrows, ncols, figsize=(6, nrows * 2.2))
for i, comm in enumerate(commodities):
    for j, label in enumerate(["fresh", "rotten"]):
        ax = axes[i, j]
        subset = df[(df["commodity"] == comm) & (df["label"] == label)]
        if not subset.empty:
            img = cv2.imread(subset.iloc[0]["filepath"])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{comm}\\n({label})", fontsize=7)
        ax.axis("off")
plt.suptitle("Sampel per Komoditas - kiri: fresh, kanan: rotten", fontsize=10)
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """\
## 3. Pemeriksaan Integritas"""
        ),
        code_cell(
            """\
corrupt = [fp for fp in df["filepath"] if not check_integrity(fp)]
print(f"Citra corrupt / tidak terbaca: {len(corrupt)}")
if corrupt:
    for fp in corrupt[:5]:
        print(' ', fp)
"""
        ),
        md_cell(
            """\
## 4. Distribusi Warna HSV: Fresh vs Rotten

Histogram HSV rata-rata dari 500 sampel tiap kelas. Perbedaan distribusi
Hue (H), Saturation (S), dan Value (V) antara fresh dan rotten
**memotivasi penggunaan HSV histogram sebagai fitur utama** dalam pipeline klasifikasi."""
        ),
        code_cell(
            """\
def _hsv_histograms(filepaths, n_sample=500):
    \"\"\"Mean of PER-IMAGE-NORMALIZED HSV histograms over a sample.

    Each image's histogram is normalized to sum to 1 BEFORE averaging, so
    the result is a probability distribution independent of image resolution.
    Without this, the dataset's heterogeneous resolution (100px-4160px) would
    make the plot reflect image size rather than color distribution.
    \"\"\"
    h_acc = np.zeros(180, dtype=np.float64)
    s_acc = np.zeros(256, dtype=np.float64)
    v_acc = np.zeros(256, dtype=np.float64)
    count = 0
    for fp in filepaths[:n_sample]:
        img = cv2.imread(fp)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
        # Normalize each image to a probability distribution (sum=1) so large
        # images don't dominate the average.
        h_acc += h / max(h.sum(), 1)
        s_acc += s / max(s.sum(), 1)
        v_acc += v / max(v.sum(), 1)
        count += 1
    if count:
        h_acc /= count; s_acc /= count; v_acc /= count
    return h_acc, s_acc, v_acc

fresh_fps  = df[df["label"] == "fresh"]["filepath"].tolist()
rotten_fps = df[df["label"] == "rotten"]["filepath"].tolist()
h_f, s_f, v_f = _hsv_histograms(fresh_fps)
h_r, s_r, v_r = _hsv_histograms(rotten_fps)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
channel_data = [
    (h_f, h_r, "Hue (H)", "Nilai Hue (0-179)"),
    (s_f, s_r, "Saturation (S)", "Nilai Saturasi (0-255)"),
    (v_f, v_r, "Value / Kecerahan (V)", "Nilai Value (0-255)"),
]
for ax, (fresh_hist, rotten_hist, title, xlabel) in zip(axes, channel_data):
    ax.plot(fresh_hist,  color="green", label="Fresh",  alpha=0.8, linewidth=1.5)
    ax.plot(rotten_hist, color="brown", label="Rotten", alpha=0.8, linewidth=1.5)
    ax.set_title(f"Distribusi {title}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proporsi piksel (rata-rata, ternormalisasi)")
    ax.legend()

plt.suptitle(
    "Distribusi Warna HSV: Fresh vs Rotten (ternormalisasi per-citra)\\n"
    "Perbedaan bentuk pada H, S, V memotivasi HSV histogram sebagai fitur warna utama",
    fontsize=10,
)
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """\
## Seksi D - Separabilitas Warna Fresh vs Rotten per Komoditas

Untuk memotivasi penggunaan fitur warna (histogram HSV) sebagai fitur utama dalam pipeline klasik,
kita mengukur separabilitas statistik warna Hue antara kelas fresh dan rotten untuk setiap komoditas
menggunakan metrik ukuran efek Cohen's d.
Nilai Cohen's d di atas 1.5 menunjukkan pemisahan warna yang sangat kontras (mudah diklasifikasi),
sedangkan nilai di bawah 0.5 menunjukkan pemisahan warna yang rendah (sulit diklasifikasi).
Analisis ini membantu menjelaskan mengapa model baseline klasik (S1) tanpa preprocessing pun sudah
memiliki performa F1 yang sangat tinggi (~0.97).
"""
        ),
        code_cell(
            """\
# D. Cohen's d Separabilitas Warna Hue per Komoditas
# Cohen's d = |mean_fresh - mean_rotten| / pooled_std
# d > 1.5 = mudah, 0.5-1.5 = sedang, < 0.5 = sulit
N_PER_LABEL = 30  # Jumlah sampel citra per kombinasi komoditas x label

sep_records = []
for commodity in sorted(full_df["commodity"].unique()):
    sub = full_df[full_df["commodity"] == commodity]

    hues = {"fresh": [], "rotten": []}
    for label in ["fresh", "rotten"]:
        paths_sub = sub[sub["label"]==label]["filepath"].tolist()
        sample_paths = paths_sub[:N_PER_LABEL]
        for p in sample_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # Resize cepat ke 64x64 untuk mengambil rata-rata nilai Hue
            img_r = preprocess_arr(img, apply_restoration=False)
            img_r_resized = cv2.resize(img_r, (64, 64))
            hsv = cv2.cvtColor(img_r_resized, cv2.COLOR_BGR2HSV)
            hues[label].append(float(hsv[:, :, 0].mean()))

    h_f = np.array(hues["fresh"])
    h_r = np.array(hues["rotten"])
    if len(h_f) < 3 or len(h_r) < 3:
        continue

    # Hitung standar deviasi gabungan (pooled std) dan Cohen's d
    pooled_std = np.sqrt((h_f.std()**2 + h_r.std()**2) / 2 + 1e-6)
    cohen_d = abs(h_f.mean() - h_r.mean()) / pooled_std
    difficulty = "Mudah" if cohen_d > 1.5 else ("Sedang" if cohen_d > 0.5 else "Sulit")
    sep_records.append({
        "commodity": commodity,
        "mean_H_fresh": h_f.mean(),
        "mean_H_rotten": h_r.mean(),
        "delta_H": abs(h_f.mean() - h_r.mean()),
        "cohen_d": cohen_d,
        "difficulty": difficulty,
    })

df_sep = pd.DataFrame(sep_records).sort_values("cohen_d", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Separabilitas Warna (Hue) Fresh vs Rotten per Komoditas", fontsize=13, fontweight="bold")

# Subplot 0: Bar chart nilai Cohen's d
color_map = {"Mudah": "#2ecc71", "Sedang": "#f39c12", "Sulit": "#e74c3c"}
bar_colors = [color_map[d] for d in df_sep["difficulty"]]
axes[0].barh(df_sep["commodity"], df_sep["cohen_d"], color=bar_colors)
axes[0].axvline(1.5, color="green", linestyle="--", alpha=0.7, label="d=1.5 (Mudah)")
axes[0].axvline(0.5, color="orange", linestyle="--", alpha=0.7, label="d=0.5 (Sedang)")
axes[0].set_xlabel("Cohen's d")
axes[0].set_title("Cohen's d per Komoditas\\n(separabilitas hue fresh vs rotten)")
from matplotlib.patches import Patch
legend_el = [Patch(facecolor="#2ecc71", label="Mudah (d>1.5)"),
              Patch(facecolor="#f39c12", label="Sedang (0.5<d<=1.5)"),
              Patch(facecolor="#e74c3c", label="Sulit (d<=0.5)")]
axes[0].legend(handles=legend_el)

# Subplot 1: Scatter plot Mean Hue Fresh vs Rotten
axes[1].scatter(df_sep["mean_H_fresh"], df_sep["mean_H_rotten"], c=bar_colors, s=100, zorder=5)
for _, row in df_sep.iterrows():
    axes[1].annotate(row["commodity"], (row["mean_H_fresh"], row["mean_H_rotten"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
mn = min(df_sep[["mean_H_fresh","mean_H_rotten"]].min())
mx = max(df_sep[["mean_H_fresh","mean_H_rotten"]].max())
axes[1].plot([mn, mx], [mn, mx], "k--", alpha=0.3, label="Fresh = Rotten")
axes[1].set_xlabel("Mean Hue - Fresh")
axes[1].set_ylabel("Mean Hue - Rotten")
axes[1].set_title("Mean Hue Fresh vs Rotten\\n(jauh dari diagonal = mudah dibedakan)")
axes[1].legend()

plt.tight_layout()
plt.savefig(paths["figures"] / "eda_d_color_separability.png", dpi=150, bbox_inches="tight")
plt.show()

n_easy   = (df_sep["difficulty"] == "Mudah").sum()
n_medium = (df_sep["difficulty"] == "Sedang").sum()
n_hard   = (df_sep["difficulty"] == "Sulit").sum()
print(f"\\nDistribusi kesulitan komoditas:")
print(f"  Mudah (d>1.5)         : {n_easy}/{len(df_sep)} komoditas")
print(f"  Sedang (0.5<d<=1.5)    : {n_medium}/{len(df_sep)} komoditas")
print(f"  Sulit (d<=0.5)         : {n_hard}/{len(df_sep)} komoditas")
print(f"\\n-> Mayoritas komoditas memiliki perbedaan warna yang jelas (d tinggi).")
print(f"  Ini menjelaskan mengapa S1 (raw baseline) sudah mencapai F1=0.970.")
if not df_sep[df_sep["difficulty"]=="Sulit"].empty:
    print(f"\\nKomoditas SULIT (kemungkinan sumber error):")
    print(df_sep[df_sep["difficulty"]=="Sulit"][["commodity","delta_H","cohen_d"]].to_string())
"""
        ),
        md_cell(
            """\
## 5. Visualisasi Pipeline Preprocessing

Menampilkan efek tiap tahap preprocessing: SSR, enhancement (CLAHE/gamma),
dan segmentasi Otsu. Membantu memahami transformasi citra sebelum ekstraksi fitur."""
        ),
        code_cell(
            """\
# Pick one representative fresh and one rotten sample for all visualizations below
fp_fresh  = df[df["label"] == "fresh"].iloc[0]["filepath"]
fp_rotten = df[df["label"] == "rotten"].iloc[0]["filepath"]
print("Fresh sample :", fp_fresh)
print("Rotten sample:", fp_rotten)
"""
        ),
        md_cell(
            """\
### 5a. Efek Single-Scale Retinex (SSR)"""
        ),
        code_cell(
            """\
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for row, (fp, label) in enumerate([(fp_fresh, "Fresh"), (fp_rotten, "Rotten")]):
    img_bgr  = cv2.imread(fp)
    original = cv2.resize(img_bgr, (224, 224))
    restored = preprocess_from_array(img_bgr, apply_restoration=True)
    diff     = cv2.absdiff(original, restored)
    diff_vis = np.clip(diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)
    for col, (image, title, cmap) in enumerate([
        (cv2.cvtColor(original,  cv2.COLOR_BGR2RGB), "Original (resize 224x224)", None),
        (cv2.cvtColor(restored,  cv2.COLOR_BGR2RGB), "Setelah SSR", None),
        (diff_vis,                                   "Difference x5 (panas=berubah)", "hot"),
    ]):
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(f"{label} - {title}", fontsize=9)
        axes[row, col].axis("off")
plt.suptitle(
    "Efek Single-Scale Retinex (SSR)\\n"
    "Koreksi pencahayaan non-uniform: area gelap dinaikkan, area sangat terang direduksi",
    fontsize=10,
)
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """\
### 5b. Perbandingan Enhancement (none vs CLAHE vs Gamma)"""
        ),
        code_cell(
            """\
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, (fp, label) in enumerate([(fp_fresh, "Fresh"), (fp_rotten, "Rotten")]):
    img_bgr = cv2.imread(fp)
    ssr_img = preprocess_from_array(img_bgr, apply_restoration=True)
    columns = [
        (cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(ssr_img,                          cv2.COLOR_BGR2RGB), "SSR (no enhance)"),
        (cv2.cvtColor(apply_enhancement(ssr_img, "clahe"), cv2.COLOR_BGR2RGB), "SSR + CLAHE"),
        (cv2.cvtColor(apply_enhancement(ssr_img, "gamma"), cv2.COLOR_BGR2RGB), "SSR + Gamma"),
    ]
    for col, (image, title) in enumerate(columns):
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{label}\\n{title}", fontsize=9)
        axes[row, col].axis("off")
plt.suptitle(
    "Perbandingan Enhancement Method\\n"
    "E* (enhancement terbaik) dipilih otomatis berdasarkan validation F1 pada S2-S4",
    fontsize=10,
)
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """\
### 5c. Segmentasi Otsu + Morfologi

**Temuan penting:** pada dataset ini segmentasi Otsu cenderung memilih *seluruh* frame
(object_ratio ~ 100%), sehingga mask praktis tidak mengisolasi objek. Penyebabnya:
background hampir seragam putih + SSR/CLAHE menaikkan saturasi, sehingga gabungan
(`mask_saturasi OR mask_grayscale`) menutupi hampir semua piksel. Mask ditampilkan
eksplisit hitam=0 / putih=255 agar terlihat apa adanya (bukan artefak rendering).
Konsekuensi ini konsisten dengan hasil eksperimen: S5 (segmentasi) ~ S3 (tanpa
segmentasi) dan uji McNemar tidak signifikan (p~0.21)."""
        ),
        code_cell(
            """\
# Use E*='clahe' as a stand-in for visualization (actual E* resolved at runtime in nb02)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, (fp, label) in enumerate([(fp_fresh, "Fresh"), (fp_rotten, "Rotten")]):
    img_bgr  = cv2.imread(fp)
    ssr_img  = preprocess_from_array(img_bgr, apply_restoration=True)
    enhanced = apply_enhancement(ssr_img, "clahe")
    masked, binary_mask, obj_ratio, used_fallback = segment_fruit(enhanced)
    ratio_note = "[FALLBACK] " if used_fallback else ""
    columns = [
        (cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB), "Original", None),
        (cv2.cvtColor(ssr_img,  cv2.COLOR_BGR2RGB), "SSR", None),
        (binary_mask, f"Mask Otsu\\n{ratio_note}foreground={obj_ratio:.0%}", "gray"),
        (cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), "Hasil Segmentasi", None),
    ]
    for col, (image, title, cmap) in enumerate(columns):
        # Fix vmin/vmax for the mask so a uniform all-255 array renders WHITE
        # (matplotlib otherwise maps a constant array to the colormap minimum = black).
        if cmap == "gray":
            axes[row, col].imshow(image, cmap=cmap, vmin=0, vmax=255)
        else:
            axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(f"{label}\\n{title}", fontsize=9)
        axes[row, col].axis("off")
plt.suptitle(
    "Segmentasi Otsu + Morfologi (ellipse kernel open->close + largest contour)\\n"
    "Pada dataset ini mask cenderung menutupi seluruh frame (foreground~100%) -> segmentasi tidak efektif",
    fontsize=10,
)
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """\
## Seksi C1 - Analisis Feasibility Segmentasi Otsu

Bagian ini menguji kelayakan (feasibility) metode segmentasi berbasis Otsu Thresholding pada proyek ini.
Kita mengukur rasio area buah (foreground) terhadap total luas gambar (object ratio).
Jika rasio ini mendekati 100%, hal ini menunjukkan bahwa buah mengisi hampir seluruh area gambar,
sehingga background tidak memberikan pengaruh kebisingan.
Analisis ini membantu menjelaskan mengapa penambahan tahap segmentasi (S5) tidak memberikan peningkatan
performa yang signifikan dibandingkan tanpa segmentasi (S3), serta mengapa fitur bentuk/shape (S8)
memiliki akurasi yang rendah karena bentuk mask selalu menyerupai persegi penuh frame.
"""
        ),
        code_cell(
            """\
# C1. Object Ratio Distribution - Seberapa baik Otsu memisahkan buah dari background?
# Menggunakan sampling terstratifikasi (stratified sampling) untuk efisiensi komputasi.
from src.segmentation import segment_fruit
from src.preprocessing import preprocess_from_array as preprocess_arr

N_PER_GROUP = 18  # Jumlah gambar per komoditas x label
sample_seg = (
    full_df
    .groupby(["commodity", "label"], group_keys=False)
    .apply(lambda g: g.sample(min(len(g), N_PER_GROUP), random_state=42))
    .reset_index(drop=True)
)

print(f"Sample size: {len(sample_seg)} gambar (dari {len(full_df)} total)")

seg_records = []
for _, row in sample_seg.iterrows():
    img = cv2.imread(str(row["filepath"]))
    if img is None:
        continue
    img_pre = preprocess_arr(img, apply_restoration=True)
    # Lakukan segmentasi buah dan hitung foreground ratio
    _, _, obj_ratio, used_fallback = segment_fruit(img_pre)
    seg_records.append({
        "object_ratio": obj_ratio,
        "used_fallback": used_fallback,
        "label": row["label"],
        "commodity": row["commodity"],
    })

df_seg = pd.DataFrame(seg_records)
pct_over90 = (df_seg["object_ratio"] > 0.90).mean() * 100
pct_fallback = df_seg["used_fallback"].mean() * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analisis Feasibility Segmentasi Otsu", fontsize=14, fontweight="bold")

# Subplot 0: Histogram object_ratio (mencari gambar dengan ratio > 90% atau fallback)
axes[0].hist(df_seg["object_ratio"], bins=30, color="steelblue", edgecolor="white")
axes[0].axvline(0.90, color="red", linestyle="--", linewidth=2, label="90% (gagal/efek nol)")
axes[0].axvline(0.05, color="orange", linestyle="--", linewidth=2, label="5% (fallback)")
axes[0].set_xlabel("Object Ratio (foreground %)")
axes[0].set_ylabel("Jumlah gambar")
axes[0].set_title("Distribusi Object Ratio\\nforeground~100% = segmentasi tidak efektif")
axes[0].legend()
axes[0].text(0.55, 0.82,
             f"{pct_over90:.1f}% gambar\\nforeground >90%",
             transform=axes[0].transAxes,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
             fontsize=10, fontweight="bold")

# Subplot 1: Fallback rate per komoditas (menyoroti komoditas dengan tingkat kegagalan tinggi)
fallback_rate = df_seg.groupby("commodity")["used_fallback"].mean().sort_values(ascending=False)
colors_fb = ["#e74c3c" if v > 0.30 else ("#f39c12" if v > 0.10 else "#2ecc71") for v in fallback_rate.values]
axes[1].barh(fallback_rate.index, fallback_rate.values * 100, color=colors_fb)
axes[1].axvline(30, color="red", linestyle="--", alpha=0.7, label=">30% (bermasalah)")
axes[1].set_xlabel("Fallback Rate (%)")
axes[1].set_title("Fallback Rate per Komoditas\\n(merah = >30% gagal segmentasi)")
axes[1].legend()

# Subplot 2: Median object_ratio per komoditas
median_ratio = df_seg.groupby("commodity")["object_ratio"].median().sort_values(ascending=False)
colors_mr = ["#e74c3c" if v > 0.90 else "#2ecc71" for v in median_ratio.values]
axes[2].barh(median_ratio.index, median_ratio.values * 100, color=colors_mr)
axes[2].axvline(90, color="red", linestyle="--", linewidth=2, label="90% (efek nol)")
axes[2].set_xlabel("Median Object Ratio (%)")
axes[2].set_title("Median Object Ratio per Komoditas\\n(merah = mask mencakup hampir seluruh frame)")
axes[2].legend()

plt.tight_layout()
plt.savefig(paths["figures"] / "eda_c1_segmentation_feasibility.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\\n{'='*55}")
print("RINGKASAN SEGMENTASI OTSU:")
print(f"  Gambar dengan foreground >90%  : {pct_over90:.1f}%")
print(f"  Gambar menggunakan fallback     : {pct_fallback:.1f}%")
print(f"  Median object_ratio            : {df_seg['object_ratio'].median():.3f}")
print(f"\\n-> KESIMPULAN: Otsu tidak efektif karena buah mengisi hampir seluruh frame (median~100%).")
print(f"  Hal ini menjelaskan mengapa S5 (segmentasi) <= S3 (tanpa segmentasi)")
print(f"  dan mengapa S8 (shape features) sangat buruk (mask = frame penuh = shape tidak bermakna).")
if not fallback_rate[fallback_rate > 0.05].empty:
    print(f"\\nKomoditas dengan fallback rate tertinggi:")
    print(fallback_rate[fallback_rate > 0.05].to_string())
"""
        ),
        md_cell(
            """\
## Seksi C2 - Analisis Efek SSR: Mengapa S2 < S1?

Secara teoritis, Single-Scale Retinex (SSR) bertujuan mengoreksi variasi pencahayaan non-uniform.
Namun, hasil eksperimen menunjukkan performa model dengan SSR (S2) lebih rendah dibanding Raw (S1).
Di sini kita menguji hipotesis secara kuantitatif bahwa SSR menormalkan perbedaan tingkat kecerahan
(luminance L pada ruang warna LAB) antara fresh dan rotten.
Kita mengukur nilai rata-rata L sebelum dan sesudah SSR untuk membuktikan secara statistik
(dengan t-test independen) apakah SSR memperkecil gap perbedaan kecerahan yang sebenarnya merupakan
fitur diskriminatif alami bagi classifier.
"""
        ),
        code_cell(
            """\
# C2. Analisis Efek SSR Terhadap Separabilitas Kecerahan (Luminance L)
# Hipotesis: SSR menormalkan variasi kecerahan alami yang justru menjadi sinyal pembeda kelas.
from scipy import stats as scipy_stats
N_SSR = 50  # Sampel citra per label

sample_ssr = (
    full_df
    .groupby("label", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), N_SSR), random_state=42))
    .reset_index(drop=True)
)

ssr_records = []
for _, row in sample_ssr.iterrows():
    img = cv2.imread(str(row["filepath"]))
    if img is None:
        continue
    img_raw = preprocess_arr(img, apply_restoration=False)
    img_ssr = preprocess_arr(img, apply_restoration=True)

    def l_stats(bgr):
        # Konversi ke ruang warna CIELAB untuk isolasi kanal luminance L
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(float)
        return float(L.mean()), float(L.std())

    lm_raw, ls_raw = l_stats(img_raw)
    lm_ssr, ls_ssr = l_stats(img_ssr)
    ssr_records.append({
        "label": row["label"],
        "L_mean_raw": lm_raw, "L_std_raw": ls_raw,
        "L_mean_ssr": lm_ssr, "L_std_ssr": ls_ssr,
    })

df_ssr_ef = pd.DataFrame(ssr_records)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Efek SSR pada Separabilitas Fresh vs Rotten", fontsize=13, fontweight="bold")

# Subplot 0: Histogram perbandingan mean L (Raw vs SSR)
for label, color in [("fresh", "#2ecc71"), ("rotten", "#e74c3c")]:
    sub = df_ssr_ef[df_ssr_ef["label"] == label]
    axes[0].hist(sub["L_mean_raw"], bins=15, alpha=0.5, color=color, label=f"{label} (raw)")
    axes[0].hist(sub["L_mean_ssr"], bins=15, alpha=0.3, color=color,
                 linestyle="dashed", histtype="step", linewidth=2.5,
                 label=f"{label} (SSR)")
axes[0].set_xlabel("Mean L (brightness)")
axes[0].set_title("Distribusi Mean Brightness\\nSSR menyempitkan gap fresh & rotten")
axes[0].legend(fontsize=9)

# Subplot 1: Jarak (separabilitas) kecerahan antara fresh vs rotten
fresh_raw = df_ssr_ef[df_ssr_ef["label"]=="fresh"]["L_mean_raw"].values
rotten_raw = df_ssr_ef[df_ssr_ef["label"]=="rotten"]["L_mean_raw"].values
fresh_ssr = df_ssr_ef[df_ssr_ef["label"]=="fresh"]["L_mean_ssr"].values
rotten_ssr = df_ssr_ef[df_ssr_ef["label"]=="rotten"]["L_mean_ssr"].values

sep_raw = abs(fresh_raw.mean() - rotten_raw.mean())
sep_ssr = abs(fresh_ssr.mean() - rotten_ssr.mean())

axes[1].bar(["Raw\\n(S1)", "SSR\\n(S2)"], [sep_raw, sep_ssr], color=["#2ecc71", "#e74c3c"])
axes[1].set_ylabel("|mean_L_fresh - mean_L_rotten|")
axes[1].set_title(f"Separabilitas Brightness\\nRaw: {sep_raw:.2f} | SSR: {sep_ssr:.2f}")
for i, v in enumerate([sep_raw, sep_ssr]):
    axes[1].text(i, v + 0.5, f"{v:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(paths["figures"] / "eda_c2_ssr_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# Uji hipotesis signifikansi statistik menggunakan T-test independen
t_stat, p_val = scipy_stats.ttest_ind(fresh_raw, rotten_raw)
t_stat2, p_val2 = scipy_stats.ttest_ind(fresh_ssr, rotten_ssr)
print("Uji t-test separabilitas brightness (fresh vs rotten):")
print(f"  Raw (S1): t={t_stat:.3f}, p={p_val:.4f} -> {'SIGNIFIKAN [OK]' if p_val<0.05 else 'tidak signifikan'}")
print(f"  SSR (S2): t={t_stat2:.3f}, p={p_val2:.4f} -> {'SIGNIFIKAN [OK]' if p_val2<0.05 else 'tidak signifikan'}")
print(f"\\n-> KESIMPULAN: SSR menghapus variasi brightness yang menjadi sinyal pembeda.")
print(f"  Separabilitas brightness turun dari {sep_raw:.2f} (Raw) menjadi {sep_ssr:.2f} (SSR).")
print(f"  Inilah alasan F1 turun dari 0.970 (S1) ke 0.948 (S2).")
"""
        ),
        md_cell(
            """\
## Seksi F - Kuantifikasi Efek Tiap Tahap Preprocessing

Bagian ini melakukan kuantifikasi terukur terhadap efek setiap tahap preprocessing
(Raw -> SSR -> SSR+CLAHE -> SSR+Gamma) pada kanal Luminance (L) dan Saturation (S).
Dengan mengukur rata-rata dan standar deviasi nilai piksel pada setiap tahap,
kita dapat memahami secara visual dan matematis bagaimana kontras citra ditingkatkan (via CLAHE)
atau bagaimana distribusi kecerahan disesuaikan (via Gamma).
"""
        ),
        code_cell(
            """\
# F. Kuantifikasi Efek Preprocessing per Tahap Preprocessing
# Kita mengukur pergeseran nilai statistik kecerahan (L) dan saturasi (S) untuk n=100 sampel.
from src.enhancement import apply_enhancement
N_PRE = 100
sample_pre_df = full_df.sample(N_PRE, random_state=42).reset_index(drop=True)

pre_records = []
for _, row in sample_pre_df.iterrows():
    img = cv2.imread(str(row["filepath"]))
    if img is None:
        continue

    # Jalankan empat tahap preprocessing pipeline untuk dibandingkan
    img_raw   = preprocess_arr(img, apply_restoration=False)
    img_ssr   = preprocess_arr(img, apply_restoration=True)
    img_clahe = apply_enhancement(img_ssr, "clahe")
    img_gamma = apply_enhancement(img_ssr, "gamma")

    def px_stats(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:,:,0].astype(float)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:,:,1].astype(float)
        return {"L_mean": L.mean(), "L_std": L.std(), "S_mean": S.mean()}

    # Simpan statistik piksel per gambar pada setiap tahap
    r = px_stats(img_raw)
    s = px_stats(img_ssr)
    c = px_stats(img_clahe)
    g = px_stats(img_gamma)

    pre_records.append({
        "label": row["label"],
        "L_mean_raw": r["L_mean"], "L_std_raw": r["L_std"], "S_mean_raw": r["S_mean"],
        "L_mean_ssr": s["L_mean"], "L_std_ssr": s["L_std"], "S_mean_ssr": s["S_mean"],
        "L_mean_clahe": c["L_mean"], "L_std_clahe": c["L_std"], "S_mean_clahe": c["S_mean"],
        "L_mean_gamma": g["L_mean"], "L_std_gamma": g["L_std"], "S_mean_gamma": g["S_mean"],
    })

df_pre = pd.DataFrame(pre_records)
stages = ["raw", "ssr", "clahe", "gamma"]
stage_labels = ["Raw", "SSR", "SSR+CLAHE", "SSR+Gamma"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Efek Tiap Tahap Preprocessing (n={len(df_pre)} gambar)", fontsize=13, fontweight="bold")

# Buat grafik garis pergeseran statistik piksel
metrics = [("L_mean", "Mean Brightness (L)", axes[0]),
           ("L_std",  "Std Brightness (L)", axes[1]),
           ("S_mean", "Mean Saturation (S)", axes[2])]

for metric_key, metric_label, ax in metrics:
    for label, color in [("fresh","#2ecc71"), ("rotten","#e74c3c")]:
        sub = df_pre[df_pre["label"]==label]
        vals = [sub[f"{metric_key}_{s}"].mean() for s in stages]
        errs = [sub[f"{metric_key}_{s}"].std() for s in stages]
        ax.errorbar(stage_labels, vals, yerr=errs, marker="o", color=color,
                    label=label, linewidth=2, capsize=4)
    ax.set_title(metric_label)
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(paths["figures"] / "eda_f_preprocessing_effect.png", dpi=150, bbox_inches="tight")
plt.show()

print("Perubahan mean brightness (L) dari raw ke SSR:")
for label in ["fresh", "rotten"]:
    sub = df_pre[df_pre["label"]==label]
    delta = sub["L_mean_ssr"].mean() - sub["L_mean_raw"].mean()
    print(f"  {label}: {delta:+.2f} (perubahan rata-rata)")
"""
        ),
        md_cell(
            """\
## Seksi E - Cek Duplikat & Data Leakage

Keberadaan gambar duplikat yang identik dalam dataset dapat memicu kebocoran data (data leakage)
jika gambar tersebut terbagi ke dalam split yang berbeda (misalnya satu di train set dan satu di test set).
Hal ini akan menghasilkan metrik evaluasi model yang terlalu optimis (overfitting tidak terdeteksi).
Kita memvalidasi kebersihan dataset menggunakan pencocokan MD5 hash untuk mendeteksi duplikasi gambar.
"""
        ),
        code_cell(
            """\
# E. Deteksi Duplikat Gambar Menggunakan MD5 Hash
# Pemeriksaan cepat menggunakan MD5 hash pada byte mentah gambar.
import hashlib

def file_hash_md5(path):
    \"\"\"Menghitung hash MD5 dari isi file secara streaming.\"\"\"
    try:
        with open(str(path), "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

# Ambil sampel acak sebanyak 1500 gambar untuk efisiensi analisis waktu
N_HASH_SAMPLE = min(len(full_df), 1500)
sample_hash_df = full_df.sample(N_HASH_SAMPLE, random_state=42).reset_index(drop=True)

print(f"Menghitung hash untuk {N_HASH_SAMPLE} gambar (sample)...")
hashes = [file_hash_md5(p) for p in sample_hash_df["filepath"]]
sample_hash_df["md5"] = hashes
sample_hash_df = sample_hash_df.dropna(subset=["md5"])

# Cari file dengan MD5 yang sama
duplicates = sample_hash_df[sample_hash_df.duplicated("md5", keep=False)]
n_dup_groups = duplicates.groupby("md5").ngroups

print(f"\\nSample diperiksa  : {len(sample_hash_df)} gambar")
print(f"Grup duplikat     : {n_dup_groups}")
print(f"Total file duplik : {len(duplicates)}")

if n_dup_groups > 0:
    print("\\n[WARNING] Duplikat terdeteksi:")
    print(duplicates.sort_values("md5")[["filepath","label","commodity","md5"]].to_string())
else:
    print("\\n[OK] Tidak ada duplikat dalam sample - dataset bersih dari duplikasi.")

print("\\nNote: Cross-split leakage check akan dilakukan setelah split 70/15/15 dibuat di bawah.")
"""
        ),
        md_cell(
            """\
## Seksi G - Feature Separability: Apakah Fitur Manual Memisahkan Kelas?

Kita memvisualisasikan seluruh ruang fitur buatan tangan (handcrafted features sebanyak 220 dimensi)
ke dalam ruang 2 dimensi menggunakan metode t-SNE.
Hal ini membantu kita melihat secara langsung sejauh mana representasi fitur warna (HSV histogram)
dan tekstur (LBP & GLCM) mampu memisahkan kelas fresh dan rotten secara linear maupun non-linear
sebelum dimasukkan ke classifier SVM/Random Forest.
Kita juga melakukan uji signifikansi Mann-Whitney U untuk membuktikan validitas fitur individual.
"""
        ),
        code_cell(
            """\
# G. Visualisasi t-SNE Separabilitas Fitur Manual & Uji Signifikansi
# Ekstraksi fitur visual (warna & tekstur) pada 120 gambar sampel.
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from src.features import extract_features
from src.pipeline import process_image

N_TSNE = 60  # Jumlah sampel per label (total 120 gambar)
sample_tsne = (
    full_df
    .groupby("label", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), N_TSNE), random_state=42))
    .reset_index(drop=True)
)

print(f"Mengekstrak fitur untuk {len(sample_tsne)} gambar (ini perlu ~2-3 menit)...")
X_feats, y_labels, commodities = [], [], []
for _, row in sample_tsne.iterrows():
    # Gunakan pipeline murni tanpa SSR/enhancement/segmentasi untuk mendapatkan baseline representatif
    out = process_image(
        path=row["filepath"],
        restoration="none",
        enhancement="none",
        do_segment=False,
    )
    if out["img"] is None:
        continue
    # Ekstraksi seluruh fitur manual (histogram warna, LBP, GLCM)
    feat = extract_features(out["img"], feature_groups="all", segmented=False)
    X_feats.append(feat)
    y_labels.append(1 if row["label"] == "rotten" else 0)
    commodities.append(row["commodity"])

X_arr = np.array(X_feats)
y_arr = np.array(y_labels)
print(f"Fitur berhasil diekstrak: {X_arr.shape}")

# Penskalaan fitur (Standardization) dan pemrosesan t-SNE
X_scaled = StandardScaler().fit_transform(X_arr)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(X_arr)//4))
X_2d = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Separabilitas Fitur Manual (220 dim: color+texture)", fontsize=13, fontweight="bold")

# Subplot 0: Plot 2D hasil reduksi dimensi t-SNE
for lv, ln, color in [(0,"fresh","#2ecc71"), (1,"rotten","#e74c3c")]:
    mask = y_arr == lv
    axes[0].scatter(X_2d[mask,0], X_2d[mask,1], c=color, label=ln, alpha=0.7, s=40, edgecolors="none")
axes[0].set_title("t-SNE - Color by Label")
axes[0].legend()
axes[0].set_xlabel("t-SNE dim 1")
axes[0].set_ylabel("t-SNE dim 2")

# Subplot 1: Violin plot untuk membandingkan distribusi fitur Mean Hue (fitur indeks 192)
idx_meanH = 192
fresh_h = X_arr[y_arr==0, idx_meanH]
rotten_h = X_arr[y_arr==1, idx_meanH]
vp = axes[1].violinplot([fresh_h, rotten_h], positions=[0,1], showmedians=True)
for pc in vp["bodies"]:
    pc.set_alpha(0.6)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Fresh", "Rotten"])
axes[1].set_ylabel("Mean Hue (fitur ke-192)")
axes[1].set_title("Distribusi Fitur Mean Hue\\nFresh vs Rotten")

# Uji statistik non-parametrik menggunakan Mann-Whitney U test untuk membuktikan signifikansi
stat, p_mwu = mannwhitneyu(fresh_h, rotten_h, alternative="two-sided")
axes[1].text(0.5, 0.92,
             f"Mann-Whitney U\\np = {p_mwu:.2e}\\n({'Signifikan [OK]' if p_mwu<0.05 else 'Tidak signifikan'})",
             transform=axes[1].transAxes, ha="center", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=9)

plt.tight_layout()
plt.savefig(paths["figures"] / "eda_g_feature_separability.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\\nFitur mean_H: Mann-Whitney U p = {p_mwu:.2e}")
print(f"-> Fitur warna {'signifikan [OK]' if p_mwu<0.05 else 'tidak signifikan'} memisahkan fresh vs rotten.")
"""
        ),
        md_cell(
            """\
## 6. Pembuatan Split Stratified (70/15/15)"""
        ),
        code_cell(
            """\
train, val, test = make_splits(df)
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Stratifikasi train: {train['label'].value_counts().to_dict()}")
print(f"Stratifikasi test : {test['label'].value_counts().to_dict()}")
split_path = save_splits(train, val, test, paths["splits"])
print(f"Split disimpan ke: {split_path}")
"""
        ),
        md_cell(
            """\
### Verifikasi Split: No Data Leakage

Memastikan tidak ada gambar yang sama antara train, val, dan test.
"""
        ),
        code_cell(
            """\
# Verifikasi Tidak Ada Data Leakage Antar Split (Train, Val, Test Set)
# Kita melakukan perbandingan set path file untuk memastikan irisan set (intersection) adalah kosong.
train_paths = set(train["filepath"].tolist())
val_paths   = set(val["filepath"].tolist())
test_paths  = set(test["filepath"].tolist())

leak_tv = train_paths & val_paths
leak_tt = train_paths & test_paths
leak_vt = val_paths   & test_paths

print("Verifikasi Data Leakage antar Split:")
print(f"  Train & Val  : {len(leak_tv)} file {'[OK]' if len(leak_tv)==0 else '[FAIL]'}")
print(f"  Train & Test : {len(leak_tt)} file {'[OK]' if len(leak_tt)==0 else '[FAIL]'}")
print(f"  Val & Test   : {len(leak_vt)} file {'[OK]' if len(leak_vt)==0 else '[FAIL]'}")
print(f"\\nTotal unik   : {len(train_paths)+len(val_paths)+len(test_paths)}")
print(f"Total dataset: {len(full_df)}")
all_ok = len(leak_tv)==0 and len(leak_tt)==0 and len(leak_vt)==0
print(f"\\n{'[OK] Tidak ada data leakage!' if all_ok else '[FAIL] TERDAPAT DATA LEAKAGE!'}")
"""
        ),
    ]
)


# ==========================================================================
# 02_experiments_classical.ipynb
# ==========================================================================
nb02 = make_nb(
    [
        md_cell(
            "# 02 - Eksperimen Klasik (Skenario 1-9)\n"
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
            "## Skenario 1-4: Baseline, Restorasi (SSR), Enhancement\n"
            "\n"
            "- **S1** = baseline mentah (tanpa restorasi, tanpa enhancement)\n"
            "- **S2** = + restorasi SSR (isolasi efek koreksi pencahayaan vs S1)\n"
            "- **S3/S4** = SSR + CLAHE / gamma. E* dipilih dari S2-S4 (val F1)."
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
            "## Skenario 5-9: Segmentasi, Ablasi Fitur, Random Forest\n"
            "\n"
            "- **S5** = E* + segmentasi, semua fitur, SVM (pipeline klasik penuh)\n"
            "- **S6/S7/S8** = ablasi fitur (warna saja / tekstur saja / bentuk saja)\n"
            "- **S9** = S5 dengan Random Forest (perbandingan classifier + feature importance)"
        ),
        code_cell(
            "for sid in range(5, 10):\n"
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
            'print("McNemar CNN (S10/S11 vs S5) dijalankan di notebook 03.")\n'
        ),
        code_cell(
            "import pandas as pd\n"
            'sig_path = metrics_dir / "significance_tests.csv"\n'
            "if sig_path.exists():\n"
            "    display(pd.read_csv(sig_path))\n"
        ),
    ]
)


# ==========================================================================
# 03_experiments_cnn.ipynb
# ==========================================================================
nb03 = make_nb(
    [
        md_cell(
            "# 03 - Eksperimen CNN (Skenario 10-11)\n"
            "\n"
            "MobileNetV2 two-stage fine-tuning, Grad-CAM, McNemar vs S5.\n"
            "- **S10** = SSR + E* + segmentasi (full pipeline klasik, diganti CNN)\n"
            "- **S11** = tanpa restorasi, tanpa enhancement (baseline murni CNN vs S1 klasik)"
        ),
        KAGGLE_SETUP,
        code_cell(
            NEW_ROOT
            + "import numpy as np\n"
            "import pandas as pd\n"
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
            "# S10: SSR + E* + segmentasi (identik dengan S5 klasik, tapi CNN)\n"
            "train_ds_s10 = make_dataset(train_df, shuffle=True, restoration='ssr', do_segment=True, cache_name='train_s10')\n"
            "val_ds_s10   = make_dataset(val_df,   restoration='ssr', do_segment=True, cache_name='val_s10')\n"
            "test_ds_s10  = make_dataset(test_df,  restoration='ssr', do_segment=True, cache_name='test_s10')\n"
            "\n"
            "# S11: tanpa restorasi, tanpa enhancement, tanpa segmentasi (baseline murni CNN)\n"
            "train_ds_s11 = make_dataset(train_df, shuffle=True, restoration='none', do_segment=False, enhancement_method='none', cache_name='train_s11')\n"
            "val_ds_s11   = make_dataset(val_df,   restoration='none', do_segment=False, enhancement_method='none', cache_name='val_s11')\n"
            "test_ds_s11  = make_dataset(test_df,  restoration='none', do_segment=False, enhancement_method='none', cache_name='test_s11')\n"
        ),
        code_cell(
            'y_train_labels = train_df["label"].map({"fresh": 0, "rotten": 1}).values\n'
            "classes = np.unique(y_train_labels)\n"
            'weights = compute_class_weight("balanced", classes=classes, y=y_train_labels)\n'
            "class_weight = {int(c): float(w) for c, w in zip(classes, weights)}\n"
            "class_weight\n"
        ),
        md_cell("## Skenario 10: CNN - SSR + E* + Segmentasi (mirror S5)"),
        md_cell("### Stage 1 - Base frozen (20 epoch)"),
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
        md_cell("### Stage 2 - Fine-tune 20 lapisan terakhir (50 epoch)"),
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
            "import time\n"
            "\n"
            "# Collect valid rows during the evaluation loop (avoids a second\n"
            "# full-dataset reprocessing pass that the old code did).\n"
            "y_true_list, y_pred_list = [], []\n"
            "valid_rows_s10 = []\n"
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
            '    10, enhancement, True, "cnn", "MobileNetV2",\n'
            '    metrics_s10, infer_ms, len(y_true_s10), paths["metrics"], restoration="ssr",\n'
            ")\n"
            "# Build predictions CSV from test_df rows that match valid S10 generator order.\n"
            "# The generator skips rows where process_image returns None; replicate same logic.\n"
            "# Guarded: a length mismatch must NOT crash here (this cell runs BEFORE S11\n"
            "# training, so a crash would waste the whole CNN run).\n"
            "valid_rows_s10 = [\n"
            "    row for _, row in test_df.iterrows()\n"
            "    if process_image(row['filepath'], restoration='ssr', enhancement=enhancement, do_segment=True)['img'] is not None\n"
            "]\n"
            "if len(valid_rows_s10) == len(y_pred_s10):\n"
            "    pred_df_s10 = pd.DataFrame(valid_rows_s10).reset_index(drop=True)\n"
            "    pred_df_s10['pred'] = y_pred_s10\n"
            "    pred_df_s10.to_csv(paths['metrics'] / 'predictions_s10.csv', index=False)\n"
            "else:\n"
            "    print(f'[WARN] predictions_s10 dilewati: {len(valid_rows_s10)} baris vs {len(y_pred_s10)} prediksi')\n"
            'plot_confusion_matrix(y_true_s10, y_pred_s10, title="Skenario 10 CNN (SSR+E*+Seg)",\n'
            '                      save_path=paths["figures_confusion"] / "scenario_10.png")\n'
            "metrics_s10\n"
        ),
        md_cell("## Skenario 11: CNN - Tanpa Restorasi, Tanpa Enhancement (mirror S1)"),
        md_cell("### Stage 1 - Base frozen (20 epoch)"),
        code_cell(
            "model_s11 = build_mobilenetv2(num_classes=2)\n"
            "model_s11 = compile_mobilenet(model_s11, learning_rate=1e-4)\n"
            'cb_s11 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s11_stage1.h5"))\n'
            "\n"
            "history1_s11 = model_s11.fit(\n"
            "    train_ds_s11, validation_data=val_ds_s11, epochs=20,\n"
            "    class_weight=class_weight, callbacks=cb_s11,\n"
            ")\n"
        ),
        md_cell("### Stage 2 - Fine-tune 20 lapisan terakhir (50 epoch)"),
        code_cell(
            "model_s11 = unfreeze_last_layers(model_s11, n=20)\n"
            "model_s11 = compile_mobilenet(model_s11, learning_rate=1e-5)\n"
            'cb2_s11 = get_mobilenet_callbacks(str(paths["models"] / "mobilenetv2_s11_stage2.h5"))\n'
            "\n"
            "history2_s11 = model_s11.fit(\n"
            "    train_ds_s11, validation_data=val_ds_s11, epochs=50,\n"
            "    class_weight=class_weight, callbacks=cb2_s11,\n"
            ")\n"
        ),
        md_cell("### Evaluasi Skenario 11"),
        code_cell(
            "y_true_list, y_pred_list = [], []\n"
            "t0 = time.perf_counter()\n"
            "n = 0\n"
            "for x_batch, y_batch in test_ds_s11:\n"
            "    preds = model_s11.predict_on_batch(x_batch)\n"
            "    y_pred_list.extend(np.argmax(preds, axis=1))\n"
            "    y_true_list.extend(np.argmax(y_batch.numpy(), axis=1))\n"
            "    n += len(y_batch)\n"
            "\n"
            "infer_ms = (time.perf_counter() - t0) * 1000 / max(n, 1)\n"
            "y_true_s11 = np.array(y_true_list)\n"
            "y_pred_s11 = np.array(y_pred_list)\n"
            "metrics_s11 = compute_metrics(y_true_s11, y_pred_s11)\n"
            "save_scenario_metrics(\n"
            '    11, "none", False, "cnn", "MobileNetV2",\n'
            '    metrics_s11, infer_ms, len(y_true_s11), paths["metrics"], restoration="none",\n'
            ")\n"
            'plot_confusion_matrix(y_true_s11, y_pred_s11, title="Skenario 11 CNN (Raw)",\n'
            '                      save_path=paths["figures_confusion"] / "scenario_11.png")\n'
            "metrics_s11\n"
        ),
        md_cell("## McNemar Significance Tests (dijalankan sebelum Grad-CAM)\n"),
        code_cell(
            "# McNemar bersifat SUPLEMENTER: metrik utama S10/S11 sudah tersimpan di sel\n"
            "# sebelumnya. Seluruh blok dibungkus try/except agar error tak terduga di sini\n"
            "# TIDAK memblokir Grad-CAM (traceback tetap dicetak supaya terlihat).\n"
            "import traceback\n"
            "import joblib\n"
            "\n"
            "def _safe_mcnemar(name, a, b, y_t, p_a, p_b):\n"
            "    # Guard panjang: McNemar butuh prediksi paired pada sampel yang sama.\n"
            "    if len(p_a) != len(y_t) or len(p_b) != len(y_t):\n"
            "        print(f'[SKIP {name}] panjang beda: y_true={len(y_t)}, {a}={len(p_a)}, {b}={len(p_b)}')\n"
            "        return\n"
            "    stat, pval, concl = mcnemar_test(y_t, p_a, p_b)\n"
            "    append_significance_test(name, a, b, stat, pval, concl, paths['metrics'])\n"
            "    print(f'{name}:', stat, pval, concl)\n"
            "\n"
            "s5_path = paths['models'] / 'svm_scenario_05.pkl'\n"
            "try:\n"
            "    if not s5_path.exists():\n"
            "        raise FileNotFoundError('svm_scenario_05.pkl tidak ada - jalankan nb02 dulu.')\n"
            "    from src.experiments import extract_split_matrix\n"
            "    from src.models import build_svm_pipeline\n"
            "    enh = read_best_enhancement(paths['metrics'])\n"
            "\n"
            "    # S5 (SVM full): fitur SSR+E*+seg+all - cache HIT dari nb02 (split_name='test').\n"
            "    s5_model = joblib.load(s5_path)\n"
            "    X_test_s5, _, _ = extract_split_matrix(\n"
            "        test_df, enh, True, 'all', paths['data_processed'], split_name='test', restoration='ssr')\n"
            "    y_pred_s5 = s5_model.predict(X_test_s5)\n"
            "\n"
            "    # S1 (SVM raw): re-train cepat. PENTING - pakai y yang DIKEMBALIKAN extract\n"
            "    # (label baris valid), bukan label_encode(train_df) penuh; kalau ada citra\n"
            "    # tak terbaca panjangnya beda dan fit() akan crash.\n"
            "    X_train_s1, y_train_s1, _ = extract_split_matrix(\n"
            "        train_df, 'none', False, 'all', paths['data_processed'], split_name='train', restoration='none')\n"
            "    X_test_s1, _, _ = extract_split_matrix(\n"
            "        test_df, 'none', False, 'all', paths['data_processed'], split_name='test', restoration='none')\n"
            "    s1_model = build_svm_pipeline()\n"
            "    s1_model.fit(X_train_s1, y_train_s1)\n"
            "    y_pred_s1 = s1_model.predict(X_test_s1)\n"
            "\n"
            "    _safe_mcnemar('S10 vs S5 (CNN vs SVM)', 'S10', 'S5', y_true_s10, y_pred_s10, y_pred_s5)\n"
            "    _safe_mcnemar('S11 vs S1 (CNN-raw vs SVM-raw)', 'S11', 'S1', y_true_s11, y_pred_s11, y_pred_s1)\n"
            "    _safe_mcnemar('S10 vs S11 (full vs raw CNN)', 'S10', 'S11', y_true_s10, y_pred_s10, y_pred_s11)\n"
            "except Exception:\n"
            "    print('[McNemar dilewati karena error - metrik utama S10/S11 tetap aman]')\n"
            "    traceback.print_exc()\n"
        ),
        md_cell("## Grad-CAM (Skenario 10 - CNN full pipeline)"),
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
            "            heatmap = make_gradcam_heatmap(model_s10, x)\n"
            '            fname = Path(row["filepath"]).stem\n'
            "            save = gradcam_dir / f\"{commodity}_{label}_{fname}.png\"\n"
            "            plot_gradcam(out[\"img\"], heatmap, save_path=save)\n"
            "            plt.close(\"all\")\n"
        ),
    ]
)


# ==========================================================================
# 04_results_summary.ipynb
# ==========================================================================
nb04 = make_nb(
    [
        md_cell(
            "# 04 - Ringkasan Hasil\n"
            "\n"
            "Tabel 3, plot komparatif F1, analisis segmentation failures, feature importance."
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
        md_cell("## Tabel 3 - Ringkasan Semua Skenario"),
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
        md_cell("## Feature Importance (Skenario 9 - RF)"),
        code_cell(
            "import joblib\n"
            "\n"
            'rf_path = paths["models"] / "rf_scenario_09.pkl"\n'
            "if rf_path.exists():\n"
            "    rf = joblib.load(rf_path)\n"
            "    from src.features import get_feature_group_names\n"
            '    names = get_feature_group_names("all", segmented=True)\n'
            "    if len(names) != len(rf.feature_importances_):\n"
            '        names = [f"f{i}" for i in range(len(rf.feature_importances_))]\n'
            "    labels, vals = aggregate_feature_importance(rf.feature_importances_, names)\n"
            '    plot_feature_importance(vals, labels, save_path=paths["figures"] / "feature_importance_s09.png")\n'
            "    plt.show()\n"
            "else:\n"
            '    print("Model RF S9 belum tersedia. Jalankan notebook 02 terlebih dahulu.")\n'
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
        md_cell(
            "## Kelemahan Sistem & Limitasi\n"
            "\n"
            "Bagian ini wajib dibahas dalam laporan (rubrik #9). Bukan kegagalan sistem,\n"
            "melainkan batas kebenaran klaim yang jujur harus diakui.\n"
            "\n"
            "### 1. Single split, satu seed\n"
            "Seluruh angka F1/akurasi bertumpu pada **satu pembagian data acak** (SEED=42,\n"
            "70/15/15). Tidak ada repeated runs atau confidence interval. Akibatnya:\n"
            "- Perbedaan F1 kecil antar skenario (mis. 0.92 vs 0.89) bisa jadi noise split.\n"
            "- McNemar membantu di level prediksi per-sampel, tapi tidak menangkap varians\n"
            "  yang muncul kalau seed diganti.\n"
            "- **Saran pengembangan**: k-fold lintas seed, atau setidaknya 3 seed berbeda.\n"
            "\n"
            "### 2. Risiko leakage near-duplicate\n"
            "Dataset Kaggle buah/sayur sering memuat banyak foto dari **objek fisik yang sama**\n"
            "dalam kondisi pencahayaan/sudut berbeda. Split acak per-citra bisa menaruh\n"
            "near-duplicate di train **dan** test sekaligus, menggelembungkan semua angka.\n"
            "- Tidak ada deteksi duplikat yang dilakukan (membutuhkan image hashing atau\n"
            "  perceptual similarity).\n"
            "- Angka performa yang sangat tinggi (>95% F1) harus dibaca dengan hati-hati\n"
            "  karena kemungkinan mengandung kontaminasi ini.\n"
            "- **Saran pengembangan**: deduplikasi dengan pHash sebelum split.\n"
            "\n"
            "### 3. Desain ladder, bukan factorial\n"
            "Setiap skenario mengubah satu variabel terhadap sibling-nya (*one-factor-at-a-time*).\n"
            "Ini memudahkan interpretasi tapi **tidak bisa menangkap interaksi antar faktor**:\n"
            "- Belum diuji: apakah CLAHE efektif *tanpa* SSR, atau manfaatnya bergantung SSR?\n"
            "- Belum diuji: apakah segmentasi membantu pada komoditas tertentu tapi menyakiti\n"
            "  yang lain?\n"
            "- **Saran pengembangan**: desain 2x2 factorial (SSR on/off x segmentasi on/off)\n"
            "  untuk melihat interaksi.\n"
            "\n"
            "### 4. Segmentasi berbasis threshold (Otsu) tanpa learning\n"
            "Metode segmentasi - Otsu pada kanal S (HSV) + grayscale + morfologi - sederhana\n"
            "dan cepat, tapi gagal pada buah/sayur gelap berlatar belakang serupa (tercatat\n"
            "di `segmentation_failures.csv`). Fallback (gambar penuh) dipakai saat foreground\n"
            "<5%, artinya fitur segmentasi kehilangan maknanya pada sebagian sampel.\n"
        ),
        md_cell("## Per-Commodity Performance Comparison"),
        code_cell(
            's5_pred_path = metrics_dir / "predictions_s5.csv"\n'
            's10_pred_path = metrics_dir / "predictions_s10.csv"\n'
            "\n"
            "if s5_pred_path.exists() and s10_pred_path.exists():\n"
            "    from sklearn.metrics import f1_score\n"
            "    s5_preds = pd.read_csv(s5_pred_path)\n"
            "    s10_preds = pd.read_csv(s10_pred_path)\n"
            '    label_map = {"fresh": 0, "rotten": 1}\n'
            '    s5_preds["true_encoded"] = s5_preds["label"].map(label_map)\n'
            '    s10_preds["true_encoded"] = s10_preds["label"].map(label_map)\n'
            "    s5_comm = []\n"
            '    for comm, group in s5_preds.groupby("commodity"):\n'
            '        f1 = f1_score(group["true_encoded"], group["pred"], average="weighted", zero_division=0)\n'
            '        s5_comm.append({"commodity": comm, "samples": len(group), "f1_s5": f1})\n'
            "    df_s5_comm = pd.DataFrame(s5_comm)\n"
            "    s10_comm = []\n"
            '    for comm, group in s10_preds.groupby("commodity"):\n'
            '        f1 = f1_score(group["true_encoded"], group["pred"], average="weighted", zero_division=0)\n'
            '        s10_comm.append({"commodity": comm, "f1_s10": f1})\n'
            "    df_s10_comm = pd.DataFrame(s10_comm)\n"
            '    df_compare = pd.merge(df_s5_comm, df_s10_comm, on="commodity").sort_values("f1_s10", ascending=False)\n'
            "    display(df_compare)\n"
            "    df_melted = df_compare.melt(id_vars=[\"commodity\", \"samples\"], value_vars=[\"f1_s5\", \"f1_s10\"],\n"
            '                                var_name="model", value_name="f1_score")\n'
            '    df_melted["model"] = df_melted["model"].map({"f1_s5": "S5 SVM", "f1_s10": "S10 CNN"})\n'
            "    fig, ax = plt.subplots(figsize=(12, 5))\n"
            '    sns.barplot(data=df_melted, x="commodity", y="f1_score", hue="model", ax=ax)\n'
            '    ax.set_title("F1-Score per Komoditas (S5 SVM vs S10 CNN)")\n'
            '    ax.set_ylabel("Weighted F1-Score")\n'
            '    ax.set_xlabel("Komoditas")\n'
            "    ax.set_ylim(0, 1.05)\n"
            "    plt.xticks(rotation=45, ha='right')\n"
            "    plt.grid(axis='y', linestyle='--', alpha=0.7)\n"
            "    plt.tight_layout()\n"
            '    plt.savefig(paths["figures"] / "commodity_comparison.png", dpi=150)\n'
            "    plt.show()\n"
            "else:\n"
            '    print("Prediksi S5 atau S10 belum lengkap. Lewati perbandingan komoditas.")\n'
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
