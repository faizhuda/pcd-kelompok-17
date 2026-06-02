# Klasifikasi Kualitas Citra Digital Buah dan Sayur

Proyek KOM1328 — Pengolahan Citra Digital, IPB.  
Klasifikasi binary **fresh** vs **rotten** dengan pipeline klasik (SVM/RF) dan CNN (MobileNetV2).

---

## Jalankan di Google Colab (Rekomendasi)

Semua notebook sudah dilengkapi sel setup otomatis untuk Google Colab.

### Persiapan sekali saja

**1. Upload proyek ke Google Drive**
- Zip folder `pcd-kelompok-17` (kecuali `data/raw/` — biarkan diunduh dari Kaggle)
- Upload dan extract ke `My Drive/pcd-kelompok-17/`

**2. Simpan Kaggle API key ke Drive**
- Buka [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
- Simpan `kaggle.json` ke Google Drive: `My Drive/.kaggle/kaggle.json`

### Urutan notebook (wajib berurutan)

Buka setiap notebook di Colab, pilih **Runtime → Change runtime type → T4 GPU**, lalu jalankan semua sel dari atas ke bawah.

| # | Notebook | Yang dihasilkan | Runtime |
|---|----------|-----------------|---------|
| 1 | `01_eda.ipynb` | `data/splits.json` | ~20 menit |
| 2 | `02_experiments_classical.ipynb` | `results/metrics/scenario_01–10.csv`, model S6 & S10 | ~4–8 jam (CPU) |
| 3 | `03_experiments_cnn.ipynb` | `results/metrics/scenario_11.csv`, model MobileNetV2 | ~2–3 jam (GPU) |
| 4 | `04_results_summary.ipynb` | Tabel 3, semua plot & visualisasi | ~5 menit |

> **Catatan:** Notebook 02 harus selesai sebelum 03 (notebook 03 memerlukan model S6).  
> Notebook 03 harus selesai sebelum 04 dapat menampilkan tabel lengkap.

**Anti-disconnect Colab** — jalankan di browser console saat training berlangsung:
```javascript
setInterval(() => document.querySelector('colab-connect-button')?.click(), 60000)
```

---

## Jalankan Lokal (Opsional)

### Persyaratan
- Python 3.10+
- ~5 GB ruang disk untuk dataset Kaggle

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Dataset
```bash
# Tanpa Kaggle — gunakan sample data 80 citra untuk development
python scripts/create_sample_data.py
python scripts/smoke_test.py
```

Untuk dataset lengkap, tambahkan API key Kaggle ke `~/.kaggle/kaggle.json`  
lalu set `DOWNLOAD_DATASET = True` di notebook 01.

---

## Testing & Linting

```bash
pytest tests/ -v          # 40 unit tests
python scripts/smoke_test.py   # integration smoke test (tanpa dataset)
pip install ruff && ruff check src/
```

---

## Struktur

```
src/           # Fungsi pipeline murni (tanpa side-effect I/O)
notebooks/     # Orchestration & visualisasi (Colab-ready)
tests/         # Unit tests (pytest)
scripts/       # Utilities (sample data, smoke test, rebuild notebooks)
data/raw/      # Dataset asli (read-only, tidak di-commit)
data/processed/# Cache fitur opsional (dapat di-regenerate)
results/       # Metrik CSV, model .pkl/.h5, figure PNG
```

Spesifikasi lengkap: [CLAUDE.md](CLAUDE.md).
