# Klasifikasi Kualitas Citra Digital Buah dan Sayur

Proyek KOM1328 — Pengolahan Citra Digital, IPB.  
Klasifikasi binary **fresh** vs **rotten** dengan pipeline klasik (SVM/RF) dan CNN (MobileNetV2).

---

## Jalankan di Kaggle Notebooks (Rekomendasi)

Semua notebook sudah dilengkapi **setup cell otomatis untuk Kaggle** (clone repo +
deteksi dataset, nol konfigurasi). Panduan lengkap: **[KAGGLE_PLAN.md](KAGGLE_PLAN.md)**.

### Persiapan (sekali, oleh koordinator)
- Pastikan repo GitHub ini **public**.
- Buat satu **Kaggle Dataset** `pcd-kelompok-17-results` (wadah hasil training
  bersama), lalu tambahkan anggota lain sebagai *collaborator*.

### Alur tim 5 orang: 1 Runner, 4 Konsumen
- **Runner (1 orang)** menjalankan pipeline penuh (`01 → 02 → 03`), lalu
  publikasikan hasil ke Dataset bersama.
- **Anggota lain** cukup attach Dataset bersama → jalankan `04` untuk analisis &
  laporan, **tanpa training**.

### Menjalankan sebuah notebook
1. **Create → New Notebook**, lalu panel kanan **Settings → Internet on**.
2. **+ Add Data** → cari `fruit and vegetable disease healthy vs rotten` → **Add**.
   (Notebook 03/04: attach juga output/Dataset hasil notebook sebelumnya.)
3. **File → Import Notebook** → pilih `.ipynb` dari repo ini.
4. Jalankan **berurutan dari atas**. Setup cell meng-clone repo & menyiapkan `src/`.

| # | Notebook | Accelerator | Estimasi | Menghasilkan |
|---|----------|-------------|----------|--------------|
| 1 | `01_eda.ipynb` | CPU | ~20 menit | EDA + split (deterministik) |
| 2 | `02_experiments_classical.ipynb` | CPU | ~4–8 jam (Save & Run All) | `scenario_01–10.csv`, model S6 & S10 |
| 3 | `03_experiments_cnn.ipynb` | **GPU P100** | ~2–3 jam | `scenario_11.csv`, model MobileNetV2 |
| 4 | `04_results_summary.ipynb` | CPU | ~5 menit | Tabel 3, semua plot |

> **Catatan:** Notebook 02 harus selesai sebelum 03 (butuh model S6); 03 sebelum 04.
> Hand-off antar-notebook: **+ Add Data → Your Work / Dataset bersama** (auto-restore
> oleh setup cell). Split tidak perlu dioper — di-regenerate deterministik tiap sesi.

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

Untuk mengunduh dataset lengkap secara lokal, tambahkan API key Kaggle ke
`~/.kaggle/kaggle.json` lalu panggil `download_kaggle_dataset()` dari `src/utils.py`.
(Di Kaggle Notebooks tidak perlu ini — dataset cukup di-*attach*.)

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
