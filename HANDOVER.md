# Handover — PCD Kelompok 17 (untuk agent berikutnya / Antigravity)

Dokumen serah-terima konteks. Project akhir KOM1328 (Pengolahan Citra Digital, IPB):
klasifikasi **fresh vs rotten** buah/sayur. Klasik (SVM/RF) + CNN (MobileNetV2).
Repo: `https://github.com/faizhuda/pcd-kelompok-17` (public). Branch kerja: **`main`**.

---

## 0. Aturan kerja yang WAJIB dipatuhi

1. **JANGAN tambahkan `Co-Authored-By` di commit** (aturan di `CLAUDE.md`). Tidak meng-co-author diri sendiri.
2. **User minta push langsung ke `main`** (bukan lewat PR lagi).
3. **Notebook di-generate dari script** `scripts/rebuild_notebooks.py` — ini **sumber kebenaran**.
   JANGAN edit file `.ipynb` langsung. Alur: edit script → `python scripts/rebuild_notebooks.py` → commit `scripts/` + `notebooks/`.
4. Komentar kode bahasa Inggris; penjelasan/markdown boleh Indonesia.

---

## 1. Cara verifikasi (lokal, Windows)

```bash
python -m ruff check src/                              # harus "All checks passed!"
python -m pytest tests/ --ignore=tests/test_evaluate.py -q   # ~51 passed
python scripts/smoke_test.py                           # "Smoke test OK."
python scripts/rebuild_notebooks.py                    # regenerate 4 notebooks
```
- **TensorFlow tidak load reliable di Windows** (DLL error di pytest) → `tests/test_evaluate.py` pakai `pytest.importorskip("tensorflow")` jadi auto-skip. Test TF (Grad-CAM dll) hanya jalan di mesin ber-TF (mis. Kaggle). Sudah pernah diverifikasi manual di TF 2.21/Keras 3 via `python -c`.
- Tidak ada dataset lokal; tes pakai sample data sintetis.

---

## 2. Arsitektur & keputusan penting (sudah selesai)

- `src/` = fungsi murni; `notebooks/` = orkestrasi; `tests/` = pytest; `scripts/` = utilities.
- **Workflow Kaggle**: model "1 Runner, 4 Konsumen". Setup cell di tiap notebook meng-`git clone` `main`, auto-detect path dataset, dan `restore_previous_outputs()` untuk mewarisi hasil notebook sebelumnya. Detail di `KAGGLE_PLAN.md`.
- **Path dataset Kaggle auto-detect**: bisa di `/kaggle/input/<slug>` ATAU `/kaggle/input/datasets/muhammad0subhan/<slug>`. Sudah dihandle di setup cell (rebuild_notebooks.py `KAGGLE_SETUP`).
- **Restore antar-notebook**: `restore_previous_outputs()` pakai `rglob("pcd-kelompok-17")` karena Kaggle mount output di `/kaggle/input/notebooks/<user>/<nb>/pcd-kelompok-17/`.
- **Split deterministik**: `build_dataset_index` (src/utils.py) **sort by filepath** → split identik lintas mesin/mount (SEED=42). Ini fix bug di mana nb02 & nb03 dapat split beda karena urutan `rglob` berbeda antar mount.
- **Cache fitur self-validating**: `extract_split_matrix` (src/experiments.py) menyimpan `df_files` di `.npz`; kalau nama file tidak cocok dengan split sekarang → recompute (mencegah IndexError & misalignment diam-diam).
- **Grad-CAM Keras 3-safe**: `make_gradcam_heatmap` (src/evaluate.py) pakai **GradientTape replay** (bukan rebuild functional sub-model) supaya jalan di model dengan nested MobileNetV2. `build_gradcam_model` lama gagal di Keras 3 untuk nested model — sudah tidak dipakai di notebook.
- **CI**: `.github/workflows/ci.yml` jalankan ruff + pytest tiap push (TF test auto-skip).
- `.gitignore`: `data/splits.json`, `scratch/`, `*.zip` di-ignore. `splits.json` sudah di-`git rm --cached` (deterministik, tak perlu di-track).

### Desain skenario SEKARANG (config.py + notebooks)
Single-variable isolation. Restorasi (SSR) kini **toggle**, jadi ada baseline raw sejati.

| ID | restoration | enhancement | seg | features | model | isolasi |
|----|------------|-------------|-----|----------|-------|---------|
| 1 | none | none | F | all | svm | **baseline raw** |
| 2 | ssr | none | F | all | svm | efek SSR (vs S1) |
| 3 | ssr | clahe | F | all | svm | enhancement |
| 4 | ssr | gamma | F | all | svm | enhancement |
| 5 | ssr | best(E*) | T | all | svm | **full klasik** (anchor McNemar) → `svm_scenario_05.pkl` |
| 6 | ssr | best | T | color | svm | ablasi fitur |
| 7 | ssr | best | T | texture | svm | ablasi fitur |
| 8 | ssr | best | T | shape | svm | ablasi fitur (bentuk) |
| 9 | ssr | best | T | all | rf | classifier + feat.importance → `rf_scenario_09.pkl` |
| 10 | ssr | best | T (CNN) | — | MobileNetV2 | mirror S5 |
| 11 | none | none | F (CNN) | — | MobileNetV2 | mirror S1 |

- E* dipilih dari val-F1 S2/S3/S4 (bukan S1, beda restorasi).
- McNemar nb02: efek SSR (S2vsS1), efek E* (S{E*}vsS2), efek segmentasi (S5 vs E*-noseg).
- McNemar nb03: S10vsS5 (CNN vs SVM full), S11vsS1 (raw), S10vsS11.
- GridSearch SVM: `C=[1,10]`, `gamma=['scale', 0.01]`, `cv=3` → 4 candidates × 3 folds = 12 fits.

---

## 3. STATUS RUNTIME (penting)

- **nb02 PERLU di-re-run.** Run sebelumnya (8.6 jam) memakai split lama (non-deterministik) + grid besar. Setelah fix determinism + grid kecil, estimasi **~1–2 jam**. Lalu nb03 (GPU, ~1.5–3 jam), lalu nb04.
- Urutan re-run: re-import notebook dari GitHub `main` → attach dataset (+ output nb sebelumnya untuk nb03/04) → **Save & Run All**.
- Hasil training dipublikasikan ke Kaggle Dataset bersama `pcd-kelompok-17-results`.

---

## 4. PENDING — keputusan metodologi (status update)

### ✅ Sudah diterapkan
1. **Gamma SVM dikembalikan.** `gamma=['scale', 0.01]`, `C=[1,10]`, `cv=3`. 4 candidates × 3 folds = 12 fits — cukup cepat, tuning adil untuk perbandingan CNN vs SVM.
2. **Skenario shape-only ditambahkan.** S8 (shape-only SVM), RF geser ke S9, CNN ke S10/S11. Semua referensi (McNemar, model-save, nb04) sudah di-update.

### 🟢 Paling bernilai untuk nilai (sebagian besar analisis, bukan training)
3. **Analisis per-komoditas** di `nb04`: pecah F1 per komoditas, dan/atau histogram warna fresh-vs-rotten per komoditas. Ini jantung "warna sebagai indikator mutu" yang sekarang kabur karena 14 komoditas digabung. Nol biaya training tambahan.
4. **Bagian "kelemahan sistem"** di laporan: akui (a) **single split / satu seed** (tak ada CI/repeated runs), (b) **risiko leakage near-duplicate** (foto objek fisik sama bisa tersebar ke train+test → menggelembungkan semua angka), (c) desain **ladder bukan factorial** (tak bisa lihat interaksi restorasi×enhancement).

---

## 5. File peta cepat

- `src/config.py` — SCENARIO_CONFIG (S1–S9 klasik, ada field `restoration`).
- `src/models.py` — `build_svm_pipeline` (GridSearch — gamma=['scale', 0.01]), `build_rf_classifier`, `build_mobilenetv2`.
- `src/preprocessing.py` — `apply_restoration` toggle (SSR).
- `src/pipeline.py` — `process_image(restoration=..., enhancement=..., do_segment=...)`, `batch_extract_features`.
- `src/experiments.py` — `extract_split_matrix` (cache + restoration), `run_classical_scenario` (model-save IDs 5 & 9).
- `src/evaluate.py` — metrics, `mcnemar_test`, `make_gradcam_heatmap` (Keras3-safe), `save_scenario_metrics` (kolom restoration), `print_summary_table`.
- `src/features.py` — color/texture/shape; helper `_foreground_mask`.
- `scripts/rebuild_notebooks.py` — **generator semua notebook** (KAGGLE_SETUP, nb01–nb04).
- `tests/test_evaluate.py` — TF tests (importorskip), termasuk regresi Grad-CAM nested-model.
- `KAGGLE_PLAN.md` — panduan workflow Kaggle lengkap.

---

## 6. Ringkas: apa yang sudah & belum

✅ Migrasi Kaggle, setup cell, auto-restore, split deterministik, cache self-validating, Grad-CAM Keras-3 fix, McNemar sebelum Grad-CAM, SSR toggle + baseline raw, CI, cleanup, KAGGLE_PLAN.md, **gamma SVM dikembalikan**, **skenario shape-only ditambahkan**, desain skenario S1–S11.

⏳ Belum: (1) re-run pipeline penuh di Kaggle dengan kode terbaru; (2) analisis per-komoditas di nb04; (3) tulis bagian kelemahan (leakage, single-split) di laporan.

Verifikasi terakhir hijau: ruff clean, 51 tests pass (non-TF), smoke OK, 4 notebook ter-generate & syntax-valid.
