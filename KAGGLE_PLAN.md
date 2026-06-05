# Rencana Migrasi ke Kaggle Notebooks - PCD Kelompok 17

Dokumen ini berisi analisis repo + rencana konkret agar project bisa dijalankan
mulus dan kolaboratif di **Kaggle Notebooks** (bukan Colab). Disusun berdasarkan
prinsip: **paling sedikit langkah, paling sedikit setup**.

> Catatan: semua kode di repo ini saat ini ditulis untuk **Google Colab**
> (mount Google Drive + `kagglehub.dataset_download`). Di Kaggle, beberapa bagian
> akan langsung *broken*. Bagian A merinci akar masalahnya.

---

## A. Root Cause Summary

Daftar masalah konkret yang ditemukan saat audit. `[BUG]` = bug aktual yang
membuat notebook gagal jalan di Kaggle.

| # | Masalah | Referensi | Dampak di Kaggle |
|---|---------|-----------|------------------|
| 1 | **[BUG] Deteksi ROOT salah di luar Colab.** Sel import memakai `ROOT = Path.cwd() if _in_colab else Path.cwd().parent`. Di Kaggle `_in_colab=False`, `cwd=/kaggle/working`, sehingga `ROOT=/kaggle`. | `notebooks/01_eda.ipynb` cell 3, juga cell 3 di notebook 02/03/04 | `sys.path` menunjuk `/kaggle`, lalu `from src...` -> **ModuleNotFoundError**. Semua notebook gagal di sel kedua. |
| 2 | **[BUG] `splits.json` menyimpan path absolut spesifik mesin.** `save_splits._to_relative` hanya membuat path relatif jika file berada di dalam project root. Dataset kagglehub berada di *luar* root, jadi path absolut dipertahankan. | `src/utils.py:210-220`; lihat isi `data/splits.json` saat ini: `C:/Users/faizn/.cache/kagglehub/...` | `splits.json` yang ter-commit ke git **tidak bisa dipakai** anggota lain / di Kaggle. `load_splits` mengembalikan path Windows yang tidak ada -> notebook 02/03/04 gagal saat baca gambar. |
| 3 | **[BUG] Setup cell khusus Colab.** `drive.mount('/content/drive')`, `PROJECT_PATH='/content/drive/MyDrive/...'`, deteksi `COLAB_RELEASE_TAG`. | cell 1 di keempat notebook | Di Kaggle, blok `if IN_COLAB` dilewati diam-diam -> tidak ada `chdir`/`sys.path` yang benar -> memperparah masalah #1. |
| 4 | **Over-install dependency.** `%pip install -q opencv-python-headless scipy scikit-image scikit-learn statsmodels matplotlib seaborn pandas tqdm joblib kagglehub tensorflow`. | cell 2 di keempat notebook | Semua paket ini **sudah pre-installed** di image Kaggle. Reinstall (terutama `tensorflow` & `opencv`) membuang 2-5 menit dan berisiko merusak TensorFlow GPU bawaan Kaggle. |
| 5 | **`kagglehub.dataset_download` dipakai padahal dataset bisa di-attach.** `DOWNLOAD_DATASET=True` default -> mengunduh ulang dataset (~beberapa GB) ke cache. | `notebooks/01_eda.ipynb` cell 4 | Di Kaggle butuh kredensial + buang waktu/kuota. Dataset yang sama bisa di-*attach* gratis ke `/kaggle/input/...` (read-only, zero-download). |
| 6 | **State antar-notebook rapuh.** Notebook 02 menghasilkan `best_enhancement.txt` + model `svm_scenario_06.pkl`; notebook 03 membacanya; notebook 04 membaca semua CSV metrik. | `src/experiments.py:153-191`, `notebooks/03...` cell 3 | Tiap notebook Kaggle = container terpisah. `results/` yang ditulis ke `/kaggle/working` **hilang** saat sesi berakhir, dan tidak otomatis terlihat oleh notebook berikutnya. Perlu strategi eksplisit (Bagian C). |
| 7 | **Notebook 02 panjang (4-8 jam) tanpa skip/resume.** `run_classical_scenario` selalu melatih ulang & menimpa CSV; tidak ada "lewati skenario yang CSV-nya sudah ada". | `src/experiments.py:70-176` | Jika sesi interaktif habis, harus mulai dari skenario 1. **Mitigasi sudah ada sebagian**: cache fitur `.npz` (`extract_split_matrix`, baris 37-65) menyimpan tahap termahal (ekstraksi fitur), jadi re-run jauh lebih cepat. Lihat Bagian E. |

**Catatan positif (sudah robust):**
- `build_dataset_index` memakai `rglob` + `IGNORE_PARTS`, jadi tahan terhadap
  struktur folder Kaggle (`/kaggle/input/.../Fruit And Vegetable Diseases Dataset/Tomato__Rotten/...`). Tidak perlu diubah.
- Cache fitur `.npz` per-split sudah ada -> kunci untuk mempercepat re-run notebook 02.
- Split bersifat **deterministik** (`SEED=42`, stratified). Artinya `splits.json`
  bisa di-regenerate ulang kapan saja dari dataset yang sama - kita manfaatkan ini
  untuk menghindari masalah #2 sepenuhnya.

---

## B. Setup Cell - Kaggle Notebooks (PRIORITAS UTAMA)

> **Cara attach dataset di Kaggle:** di panel kanan klik **+ Add Data** -> ketik
> `fruit and vegetable disease healthy vs rotten` -> klik **Add**. Dataset akan
> muncul otomatis di `/kaggle/input/fruit-and-vegetable-disease-healthy-vs-rotten`.
> (Tidak perlu API key, tidak perlu download.)

Tempel satu sel ini **paling atas** di setiap notebook saat dijalankan di Kaggle
(ganti sel "Google Colab Setup" + sel "Install dependencies" yang lama):

```python
# ============================================================
# Setup cell - Kaggle Notebooks (SATU-SATUNYA jalur, Kaggle-only)
# Jalankan PALING ATAS di setiap notebook, sekali per sesi.
# ============================================================
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 1. Clone repo dari GitHub (atau pull jika sudah ada di sesi ini)
REPO_URL = "https://github.com/faizhuda/pcd-kelompok-17.git"
PROJECT_DIR = Path("/kaggle/working/pcd-kelompok-17")
if not PROJECT_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=False)

# 2. Set working directory ke root project + tambahkan ke sys.path
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 3. Dependencies: SEMUA library inti SUDAH pre-installed di Kaggle
#    (tensorflow, opencv, scikit-learn, scikit-image, scipy, statsmodels,
#     pandas, numpy, matplotlib, seaborn, joblib, tqdm, kagglehub).
#    => TIDAK perlu pip install apa pun.

# 4. Dataset gambar (read-only, sudah di-attach via + Add Data)
RAW_DATA_DIR = Path("/kaggle/input/fruit-and-vegetable-disease-healthy-vs-rotten")
assert RAW_DATA_DIR.exists(), (
    "Dataset belum di-attach. Klik '+ Add Data' di panel kanan, cari "
    "'fruit and vegetable disease healthy vs rotten', lalu Add."
)

# 5. Auto-restore hasil notebook sebelumnya (untuk notebook 03 & 04).
#    Cukup attach output run lama: + Add Data -> Your Work -> pilih run.
#    Helper menyalin results/ & data/processed/ ke project secara otomatis,
#    tanpa perlu mengetik path apa pun.
def restore_previous_outputs() -> list[str]:
    restored: list[str] = []
    for inp in Path("/kaggle/input").glob("*"):
        repo = inp / "pcd-kelompok-17"  # output Kaggle = isi /kaggle/working
        if not repo.is_dir():
            continue
        for sub in ("results", "data/processed"):
            src_dir = repo / sub
            if src_dir.exists():
                shutil.copytree(src_dir, PROJECT_DIR / sub, dirs_exist_ok=True)
                restored.append(f"{inp.name}/{sub}")
    return restored

restored = restore_previous_outputs()
print("Project :", PROJECT_DIR)
print("Dataset :", RAW_DATA_DIR)
print("Restore :", restored or "(tidak ada output sebelumnya - mulai dari nol)")
```

**Kenapa ini lebih sederhana & lebih konsisten dari versi Colab:**
- **Kaggle-only**: tidak ada percabangan `if _in_colab` / `cwd().parent` yang bisa
  salah tebak. Satu jalur, satu perilaku, di semua mesin anggota.
- Tidak mount Drive, tidak butuh `kaggle.json`, tidak download dataset.
- Nol `pip install` -> sel selesai dalam hitungan detik, bukan menit.
- **State antar-notebook otomatis**: `restore_previous_outputs()` menghapus
  langkah copy manual (lihat Bagian C) - anggota cukup *Add Data -> Your Work*.

> **Repo private?** Tambahkan token: ganti `REPO_URL` menjadi
> `https://<TOKEN>@github.com/faizhuda/pcd-kelompok-17.git` dan simpan token via
> **Add-ons -> Secrets**. Untuk repo *public* (rekomendasi), tidak perlu apa-apa.

---

## C. Strategi Simpan Hasil Training

Di Kaggle, semua yang ditulis ke `/kaggle/working/` **otomatis tersimpan** saat
kamu menekan **Save Version**. Karena repo kita di-clone ke
`/kaggle/working/pcd-kelompok-17`, maka `results/` dan `data/processed/` otomatis
ikut tersimpan. Alurnya:

1. **Jalankan notebook sampai selesai.** Output (model `.pkl`/`.h5`, CSV metrik,
   `best_enhancement.txt`, figur PNG) tertulis ke
   `/kaggle/working/pcd-kelompok-17/results/...` dan cache fitur ke
   `/kaggle/working/pcd-kelompok-17/data/processed/`.
2. **Klik "Save Version"** (pojok kanan atas) -> pilih:
   - **Save & Run All (Commit)** - menjalankan ulang notebook dari nol secara
     *headless* (boleh ditutup browsernya, kuat sampai 12 jam). **Pilih ini untuk
     notebook 02 & 03** supaya training tidak terganggu disconnect.
   - **Quick Save** - menyimpan kondisi sel saat ini tanpa run ulang (untuk
     notebook yang sudah selesai dijalankan interaktif).
3. **Akses kembali hasil** dari tab **Output** di halaman notebook. Semua file di
   `/kaggle/working` ada di sana dan bisa di-download.
4. **Bagikan ke anggota lain:** share **link notebook** (Settings -> Sharing), atau
   anggota lain **+ Add Data -> Notebook Output** untuk meng-attach output kamu
   sebagai input di notebook mereka.

### Mewariskan state antar-notebook (mengatasi masalah #6)

Artefak yang **wajib** diteruskan: dari notebook 02 -> `best_enhancement.txt`, model
`svm_scenario_06.pkl`, semua CSV `results/metrics/`; dari notebook 03 -> CSV skenario
11 + prediksi CNN untuk notebook 04.

**Cara yang dipakai (satu klik, nol kode tambahan):** setup cell sudah memuat
`restore_previous_outputs()`. Jadi prosedurnya hanya:

1. Selesaikan notebook 02 -> **Save Version (Save & Run All)**. Beri judul jelas,
   mis. `pcd-02-classical`.
2. Buka notebook 03 -> **+ Add Data -> Your Work ->** pilih run `pcd-02-classical`.
3. Jalankan setup cell. `restore_previous_outputs()` otomatis menyalin
   `results/` + `data/processed/` dari output 02 ke project. **Tidak perlu mengetik
   path apa pun.** Notebook 03 langsung menemukan `best_enhancement.txt` & model S6.
4. Notebook 04: attach **kedua** output (`pcd-02-classical` + `pcd-03-cnn`) lewat
   *Add Data -> Your Work*; setup cell merestore keduanya sekaligus.

**Split (`splits.json`) tidak perlu dioper sama sekali** - deterministik, cukup
di-*generate* ulang di tiap notebook dari `RAW_DATA_DIR` (lihat Bagian E #1).

> **Alternatif paling ringkas (opsional):** gabung rantai **02->03 dalam satu
> notebook** GPU. Karena S11 (CNN) butuh `best_enhancement.txt` + model S6 dari
> klasik, menjalankannya berurutan dalam satu sesi menghapus hand-off sepenuhnya.
> Trade-off: bagian klasik 4-8 jam ikut memakai kuota GPU. Untuk laporan dengan 4
> notebook terpisah (sesuai struktur bab), tetap pakai cara *Add Data -> Your Work*
> di atas.

---

## D. Workflow Praktis Tim 5 Orang (Rekomendasi Utama)

Inti masalahnya bukan teknis kode, tapi **jangan sampai 5 orang masing-masing
re-run training 4-8 jam.** Solusi paling praktis: **1 orang menjalankan training,
4 orang lain memakai hasilnya.** Pipeline ini sekuensial (02->03->04 saling
bergantung), jadi yang bisa diparalelkan adalah *pengembangan & penulisan*, bukan
training-nya.

### Prinsip emas (3 hal saja)
1. **Kode -> GitHub** (satu sumber kebenaran kode). Edit di laptop, push dari laptop.
2. **Hasil training -> satu Kaggle Dataset bersama** (satu sumber kebenaran hasil).
3. **Kaggle hanya untuk menjalankan**, bukan mengedit kode atau push Git.

### Pembagian peran (5 orang)
Fleksibel, tapi setiap file/notebook punya **satu pemilik** agar tidak bentrok saat
push (notebook `.ipynb` itu JSON - hampir mustahil di-merge kalau 2 orang edit bareng).

| Orang | Peran | Pemilik file |
|-------|-------|--------------|
| 1 | **Koordinator & Runner** - menjalankan pipeline penuh di Kaggle, publikasikan hasil | `notebooks/01_eda.ipynb`, koordinasi merge |
| 2 | **Engineer Klasik** | `notebooks/02_*.ipynb`, `src/features.py`, `src/enhancement.py`, `src/segmentation.py` |
| 3 | **Engineer CNN** | `notebooks/03_*.ipynb`, `src/models.py`, `src/pipeline.py` |
| 4 | **Analis Hasil** | `notebooks/04_*.ipynb`, `src/evaluate.py` - bikin tabel & plot untuk laporan |
| 5 | **Penulis Laporan & QA** | uji lokal pakai sample data, validasi angka, tulis laporan |

> Semua tetap bisa mengembangkan & menguji **tanpa dataset penuh** di laptop:
> `python scripts/create_sample_data.py` (80 citra) lalu `python scripts/smoke_test.py`.
> Iterasi cepat, nol kuota Kaggle.

### Alur kerja (4 fase)

**Fase 1 - Setup sekali (Koordinator):**
- Pastikan repo GitHub **public**: `https://github.com/faizhuda/pcd-kelompok-17`.
- Buat **satu Kaggle Dataset** bernama `pcd-kelompok-17-results` (boleh kosong dulu),
  lalu **Settings -> Collaborators -> tambahkan 4 anggota** (akses *Edit*). Ini wadah
  bersama untuk model `.pkl`/`.h5` + CSV metrik. Dibuat **sekali**.

**Fase 2 - Pengembangan (semua, paralel, di laptop):**
```bash
git pull --ff-only              # selalu tarik dulu sebelum kerja
# edit file milikmu saja (lihat tabel peran)
git add src/ notebooks/         # hanya kode - bukan hasil/dataset
git commit -m "deskripsi singkat"
git push
```

**Fase 3 - Training (cukup 1 Runner, ~1x per pekan):**
1. Buka Kaggle Notebook, jalankan **setup cell** (Bagian B) -> kode terbaru ter-clone.
2. Jalankan `01` -> `02` (CPU, **Save & Run All**, tahan 12 jam) -> `03` (GPU P100).
   Antar-notebook pakai **+ Add Data -> Your Work** (auto-restore, Bagian C).
   > Kuota cukup: `02` di CPU tidak memakai kuota GPU; `03` ~2-3 jam dari jatah
   > 30 jam GPU/pekan per akun. Satu Runner sanggup seluruh pipeline dalam 1 pekan.
3. Setelah `03` selesai, **publikasikan hasil** ke Dataset bersama: dari tab
   **Output** notebook -> **New Dataset / Update** -> pilih `pcd-kelompok-17-results`
   (atau download folder `results/` lalu unggah ke dataset itu via web). Sekali klik
   per training run.

**Fase 4 - Analisis & laporan (semua, paralel, tanpa training):**
1. Buka Kaggle Notebook (atau di laptop), jalankan setup cell.
2. **+ Add Data -> `pcd-kelompok-17-results`** -> setup cell auto-restore ke
   `results/`. Notebook `04` langsung jalan dalam ~5 menit menghasilkan tabel & plot.
3. Ambil figur/tabel untuk laporan. **Nol training, nol tunggu berjam-jam.**

### Aturan commit (ringkas)
- **WAJIB di-commit:** perubahan di `src/` dan `notebooks/` (kode saja).
- **JANGAN di-commit** (sudah di `.gitignore`): `data/raw/`, `data/processed/`,
  `results/models/*`, `results/metrics/*.csv`, figur PNG. Hasil training mengalir
  lewat **Kaggle Dataset bersama**, bukan Git.
- `splits.json` **tidak perlu** di-commit (deterministik, selalu di-regenerate).
- **Jangan `git push` dari Kaggle.** Kaggle = menjalankan; laptop = edit & push.
- **Selalu `git pull --ff-only` sebelum mulai**, dan edit hanya file milikmu ->
  praktis tidak akan ada konflik.

---

## E. Rekomendasi Perubahan Kode

| Prioritas | Perubahan | File yang Diubah |
|-----------|-----------|------------------|
| Kritis | Ganti setup cell Colab (drive.mount) dengan setup cell Kaggle Bagian B | `notebooks/01..04` cell 1 |
| Kritis | Hapus sel `%pip install ...` (semua sudah pre-installed di Kaggle) | `notebooks/01..04` cell 2 |
| Kritis | Perbaiki deteksi `ROOT` agar tidak menebak `cwd().parent` | `notebooks/01..04` cell 3 |
| Kritis | Arahkan `RAW_DATA_DIR` ke `/kaggle/input/...` (jangan `kagglehub.dataset_download`) | `notebooks/01_eda.ipynb` cell 4 |
| Penting | Regenerate `splits.json` dari `RAW_DATA_DIR` di awal notebook 02/03/04 (jangan andalkan `load_splits` dari git) | `notebooks/02..04` cell 3 |
| Penting | Aktifkan & persist cache fitur `.npz` untuk resume notebook 02 | `notebooks/02_experiments_classical.ipynb` |
| Penting | Skip skenario yang CSV-nya sudah ada (resume halus) | `src/experiments.py` (opsional) |
| Nice-to-have | Update `README.md`: jadikan Kaggle **satu-satunya** jalur terdokumentasi; hapus seksi Colab (mount Drive + `kaggle.json`) agar tidak ada dua sumber kebenaran | `README.md` |
| Nice-to-have | Anti-idle interaktif Kaggle (sebenarnya cukup pakai "Save & Run All") | dokumentasi |

### Kritis #1 - Setup cell

Sudah dijabarkan di **Bagian B** (ganti cell 1 + cell 2 lama dengan satu setup cell).

### Kritis #2 - Deteksi ROOT (cell 3)

**Before** (`notebooks/01_eda.ipynb` cell 3, sama di 02/03/04):
```python
_in_colab = "COLAB_RELEASE_TAG" in os.environ
ROOT = Path.cwd() if _in_colab else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

**After** (Kaggle-only - `PROJECT_DIR` sudah di-set setup cell; tidak menebak):
```python
# Setup cell sudah chdir ke PROJECT_DIR dan memasukkannya ke sys.path.
# Cukup pakai itu langsung - tanpa percabangan Colab/lokal.
ROOT = Path("/kaggle/working/pcd-kelompok-17")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

> Sel ini sebenarnya **boleh dihapus seluruhnya** karena setup cell sudah
> mengurus `cwd` + `sys.path`. Dipertahankan hanya agar `ROOT` tersedia sebagai
> variabel jika dipakai di sel lain.

### Kritis #3 - Sumber dataset (notebook 01 cell 4)

**Before** (`notebooks/01_eda.ipynb` cell 4):
```python
import kagglehub
DOWNLOAD_DATASET = True
if DOWNLOAD_DATASET:
    _in_colab = "COLAB_RELEASE_TAG" in os.environ
    if _in_colab:
        _kaggle_src = Path('/content/drive/MyDrive/.kaggle/kaggle.json')
        ...
    kaggle_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    RAW_DATA_DIR = kaggle_path
else:
    RAW_DATA_DIR = paths['data_raw']
```

**After** (pakai dataset yang sudah di-attach - nol download):
```python
# Di Kaggle: dataset di-attach langsung (lihat setup cell). Tidak ada download.
RAW_DATA_DIR = Path("/kaggle/input/fruit-and-vegetable-disease-healthy-vs-rotten")
assert RAW_DATA_DIR.exists(), "Attach dataset dulu: + Add Data -> cari nama dataset -> Add."
print("Dataset:", RAW_DATA_DIR)
```

### Penting #1 - Regenerate splits di notebook 02/03/04

Karena `data/splits.json` menyimpan path absolut spesifik-mesin (masalah #2) dan
split bersifat deterministik, **jangan baca `splits.json` dari git**. Bangun ulang
di tempat. Ini juga menghapus kebutuhan mengoper `splits.json` antar-notebook.

**Before** (`notebooks/02_experiments_classical.ipynb` cell 3, mirip di 03):
```python
from src.utils import get_project_paths, load_splits, set_seed
set_seed(42)
paths = get_project_paths()
train, val, test = load_splits(paths["splits"])
```

**After** (regenerate dari dataset yang di-attach - selalu konsisten lintas mesin):
```python
from src.utils import build_dataset_index, make_splits, get_project_paths, set_seed
set_seed(42)
paths = get_project_paths()
RAW_DATA_DIR = Path("/kaggle/input/fruit-and-vegetable-disease-healthy-vs-rotten")
df = build_dataset_index(RAW_DATA_DIR)
train, val, test = make_splits(df)   # deterministik (SEED=42), identik di tiap sesi
print(len(train), len(val), len(test))
```

> Karena `make_splits` memakai `random_state=SEED`, split akan **identik** di
> notebook 01, 02, 03, 04, dan di laptop semua anggota - selama dataset Kaggle
> versinya sama. Inilah cara termudah menjaga konsistensi train/val/test.

### Penting #2 - Resume notebook 02 (4-8 jam)

**Cara paling mudah (rekomendasi): "Save & Run All (Commit)".** Jalankan notebook 02
via Save Version -> *Save & Run All*. Ini berjalan headless hingga **12 jam**, cukup
untuk menampung 4-8 jam tanpa terganggu disconnect browser. Tidak perlu resume.

**Lapisan tambahan (kalau ingin aman): persist cache fitur `.npz`.**
`extract_split_matrix` (`src/experiments.py:37-65`) sudah menyimpan fitur per-split
ke `cache_dir` (= `data/processed/`). Tahap ekstraksi fitur inilah yang termahal;
training SVM/RF relatif cepat. Karena `data/processed/` ada di `/kaggle/working`,
*Save Version* akan mengawetkannya. Saat re-run, **+ Add Data -> Your Work** pilih
run sebelumnya; setup cell (`restore_previous_outputs()`) otomatis memulihkan
`data/processed/` -> ekstraksi fitur dilewati, skenario tinggal melatih ulang cepat.

**Opsional - skip skenario yang sudah selesai** (`src/experiments.py`, dalam
`run_classical_scenario` setelah `cfg` di-resolve):
```python
# Resume: lewati skenario jika baris CSV-nya sudah ada
metrics_csv = metrics_dir / f"scenario_{scenario_id:02d}.csv"
if metrics_csv.exists():
    print(f"Skenario {scenario_id} sudah ada - dilewati.")
    return {"scenario_id": scenario_id, "skipped": True}
```
> Catatan: jika di-skip, `scenario_results[sid]` tidak akan punya `y_pred`/`y_test`
> untuk uji McNemar. Untuk akademik yang butuh tabel lengkap, lebih aman jalankan
> sekali penuh via *Save & Run All* daripada mengandalkan skip parsial.

---

## F. Quick-Start Guide (1 Halaman)

Ada **dua jalur** sesuai peran (lihat Bagian D). Mayoritas anggota cukup Jalur A.

### Jalur A - Anggota Analis/Penulis (4 dari 5 orang) - TANPA training
**"Saya cuma butuh hasil untuk analisis & laporan."**
1. Login [kaggle.com](https://www.kaggle.com) -> **Create -> New Notebook**.
2. Panel kanan -> **Settings -> Internet on**.
3. **+ Add Data** -> tambahkan dua hal: dataset
   `fruit and vegetable disease healthy vs rotten` **dan** Dataset bersama
   `pcd-kelompok-17-results`.
4. Tempel **setup cell** (Bagian B) di sel teratas -> **Run**. Hasil training otomatis
   ter-restore ke `results/`.
5. Buka `notebooks/04_results_summary.ipynb` (**File -> Open**) -> **Run All**.
   ~5 menit -> tabel & plot siap untuk laporan. **Selesai, tanpa training.**

### Jalur B - Runner (1 orang) - menjalankan training
**"Saya yang menjalankan pipeline penuh."**
1. Login -> **Create -> New Notebook** -> **Settings -> Internet on**.
2. **+ Add Data** -> dataset `fruit and vegetable disease healthy vs rotten`.
3. Tempel **setup cell** (Bagian B) -> **Run** (repo ter-clone, nol install).
4. Jalankan `01_eda.ipynb` -> **Run All** (~15-20 mnt, CPU).
5. `02_experiments_classical.ipynb`: **Save Version -> Save & Run All** (CPU, headless
   <=12 jam). Selesai -> **+ Add Data -> Your Work** pilih run `02` untuk notebook `03`.
6. `03_experiments_cnn.ipynb`: **Settings -> Accelerator -> GPU P100**, attach output
   `02`, lalu **Save & Run All** (~2-3 jam).
7. **Publikasikan hasil:** tab **Output** -> unggah/Update ke Dataset bersama
   `pcd-kelompok-17-results` (Bagian D, Fase 3). Beri tahu grup -> semua pakai Jalur A.

**Urutan menjalankan keseluruhan (wajib berurutan):**

| # | Notebook | Accelerator | Estimasi | Menghasilkan |
|---|----------|-------------|----------|--------------|
| 1 | `01_eda.ipynb` | CPU | ~15-20 mnt | split (di-regenerate, deterministik) |
| 2 | `02_experiments_classical.ipynb` | CPU | 4-8 jam -> pakai *Save & Run All* | CSV skenario 1-10, `best_enhancement.txt`, model S6 & S10 |
| 3 | `03_experiments_cnn.ipynb` | **GPU P100** | 2-3 jam | CSV skenario 11, model MobileNetV2 |
| 4 | `04_results_summary.ipynb` | CPU | ~5 mnt | Tabel 3, semua plot |

> Notebook 03 butuh output notebook 02 (`best_enhancement.txt` + model S6), dan
> notebook 04 butuh semua CSV dari 02 & 03. Caranya satu klik: **+ Add Data -> Your
> Work** pilih run sebelumnya, lalu jalankan setup cell - hasil lama dipulihkan
> otomatis (lihat Bagian C). Tidak ada path yang perlu diketik.

---

### Ringkasan: kenapa pendekatan ini paling praktis untuk tim 5 orang
- **1 Runner, 4 konsumen**: hanya 1 orang menjalankan training 4-8 jam; 4 lainnya
  langsung memakai hasil (Jalur A) -> nol pemborosan waktu & kuota.
- **Dua sumber kebenaran yang jelas**: kode di GitHub, hasil di satu Kaggle Dataset
  bersama. Tidak ada kebingungan "versi siapa yang benar".
- **Anti-bentrok**: setiap notebook/modul punya satu pemilik -> push tanpa konflik.
- **Nol konfigurasi**: tanpa Google Drive, tanpa `kaggle.json`, tanpa `pip install`.
- **Dataset di-attach** & **split deterministik di-regenerate** -> tidak ada file
  rapuh yang dioper antar-orang.
- **Hand-off satu klik**: *+ Add Data -> Your Work* / Dataset bersama, auto-restore.

---

## G. Versi Paket Kaggle (Reproducibility)

Diverifikasi pada **Kaggle Notebooks, Juni 2026** (image `Latest Container Image`):

| Paket | Versi di Kaggle | Catatan |
|-------|----------------|---------|
| Python | 3.12 | |
| TensorFlow / Keras | 2.16+ (Keras 3) | `Layer.output_shape` dihapus → kode Grad-CAM pakai `.output.shape` |
| scikit-learn | 1.4+ | |
| opencv-python | 4.9+ | |
| pandas | 2.2+ | |
| numpy | 1.26+ | |

> Jika Kaggle mengupgrade image dan terjadi error tak terduga, cek versi paket
> di notebook dengan: `import sklearn; print(sklearn.__version__)` dll.
