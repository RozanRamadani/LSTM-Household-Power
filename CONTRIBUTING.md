# Panduan Kontribusi & Kolaborasi Git — LSTM-Household-Power

Dokumen ini menjadi standar kerja kelompok (**M. Faris Sabiq**, **Bagus Haryo Prasojo**, **Rozan Aiman Ramadani**) untuk menjaga kebersihan riwayat Git, mencegah kebocoran *dataset* maupun kredensial, serta menjaga integritas eksperimen *Machine Learning*.

---

## 1. Strategi Branching

| Branch | Peran | Aturan |
| :--- | :--- | :--- |
| `main` | Branch utama yang stabil, teruji, dan siap dipresentasikan | **Dilarang push langsung**; setiap perubahan masuk melalui Pull Request |
| `feature/<topik>` | Pengembangan fitur baru atau dokumen per anggota | Bercabang dari `main` terbaru, misal `feature/inference-cli`, `feature/hyperparameter-tuning`, `feature/docs-readme` |
| `fix/<topik>` | Perbaikan bug atau dokumentasi spesifik | Bercabang dari `main` terbaru, misal `fix/memory-telemetry`, `fix/readme-links` |

---

## 2. Alur Kerja Harian Anggota

1. **Sinkronkan `main` lokal sebelum mulai bekerja:**
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Buat branch fitur baru:**
   ```bash
   git checkout -b feature/<nama-fitur-singkat>
   ```
3. **Kerjakan perubahan secara terfokus & periksa `git status`:**
   ```bash
   git status
   git diff
   ```
   Pastikan tidak ada file `.env`, `.venv/`, dataset mentah `data/raw/`, atau checkpoint eksperimen sementara yang ikut terbawa.
4. **Buat commit dengan pesan konvensional yang jelas:**
   ```bash
   git add <file-yang-relevan>
   git commit -m "feat(ml): deskripsi singkat perubahan"
   ```
   Konvensi *prefix* commit:
   - `feat(...)`: penambahan skrip, fitur, atau komponen model baru
   - `fix(...)`: perbaikan bug atau kesalahan konfigurasi
   - `docs(...)`: pembaruan `README.md`, `CONTRIBUTING.md`, atau laporan
   - `chore(...)`: konfigurasi `.gitignore`, template PR, atau pemeliharaan repo
5. **Push branch fitur ke GitHub dan buka Pull Request:**
   ```bash
   git push -u origin feature/<nama-fitur-singkat>
   ```
6. **Review & Merge Pull Request:**
   - Isi checklist pada `.github/PULL_REQUEST_TEMPLATE.md`.
   - Minta minimal **1 persetujuan (*approval*)** dari anggota kelompok lain sebelum di-merge ke `main`.
   - Setelah PR di-merge, hapus branch fitur di GitHub dan perbarui `main` lokal (`git checkout main && git pull origin main`).

---

## 3. Rekomendasi GitHub Branch Protection (`main`)

Pemilik repository disarankan mengaktifkan pengaturan berikut di **GitHub → Settings → Branches → Add branch protection rule (`main`)**:
- Centang **Require a pull request before merging** → **Require approvals: 1**.
- Centang **Dismiss stale pull request approvals when new commits are pushed**.
- Jangan aktifkan *allow force pushes* maupun *allow deletions* pada `main`.

---

## 4. Aturan Penyimpanan File Machine Learning

1. **Jangan pernah commit ke Git:**
   - `data/raw/*` (`household_power_consumption.txt` berukuran `126.80 MB` dan melebihi batas file 100 MB GitHub).
   - `data/processed/*.csv` dan `*.npz` (`hourly_dataset.csv` `7.10 MB` dan file sequence `.npz` dapat direproduksi lewat `python src/preprocess.py`).
   - `.venv/`, `__pycache__/`, `.env`, token API, maupun file kunci privat (`*.pem`, `*.key`).
   - Checkpoint latihan sementara (`checkpoints/`, file `.keras` eksperimental di luar `models/stacked_lstm_best.keras`).
2. **File yang wajib ter-track di Git:**
   - Kode sumber (`src/*.py`), dependensi (`requirements.txt`), konfigurasi (`.gitignore`, `.github/`, `README.md`, `CONTRIBUTING.md`).
   - Model tervalidasi (`models/stacked_lstm_best.keras` ~430 KB & `models/model_summary.txt`) dan scaler produksi (`data/processed/scalers/*.joblib` + `data/processed/preprocessing_metadata.json`) agar anggota lain dapat melakukan *inference* langsung tanpa melatih ulang.
   - Laporan resmi (`reports/*.md`, `reports/laporan_akhir.docx`, `reports/laporan_akhir.pdf`), metrik (`evaluation_metrics.*`, `training_history.*`, `eda_metrics.json`), dan grafik (`reports/figures/*.png`).
3. **Kapan memakai Git LFS atau Google Drive:**
   - Gunakan **Google Drive** (atau unduh langsung dari tautan UCI pada `README.md`) untuk berbagi dataset mentah `data/raw/` dan arsip `data/processed/` jika anggota membutuhkan data siap pakai tanpa mengulang preprocessing.
   - Pertimbangkan **Git LFS** hanya jika ke depan tim menyimpan bobot model berkapasitas besar (`> 5–10 MB`) atau melakukan banyak revisi biner `*.keras` / `*.pdf` / `*.docx` secara berulang (perhatikan kuota gratis penyimpanan/bandwidth GitHub LFS sebesar 1 GB).

---

## 5. Checklist Integritas Eksperimen Sebelum Pull Request

Apabila PR menyentuh `src/preprocess.py` atau `src/train.py`:
- [ ] Pemisahan `Train / Validation / Test` dilakukan secara **kronologis murni** (tidak diacak lintas waktu).
- [ ] `MinMaxScaler` hanya di-*fit* pada baris valid split `Train`, lalu di-*transform* ke `Validation` dan `Test`.
- [ ] *Sliding window* tidak melintasi *gap* `NaN` maupun batas antar-split.
- [ ] Fitur kolinear `Global_intensity` tetap dikecualikan kecuali PR memang mendokumentasikan eksperimen ablasi terpisah.
- [ ] Evaluasi metrik (MAE, MSE, RMSE, R²) dihitung ulang dalam **satuan kW asli** setelah *inverse scaling*.
- [ ] Perubahan kode tetap menjaga hasil baseline eksperimen pertama terdokumentasi di `reports/`.
