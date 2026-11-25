## Ringkasan Perubahan

<!-- Jelaskan secara singkat apa yang diubah atau ditambahkan pada PR ini -->

## Jenis Perubahan

- [ ] `feat`: Penambahan fitur, skrip, atau komponen baru
- [ ] `fix`: Perbaikan bug atau konfigurasi
- [ ] `experiment`: Eksperimen model / analisis tambahan
- [ ] `docs`: Dokumentasi, README, atau laporan praktikum
- [ ] `chore`: Pemeliharaan repository / `.gitignore` / dependensi

## Dampak pada Eksperimen & Artefak

<!-- Jika mengubah kode preprocessing/training, tuliskan perubahan metrik (MAE, RMSE, R²) atau artefak terkait. Jika tidak berdampak, tulis "Tidak ada perubahan pada model/metrik eksperimen pertama". -->

## Checklist Pra-Merge

- [ ] Branch dibuat dari `main` terbaru dan tidak memiliki *merge conflict*.
- [ ] Tidak ada file sensitif (`.env`, API key, token, kredensial) atau *path* lokal absolut spesifik pengguna.
- [ ] Tidak ada dataset mentah (`data/raw/`), `.venv/`, maupun file biner > 50 MB yang ikut ter-stage.
- [ ] `git status` bersih dari file sementara (`__pycache__/`, `~$*.docx`, `*.tmp`, `*.log`).
- [ ] Jika mengubah kode ML: aturan *anti-data-leakage* (split kronologis, `MinMaxScaler` fit khusus `Train`, tanpa *gap crossing*) tetap terjaga dan sudah diuji secara lokal.
