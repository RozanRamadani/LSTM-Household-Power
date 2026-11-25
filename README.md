# Prediksi Konsumsi Energi Listrik Rumah Tangga Menggunakan Metode Stacked Long Short-Term Memory (LSTM)

Proyek **Praktikum Pembelajaran Mesin** — Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga (Tahun 2026).

**Anggota Kelompok:**
1. M. Faris Sabiq
2. Bagus Haryo Prasojo
3. Rozan Aiman Ramadani

---

## Ringkasan Proyek & Hasil Utama

Proyek ini memprediksi konsumsi daya aktif rata-rata **1 jam ke depan** (`Global_active_power`, satuan kW) pada rumah tangga individu menggunakan riwayat multivariat **60 jam sebelumnya** (`input shape: (60, 13)`). Dataset berasal dari [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (`2,075,259` observasi 1-menit pada rentang `2006-12-16` hingga `2010-11-26`).

### Arsitektur & Konfigurasi Utama
- **Resolusi & Target:** Agregasi *hourly* (minimal `45/60` menit valid per jam) untuk memprediksi rata-rata daya aktif pada jam berikutnya (`H = 1`).
- **Fitur Input (`13` fitur):** `Global_active_power`, `Global_reactive_power`, `Voltage`, `Sub_metering_1..3` (terskala `MinMaxScaler(0, 1)` khusus data `Train`), ditambah 6 fitur siklik waktu (`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`) dan `is_weekend`. Fitur `Global_intensity` dikecualikan karena kolinearitas ekstrem dengan target (*Pearson r* = `0.9989`).
- **Anti-Data-Leakage:** Split kronologis murni **70% / 15% / 15%** (`Train: 23,713`, `Validation: 4,922`, `Test: 4,801` *sequence*), `MinMaxScaler` hanya di-*fit* pada `24,073` baris valid `Train`, serta *sliding window* dilarang melintasi *gap* `NaN`.
- **Model Stacked LSTM (`32,929` parameter):** `LSTM(64, return_sequences=True)` → `Dropout(0.2)` → `LSTM(32)` → `Dropout(0.2)` → `Dense(16, relu)` → `Dense(1, linear)` dengan `Adam(lr=0.001, clipnorm=1.0)`, `EarlyStopping(patience=8)` (konvergen terbaik pada **Epoch 25**, `val_loss = 0.006625`).

### Ringkasan Metrik Evaluasi (Satuan Asli kW)

| Split | Model | MAE (kW) | MSE (kW²) | RMSE (kW) | R² |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Validation (`n=4,922`) | Persistence Baseline | 0.4630 | 0.4761 | 0.6900 | 0.3953 |
| Validation (`n=4,922`) | Seasonal Naive 24-hour | 0.6644 | 0.9138 | 0.9560 | -0.1605 |
| Validation (`n=4,922`) | **Stacked LSTM** | **0.3618** | **0.2745** | **0.5239** | **0.6514** |
| Test (`n=4,801`) | Persistence Baseline | 0.3908 | 0.3484 | 0.5903 | 0.3177 |
| Test (`n=4,801`) | Seasonal Naive 24-hour | 0.5089 | 0.5738 | 0.7575 | -0.1238 |
| Test (`n=4,801`) | **Stacked LSTM** | **0.3159** | **0.2149** | **0.4635** | **0.5792** |

Pada split **Test**, Stacked LSTM menurunkan MAE sebesar **19.18%** dan RMSE sebesar **21.47%** dibanding *Persistence*, serta menurunkan MAE **37.93%** dibanding *Seasonal Naive 24-hour*. Catatan audit teknis: model cenderung *underprediction* pada beban puncak `> 2.0 kW` (*mean bias* `+0.7253 kW`), terdapat `5` jam interpolasi dua arah yang terisolasi pada split `Train` (`0` pada `Validation`/`Test`), dan evaluasi `Train` dengan mode *Dropout OFF* belum tersedia sehingga kemungkinan *mild overfitting* belum dikesampingkan sepenuhnya.

---

## Struktur Direktori & Kebijakan Penyimpanan

```text
LSTM-Household-Power/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md     # Template Pull Request kelompok
├── data/
│   ├── raw/                         # Dataset mentah UCI (.gitignore, >100 MB)
│   └── processed/
│       ├── preprocessing_metadata.json  # Metadata & audit validasi (tracked)
│       ├── scalers/                 # feature_scaler.joblib & target_scaler.joblib (tracked)
│       └── *.csv, *.npz             # Artefak besar hasil preprocessing (.gitignore)
├── models/
│   ├── stacked_lstm_best.keras      # Model terlatih terbaik 430 KB (tracked untuk inference)
│   └── model_summary.txt            # Ringkasan layer & parameter (tracked)
├── reports/
│   ├── figures/                     # 13 grafik PNG EDA, preprocessing, & evaluasi (tracked)
│   ├── laporan_akhir.{md,docx,pdf}  # Laporan praktikum lengkap (tracked)
│   ├── eda_initial.md               # Laporan EDA awal (tracked)
│   ├── preprocessing_report.md      # Laporan preprocessing (tracked)
│   ├── model_experiment_report.md   # Laporan eksperimen & audit teknis (tracked)
│   ├── evaluation_metrics.{json,csv}# Metrik lengkap & residual (tracked)
│   ├── training_history.{json,csv}  # Histori loss per epoch (tracked)
│   └── predictions_{val,test}.csv   # Hasil prediksi Validation & Test (tracked)
├── src/
│   ├── eda_initial.py               # Exploratory Data Analysis (read-only raw data)
│   ├── preprocess.py                # Agregasi hourly, feature engineering, windowing
│   └── train.py                     # Baseline, training Stacked LSTM CPU, & evaluasi
├── .gitignore
├── CONTRIBUTING.md                  # Panduan branching, PR, & integritas eksperimen
├── README.md
└── requirements.txt
```

---

## Panduan Setup Lingkungan Lokal (Untuk Anggota Kelompok)

### 1. Prasyarat
- **Python:** `3.10.6` (direkomendasikan agar konsisten dengan lingkungan eksperimen)
- **Git:** Pastikan terpasang (`git --version`)

### 2. Clone Repository & Buat Virtual Environment
```bash
git clone https://github.com/RozanRamadani/LSTM-Household-Power.git
cd LSTM-Household-Power

# Buat & aktifkan virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instal dependensi
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> *Catatan dependensi:* `src/preprocess.py` dan `src/train.py` mengimpor `joblib` (`joblib==1.6.0` pada `.venv` utama), yang terpasang otomatis sebagai dependensi dari `scikit-learn==1.7.2`.

### 3. Skenario Penggunaan

#### Opsi A — Langsung Melakukan Inference / Analisis Hasil (Tanpa Mengunduh Dataset 126.80 MB)
Repository ini sudah menyertakan:
- Bobot model terbaik: `models/stacked_lstm_best.keras` (`429.96 KB`)
- Scaler fit Train: `data/processed/scalers/feature_scaler.joblib` & `target_scaler.joblib`
- Metadata & laporan metrik lengkap di `data/processed/preprocessing_metadata.json` dan `reports/`

Sehingga anggota dapat langsung memuat model (`tf.keras.models.load_model("models/stacked_lstm_best.keras")`) dan scaler (`joblib.load(...)`) untuk pengembangan skrip *inference* maupun analisis residual.

#### Opsi B — Reproduksi Penuh (EDA, Preprocessing, hingga Training Ulang)
1. Unduh dataset **Individual Household Electric Power Consumption** dari [UCI Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (atau dari tautan Google Drive internal kelompok jika dibagikan terpisah).
2. Ekstrak berkas teks sehingga terletak tepat di:
   ```text
   data/raw/household_power_consumption.txt
   ```
3. Jalankan pipeline secara berurutan dari root proyek:
   ```bash
   python src/eda_initial.py
   python src/preprocess.py
   python src/train.py
   ```

---

## Dokumentasi & Kolaborasi Tim

- **Laporan Praktikum Lengkap:** [reports/laporan_akhir.md](reports/laporan_akhir.md) (tersedia juga versi `.docx` dan `.pdf`)
- **Laporan Audit & Eksperimen:** [reports/model_experiment_report.md](reports/model_experiment_report.md)
- **Laporan Preprocessing:** [reports/preprocessing_report.md](reports/preprocessing_report.md)
- **Laporan EDA Awal:** [reports/eda_initial.md](reports/eda_initial.md)
- **Aturan Branching, PR, dan Kolaborasi Git:** Baca [CONTRIBUTING.md](CONTRIBUTING.md) sebelum membuat perubahan ke repository.
