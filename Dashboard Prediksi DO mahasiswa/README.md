# Dashboard Prediksi DO Mahasiswa

Aplikasi dashboard untuk memprediksi kemungkinan Drop Out (DO) mahasiswa menggunakan model Machine Learning.

## Anggota Kelompok 10
- Arya Mulahernawan (G1A022029)
- Yebi Depriansyah (G1A022063)
- Aisyah Amelia Zarah Juaita (G1A022075)

## Persyaratan Sistem
- Python 3.8 atau lebih baru
- pip (Python package manager)

## Cara Instalasi

1. Clone repository ini atau download source code

2. Buat virtual environment Python:
```bash
python -m venv venv
```

3. Aktifkan virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies yang diperlukan:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan Aplikasi

1. Pastikan virtual environment sudah aktif

2. Jalankan aplikasi dengan perintah:
```bash
streamlit run app.py
```

3. Buka browser dan akses URL yang ditampilkan di terminal (biasanya http://localhost:8501)

## Fitur Aplikasi
- Input data akademik mahasiswa (IPK, kehadiran, dll)
- Input data pribadi mahasiswa
- Prediksi kemungkinan DO
- Rekomendasi berdasarkan hasil prediksi

## File yang Diperlukan
- `app.py` - File utama aplikasi
- `random_forest_model.joblib` - Model machine learning
- `label_encoders.pkl` - Encoder untuk data kategorikal
- `feature_columns.pkl` - Daftar kolom fitur
- `label_Sacler.pkl` - Scaler untuk normalisasi data
- `requirements.txt` - Daftar package yang diperlukan 