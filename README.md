# Analisis Relevansi Jurusan PTN & Industri 4.0 di Indonesia

Aplikasi ini adalah hasil proyek Kerja Praktik yang menganalisis relevansi kompetensi lulusan S1 dari 20 Perguruan Tinggi Negeri teratas di Indonesia terhadap tuntutan kompetensi digital di era Industri 4.0. Aplikasi ini juga mendemonstrasikan penerapan visualisasi interaktif menggunakan Streamlit dan model klasifikasi Machine Learning.

## Tujuan Penelitian
- Menganalisis kesesuaian kompetensi lulusan PTN dengan kompetensi digital yang dibutuhkan di era Industri 4.0.
- Mengidentifikasi pola dan hubungan dalam data untuk memberikan gambaran objektif mengenai relevansi pendidikan tinggi dengan tuntutan industri digital.

## Fitur Aplikasi
- **Visualisasi Data Interaktif:** Menampilkan berbagai grafik seperti scatter plot, histogram, bar chart, dan heatmap untuk eksplorasi data.
- **Analisis Relevansi:** Visualisasi untuk menganalisis rasio dosen/mahasiswa, keselarasan horizontal lulusan per akreditasi, dan distribusi lowongan pekerjaan.
- **Model Machine Learning:** Implementasi model Random Forest Classifier untuk memprediksi waktu tunggu lulusan ("Cepat" atau "Lama") berdasarkan karakteristik program studi.
- **Input Prediksi:** Pengguna dapat memasukkan kriteria program studi untuk mendapatkan prediksi waktu tunggu.
- **Feature Importance:** Menampilkan fitur-fitur yang paling berpengaruh dalam prediksi model.

## Sumber Data
Penelitian ini menggunakan tiga sumber data utama:
1.  **Pangkalan Data Pendidikan Tinggi (PDDikti) Kemdikbudristek (2023):** Informasi program studi S1 PTN, akreditasi, jumlah mahasiswa, jumlah dosen, dan rasio dosen/mahasiswa. 
2.  **Studi Tracer Alumni Kemendikbudristek (2023):** Data masa tunggu kerja alumni dan tingkat keselarasan pekerjaan (vertikal dan horizontal). 
3.  **Dataset Terbuka Jobstreet Indonesia (2021) dari Kaggle:** Informasi lowongan kerja, kebutuhan pasar digital, jenis pekerjaan, lokasi, dan keterampilan yang dicari. 

## Metode Penelitian
1.  **Preprocessing Data:** Pembersihan, imputasi, dan standarisasi data dari ketiga sumber.
2.  **Eksplorasi dan Visualisasi Data:** Penggunaan berbagai jenis grafik untuk mengidentifikasi pola dan hubungan antar variabel.
3.  **Pembangunan Model Machine Learning:** Model Random Forest Classifier digunakan untuk mengklasifikasikan masa tunggu kerja lulusan. 

## Struktur Proyek
VISUALISASI_DATA/
├── data/
│   ├── combined_tracer_study.csv
│   ├── jobs_crawling_cleaned.csv
│   ├── merged_data_cleaned.csv
│   └── Pddikti_Combined.xlsx
├── app.py
├── README.md
└── requirements.txt

## Cara Menjalankan Aplikasi
1.  **Pastikan Anda memiliki Python 3.8+ terinstal.**
2.  **Instal dependensi:**
    Navigasi ke direktori `VISUALISASI_DATA` di terminal Anda dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Jalankan aplikasi Streamlit:**
    Dari direktori `VISUALISASI_DATA`, jalankan:
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan terbuka di browser web default Anda.

## Hasil Singkat (dari penelitian)
- Rata-rata keselarasan horizontal lulusan: 62%.
- Rata-rata masa tunggu kerja lulusan: 4.2 bulan.
- Akurasi klasifikasi model Random Forest: 85%.