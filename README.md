# Analisis Persetujuan KPR Oregon 2023

**Prediksi Penolakan KPR di Oregon (HMDA 2023)**

Aplikasi Streamlit: https://pnilbb6bpsgpywly3zue2r.streamlit.app/

Repository ini berisi aplikasi deployment dari Milestone 2. Repository lengkap (notebook, dataset, dan dokumentasi): https://github.com/FTDS-Assignment-Bay-2/P1-M2-FTDS-061-RMT-AmanSegavo

## Repository Outline (Milestone 2)
```
1. README.md - Penjelasan tugas Milestone 2
2. description.md - Dokumentasi project
3. P1M2_Abdurrahman_Assegaf.ipynb - Notebook EDA, preprocessing, modeling, dan kesimpulan
4. P1M2_Abdurrahman_Assegaf_inf.ipynb - Notebook model inference pada data baru
5. P1M2_Abdurrahman_Assegaf_conceptual.txt - Jawaban conceptual problem
6. actions_taken_1-3_loan_purposes_1_state_OR.csv - Dataset HMDA Oregon 2023
7. artefak_preprocessing.joblib - Parameter preprocessing dari data train
8. model_ditolak.joblib - Model terbaik beserta threshold
9. url.txt - Link dataset, model, dan deployment
10. deployment/ - Aplikasi Streamlit (halaman EDA dan Prediction), requirements.txt, artifacts, dan data
```

## Problem Background
Pemberi KPR menerima banyak aplikasi dan perlu menyaringnya dengan cepat. Aplikasi yang akhirnya ditolak memakan waktu tinjau yang sama dengan aplikasi yang disetujui. Project ini menganalisis data aplikasi KPR pembelian rumah di Oregon tahun 2023 untuk mengetahui faktor yang terkait dengan penolakan, lalu membangun model yang memprediksi apakah sebuah aplikasi berisiko ditolak.

## Project Output
1. Model klasifikasi (HistGradientBoosting) beserta threshold keputusan 0,329.
2. Aplikasi Streamlit dengan dua halaman: **EDA** dan **Prediction** (input lewat form atau unggah CSV).

Model hanya alat bantu penyaringan, bukan pengambil keputusan otomatis.

## Data
- Sumber: HMDA (Home Mortgage Disclosure Act) dari FFIEC, KPR pembelian rumah di Oregon tahun 2023.
- Ukuran awal: 45.034 baris dan 99 kolom. Setelah 89 duplikat dibuang tersisa 44.945 baris.
- Target: `ditolak` (1 = ditolak, 0 = disetujui). Proporsi ditolak sekitar 11,45% (data tidak seimbang).
- Pembagian data: 35.956 train dan 8.989 test (stratified).
- Kolom yang bocor (leakage) seperti `denial_reason-1`, `interest_rate`, dan `purchaser_type` dibuang. Kolom demografi tidak dipakai sebagai fitur.

## Method
Supervised learning (klasifikasi biner) dengan alur:
1. Cleaning, EDA, dan feature engineering (`dti_numerik`, `loan_to_income`, `income_to_msa_median`, `punya_coapplicant`).
2. Preprocessing: missing value, winsorizing outlier, transformasi log, one-hot encoding, scaling, dan seleksi fitur.
3. Perbandingan 5 algoritma dengan 5-fold stratified CV: Decision Tree, KNN, SVM, Random Forest, dan HistGradientBoosting.
4. Hyperparameter tuning (`RandomizedSearchCV`) pada model terbaik, lalu pemilihan threshold dari prediksi out-of-fold.
5. Evaluasi pada data test dan interpretasi dengan permutation importance.

Hasil pada data test (threshold 0,329): PR-AUC 0,720, ROC-AUC 0,890, precision 0,742, recall 0,616, F1 0,673. Fitur terpenting: `dti_numerik`, `aus-1`, jenis produk pinjaman, dan `property_value`.

## Stacks
- Bahasa: Python
- Library: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, joblib
- Deployment: Streamlit

## Cara Menjalankan
```
pip install -r deployment/requirements.txt
cd deployment/src
streamlit run streamlit_app.py
```
Notebook dijalankan berurutan: modeling dahulu, lalu notebook inference (membutuhkan dua file `.joblib`).

## Reference
- Dataset: [HMDA Data Browser, FFIEC](https://ffiec.cfpb.gov/data-browser/)
- Deployment: https://pnilbb6bpsgpywly3zue2r.streamlit.app/
