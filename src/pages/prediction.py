"""Halaman Prediction: memprediksi apakah sebuah aplikasi KPR akan ditolak.

Bagian atas file berisi fungsi preprocessing yang sama dengan notebook (cleaning 1.5,
feature creation 3.1, preprocess 4.1 sampai 4.6). Parameter hasil fit dibaca dari folder
artifacts/. Bagian bawah (run) adalah tampilan Streamlit-nya.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# folder artifacts ada di src/artifacts, satu tingkat di atas folder pages
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / 'artifacts'

# titik tengah untuk rentang DTI (sama dengan notebook 3.1)
PETA_DTI = {'<20%': 15, '20%-<30%': 25, '30%-<36%': 33, '50%-60%': 55, '>60%': 65}

# pilihan DTI untuk form, berurutan dari rendah ke tinggi
OPSI_DTI = {
    '<20%': 15.0, '20%-<30%': 25.0, '30%-<36%': 33.0,
    **{f'{i}%': float(i) for i in range(36, 50)},
    '50%-60%': 55.0, '>60%': 65.0,
    'Tidak diketahui': np.nan,
}

KOLOM_NUMERIK_MENTAH = [
    'loan_amount', 'income', 'loan_to_value_ratio', 'loan_term', 'property_value',
    'tract_population', 'tract_minority_population_percent', 'tract_owner_occupied_units',
    'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units',
    'tract_to_msa_income_percentage', 'ffiec_msa_md_median_family_income',
]
KOLOM_TRACT = [
    'tract_population', 'tract_minority_population_percent', 'tract_owner_occupied_units',
    'tract_one_to_four_family_homes', 'tract_median_age_of_housing_units',
    'tract_to_msa_income_percentage',
]
BINER = ['punya_coapplicant']
KOLOM_TURUNAN = ['dti_numerik', 'punya_coapplicant', 'loan_to_income', 'income_to_msa_median']
KOLOM_PENDUKUNG = ['debt_to_income_ratio', 'co-applicant_age', 'census_tract']


def muat_artefak(folder=ARTIFACT_DIR):
    """Memuat artefak preprocessing, paket model, dan nilai default form."""
    folder = Path(folder)
    art = joblib.load(folder / 'artefak_preprocessing.joblib')
    paket = joblib.load(folder / 'model_ditolak.joblib')
    with open(folder / 'default_form.json', encoding='utf-8') as f:
        default = json.load(f)
    return art, paket, default


def kolom_mentah_wajib(art):
    """Kolom yang harus ada pada CSV mentah (format FFIEC)."""
    langsung = [k for k in art['fitur_kontinu'] + art['fitur_kategorik'] if k not in KOLOM_TURUNAN]
    return sorted(set(langsung + KOLOM_PENDUKUNG))


def opsi_kategori(art):
    """Daftar kategori per kolom kategorik, diambil dari encoder."""
    ohe = art['ohe']
    return {k: [str(c) for c in cats] for k, cats in zip(ohe.feature_names_in_, ohe.categories_)}


def ke_dti_numerik(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in PETA_DTI:
        return float(PETA_DTI[x])
    try:
        return float(x)
    except ValueError:
        return np.nan


def ke_teks(seri):
    if pd.api.types.is_float_dtype(seri):
        seri = seri.astype('Int64')          # 41039.0 -> 41039
    return seri.astype('string').fillna('Unknown').astype(str)


def bersihkan_hmda(raw, art):
    """Data mentah HMDA (format FFIEC) -> tabel fitur sebelum preprocessing.

    Sama dengan bagian 1.5 dan 3.1 di notebook, tanpa menghapus baris apa pun.
    """
    df = raw.copy()

    # 'Exempt' berarti tidak wajib dilaporkan -> NaN
    kolom_teks = df.select_dtypes(exclude='number').columns
    df[kolom_teks] = df[kolom_teks].replace('Exempt', np.nan)

    for k in KOLOM_NUMERIK_MENTAH:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce')

    # income nol atau negatif tidak masuk akal
    df.loc[df['income'] <= 0, 'income'] = np.nan

    # nol pada baris tanpa lokasi berarti "tidak tersedia"
    tanpa_tract = df['census_tract'].isna()
    kolom_tract = [k for k in KOLOM_TRACT if k in df.columns]
    df.loc[tanpa_tract, kolom_tract] = np.nan
    df['ffiec_msa_md_median_family_income'] = df['ffiec_msa_md_median_family_income'].replace(0, np.nan)

    # feature creation (3.1)
    df['dti_numerik'] = df['debt_to_income_ratio'].map(ke_dti_numerik).astype(float)
    umur_co = df['co-applicant_age'].astype('string').str.replace(r'\.0$', '', regex=True).fillna('')
    df['punya_coapplicant'] = (umur_co != '9999').astype(int)
    df['loan_to_income'] = df['loan_amount'] / (df['income'] * 1000)
    df['income_to_msa_median'] = (df['income'] * 1000) / df['ffiec_msa_md_median_family_income']

    return df[art['fitur_kontinu'] + BINER + art['fitur_kategorik']]


def preprocess(X_mentah, art):
    """Tabel fitur -> matriks siap model (bagian 4.1 sampai 4.6)."""
    kontinu = art['fitur_kontinu']
    kategorik = art['fitur_kategorik']
    X = X_mentah.copy()

    # 4.1 missing value
    for k in art['kolom_flag_missing']:
        X[f'{k}_missing'] = X[k].isna().astype(int)
    X[kontinu] = X[kontinu].fillna(art['median_train'])
    for k in kategorik:
        X[k] = ke_teks(X[k])

    # 4.2 outlier (winsorizing dengan batas dari data train)
    X[kontinu] = X[kontinu].clip(lower=art['batas_bawah'], upper=art['batas_atas'], axis=1)

    # 4.3 transformasi log
    X[art['kolom_log']] = np.log1p(X[art['kolom_log']])

    # 4.4 one-hot encoding
    enc = pd.DataFrame(art['ohe'].transform(X[kategorik]), columns=art['nama_ohe'], index=X.index)
    X = pd.concat([X[kontinu + art['fitur_biner_akhir']], enc], axis=1)

    # 4.5 scaling
    X[kontinu] = art['scaler'].transform(X[kontinu])

    # 4.6 fitur final
    return X[art['fitur_final']]


def prediksi_proba(X_bersih, art, paket):
    """Probabilitas ditolak untuk tabel fitur hasil bersihkan_hmda atau buat_baris_form."""
    return paket['model'].predict_proba(preprocess(X_bersih, art))[:, 1]


def buat_baris_form(isian, default, art):
    """Satu baris fitur dari isian form. Kolom yang tidak diisi memakai nilai default."""
    baris = {**default, **isian}
    income_usd = baris['income'] * 1000
    baris['loan_to_income'] = baris['loan_amount'] / income_usd
    baris['income_to_msa_median'] = income_usd / baris['ffiec_msa_md_median_family_income']

    kolom = art['fitur_kontinu'] + BINER + art['fitur_kategorik']
    X = pd.DataFrame([{k: baris[k] for k in kolom}])
    X[art['fitur_kontinu']] = X[art['fitur_kontinu']].astype(float)
    X['punya_coapplicant'] = X['punya_coapplicant'].astype(int)
    return X

# ---------------------------------------------------------------------------
# bagian tampilan Streamlit
# ---------------------------------------------------------------------------

# label yang lebih mudah dibaca untuk kode HMDA (kode lain ditampilkan apa adanya)
LABEL_AUS = {
    '1': '1 - Desktop Underwriter (DU)', '2': '2 - Loan Prospector (LP)', '3': '3 - TOTAL Scorecard',
    '4': '4 - GUS', '5': '5 - Lainnya', '6': '6 - Tidak berlaku', '7': '7 - Sistem internal lembaga',
    '1111': '1111 - Exempt',
}
LABEL_HUNIAN = {'1': '1 - Tempat tinggal utama', '2': '2 - Rumah kedua', '3': '3 - Properti investasi'}
KODE_LOAN_TYPE = {'Conventional': '1', 'FHA': '2', 'VA': '3', 'FSA/RHS': '4'}


@st.cache_resource(show_spinner=False)
def muat_semua():
    # artefak dimuat sekali saja, bukan setiap kali tombol ditekan
    return muat_artefak()


def indeks_default(pilihan, nilai):
    # posisi nilai default pada daftar pilihan (0 bila tidak ada)
    return pilihan.index(nilai) if nilai in pilihan else 0


def catatan_data(isian, dti_label):
    """Catatan singkat dari EDA untuk isian ini (angka historis, bukan sebab-akibat)."""
    catatan = []
    if dti_label == '>60%':
        catatan.append('DTI di atas 60%: sekitar 81% aplikasi seperti ini ditolak pada data.')
    elif dti_label == '50%-60%':
        catatan.append('DTI 50% sampai 60%: sekitar 20% aplikasi seperti ini ditolak pada data.')
    elif dti_label == 'Tidak diketahui':
        catatan.append('DTI kosong: ditolak 18,9% pada data, dibanding 11,3% bila DTI terisi.')
    if isian['aus-1'] == '6':
        catatan.append('aus-1 kode 6 (tidak berlaku): ditolak 38,8% pada data.')
    if 'Subordinate' in isian['derived_loan_product_type']:
        catatan.append('Pinjaman lien kedua: ditolak 29,5% pada data (lien pertama konvensional 11,2%).')
    if 'Manufactured' in isian['derived_dwelling_category']:
        catatan.append('Rumah manufaktur: ditolak sekitar 39% pada data (site-built 8,0%).')
    return catatan


def tampilkan_hasil(proba, threshold, catatan):
    # kartu hasil: probabilitas, keputusan menurut threshold, dan catatan
    k1, k2 = st.columns(2)
    k1.metric('Probabilitas ditolak', f'{proba:.1%}')
    k2.metric('Threshold model', f'{threshold:.1%}')
    st.progress(float(min(max(proba, 0.0), 1.0)))

    if proba >= threshold:
        st.error('**Berisiko ditolak.** Probabilitas berada di atas threshold, sebaiknya aplikasi ditinjau lebih teliti.')
    else:
        st.success('**Kemungkinan disetujui.** Probabilitas berada di bawah threshold.')

    if catatan:
        st.markdown('**Catatan dari data historis**')
        for c in catatan:
            st.markdown(f'- {c}')
    st.caption('Hasil ini hanya alat bantu penyaringan, bukan keputusan akhir. '
               'Isian yang tidak ada di form memakai nilai tipikal dari data train.')


def tab_form(art, paket, default):
    opsi = opsi_kategori(art)
    threshold = float(paket['threshold'])
    st.markdown('Isi data aplikasi, lalu tekan **Prediksi**.')

    with st.form('form_prediksi'):
        kiri, kanan = st.columns(2)
        with kiri:
            loan_amount = st.number_input('Jumlah pinjaman (USD)', min_value=5000, max_value=5_000_000,
                                          value=int(default['loan_amount']), step=5000)
            income = st.number_input('Pendapatan tahunan pemohon (ribu USD)', min_value=1, max_value=5000,
                                     value=int(default['income']), step=1)
            property_value = st.number_input('Nilai properti (USD)', min_value=5000, max_value=10_000_000,
                                             value=int(default['property_value']), step=5000)
            tenor = st.number_input('Tenor (bulan)', min_value=12, max_value=480,
                                    value=int(default['loan_term']), step=12)
            dti_label = st.selectbox('Debt-to-income ratio (DTI)', list(OPSI_DTI),
                                     index=list(OPSI_DTI).index('40%'))
        with kanan:
            co = st.radio('Co-applicant', ['Ada', 'Tidak ada'], horizontal=True)
            produk = st.selectbox('Jenis produk pinjaman', opsi['derived_loan_product_type'],
                                  index=indeks_default(opsi['derived_loan_product_type'],
                                                       default['derived_loan_product_type']))
            rumah = st.selectbox('Jenis rumah', opsi['derived_dwelling_category'],
                                 index=indeks_default(opsi['derived_dwelling_category'],
                                                      default['derived_dwelling_category']))
            aus = st.selectbox('Sistem underwriting otomatis (aus-1)', opsi['aus-1'],
                               index=indeks_default(opsi['aus-1'], default['aus-1']),
                               format_func=lambda k: LABEL_AUS.get(k, k))
            hunian = st.selectbox('Tujuan hunian', opsi['occupancy_type'],
                                  index=indeks_default(opsi['occupancy_type'], default['occupancy_type']),
                                  format_func=lambda k: LABEL_HUNIAN.get(k, k))
            skor = st.selectbox('Jenis skor kredit pemohon (kode HMDA)', opsi['applicant_credit_score_type'],
                                index=indeks_default(opsi['applicant_credit_score_type'],
                                                     default['applicant_credit_score_type']))
        kirim = st.form_submit_button('Prediksi')

    if not kirim:
        return

    # LTV dihitung dari pinjaman dan nilai properti
    ltv = loan_amount / property_value * 100
    st.caption(f'Loan-to-value (dihitung otomatis): {ltv:.1f}%')

    # jenis pinjaman dan status lien mengikuti jenis produk agar konsisten
    awalan, lien = produk.split(':')
    isian = {
        'loan_amount': float(loan_amount), 'income': float(income), 'property_value': float(property_value),
        'loan_term': float(tenor), 'loan_to_value_ratio': float(ltv),
        'dti_numerik': OPSI_DTI[dti_label], 'punya_coapplicant': 1 if co == 'Ada' else 0,
        'derived_loan_product_type': produk, 'derived_dwelling_category': rumah,
        'loan_type': KODE_LOAN_TYPE.get(awalan, default['loan_type']),
        'lien_status': '2' if 'Subordinate' in lien else '1',
        'construction_method': '2' if 'Manufactured' in rumah else '1',
        'aus-1': aus, 'occupancy_type': hunian, 'applicant_credit_score_type': skor,
    }

    X = buat_baris_form(isian, default, art)
    proba = float(prediksi_proba(X, art, paket)[0])
    tampilkan_hasil(proba, threshold, catatan_data(isian, dti_label))


def tab_csv(art, paket):
    threshold = float(paket['threshold'])
    st.markdown('Unggah file CSV mentah berformat FFIEC/HMDA (kolom sama seperti dataset asli) '
                'untuk memprediksi banyak aplikasi sekaligus.')
    berkas = st.file_uploader('Pilih file CSV', type='csv')
    if berkas is None:
        return

    raw = pd.read_csv(berkas, low_memory=False)
    kurang = [k for k in kolom_mentah_wajib(art) if k not in raw.columns]
    if kurang:
        st.error('Kolom berikut tidak ditemukan di file: ' + ', '.join(kurang))
        return

    with st.spinner('Menghitung prediksi...'):
        proba = prediksi_proba(bersihkan_hmda(raw, art), art, paket)

    hasil = raw.copy()
    hasil['proba_ditolak'] = proba.round(4)
    hasil['prediksi'] = np.where(proba >= threshold, 'Ditolak', 'Disetujui')

    k1, k2, k3 = st.columns(3)
    k1.metric('Jumlah aplikasi', f'{len(hasil):,}')
    k2.metric('Diprediksi ditolak', f"{(hasil['prediksi'] == 'Ditolak').sum():,}")
    k3.metric('Persentase ditolak', f"{(proba >= threshold).mean() * 100:.1f}%")

    kolom_tampil = ['proba_ditolak', 'prediksi'] + [k for k in raw.columns[:6]]
    st.dataframe(hasil[kolom_tampil].head(200))
    st.download_button('Unduh hasil (CSV)', hasil.to_csv(index=False).encode('utf-8'),
                       file_name='hasil_prediksi_kpr.csv', mime='text/csv')


def run():
    st.title('Prediksi Penolakan KPR')
    st.markdown('## Prediction')
    st.caption('created by: Abdurrahman Assegaf | Batch FTDS-RMT-061')
    st.markdown('Model memperkirakan peluang sebuah aplikasi KPR di Oregon (data HMDA 2023) **ditolak**. '
                'Model dilatih pada data historis dan hanya menangkap hubungan statistik, bukan sebab-akibat.')
    st.markdown('---')

    try:
        art, paket, default = muat_semua()
    except FileNotFoundError as e:
        st.error(f'File artefak tidak ditemukan: {e}. Pastikan folder artifacts berisi '
                 'artefak_preprocessing.joblib, model_ditolak.joblib, dan default_form.json.')
        st.stop()

    k1, k2 = st.columns(2)
    k1.metric('Model', str(paket['nama_model']))
    k2.metric('Threshold keputusan', f"{float(paket['threshold']):.3f}")

    tab1, tab2 = st.tabs(['Input manual', 'Unggah CSV'])
    with tab1:
        tab_form(art, paket, default)
    with tab2:
        tab_csv(art, paket)


if __name__ == '__main__':
    run()
