from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title='EDA KPR Oregon',
    layout='wide',
    initial_sidebar_state='expanded'
)

# lokasi data hasil cleaning (diekspor dari notebook, lihat bagian 1.5 dan 3.1)
DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'hmda_oregon_bersih.csv'

KOLOM_KATEGORI = {
    'Jenis produk pinjaman': 'derived_loan_product_type',
    'Jenis rumah': 'derived_dwelling_category',
    'Hasil AUS (aus-1)': 'aus-1',
    'Status lien': 'lien_status',
    'Jenis hunian': 'occupancy_type',
    'Ada co-applicant (1 = ada)': 'punya_coapplicant',
}

KOLOM_NUMERIK = {
    'Loan amount (USD)': 'loan_amount',
    'Income (ribu USD)': 'income',
    'Nilai properti (USD)': 'property_value',
    'Loan-to-value (%)': 'loan_to_value_ratio',
    'DTI (numerik)': 'dti_numerik',
    'Tenor (bulan)': 'loan_term',
}

BATAS_DTI = [0, 20, 30, 36, 40, 44, 47, 50, 60, 100]
LABEL_DTI = ['<20', '20-29', '30-35', '36-39', '40-43', '44-46', '47-49', '50-59', '60+']


@st.cache_data
def muat_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['status'] = df['ditolak'].map({0: 'Disetujui', 1: 'Ditolak'})
    df['kelompok_dti'] = pd.cut(df['dti_numerik'], bins=BATAS_DTI, labels=LABEL_DTI, right=False)
    return df


def tingkat_penolakan(df, kolom, min_n=100):
    # persentase ditolak per kategori, kategori dengan sampel kecil dibuang
    t = df.groupby(df[kolom].astype('string').fillna('Missing'))['ditolak'].agg(['count', 'mean'])
    t = t[t['count'] >= min_n].sort_values('mean', ascending=False)
    t['persen_ditolak'] = t['mean'] * 100
    return t.rename(columns={'count': 'jumlah'}).reset_index(names='kategori')


def run():
    st.title('Analisis Persetujuan KPR Oregon 2023')
    st.markdown('## Exploratory Data Analysis')
    st.caption('created by: Abdurrahman Assegaf | Batch FTDS-RMT-061')

    '''
    Dataset berasal dari HMDA (Home Mortgage Disclosure Act) yang dipublikasikan FFIEC, berisi aplikasi
    KPR untuk pembelian rumah di Oregon tahun 2023 yang berakhir **disetujui** atau **ditolak**.

    Tujuan project ini adalah memahami faktor yang berkaitan dengan penolakan KPR dan membangun model
    untuk memprediksi apakah sebuah aplikasi akan ditolak. Semua temuan di halaman ini adalah hubungan
    statistik, bukan sebab-akibat.
    '''

    st.markdown('---')

    if not DATA_PATH.exists():
        st.error(f'File data tidak ditemukan: {DATA_PATH}. Ekspor dari notebook terlebih dahulu.')
        st.stop()

    df = muat_data()

    # ringkasan
    k1, k2, k3 = st.columns(3)
    k1.metric('Jumlah aplikasi', f'{len(df):,}')
    k2.metric('Ditolak', f"{df['ditolak'].mean() * 100:.1f}%")
    k3.metric('Disetujui : ditolak', f"{(1 - df['ditolak'].mean()) / df['ditolak'].mean():.1f} : 1")

    with st.expander('Lihat contoh data'):
        st.dataframe(df.drop(columns=['status', 'kelompok_dti']).head(100))

    # first viz: tingkat penolakan per kategori
    st.header('Tingkat Penolakan per Kategori')
    pilihan_kat = st.selectbox('Pilih kategori:', list(KOLOM_KATEGORI))
    kolom = KOLOM_KATEGORI[pilihan_kat]
    t = tingkat_penolakan(df, kolom)
    rata2 = df['ditolak'].mean() * 100

    fig = px.bar(t, x='persen_ditolak', y='kategori', orientation='h', hover_data=['jumlah'],
                 labels={'persen_ditolak': '% ditolak', 'kategori': ''})
    fig.add_vline(x=rata2, line_dash='dash', annotation_text=f'rata-rata {rata2:.1f}%')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig)
    st.caption('Kategori dengan kurang dari 100 aplikasi tidak ditampilkan. '
               'Pada aus-1, kode 6 berarti tidak berlaku dan mungkin mencerminkan aplikasi '
               'yang ditolak sebelum diproses sistem underwriting.')

    # second viz: histogram
    st.header('Distribusi Variabel Numerik')
    pilihan_num = st.selectbox('Pilih kolom:', list(KOLOM_NUMERIK))
    kolom_num = KOLOM_NUMERIK[pilihan_num]
    batas_atas = df[kolom_num].quantile(0.99)
    data_plot = df[df[kolom_num] <= batas_atas]

    fig = plt.figure(figsize=(16, 6))
    sns.histplot(data=data_plot, x=kolom_num, hue='status', bins=40, stat='density',
                 common_norm=False, element='step', palette={'Disetujui': '#4C78A8', 'Ditolak': '#E45756'})
    plt.xlabel(pilihan_num)
    st.pyplot(fig)
    st.caption('Nilai di atas persentil ke-99 tidak ditampilkan agar bentuk sebaran terbaca.')

    # third viz: pola DTI
    st.header('Debt-to-Income Ratio vs Penolakan')
    pisah = st.radio('Tampilkan:', ['Semua aplikasi', 'Per jenis produk (3 terbanyak)'], horizontal=True)

    if pisah == 'Semua aplikasi':
        g = (df.groupby('kelompok_dti', observed=True)['ditolak']
               .agg(['count', 'mean']).reset_index())
        g['persen_ditolak'] = g['mean'] * 100
        fig = px.bar(g, x='kelompok_dti', y='persen_ditolak', hover_data=['count'],
                     labels={'kelompok_dti': 'DTI (%)', 'persen_ditolak': '% ditolak', 'count': 'jumlah'})
    else:
        produk = df['derived_loan_product_type'].value_counts().head(3).index
        g = (df[df['derived_loan_product_type'].isin(produk)]
             .groupby(['kelompok_dti', 'derived_loan_product_type'], observed=True)['ditolak']
             .agg(['count', 'mean']).reset_index())
        g = g[g['count'] >= 30]  # kelompok kecil tidak stabil
        g['persen_ditolak'] = g['mean'] * 100
        fig = px.bar(g, x='kelompok_dti', y='persen_ditolak', color='derived_loan_product_type',
                     barmode='group', hover_data=['count'],
                     labels={'kelompok_dti': 'DTI (%)', 'persen_ditolak': '% ditolak',
                             'derived_loan_product_type': 'Jenis produk', 'count': 'jumlah'})
    fig.add_hline(y=rata2, line_dash='dash', annotation_text=f'rata-rata {rata2:.1f}%')
    st.plotly_chart(fig)
    st.caption('Rentang DTI diwakili titik tengahnya (dti_numerik). Kelompok dengan DTI kosong tidak ditampilkan.')

    # fourth viz: scatterplot
    st.header('Income vs Loan Amount')
    contoh = df.dropna(subset=['income', 'loan_amount']).sample(
        n=min(5000, len(df)), random_state=42)
    fig = px.scatter(contoh, x='income', y='loan_amount', color='status',
                     hover_data=['property_value', 'dti_numerik'], opacity=0.5,
                     log_x=True, log_y=True,
                     color_discrete_map={'Disetujui': '#4C78A8', 'Ditolak': '#E45756'},
                     labels={'income': 'Income (ribu USD, skala log)',
                             'loan_amount': 'Loan amount (USD, skala log)'})
    st.plotly_chart(fig)
    st.caption('Sampel acak 5.000 aplikasi agar grafik tetap ringan.')

    st.markdown('---')
    st.header('Temuan Utama')
    '''
    - **DTI** punya hubungan terkuat dengan keputusan (Cramér\'s V = 0,48) dan polanya berbentuk U.
    - **Hasil AUS (`aus-1`)** (V = 0,39) dan **jenis rumah** (manufaktur vs site-built, V = 0,30) juga kuat.
    - Aplikasi yang ditolak punya median income, loan amount, dan nilai properti lebih rendah
      (sekitar 83 vs 121 ribu, 255 vs 385 ribu, dan 355 vs 475 ribu dolar).
    - Jenis produk memengaruhi efek DTI, sehingga DTI tidak bisa dibaca terpisah dari jenis pinjaman.
    - Halaman ini sengaja tidak memuat variabel demografi (ras, etnis, jenis kelamin, umur).
    '''


if __name__ == '__main__':
    run()