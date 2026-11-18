import streamlit as st

import pages.eda as eda
import pages.prediction as prediction

# set_page_config hanya dipanggil di sini, jangan di pages/eda.py maupun pages/prediction.py
st.set_page_config(
    page_title='KPR Oregon 2023',
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# st.navigation menggantikan navigasi otomatis dari folder pages/,
# jadi menu bawaan Streamlit tidak muncul lagi
navigasi = st.navigation([
    st.Page(eda.run, title='EDA', icon='📊', url_path='eda', default=True),
    st.Page(prediction.run, title='Prediction', icon='🤖', url_path='prediction'),
])

with st.sidebar:
    st.markdown('---')
    st.markdown('**Analisis Persetujuan KPR Oregon 2023**')
    st.caption('Milestone 2 | Batch FTDS-RMT-061')
    st.caption('Abdurrahman Assegaf')
    st.caption('Data: HMDA / FFIEC')

navigasi.run()
