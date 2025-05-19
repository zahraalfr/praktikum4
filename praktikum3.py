import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi tampilan
st.set_page_config(layout="wide")
st.title("Dashboard Analisis COVID-19 Indonesia")

# Load data
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")

# Ambil data terakhir dari tiap lokasi
latest = df.sort_values('Date').groupby('Location').tail(1)

# Bersihkan data
latest["Case Fatality Rate"] = latest["Case Fatality Rate"].str.replace('%','').astype(float)
latest = latest[latest['Location Level'] == 'Province']

# Fitur yang digunakan
features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']

# Clustering dengan KMeans
X_scaled = StandardScaler().fit_transform(latest[features])
latest['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

# ✅ Tambahkan kolom Risiko berdasarkan CFR
def risiko(row):
    if row < 1:
        return 'Rendah'
    elif row < 3:
        return 'Sedang'
    else:
        return 'Tinggi'

latest['Risiko Wilayah'] = latest['Case Fatality Rate'].apply(risiko)

# ========================
# 🔸 Peta Interaktif
# ========================
st.subheader("Peta Clustering Wilayah")
fig_map = px.scatter_mapbox(
    latest,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    hover_name="Location",
    zoom=4,
    mapbox_style="carto-positron",
    title="Peta Clustering Lokasi COVID-19"
)
st.plotly_chart(fig_map, use_container_width=True)

# ========================
# 🔸 Grafik Tren Harian
# ========================
st.subheader("Tren Kasus Harian")
provinsi = st.selectbox("Pilih Provinsi", latest['Location'].unique())
chart_data = df[df['Location'] == provinsi]
st.line_chart(chart_data.set_index('Date')[['New Cases', 'New Deaths']])

# ========================
# 🔸 Tabel Ringkasan Risiko
# ========================
st.subheader("Ringkasan Risiko Wilayah Berdasarkan Case Fatality Rate")
st.dataframe(latest[['Location', 'Total Cases', 'Total Deaths', 'Case Fatality Rate', 'Risiko Wilayah']].sort_values(by='Case Fatality Rate', ascending=False))
