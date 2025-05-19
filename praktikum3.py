import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

df = pd.read_csv("covid_19_indonesia_time_series_all.csv")

# Hanya ambil data terakhir untuk tiap lokasi (asumsi: data harian)
latest = df.sort_values('Date').groupby('Location').tail(1)

# Bersihkan kolom persen dan ubah menjadi float
latest["Case Fatality Rate"] = latest["Case Fatality Rate"].str.replace('%','').astype(float)
latest["Case Recovered Rate"] = latest["Case Recovered Rate"].str.replace('%','').astype(float)

# Hapus lokasi selain provinsi (misal: Indonesia, DKI Jakarta bisa tetap)
latest = latest[latest['Location Level'] == 'Province']

features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
target = 'Total Cases'

X = latest[features]
y = latest[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
predictions = model.predict(X_test)

# Skala data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(latest[features])

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Simpan hasil cluster
latest['Cluster'] = clusters

# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Dashboard Analisis COVID-19 Indonesia")

# Load data yang sudah diproses
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
latest = df.sort_values('Date').groupby('Location').tail(1)
latest["Case Fatality Rate"] = latest["Case Fatality Rate"].str.replace('%','').astype(float)
latest = latest[latest['Location Level'] == 'Province']

# Feature Engineering
features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(latest[features])

from sklearn.cluster import KMeans
latest['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

# Map cluster
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
st.plotly_chart(fig_map)

# Tren kasus harian (visualisasi manual)
st.subheader("Tren Kasus Harian (Contoh: Jawa Barat)")
location = st.selectbox("Pilih Provinsi", latest['Location'].unique())
chart_data = df[df['Location'] == location]
st.line_chart(chart_data.set_index('Date')[['New Cases', 'New Deaths']])

# Risiko wilayah
st.subheader("Ringkasan Risiko Wilayah")
st.dataframe(latest[['Location', 'Total Cases', 'Case Fatality Rate', 'Cluster']])
