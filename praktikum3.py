import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Konfigurasi tampilan
st.set_page_config(layout="wide")
st.title("📊 Dashboard COVID-19 Indonesia")

# Load data
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")

# Ambil data terakhir dari tiap lokasi
latest = df.sort_values('Date').groupby('Location').tail(1)
latest["Case Fatality Rate"] = latest["Case Fatality Rate"].str.replace('%','').astype(float)
latest = latest[latest['Location Level'] == 'Province']

# Fitur untuk clustering
features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
X_scaled = StandardScaler().fit_transform(latest[features])
latest['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

# Tambahkan kolom Risiko Wilayah berdasarkan CFR
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
st.subheader("🗺️ Peta Clustering Wilayah")
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
st.subheader("📈 Tren Kasus Harian")
provinsi = st.selectbox("Pilih Provinsi", latest['Location'].unique())
chart_data = df[df['Location'] == provinsi]
st.line_chart(chart_data.set_index('Date')[['New Cases', 'New Deaths']])

# ========================
# 🔸 Tabel Ringkasan Risiko
# ========================
st.subheader("🚨 Ringkasan Risiko Wilayah Berdasarkan Case Fatality Rate")
st.dataframe(latest[['Location', 'Total Cases', 'Total Deaths', 'Case Fatality Rate', 'Risiko Wilayah']].sort_values(by='Case Fatality Rate', ascending=False))

# ========================
# 🔸 Supervised Learning - Prediksi Total Kasus
# ========================
st.subheader("🧠 Supervised Learning: Prediksi Total Kasus")

# Fitur dan target
X = latest[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
y = latest['Total Cases']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model regresi
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Tampilkan metrik
st.markdown(f"**R² Score:** `{r2}`")
st.markdown(f"**RMSE:** `{rmse}`")

# Plot Prediksi vs Aktual
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Prediksi')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
ax.set_xlabel("Aktual Total Kasus")
ax.set_ylabel("Prediksi Total Kasus")
ax.set_title("Prediksi vs Aktual Total Kasus COVID-19")
ax.legend()
st.pyplot(fig)
