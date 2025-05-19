import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Dashboard COVID-19 Indonesia", layout="wide")

# ============================
# 📌 Pendahuluan & Tujuan
# ============================
st.title("🦠 Dashboard COVID-19 Indonesia")

st.markdown("""
### 📖 Pendahuluan
Pandemi COVID-19 memberikan tantangan besar dalam pengambilan keputusan berbasis data. Melalui pendekatan machine learning, kita dapat memprediksi jumlah kasus dan mengelompokkan wilayah berdasarkan kemiripan karakteristik epidemiologis. Dashboard ini menggabungkan metode **supervised learning** dan **unsupervised learning** untuk membantu visualisasi dan analisis.

### 🎯 Tujuan Praktikum
Praktikum ini bertujuan untuk menerapkan metode machine learning dalam analisis data COVID-19 di Indonesia. Metode supervised learning (regresi linier) digunakan untuk memprediksi jumlah total kasus berdasarkan fitur-fitur epidemiologis. Selain itu, metode unsupervised learning (KMeans Clustering) diterapkan untuk mengelompokkan wilayah berdasarkan karakteristik data kasus. Seluruh hasil analisis disajikan dalam bentuk dashboard interaktif yang memudahkan pengguna dalam memahami tren dan pola data.
""")

# ============================
# 📊 Supervised Learning
# ============================
st.markdown("""
---
## 📈 Supervised Learning: Prediksi Total Kasus
""")

# Load data
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
latest = df.sort_values('Date').groupby('Location').tail(1)
latest = latest[latest['Location Level'] == 'Province']
latest["Case Fatality Rate"] = latest["Case Fatality Rate"].str.replace('%','').astype(float)

# Model
X = latest[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
y = latest['Total Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Metrik evaluasi
st.markdown(f"""
**R² Score:** <span style='color:green; font-size:20px'><b>{r2:.6f}</b></span>  
**RMSE:** <span style='color:green; font-size:20px'><b>{rmse:,.2f}</b></span>
""", unsafe_allow_html=True)

# Plot prediksi vs aktual
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test, y_pred, color='blue', s=20, label='Prediksi')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
ax.set_xlabel("Aktual Total Kasus")
ax.set_ylabel("Prediksi Total Kasus")
ax.set_title("Prediksi vs Aktual Total Kasus COVID-19", fontsize=14)
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ============================
# 🔍 Unsupervised Learning
# ============================
st.markdown("""
---
## 🧠 Unsupervised Learning: Clustering Wilayah
""")

cluster_data = latest[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
latest['Cluster'] = kmeans.labels_

fig_map = px.scatter_geo(latest,
                         locations="Location",
                         locationmode="country names",
                         color="Cluster",
                         size="Total Cases",
                         hover_name="Location",
                         title="🌍 Peta Interaktif Clustering Kasus COVID-19",
                         projection="natural earth")
st.plotly_chart(fig_map, use_container_width=True)

# ============================
# ⚠️ Ringkasan Risiko Wilayah
# ============================
st.markdown("""
---
## ⚠️ Ringkasan Risiko Wilayah Berdasarkan Case Fatality Rate
""")

bins = [0, 1, 3, 100]
labels = ['Rendah', 'Sedang', 'Tinggi']
latest['Tingkat Risiko'] = pd.cut(latest['Case Fatality Rate'], bins=bins, labels=labels, include_lowest=True)
st.dataframe(latest[['Location', 'Case Fatality Rate', 'Tingkat Risiko']].sort_values(by='Case Fatality Rate', ascending=False))

# ============================
# 📌 Kesimpulan
# ============================
st.markdown("""
---

