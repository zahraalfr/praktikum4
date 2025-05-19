# Judul: Analisis Data COVID-19 Indonesia
# 📚 Modul 3 — Supervised & Unsupervised Learning

# ============================================
# 📦 IMPORT LIBRARY
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 📁 LOAD DAN PRA-PROSES DATA
# ============================================
df = pd.read_csv("covid_19_indonesia_time_series_all.csv")

# Konversi CFR ke desimal
df['Case Fatality Rate'] = df['Case Fatality Rate'].astype(str).str.replace('%', '', regex=False).str.strip()
df['Case Fatality Rate'] = pd.to_numeric(df['Case Fatality Rate'], errors='coerce') / 100.0

# Konversi tanggal
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Ambil kolom penting dan hapus NaN
df = df[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Total Recovered',
         'Population Density', 'Case Fatality Rate']].dropna()

df.head()

# 🔢 1. Supervised Learning: Prediksi Total Kasus

# Definisi fitur dan target
X = df[['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']]
y = df['Total Cases']

# Split data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi performa
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualisasi hasil
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Total Cases")
plt.ylabel("Predicted Total Cases")
plt.title("Prediksi vs Aktual Total Kasus COVID-19")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧠 2. Unsupervised Learning: Clustering Lokasi

# Agregasi data per lokasi
df_grouped = df.groupby('Location').agg({
    'Total Cases': 'max',
    'Total Deaths': 'max',
    'Total Recovered': 'max',
    'Population Density': 'mean'
}).reset_index()

# Ambil fitur numerik
features = df_grouped.iloc[:, 1:]

# KMeans clustering (3 klaster)
kmeans = KMeans(n_clusters=3, random_state=0)
df_grouped['Cluster'] = kmeans.fit_predict(features)

# Visualisasi hasil clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_grouped, x='Total Cases', y='Total Deaths', hue='Cluster', palette='Set2', s=100)

# Tambah label lokasi
for i in range(len(df_grouped)):
    plt.text(df_grouped['Total Cases'][i] + 300, df_grouped['Total Deaths'][i] + 30,
             df_grouped['Location'][i], fontsize=8, alpha=0.7)

plt.title("Clustering Lokasi berdasarkan Total Kasus & Kematian")
plt.xlabel("Total Cases")
plt.ylabel("Total Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🚨 3. Ringkasan Risiko Wilayah Berdasarkan CFR

# Rata-rata CFR per lokasi
df_risk = df.groupby('Location')['Case Fatality Rate'].mean().reset_index()

# Klasifikasi risiko berdasarkan CFR
df_risk['Risk Level'] = pd.cut(df_risk['Case Fatality Rate'],
                               bins=[0, 0.01, 0.03, 1],
                               labels=['Low', 'Medium', 'High'])

# Tampilkan data risiko
print(df_risk.head())

# Visualisasi distribusi tingkat risiko
sns.countplot(x='Risk Level', data=df_risk, palette='coolwarm')
plt.title("Distribusi Risiko Wilayah Berdasarkan Case Fatality Rate")
plt.xlabel("Risk Level")
plt.ylabel("Jumlah Lokasi")
plt.tight_layout()
plt.show()
