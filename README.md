# 💼 Prediksi Gaji Berdasarkan Pengalaman dan Tingkat Pendidikan — Regresi Linier Berganda

Proyek ini merupakan implementasi **Machine Learning dasar** menggunakan **Regresi Linier Berganda** untuk memprediksi **pendapatan seseorang (dalam Rupiah)** berdasarkan dua faktor utama:
1. **Pengalaman kerja (dalam tahun)**
2. **Tingkat pendidikan (SMA, D3, S1, S2)**

Proyek ini merupakan pengembangan dari konsep *Gradient Descent* pada kursus *Machine Learning Andrew Ng*, namun disesuaikan agar relevan dengan konteks di Indonesia.

---

## 📘 Deskripsi Singkat

Model ini bertujuan untuk memahami hubungan linear antara pengalaman dan pendidikan terhadap pendapatan seseorang.  
Fungsi model yang digunakan adalah:

\[
f_{w,b}(x) = w_1 \times x_1 + w_2 \times x_2 + b
\]

di mana:
- `x₁` = pengalaman kerja (tahun)  
- `x₂` = tingkat pendidikan (1–4)  
- `w₁, w₂` = parameter bobot  
- `b` = bias (intersep)  

Model ini dilatih menggunakan algoritma **Gradient Descent** yang diimplementasikan secara **manual (tanpa library ML)** dengan bantuan **NumPy** dan **Matplotlib**.

---

## ⚙️ Fitur Utama

✅ Implementasi manual **gradient descent**  
✅ Menggunakan **NumPy** dan **Matplotlib**  
✅ Dataset buatan dengan konteks realistis  
✅ Visualisasi **penurunan cost function**  
✅ Prediksi gaji berdasarkan input pengguna  
✅ Hasil dan satuan menggunakan **Rupiah (Rp)**  

---

## 📊 Dataset Contoh

| Pengalaman (Tahun) | Pendidikan (Level) | Gaji (Rp)     |
|--------------------:|------------------:|---------------:|
| 1                  | 1 (SMA)          | 4,000,000      |
| 2                  | 1 (SMA)          | 5,000,000      |
| 2                  | 2 (D3)           | 6,000,000      |
| 3                  | 3 (S1)           | 8,000,000      |
| 5                  | 3 (S1)           | 10,000,000     |
| 7                  | 4 (S2)           | 15,000,000     |
| 8                  | 4 (S2)           | 17,000,000     |

Model akan mempelajari pola data ini untuk memperkirakan hubungan antara pengalaman, pendidikan, dan gaji.

---

## 🧠 Algoritma yang Digunakan

### 1️⃣ Model Linier
\[
f_{w,b}(x) = w_1 \times x_1 + w_2 \times x_2 + b
\]

### 2️⃣ Fungsi Biaya (Cost Function)
Menggunakan **Mean Squared Error (MSE)**:
\[
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2
\]

### 3️⃣ Gradient Descent
Proses optimasi untuk memperbarui parameter:
\[
w_j := w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}
\]
\[
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
\]

---

## 🚀 Cara Menjalankan

### Persyaratan
Pastikan Python dan library berikut sudah terpasang:
```bash
pip install numpy matplotlib
