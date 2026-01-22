# Course Recommender System 🎓

Proyek ini adalah sistem rekomendasi kursus yang cerdas, dirancang untuk membantu pengguna menemukan materi pembelajaran yang paling relevan dengan minat dan kebutuhan mereka. Menggabungkan berbagai algoritma Machine Learning untuk memberikan rekomendasi yang akurat dan personal.

## 🚀 Fitur Unggulan

*   **🔍 Jelajah Topik**: Temukan kursus populer berdasarkan kategori minat Anda (Bisnis, Data Science, AI, dll).
*   **🔗 Cari Kemiripan**: Masukkan judul kursus yang Anda sukai, dan sistem akan mencari kursus lain yang memiliki karakteristik serupa.
*   **🤖 Multi-Model Engine**: Menggunakan kekuatan 5 algoritma sekaligus:
    *   **Content-Based**: Analisis deskripsi dan silabus kursus.
    *   **Collaborative Filtering (KNN)**: Berdasarkan pola kemiripan antar item.
    *   **NMF & Neural Embedding**: Teknik faktorisasi matriks untuk prediksi yang lebih dalam.
    *   **Clustering**: Rekomendasi berdasarkan komunitas pengguna serupa.
*   **📊 Dashboard Analitik**: Visualisasi performa model (Precision, Recall, F1-Score) untuk transparansi akurasi.
*   **🎨 Antarmuka Interaktif**: Dibangun dengan Streamlit untuk pengalaman pengguna yang mulus dan modern.

## 🛠️ Teknologi yang Digunakan

*   **Bahasa Utama**: Python
*   **Web Framework**: Streamlit
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn, TensorFlow/Keras (untuk Neural Embedding)
*   **Visualisasi**: Matplotlib, Seaborn

## 💻 Cara Menjalankan

Ikuti langkah mudah ini untuk menjalankan aplikasi di komputer Anda:

1.  **Install Dependensi**
    Pastikan Python sudah terinstal, lalu jalankan perintah berikut di terminal:
    ```bash
    pip install -r app_streamlit/requirements.txt
    ```

2.  **Jalankan Aplikasi**
    Mulai aplikasi Streamlit dengan perintah:
    ```bash
    streamlit run app_streamlit/app.py
    ```

3.  **Akses Browser**
    Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

## 📂 Struktur Proyek

*   `app_streamlit/`: Source code aplikasi web (Streamlit).
*   `src/`: Kode inti untuk model Machine Learning dan pemrosesan data.
*   `data/`: Dataset kursus dan rating.
*   `outputs/`: Hasil visualisasi dan evaluasi model.

---
**Dibuat oleh Ridlo Fanata Wicaksana**
