# Capstone Project — Course Recommender System (Final Task)

**Nama:** Ridlo Fanata Wicaksana  
**Tanggal:** 22 Jan 2026 (Asia/Jakarta)

Project ini dibuat untuk memenuhi **Capstone Project: Final Task** (presentasi) — berisi:
- EDA (grafik + insight)
- Content-based recommender (3 metode)
- Collaborative filtering (KNN, NMF, Neural Embedding)
- Evaluasi model (Precision@10, Recall@10, RMSE)
- (Opsional) Demo Streamlit untuk bonus kreativitas

---

## 1) Setup cepat

### Opsi A — pakai notebook
Buka `notebook.ipynb` lalu jalankan semua sel.

### Opsi B — pakai script
Buat environment (opsional) dan install dependensi:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2) Jalankan EDA + model

### A) EDA (menghasilkan gambar untuk slide)
```bash
python src/eda.py
```

Output ke folder `outputs/`:
- `genre_counts.png`
- `rating_dist.png`
- `enrollment_dist.png` (jika kolom enrollment ada)
- `top20_courses.png`
- `title_words.png`
- `wordcloud_like.png`
- `data_summary.txt`

### B) Jalankan tiap model (demo singkat)
```bash
python src/content_user_profile.py
python src/content_similarity.py
python src/content_clustering.py

python src/cf_knn.py
python src/cf_nmf.py
python src/cf_neural_embedding.py
```

### C) Evaluasi & perbandingan model (untuk slide evaluasi)
```bash
python src/evaluate.py
```

Output:
- `outputs/metrics_summary.csv`
- `outputs/metrics_summary.md`
- `outputs/metrics_precision_at10.png`
- `outputs/metrics_recall_at10.png`
- `outputs/metrics_rmse.png` (jika ada)

---

## 3) Isi PPT template Coursera

Di dalam zip ini ada template: **`ml-capstone-template-coursera.pptx`**.

Cara pakai:
1. Buka template PPT
2. Replace placeholder dengan:
   - Grafik dari folder `outputs/`
   - Tabel dari `outputs/metrics_summary.csv` (atau `metrics_summary.md`)
3. Export PPT → PDF untuk submission Coursera

---

## 4) (Opsional) Deploy Streamlit untuk bonus kreativitas

> Aku tidak bisa deploy dari sini karena butuh akses akun GitHub/Streamlit kamu.
> Kamu yang deploy, lalu ambil screenshot + link untuk dimasukkan ke slide.

### Jalankan lokal
```bash
pip install -r app_streamlit/requirements.txt
streamlit run app_streamlit/app.py
```

### Deploy ke Streamlit Cloud
1. Push folder project ini ke GitHub repo
2. Streamlit Cloud → New app → pilih repo
3. Entry point: `app_streamlit/app.py`
4. Requirements: `app_streamlit/requirements.txt`

Setelah online:
- ambil screenshot UI
- copy link aplikasi
- masukkan ke slide “Creativity / Demo”

---

## 5) Dataset
Folder `data/` berisi dataset **sintetis** agar project bisa langsung jalan.
Kalau kamu punya dataset resmi dari lab Coursera, replace file:
- `data/courses.csv`
- `data/ratings.csv`

Lihat format di: `data/README_dataset.md`

---

## Struktur folder
- `src/` : semua script model + evaluasi
- `outputs/` : gambar + tabel siap untuk slide
- `app_streamlit/` : demo web app Streamlit
- `notebook.ipynb` : all-in-one notebook
