
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import core modules from src
import sys
# Add parent dir to sys.path to allow importing from src
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src")) # Fix for 'ModuleNotFoundError: No module named utils'

from src.utils import load_data, course_genre_matrix, build_mappings, build_sparse_matrix, explode_genres
from src.cf_knn import fit_item_knn, recommend_user_item_knn
from src.cf_nmf import fit_nmf, recommend_user_nmf
from src.cf_neural_embedding import train_mf_sgd, recommend_user as recommend_user_neural
from src.content_clustering import build_user_profile_matrix, cluster_users, cluster_popular_recs

# --------------------------------------------------------------------------
# CONFIG & STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Course Recommender Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #444; font-weight: 600;}
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #333333; /* Fix: Force dark text for visibility */
    }
    .card h4 {
        color: #000000;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #eef4ff;
        border-radius: 8px;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------
# LOAD DATA & TRAIN MODELS (CACHED)
# --------------------------------------------------------------------------
@st.cache_resource
def get_data_and_models():
    # 1. Load Data
    data = load_data(DATA_DIR)
    courses = data.courses
    ratings = data.ratings
    
    # 2. Precompute Content-Based (TF-IDF)
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    text = (courses_sorted["title"].fillna("") + " " + courses_sorted["genres"].fillna("").str.replace("|", " ", regex=False)).astype(str)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    tfidf_matrix = tfidf.fit_transform(text)
    cid_list = courses_sorted["course_id"].tolist()
    
    # 3. Train Collaborative Filtering Models (KNN, NMF, Neural)
    # Note: For Capstone scale (small data), we can train on the fly.
    # Typically we would load saved models.
    u2i, it2i = build_mappings(ratings)
    R_csr = build_sparse_matrix(ratings, u2i, it2i, kind="csr")
    
    # KNN
    knn_model = fit_item_knn(R_csr, n_neighbors=50)
    
    # NMF
    W_nmf, H_nmf = fit_nmf(R_csr, n_factors=20, seed=42)
    
    # Neural Embedding (Simplified Numpy implementation)
    user_emb, item_emb, user_bias, item_bias, gmean = train_mf_sgd(ratings, u2i, it2i, n_factors=32, n_epochs=5)
    neural_params = (user_emb, item_emb, user_bias, item_bias, gmean)
    
    # Clustering
    U_prof, u_ids = build_user_profile_matrix(ratings, courses)
    cluster_labels = cluster_users(U_prof, n_clusters=5)
    
    return {
        "courses": courses,
        "ratings": ratings,
        "tfidf_matrix": tfidf_matrix,
        "cid_list": cid_list,
        "u2i": u2i,
        "it2i": it2i,
        "R_csr": R_csr,
        "knn_model": knn_model,
        "nmf_model": (W_nmf, H_nmf),
        "neural_params": neural_params,
        "cluster_data": (U_prof, cluster_labels, u_ids)
    }

assets = get_data_and_models()

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_course_title(cid):
    row = assets["courses"][assets["courses"]["course_id"] == cid]
    if not row.empty:
        return row.iloc[0]["title"]
    return f"Unknown ({cid})"

def get_representative_user_for_genre(genre_indonesian):
    # Map Indonesian Topic -> actual genre word
    map_genre = {
        "Bisnis & Keuangan": "Business",
        "Data Science & Analisis": "Data Science",
        "Pengembangan Web & DevOps": "DevOps",
        "Kecerdasan Buatan (AI)": "Machine Learning",
        "Pemrograman Python": "Python",
        "Big Data": "Big Data"
    }
    
    target_genre = map_genre.get(genre_indonesian, "Business")
    
    # Logic: Find a user who has rated many courses in this genre with high scores
    # 1. Identify courses in this genre
    all_courses = assets["courses"]
    mask = all_courses["genres"].fillna("").str.contains(target_genre, case=False)
    target_cids = set(all_courses[mask]["course_id"])
    
    # 2. Filter ratings
    relevant_ratings = assets["ratings"][assets["ratings"]["course_id"].isin(target_cids)]
    
    # 3. Find top user
    if relevant_ratings.empty:
        return assets["ratings"]["user_id"].iloc[0] # Fallback
        
    top_user = relevant_ratings["user_id"].value_counts().index[0]
    return top_user

def show_recommendations(recs, model_name):
    if not recs:
        st.warning("Belum ada rekomendasi yang ditemukan.")
        return
        
    st.subheader(f"✨ Rekomendasi dari: {model_name}")
    
    # Convert list of IDs to DataFrame
    res_df = assets["courses"][assets["courses"]["course_id"].isin(recs)].copy()
    # Enforce order
    res_df = res_df.set_index("course_id").loc[recs].reset_index()
    
    for i, row in res_df.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="card">
                <h4 style="margin:0;">{i+1}. {row['title']}</h4>
                <p style="color:grey; font-size:0.9em;">Genre: {row['genres']} | 👥 {row.get('enrollment', 'N/A')} enrolled</p>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------

def page_home():
    st.markdown('<p class="main-header">🎓 Course Recommender System</p>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1501504905252-473c47e087f8?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Selamat Datang!")
        st.write("""
        Aplikasi ini menggunakan **Machine Learning** canggih untuk membantu Anda menemukan kursus online terbaik. 
        Kami menggabungkan berbagai metode AI untuk memastikan Anda mendapatkan rekomendasi yang paling relevan.
        """)
        
        st.info("👈 **Mulai dengan memilih menu di Sidebar!**")
        
        st.write("#### 🚀 Fitur Unggulan:")
        st.write("1. **Jelajah Topik**: Cari kursus populer berdasarkan minat Anda.")
        st.write("2. **Cari Kemiripan**: Suka satu kursus? Kami carikan yang mirip!")
        st.write("3. **Multi-Model**: Menggunakan 5 algoritma AI sekaligus (KNN, Content-Based, Neural Net, dll).")

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Total Kursus", len(assets["courses"]))
        st.metric("Total Rating User", len(assets["ratings"]))
        st.markdown('</div>', unsafe_allow_html=True)

def page_eda():
    st.markdown('<p class="main-header">📊 Analisis Data (EDA)</p>', unsafe_allow_html=True)
    st.write("Sebelum masuk ke prediksi, mari kita lihat statistik data kursus yang kita miliki.")
    
    courses = assets["courses"]
    ratings = assets["ratings"]
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Genre", "Distribusi Rating", "Word Cloud"])
    
    with tab1:
        st.subheader("Topik/Genre Paling Populer")
        g_long = explode_genres(courses).reset_index(drop=True) # Fix: Reset index to avoid 'cannot reindex with duplicate labels' error
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=g_long["genre"], order=g_long["genre"].value_counts().index[:10], ax=ax, palette="viridis")
        st.pyplot(fig)
        st.caption("Grafik menunjukkan kategori kursus yang paling banyak tersedia di database.")

    with tab2:
        st.subheader("Kepuasan Pengguna (Rating)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(ratings["rating"], bins=10, kde=True, color="orange", ax=ax)
        st.pyplot(fig)
        st.caption("Mayoritas user memberikan rating tinggi, menandakan kualitas kursus yang baik.")
    
    with tab3:
        st.subheader("Kata Kunci Populer")
        # Simple frequency count
        from collections import Counter
        all_words = " ".join(courses["title"].astype(str)).lower().split()
        # Filter stopwords simple
        stops = {"and", "to", "of", "intro", "in", "the", "for", "with", "complete", "essential", "applied"}
        filtered = [w for w in all_words if w not in stops and len(w) > 2]
        counts = Counter(filtered).most_common(15)
        
        # Plot bar chart style for word cloud to avoid complex text processing
        words, vals = zip(*counts)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=list(vals), y=list(words), palette="magma", ax=ax)
        st.pyplot(fig)

def page_recommender():
    st.markdown('<p class="main-header">🎯 Sistem Rekomendasi</p>', unsafe_allow_html=True)
    
    choice = st.radio("Pilih Cara Belajar Anda:", 
                      ["Saya ingin belajar Topik tertentu...", "Saya suka Kursus tertentu, cari yang mirip..."],
                      horizontal=True)
    
    # --------------------------------------------------------------------------
    # MODE 1: BY TOPIC
    # --------------------------------------------------------------------------
    if choice == "Saya ingin belajar Topik tertentu...":
        st.markdown("---")
        st.subheader("🔍 Jelajah Topik")
        
        genre_topic = st.selectbox("Pilih Bidang Minat:", [
            "Bisnis & Keuangan", 
            "Data Science & Analisis", 
            "Pengembangan Web & DevOps", 
            "Kecerdasan Buatan (AI)",
            "Pemrograman Python",
            "Big Data"
        ])
        
        if st.button("Tampilkan Rekomendasi"):
            rep_user = get_representative_user_for_genre(genre_topic)
            
            # 1. Popularity (Simple Logic: Filter by Genre text and sort by enrollment/rating)
            # Improved mapping to handle partial matches more robustly
            # If map fails, we search for ANY match or fallback to all
            map_keyword = {
                "Bisnis & Keuangan": ["Business", "Finance"],
                "Data Science & Analisis": ["Data Science", "Analysis", "Data Engineering"],
                "Pengembangan Web & DevOps": ["Web", "DevOps", "Cloud"],
                "Kecerdasan Buatan (AI)": ["Machine Learning", "Deep Learning", "Computer Vision"],
                "Pemrograman Python": ["Python"],
                "Big Data": ["Big Data"]
            }
            keywords = map_keyword.get(genre_topic, ["Business"])
            
            # Construct Regex mask for flexible matching
            pattern = "|".join(keywords)
            mask = assets["courses"]["genres"].fillna("").str.contains(pattern, case=False, regex=True)
            
            # Fallback if specific genre yields 0 results (unlikely but safe)
            if not mask.any():
                mask = [True] * len(assets["courses"])
                
            pop_recs = assets["courses"][mask].sort_values("enrollment", ascending=False).head(5)["course_id"].tolist()
            
            # 2. Communities (Clustering)
            # Find cluster of representative user and recommend
            U_prof, cluster_labels, u_ids = assets["cluster_data"]
            recs_dict = cluster_popular_recs(assets["ratings"], cluster_labels, [rep_user], top_k=5)
            cluster_recs = recs_dict.get(rep_user, [])
            
            # 3. Neural Network (AI Simulation)
            user_emb, item_emb, user_bias, item_bias, gmean = assets["neural_params"]
            neural_recs = recommend_user_neural(rep_user, assets["ratings"], assets["u2i"], assets["it2i"],
                                               user_emb, item_emb, user_bias, item_bias, gmean, top_k=5)
            
            # DISPLAY
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.info("🔥 Terpopuler (Statistik)")
                show_recommendations(pop_recs, "Market Best-Sellers")
            with col_b:
                st.success("👥 Pilihan Komunitas (Clustering)")
                if not cluster_recs:
                     st.warning("Data komunitas belum cukup.")
                else:
                     show_recommendations(cluster_recs, f"User Group ({genre_topic})")
            with col_c:
                st.warning("🧠 Pilihan AI (Neural Net)")
                if not neural_recs:
                     st.warning("AI butuh lebih banyak data.")
                else:
                     show_recommendations(neural_recs, "Deep Learning Model")
                st.caption(f"Disimulasikan untuk User #{rep_user}")

    # --------------------------------------------------------------------------
    # MODE 2: BY SIMILAR COURSE
    # --------------------------------------------------------------------------
    else:
        st.markdown("---")
        st.subheader("🔗 Cari Kemiripan")
        
        # Searchable selectbox
        course_titles = dict(zip(assets["courses"]["title"], assets["courses"]["course_id"]))
        selected_title = st.selectbox("Cari Judul Kursus yang Anda Suka:", options=list(course_titles.keys()))
        selected_cid = course_titles[selected_title]
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown("#### 🤖 Pilih Metode AI")
            model_mode = st.selectbox("Algoritma:", 
                                     ["Content-Based (Berdasarkan Deskripsi)", 
                                      "Item-KNN (Berdasarkan Pola User Lain)"])
        
        if st.button("Cari Kursus Serupa"):
            st.write(f"Mencari kemiripan dengan: **{selected_title}**")
            
            final_recs = []
            
            if "Content-Based" in model_mode:
                # TD-IDF Logic
                idx = assets["cid_list"].index(selected_cid)
                sims = cosine_similarity(assets["tfidf_matrix"][idx], assets["tfidf_matrix"]).ravel()
                # Sort
                ranked_indices = sims.argsort()[::-1][1:6] # Skip self (index 0)
                final_recs = [assets["cid_list"][i] for i in ranked_indices]
                
            else: # Item-KNN
                # Check if item in training set
                if selected_cid in assets["it2i"]:
                    it_idx = assets["it2i"][selected_cid]
                    dists, neighs = assets["knn_model"].kneighbors(assets["R_csr"].T[it_idx], n_neighbors=6)
                    # Skip self
                    neigh_indices = neighs.ravel()[1:] 
                    # Map back to course_id
                    idx_to_cid = {v:k for k,v in assets["it2i"].items()}
                    final_recs = [idx_to_cid[x] for x in neigh_indices]
                else:
                    st.error("Kursus ini belum cukup data ratingnya untuk KNN.")
            
            show_recommendations(final_recs, model_mode)

def page_analytics():
    st.markdown('<p class="main-header">📈 Analitik Performa Model</p>', unsafe_allow_html=True)
    st.write("Halaman ini membandingkan kecerdasan (akurasi) dari setiap model yang kita gunakan.")
    
    # Check if metrics file exists
    metrics_path = ROOT / "outputs" / "metrics_summary.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Grafik Perbandingan")
        
        # Calculate F1 Score
        df["f1_score"] = 2 * (df["precision@10"] * df["recall@10"]) / (df["precision@10"] + df["recall@10"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
             st.markdown("##### Precision@10 (Ketepatan)")
             st.bar_chart(df.set_index("model")["precision@10"], color="#4F8BF9")
        with col2:
             st.markdown("##### Recall@10 (Kelengkapan)")
             st.bar_chart(df.set_index("model")["recall@10"], color="#FF6B6B")
        with col3:
             st.markdown("##### F1-Score (Keseimbangan)")
             st.bar_chart(df.set_index("model")["f1_score"], color="#2ECC71")
             
        st.success("""
        **Penjelasan Metrik:**
        - **Precision**: Seberapa akurat tebakan model (sedikit salah tebak).
        - **Recall**: Seberapa banyak item relevan yang berhasil ditemukan (sedikit yang terlewat).
        - **F1-Score**: Nilai gabungan rata-rata. Model dengan F1 tertinggi adalah yang paling seimbang.
        """)
        
    else:
        st.warning("File metrics_summary.csv belum ditemukan. Silakan jalankan `python src/evaluate.py` terlebih dahulu.")
        if st.button("Jalankan Evaluasi (Mungkin butuh waktu)"):
            import subprocess
            subprocess.run(["python", "src/evaluate.py"])
            st.experimental_rerun()

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Navigasi")
    page = st.radio("Pergi ke:", ["🏠 Beranda", "📊 Analisis Data", "🎯 Sistem Rekomendasi", "📈 Performa Model"])
    
    st.markdown("---")
    st.caption("Capstone Project Recommender\nCreated by Ridlo Fanata W.")

if page == "🏠 Beranda":
    page_home()
elif page == "📊 Analisis Data":
    page_eda()
elif page == "🎯 Sistem Rekomendasi":
    page_recommender()
elif page == "📈 Performa Model":
    page_analytics()
