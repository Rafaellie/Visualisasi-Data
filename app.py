import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import io
import re

# Inisialisasi st.session_state untuk df_job_original
if 'df_job_original' not in st.session_state:
    st.session_state.df_job_original = pd.DataFrame()

# Judul Aplikasi
st.set_page_config(layout="wide")
st.title("Analisis Relevansi Jurusan PTN & Industri 4.0 di Indonesia")
st.write("Aplikasi ini menampilkan visualisasi dan model Machine Learning berdasarkan data PDDikti(jurusan), Tracer study(lulusan pada dunia kerja), dan Jobstreet (kompetensi kerja yang dibutuhkan di era Industri 4.0)")

# --- Fungsi Global untuk Kategorisasi ---
def get_category(name):
    if pd.isna(name):
        return 'Lainnya'
    name_lower = str(name).lower()
    if 'teknik' in name_lower or 'sipil' in name_lower or 'mesin' in name_lower or 'lingkungan' in name_lower or 'engineering' in name_lower:
        return 'Teknik'
    elif 'informatika' in name_lower or 'komputer' in name_lower or 'teknologi informasi' in name_lower or 'software' in name_lower or 'it ' in name_lower:
        return 'Komputer/Teknologi Informasi'
    elif 'akuntansi' in name_lower or 'keuangan' in name_lower or 'finance' in name_lower or 'accounting' in name_lower:
        return 'Akuntansi / Keuangan'
    elif 'penjualan' in name_lower or 'pemasaran' in name_lower or 'marketing' in name_lower or 'sales' in name_lower:
        return 'Penjualan / Pemasaran'
    elif 'pendidikan' in name_lower or 'teacher' in name_lower or 'education' in name_lower:
        return 'Pendidikan'
    elif 'agribisnis' in name_lower or 'perikanan' in name_lower or 'pertanian' in name_lower or 'agriculture' in name_lower:
        return 'Agribisnis'
    elif 'kedokteran' in name_lower or 'biomedik' in name_lower or 'medical' in name_lower or 'doctor' in name_lower:
        return 'Kedokteran'
    elif 'hukum' in name_lower or 'legal' in name_lower or 'law' in name_lower:
        return 'Hukum'
    elif 'seni' in name_lower or 'desain' in name_lower or 'media' in name_lower or 'komunikasi' in name_lower or 'art' in name_lower or 'design' in name_lower:
        return 'Seni/Media/Komunikasi'
    elif 'administrasi' in name_lower or 'administration' in name_lower or 'admin' in name_lower:
        return 'Administrasi'
    elif 'manajemen' in name_lower or 'management' in name_lower:
        return 'Manajemen'
    elif 'human resources' in name_lower or 'hrd' in name_lower or 'personalia' in name_lower:
        return 'Sumber Daya Manusia'
    elif 'bisnis' in name_lower or 'business' in name_lower:
        return 'Bisnis'
    elif 'data' in name_lower or 'analis' in name_lower or 'analyst' in name_lower or 'ilmuwan' in name_lower:
        return 'Data Science/Analisis'
    else:
        return 'Lainnya'

def map_job_category(job_title):
    title = str(job_title).lower()
    if 'developer' in title or 'programmer' in title:
        return 'IT – Software Development'
    elif 'engineer' in title:
        return 'Engineering / Maintenance'
    elif 'finance' in title or 'accounting' in title:
        return 'Finance / Accounting'
    elif 'marketing' in title:
        return 'Marketing / Digital Marketing'
    elif 'agri' in title:
        return 'Agriculture / Agroindustry'
    else:
        return 'Other'

def categorize_industry(title: str) -> str:
    title_lower = str(title).lower()
    industry_map = {
        'Artificial Intelligence / Machine Learning': [
            'machine learning', 'artificial intelligence', 'ai', 'deep learning', 'computer vision'
        ],
        'Data & Big Data': [
            'data scientist', 'data engineer', 'big data', 'data analyst'
        ],
        'Internet of Things (IoT)': [
            'iot', 'internet of things', 'embedded'
        ],
        'Cloud Computing': [
            'cloud', 'aws', 'azure', 'gcp', 'google cloud', 'cloud architect'
        ],
        'Cybersecurity': [
            'security', 'cybersecurity', 'information security', 'infosec'
        ],
        'Robotics & Automation': [
            'robot', 'automation', 'robotics', 'mechatronics'
        ],
        'Augmented/Virtual Reality': [
            'augmented reality', 'virtual reality', 'vr', 'ar'
        ],
        'Blockchain': [
            'blockchain', 'crypto', 'ethereum', 'smart contract'
        ],
        'Digital Manufacturing / Industry 4.0': [
            'additive manufacturing', '3d printing', 'smart factory', 'industry 4.0'
        ],
        'Software Development': [
            'developer', 'programmer', 'full stack', 'backend', 'frontend', 'software'
        ]
    }
    for category, keywords in industry_map.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
                return category
    return 'Other / Non-4.0'

# --- Fungsi Pra-pemrosesan Data untuk Visualisasi ---
def process_ratio_column(ratio_str):
    if pd.isna(ratio_str) or ratio_str == '' or ratio_str == 'N/A':
        return np.nan
    if isinstance(ratio_str, str):
        if ':' in ratio_str:
            parts = ratio_str.split(':')
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return np.nan
        try:
            return float(ratio_str)
        except ValueError:
            return np.nan
    return float(ratio_str) if not pd.isna(ratio_str) else np.nan

def calculate_average_waiting_time(row):
    mtk_selaras = row['mtk_selaras_jumlah'] if not pd.isna(row['mtk_selaras_jumlah']) else 0
    mtk_tidak_selaras = row['mtk_tidak_selaras_jumlah'] if not pd.isna(row['mtk_tidak_selaras_jumlah']) else 0
    total = mtk_selaras + mtk_tidak_selaras

    if total == 0:
        return np.nan

    avg_waiting_time = (mtk_selaras * 3 + mtk_tidak_selaras * 9) / total
    return avg_waiting_time

def prepare_data(df_input): 
    df_clean = df_input.copy()

    expected_numeric_cols = [
        'Jumlah dosen', 'Jumlah Pendidik Tetap', 'Jumlah pendidik tidak tetap',
        'Total pendidik', 'Jumlah mahasiswa ',
        'hor_selaras_jumlah', 'hor_tidak_selaras_jumlah',
        'mtk_selaras_jumlah', 'mtk_tidak_selaras_jumlah',
        'vert_tinggi_jumlah', 'vert_sama_jumlah', 'vert_rendah_jumlah',
        'hor_selaras_pct', 'hor_tidak_selaras_pct',
        'Rasio dosen/mahasiswa',
        'job_count'
    ]

    for col in expected_numeric_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0
            st.warning(f"Kolom '{col}' tidak ditemukan di data utama (merged_data_cleaned.csv). Diinisialisasi dengan 0.")
        else:
            if col not in ['Rasio dosen/mahasiswa']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    df_clean['Rasio dosen/mahasiswa'] = df_clean['Rasio dosen/mahasiswa'].astype(str)
    df_clean['ratio_numeric'] = df_clean['Rasio dosen/mahasiswa'].apply(process_ratio_column)
    df_clean['ratio_numeric'] = pd.to_numeric(df_clean['ratio_numeric'], errors='coerce').fillna(0)

    df_clean['avg_waiting_time'] = df_clean.apply(calculate_average_waiting_time, axis=1)
    df_clean['avg_waiting_time'] = pd.to_numeric(df_clean['avg_waiting_time'], errors='coerce').fillna(df_clean['avg_waiting_time'].median() if not df_clean['avg_waiting_time'].empty and df_clean['avg_waiting_time'].median() == df_clean['avg_waiting_time'].median() else 0)

    for col in ['hor_selaras_pct', 'hor_tidak_selaras_pct']:
        if col in df_clean.columns and df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].replace('NaN', '0%').str.rstrip('%').astype(float) / 100
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    df_clean['hor_selaras_pct_clean'] = pd.to_numeric(df_clean['hor_selaras_pct'], errors='coerce').fillna(0)
    df_clean['hor_tidak_selaras_pct_clean'] = pd.to_numeric(df_clean['hor_tidak_selaras_pct'], errors='coerce').fillna(0)

    subset_for_dropna = ['ratio_numeric', 'avg_waiting_time', 'hor_selaras_pct_clean']
    actual_subset_for_dropna = [col for col in subset_for_dropna if col in df_clean.columns and not df_clean[col].isnull().all()]
    if actual_subset_for_dropna and not df_clean[actual_subset_for_dropna].empty:
        df_clean = df_clean.dropna(subset=actual_subset_for_dropna).reset_index(drop=True)
    elif df_clean.empty:
        return pd.DataFrame()

    if 'ratio_numeric' in df_clean.columns and not df_clean['ratio_numeric'].isnull().all() and not df_clean['ratio_numeric'].empty:
        min_ratio = df_clean['ratio_numeric'].min()
        max_ratio = df_clean['ratio_numeric'].max()
        
        if max_ratio > 30 and max_ratio > 20 and max_ratio > min_ratio:
            bins = [min_ratio - 0.1, 20, 30, max_ratio + 0.1]
            labels = ['Rendah (≤20)', 'Sedang (20-30)', 'Tinggi (>30)']
        elif max_ratio > 20 and max_ratio > min_ratio:
            bins = [min_ratio - 0.1, 20, max_ratio + 0.1]
            labels = ['Rendah (≤20)', 'Sedang (>20)']
        elif max_ratio > min_ratio:
            bins = [min_ratio - 0.1, max_ratio + 0.1]
            labels = ['Rendah']
        else:
            bins = [min_ratio - 0.1, min_ratio + 0.1]
            labels = ['Rendah']

        if len(bins) - 1 != len(labels):
            st.warning(f"Jumlah bin ({len(bins)}) tidak sesuai dengan jumlah label ({len(labels)}). Kategorisasi rasio mungkin tidak akurat.")
            df_clean['ratio_category'] = 'Tidak Diketahui'
        else:
            df_clean['ratio_category'] = pd.cut(df_clean['ratio_numeric'],
                                                bins=bins,
                                                labels=labels,
                                                right=True)
            df_clean['ratio_category'] = df_clean['ratio_category'].astype(object).fillna('Tidak Diketahui')
    else:
        df_clean['ratio_category'] = 'Tidak Diketahui'

    if 'hor_selaras_pct_clean' in df_clean.columns and not df_clean['hor_selaras_pct_clean'].isnull().all() and not df_clean['hor_selaras_pct_clean'].empty:
        median_hor_selaras_pct = df_clean['hor_selaras_pct_clean'].median()
        df_clean['waktu_tunggu_label'] = df_clean['hor_selaras_pct_clean'].apply(lambda x: 'Cepat' if x >= median_hor_selaras_pct else 'Lama')
    else:
        df_clean['waktu_tunggu_label'] = 'Lama'

    return df_clean

def prepare_data_for_stacked_bar(df_input):
    """
    Menyiapkan data untuk stacked bar chart keselarasan horizontal per akreditasi.
    """
    df_clean = df_input.copy()

    # Bersihkan kolom akreditasi
    df_clean['Akreditasi'] = df_clean['Akreditasi'].fillna('Tidak Diketahui')

    # Konversi kolom persentase menjadi numerik
    for col in ['hor_selaras_pct', 'hor_tidak_selaras_pct']:
        if col not in df_clean.columns:
            st.warning(f"Kolom '{col}' tidak ditemukan untuk stacked bar chart. Diinisialisasi dengan 0.")
            df_clean[col] = 0
        elif df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].replace('NaN', '0%').str.rstrip('%').astype(float) / 100
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    # Filter data yang valid (tidak NaN)
    subset_for_stacked_bar = ['hor_selaras_pct', 'hor_tidak_selaras_pct']
    existing_subset_for_stacked_bar = [col for col in subset_for_stacked_bar if col in df_clean.columns and not df_clean[col].isnull().all()]
    if existing_subset_for_stacked_bar and not df_clean[existing_subset_for_stacked_bar].empty:
        df_clean = df_clean.dropna(subset=existing_subset_for_stacked_bar)
    else:
        st.warning("Tidak ada kolom yang cukup untuk filter data untuk stacked bar chart. Data mungkin mengandung NaN atau kosong.")
        return pd.DataFrame(), pd.DataFrame() 


    # Pastikan persentase selaras + tidak selaras = 100% (normalisasi jika perlu)
    total_pct = df_clean['hor_selaras_pct'] + df_clean['hor_tidak_selaras_pct']
    # Filter out zero totals to avoid division by zero
    mask_valid_total = (total_pct > 0) & (total_pct <= 1.05) 
    if mask_valid_total.sum() == 0:
        st.warning("Semua total persentase horizontal adalah nol atau tidak valid. Tidak dapat normalisasi.")
        return pd.DataFrame(), pd.DataFrame() 

    df_clean = df_clean[mask_valid_total].copy() 

    df_clean['hor_selaras_pct_norm'] = (df_clean['hor_selaras_pct'] / total_pct[mask_valid_total]) * 100
    df_clean['hor_tidak_selaras_pct_norm'] = (df_clean['hor_tidak_selaras_pct'] / total_pct[mask_valid_total]) * 100
    if df_clean['hor_selaras_pct_norm'].isnull().all() and df_clean['hor_tidak_selaras_pct_norm'].isnull().all():
        st.warning("Normalisasi menghasilkan semua NaN. Data mungkin tidak cocok untuk stacked bar chart.")
        return pd.DataFrame(), pd.DataFrame()


    # Kelompokkan berdasarkan akreditasi dan hitung rata-rata
    if 'Akreditasi' not in df_clean.columns or df_clean['Akreditasi'].isnull().all() or df_clean['Akreditasi'].empty:
        st.warning("Kolom 'Akreditasi' tidak ditemukan atau kosong untuk pengelompokan. Tidak dapat membuat grouped data.")
        return pd.DataFrame(), pd.DataFrame() 

    grouped_data = df_clean.groupby('Akreditasi').agg({
        'hor_selaras_pct_norm': 'mean',
        'hor_tidak_selaras_pct_norm': 'mean',
        'hor_selaras_pct': 'mean',  # Data asli untuk perbandingan
        'hor_tidak_selaras_pct': 'mean',
        'Prodi': 'count'   # Jumlah program studi per akreditasi
    }).round(2)

    # Rename kolom untuk kemudahan
    grouped_data.columns = ['Selaras_Norm', 'Tidak_Selaras_Norm', 'Selaras_Asli', 'Tidak_Selaras_Asli', 'Jumlah_Prodi']

    # Urutkan berdasarkan tingkat akreditasi (jika menggunakan standar Indonesia)
    akreditasi_order = ['Unggul', 'A', 'Baik Sekali', 'B', 'Baik', 'C', 'Tidak Diketahui']

    # Filter dan urutkan berdasarkan akreditasi yang ada
    available_akreditasi = [akred for akred in akreditasi_order if akred in grouped_data.index]
    other_akreditasi = [akred for akred in grouped_data.index if akred not in akreditasi_order]
    final_order = available_akreditasi + sorted(other_akreditasi)

    grouped_data = grouped_data.reindex(final_order)

    return grouped_data, df_clean

# --- Load Data ---
@st.cache_data(show_spinner=True) # Tambahkan show_spinner untuk feedback visual
def load_data():
    """
    Memuat data yang sudah diproses sebelumnya (merged_data_cleaned.csv)
    dan data jobs_crawling_cleaned.csv secara terpisah.
    """
    # Memuat data gabungan utama
    try:
        df_final_merged = pd.read_csv('data/merged_data_cleaned.csv')
    except FileNotFoundError:
        st.error("File 'merged_data_cleaned.csv' tidak ditemukan. Pastikan sudah ada di folder 'data/'.")
        st.stop()
    if df_final_merged.empty:
        st.error("File 'merged_data_cleaned.csv' kosong. Pastikan berisi data yang valid.")
        st.stop()

    # Memuat dan memproses data jobs_crawling_cleaned.csv secara terpisah untuk df_job_original
    try:
        df_job = pd.read_csv('data/jobs_crawling_cleaned.csv') 
    except FileNotFoundError:
        st.error("File 'jobs_crawling_cleaned.csv' tidak ditemukan. Pastikan sudah ada di folder 'data/'.")
        st.stop()
    if df_job.empty:
        st.error("File 'jobs_crawling_cleaned.csv' kosong. Pastikan berisi data yang valid.")
        st.stop()

    # --- PERBAIKAN: Memastikan kolom-kolom kritis df_job_original selalu ada dan valid ---
    # List kolom-kolom yang diharapkan dari jobs_crawling_cleaned.csv untuk visualisasi
    critical_job_cols = ['jobTitle', 'categoriesName', 'locations']
    for col in critical_job_cols:
        if col not in df_job.columns:
            df_job[col] = "" # Tambahkan kolom kosong jika tidak ada
            st.warning(f"Kolom '{col}' tidak ditemukan di jobs_crawling_cleaned.csv. Visualisasi yang menggunakannya mungkin tidak akurat.")
        # Ensure they are string type for text processing
        df_job[col] = df_job[col].astype(str)

    # Menentukan kolom Job Title yang sebenarnya di df_job untuk map_job_category
    JOB_TITLE_COL_FOR_MAP = 'jobTitle' # Default ke jobTitle
    if 'Job Title' in df_job.columns and not df_job['Job Title'].empty:
        JOB_TITLE_COL_FOR_MAP = 'Job Title'
    elif 'Pekerjaan' in df_job.columns and not df_job['Pekerjaan'].empty: 
        JOB_TITLE_COL_FOR_MAP = 'Pekerjaan'
    elif 'jobTitle' in df_job.columns and not df_job['jobTitle'].empty:
        JOB_TITLE_COL_FOR_MAP = 'jobTitle'
    else: # Fallback jika semua kolom judul pekerjaan kosong atau tidak ada
        JOB_TITLE_COL_FOR_MAP = 'temp_job_title_for_mapping'
        df_job['temp_job_title_for_mapping'] = ""
        st.warning("Tidak ada kolom judul pekerjaan yang valid (Job Title/Pekerjaan/jobTitle) di jobs_crawling_cleaned.csv. Mapping kategori spesifik mungkin tidak akurat.")

    # Apply category_from_job_data (from categoriesName)
    if 'categoriesName' in df_job.columns and not df_job['categoriesName'].empty:
        df_job['category_from_job_data'] = df_job['categoriesName'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notnull(x) and ',' in str(x) else str(x).strip()
        )
    else:
        df_job['category_from_job_data'] = 'Lainnya' 
        st.warning("Kolom 'categoriesName' tidak valid atau kosong di jobs_crawling_cleaned.csv. Kategori untuk merge/visualisasi mungkin tidak akurat.")

    # categorize_industry diterapkan pada kolom yang berisi judul pekerjaan asli
    df_job['industri4_0'] = df_job[JOB_TITLE_COL_FOR_MAP].apply(categorize_industry)

    # df_job['job_mapped_category']
    df_job['job_mapped_category'] = df_job[JOB_TITLE_COL_FOR_MAP].apply(map_job_category)
    
    # Simpan df_job yang sudah diproses ke session state
    st.session_state.df_job_original = df_job 

    return df_final_merged

# --- Load Data ---
df_raw = load_data()
if df_raw.empty:
    st.error("Data utama (merged_data_cleaned.csv) tidak dapat dimuat atau kosong setelah pemrosesan awal. Pastikan file ada dan berisi data valid.")
    st.stop() 

df_clean = prepare_data(df_raw) 
if df_clean.empty:
    st.error("Data bersih (df_clean) kosong setelah persiapan. Pastikan data input valid dan tidak menyebabkan semua baris dihapus.")
    st.stop()

df_job_original = st.session_state.df_job_original

# --- Sidebar Navigation ---
st.sidebar.title("Navigasi")
selected_tab = st.sidebar.radio(
    "Pilih Bagian Analisis:",
    [
        "Analisis Rasio Dosen/Mahasiswa & Masa Tunggu",
        "Analisis Keselarasan Horizontal per Akreditasi",
        "Analisis Data Lowongan Pekerjaan",
        "Heatmap Korelasi Metrik Utama",
        "Model Machine Learning"
    ]
)

# --- Bagian Konten berdasarkan Pilihan Sidebar ---
if selected_tab == "Analisis Rasio Dosen/Mahasiswa & Masa Tunggu":
    st.header("Visualisasi Data")
    st.subheader("Analisis Rasio Dosen/Mahasiswa vs Masa Tunggu Kerja Alumni")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analisis Rasio Dosen/Mahasiswa vs Masa Tunggu Kerja Alumni', fontsize=16, fontweight='bold', y=0.98)

    scatter_size_factor = 25 
    scatter = ax1.scatter(df_clean['ratio_numeric'],
                            df_clean['avg_waiting_time'],
                            s=df_clean['hor_selaras_pct_clean'] * scatter_size_factor,
                            c=df_clean['hor_selaras_pct_clean'],
                            alpha=0.6,
                            cmap='RdYlGn',
                            edgecolors='black',
                            linewidth=0.5)

    ax1.set_xlabel('Rasio Dosen/Mahasiswa (Mahasiswa per Dosen)')
    ax1.set_ylabel('Rata-rata Masa Tunggu Kerja (Bulan)')
    ax1.set_title('Scatter Plot dengan Ukuran = Persentase Keselarasan Horizontal')
    ax1.grid(True, alpha=0.3)
    cbar1 = fig.colorbar(scatter, ax=ax1) 
    cbar1.set_label('Persentase Keselarasan Horizontal (%)')
    if len(df_clean) > 1 and not df_clean['ratio_numeric'].isnull().all() and not df_clean['avg_waiting_time'].isnull().all():
        try:
            z = np.polyfit(df_clean['ratio_numeric'], df_clean['avg_waiting_time'], 1)
            p = np.poly1d(z)
            ax1.plot(df_clean['ratio_numeric'], p(df_clean['ratio_numeric']),
                     "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
            ax1.legend()
        except np.linalg.LinAlgError:
            st.warning("Tidak dapat menghitung trend line (singular matrix). Mungkin data terlalu sedikit atau tidak ada variasi.")


    if 'Akreditasi' in df_clean.columns and not df_clean['Akreditasi'].isnull().all():
        akreditasi_colors = {'A': 'green', 'B': 'orange', 'C': 'red', 'Unggul': 'purple', 'Baik Sekali': 'blue', 'Sangat Baik': 'cyan', 'Tidak Diketahui': 'gray'}
        for akred in df_clean['Akreditasi'].unique():
            if pd.notna(akred):
                subset = df_clean[df_clean['Akreditasi'] == akred]
                color = akreditasi_colors.get(akred, 'gray')
                ax2.scatter(subset['ratio_numeric'], subset['avg_waiting_time'],
                            alpha=0.7, label=f'Akreditasi {akred}', color=color, s=60)
    ax2.set_xlabel('Rasio Dosen/Mahasiswa (Mahasiswa per Dosen)')
    ax2.set_ylabel('Rata-rata Masa Tunggu Kerja (Bulan)')
    ax2.set_title('Scatter Plot Berdasarkan Akreditasi Program Studi')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.hist(df_clean['ratio_numeric'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Rasio Dosen/Mahasiswa (Mahasiswa per Dosen)')
    ax3.set_ylabel('Frekuensi')
    ax3.set_title('Distribusi Rasio Dosen/Mahasiswa')
    ax3.grid(True, alpha=0.3)
    mean_ratio = df_clean['ratio_numeric'].mean()
    median_ratio = df_clean['ratio_numeric'].median()
    # Format string untuk mean dan median 
    ax3.axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.1f}')
    ax3.axvline(median_ratio, color='orange', linestyle='--', label=f'Median: {median_ratio:.1f}')
    ax3.legend()

    # Plot 4: Box plot masa tunggu berdasarkan kategori rasio
    if 'ratio_category' in df_clean.columns and not df_clean['ratio_category'].isna().all() and not df_clean['ratio_category'].empty:
        # Filter order untuk hanya kategori yang ada di data
        existing_categories_order = [cat for cat in ['Rendah (≤20)', 'Sedang (20-30)', 'Tinggi (>30)'] if cat in df_clean['ratio_category'].unique()]
        if existing_categories_order:
            sns.boxplot(data=df_clean, x='ratio_category', y='avg_waiting_time', ax=ax4, order=existing_categories_order)
    ax4.set_xlabel('Kategori Rasio Dosen/Mahasiswa')
    ax4.set_ylabel('Rata-rata Masa Tunggu Kerja (Bulan)')
    ax4.set_title('Distribusi Masa Tunggu per Kategori Rasio')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    Visualisasi ini menyajikan hubungan antara rasio dosen/mahasiswa dengan masa tunggu kerja alumni dari berbagai perspektif:
    ### 1. Subplot Kiri Atas (Top-Left)
    Visualisasi: Scatter plot dari rasio dosen/mahasiswa vs masa tunggu kerja, dengan warna dan ukuran titik merepresentasikan persentase keselarasan horizontal.

    **Analisis Mendalam:**
    - Titik-titik menunjukkan hubungan antara rasio dosen/mahasiswa dengan masa tunggu kerja alumni.
    - Ukuran dan warna titik menunjukkan keselarasan horizontal (semakin besar dan hijau = semakin selaras).
    - Terlihat *overlap* signifikan antara titik berwarna merah (keselarasan rendah) dan hijau (tinggi), serta tidak ada pola linear yang kuat.

    ### 2. Subplot Kanan Atas (Top-Right)
    Visualisasi: Scatter plot yang sama, namun diberi pewarnaan berdasarkan Akreditasi Program Studi.

    **Analisis Mendalam:**
    - Warna titik berdasarkan kategori Akreditasi (A, B, C, Unggul, dst).
    - Tampak bahwa program studi dengan Akreditasi A dan Unggul cenderung berada di area dengan masa tunggu kerja lebih rendah, meskipun belum absolut.
    - Program dengan Akreditasi C atau B tersebar lebih luas, bahkan ada yang di masa tunggu tinggi.

    ### 3. Subplot Kiri Bawah (Bottom-Left)
    Visualisasi: Histogram distribusi rasio dosen/mahasiswa.

    **Analisis Mendalam:**
    - Menunjukkan sebaran jumlah mahasiswa per dosen di seluruh program studi.
    - Tampak bahwa mayoritas program studi memiliki rasio di sekitar nilai tertentu.

    **Ditandai garis vertikal:**
    - Merah: Mean
    - Oranye: Median
 
    **Interpretasi:**
    - Jika *mean* > *median* → distribusi sedikit condong ke kanan (*right-skewed*), menandakan ada beberapa program dengan rasio sangat tinggi.
    - Bisa dipakai untuk *benchmarking* → apakah program sudah memenuhi rasio ideal sesuai standar (misal $\leq25$ mahasiswa per dosen).
 
    ### 4. Subplot Kanan Bawah (Bottom-Right)
    Visualisasi: Boxplot masa tunggu kerja berdasarkan kategori rasio dosen/mahasiswa (rendah, sedang, tinggi).

    **Analisis Mendalam:**
    - Rasio dikategorikan ke dalam 3 kelas:
        - Rendah ($\leq20$)
        - Sedang (20–30)
        - Tinggi ($>30$)

    **Boxplot menunjukkan:**
    - Semakin tinggi rasio, median masa tunggu cenderung meningkat.
    - Variasi (IQR) terbesar ada pada kelompok rasio tinggi → lebih tidak stabil.
    - Terdapat *outlier* di semua kelompok, tetapi jumlahnya lebih banyak di kelompok “Tinggi”.

    **Interpretasi:**
    - Rasio dosen/mahasiswa tinggi bisa berdampak negatif terhadap waktu alumni mendapatkan pekerjaan.
    - Beban pengajaran yang terlalu tinggi mungkin berdampak pada kualitas pembelajaran atau pembimbingan karir.
    """)

elif selected_tab == "Analisis Keselarasan Horizontal per Akreditasi":
    st.header("Visualisasi Data")
    st.subheader("Analisis Keselarasan Horizontal Lulusan per Akreditasi Program Studi")

    grouped_data, _ = prepare_data_for_stacked_bar(df_raw)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analisis Keselarasan Horizontal Lulusan per Akreditasi Program Studi', fontsize=16, fontweight='bold', y=0.98)

    categories = grouped_data.index
    selaras_values = grouped_data['Selaras_Norm']
    tidak_selaras_values = grouped_data['Tidak_Selaras_Norm']
    colors = ['#2E8B57', '#FF6B6B']

    # Subplot Kiri Atas (Top-Left): Stacked bar chart
    bars1 = ax1.bar(categories, selaras_values, color=colors[0], alpha=0.8, label='Selaras Horizontal', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(categories, tidak_selaras_values, bottom=selaras_values, color=colors[1], alpha=0.8, label='Tidak Selaras Horizontal', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Kategori Akreditasi Program Studi')
    ax1.set_ylabel('Persentase Lulusan (%)')
    ax1.set_title('Persentase Keselarasan Horizontal per Akreditasi')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height1 > 5:
            ax1.text(bar1.get_x() + bar1.get_width()/2, height1/2, f'{height1:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        if height2 > 5:
            ax1.text(bar2.get_x() + bar2.get_width()/2, height1 + height2/2, f'{height2:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)

    # Subplot Kanan Atas (Top-Right): Side-by-side comparison chart
    x_pos = np.arange(len(categories))
    width = 0.35
    ax2.bar(x_pos - width/2, selaras_values, width, label='Selaras Horizontal', color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.bar(x_pos + width/2, tidak_selaras_values, width, label='Tidak Selaras Horizontal', color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Kategori Akreditasi Program Studi')
    ax2.set_ylabel('Persentase Lulusan (%)')
    ax2.set_title('Perbandingan Keselarasan Horizontal (Side-by-Side)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot Kiri Bawah (Bottom-Left): Pie chart
    jumlah_prodi = grouped_data['Jumlah_Prodi']
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax3.pie(jumlah_prodi, labels=categories, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax3.set_title('Distribusi Jumlah Program Studi per Akreditasi')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Subplot Kanan Bawah (Bottom-Right): Horizontal bar chart
    y_pos = np.arange(len(categories))
    ax4.barh(y_pos, selaras_values, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(categories)
    ax4.set_xlabel('Persentase Keselarasan Horizontal (%)')
    ax4.set_title('Ranking Keselarasan Horizontal per Akreditasi')
    ax4.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(selaras_values):
        ax4.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    Visualisasi ini menganalisis keselarasan horizontal lulusan per akreditasi program studi:

    ### 1. Subplot Kiri Atas (Top-Left)
    Visualisasi: Stacked bar chart menunjukkan persentase keselarasan horizontal per kategori akreditasi dengan pembagian "Selaras Horizontal" (hijau) dan "Tidak Selaras Horizontal" (merah muda).
    **Analisis Mendalam:**
    - Akreditasi "Baik" menunjukkan proporsi hijau tertinggi (82.4%), mengindikasikan efektivitas terbaik dalam keselarasan horizontal.
    - Paradoks terlihat pada akreditasi "Unggul" yang hanya mencapai 77.8%, lebih rendah dari "Baik", "Baik Sekali", dan "B".
    - Rentang keselarasan yang relatif sempit (78.6%-82.4%) menunjukkan sistem akreditasi memiliki *baseline* yang konsisten.
    - Pola tidak linear antara "level" akreditasi dengan keselarasan menunjukkan perlu evaluasi ulang kriteria akreditasi.

    ### 2. Subplot Kanan Atas (Top-Right)
    Visualisasi: *Side-by-side comparison chart* yang memisahkan secara visual proporsi lulusan selaras vs tidak selaras per kategori akreditasi.
    **Analisis Mendalam:**
    - Dominasi warna hijau di semua kategori ($>75%$) menunjukkan sistem pendidikan tinggi yang relatif efektif.
    - Konsistensi visual yang tinggi mengindikasikan tidak ada kategori akreditasi yang "gagal total".
    - *Gap* terkecil antara selaras-tidak selaras terlihat pada akreditasi "Baik" (sekitar 65% berbanding 17%).
    - Visualisasi ini memperkuat temuan bahwa perbedaan performa antar akreditasi tidak drastis.

    ### 3. Subplot Kiri Bawah (Bottom-Left)
    Visualisasi: *Pie chart* menunjukkan distribusi jumlah program studi berdasarkan kategori akreditasi dengan warna berbeda untuk setiap kategori.
    **Analisis Mendalam:**
    - Dominasi masif akreditasi "Unggul" (61.7%) menciptakan ketimpangan distribusi yang signifikan.
    - Akreditasi "A" (10.7%) dan "B" (5.6%) menunjukkan distribusi yang tidak proporsional dengan kualitas keselarasannya.
    - "Terakreditasi Sementara" dengan proporsi terkecil sekitar (5.6%) mengindikasikan sistem *filtering* yang baik.
    - Distribusi ini menunjukkan kemungkinan "*grade inflation*" pada akreditasi "Unggul".

    ### 4. Subplot Kanan Bawah (Bottom-Right)
    Visualisasi: *Horizontal bar chart* ranking keselarasan horizontal dari tertinggi ke terendah per kategori akreditasi.
    **Analisis Mendalam:**
    - *Ranking* terbalik antara "nama" akreditasi dengan performa aktual (Baik > Unggul).
    - *Clustering* performa di kisaran 76-82% menunjukkan sistem yang *mature*.
    - *Visual gap* yang jelas antara *performer* terbaik (Baik: 81.7%) dan terburuk (B: 76.4%).
    """)

elif selected_tab == "Analisis Data Lowongan Pekerjaan":
    st.header("Visualisasi Data")
    st.subheader("Analisis Data Lowongan Pekerjaan")

    # --- Visualisasi: Distribusi Top 10 Aliran Lulusan ke Kategori Pekerjaan ---
    st.markdown("### Distribusi Top 10 Aliran Lulusan ke Kategori Pekerjaan")

    # Mapping kategori sederhana agar hanya 3 kategori utama
    def simple_job_category(cat):
        cat = str(cat).lower()
        if 'akuntansi' in cat or 'keuangan' in cat or 'finance' in cat or 'accounting' in cat:
            return 'Akuntansi / Keuangan'
        elif 'informatika' in cat or 'komputer' in cat or 'teknologi informasi' in cat or 'it' in cat:
            return 'Komputer/Teknologi Informasi'
        elif 'teknik' in cat:
            return 'Teknik'
        else:
            return 'Lainnya'

    if 'program_studi' in df_raw.columns and 'category' in df_raw.columns and 'job_count' in df_raw.columns:
        df_flow_data = df_raw[['program_studi', 'category', 'job_count']].copy()
        df_flow_data['job_flow_category'] = df_flow_data['category'].apply(simple_job_category)

        grouped_flows = df_flow_data.groupby(['program_studi', 'job_flow_category'])['job_count'].sum().reset_index()
        top_flows = grouped_flows.sort_values(by='job_count', ascending=False).head(10)

        if not top_flows.empty:
            fig_grad_job_flow, ax_grad_job_flow = plt.subplots(figsize=(12, 7))

            # Palette dan urutan legend sesuai gambar
            custom_flow_palette = {
                'Akuntansi / Keuangan': '#1f77b4', 
                'Komputer/Teknologi Informasi': '#ff7f0e', 
                'Teknik': '#2ca02c', 
                'Lainnya': 'gray'
            }
            order_kategori = ['Akuntansi / Keuangan', 'Komputer/Teknologi Informasi', 'Teknik']

            sns.barplot(
                data=top_flows,
                x='job_count',
                y='program_studi',
                hue='job_flow_category',
                palette=custom_flow_palette,
                ax=ax_grad_job_flow,
                dodge=False,
                hue_order=order_kategori
            )
            ax_grad_job_flow.set_title("Distribusi Top 10 Aliran Lulusan ke Kategori Pekerjaan", fontsize=14)
            ax_grad_job_flow.set_xlabel("Jumlah Lulusan")
            ax_grad_job_flow.set_ylabel("Program Studi")
            ax_grad_job_flow.legend(title="Kategori Pekerjaan", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_grad_job_flow)
            st.markdown("""
            Visualisasi: *Horizontal bar chart* menunjukkan top 10 program studi berdasarkan jumlah lulusan yang terserap ke pasar kerja, dengan *color coding* berdasarkan kategori pekerjaan (Akuntansi/Keuangan: biru, Komputer/Teknologi Informasi: oranye, Teknik: hijau).
                                    
            **Insight : **
            1. **Akuntansi / Keuangan Dominan**: Program studi Akuntansi/Keuangan memiliki jumlah lulusan yang paling banyak terserap, menunjukkan permintaan tinggi di sektor ini.
            2. **Komputer/Teknologi Informasi Stabil**: Program studi Komputer/Teknologi Informasi juga menunjukkan jumlah lulusan yang signifikan, mencerminkan kebutuhan berkelanjutan di sektor IT.  
            3. **Teknik Tumbuh**: Program studi Teknik menunjukkan pertumbuhan yang stabil, menandakan sektor teknik masih diminati.
            4. **Lainnya**: Kategori 'Lainnya' mencakup program studi yang tidak masuk dalam tiga kategori utama, namun tetap menunjukkan kontribusi penting terhadap aliran lulusan ke pasar kerja.
            """)
        else:
            st.info("Tidak ada data aliran lulusan yang cukup untuk ditampilkan.")
    else:
        st.info("Kolom yang diperlukan untuk visualisasi aliran lulusan (program_studi, category, job_count) tidak ditemukan di data gabungan. Pastikan data Anda sudah lengkap.")

    # --- Visualisasi: Top 10 Kategori Lowongan Pekerjaan (dari jobs_crawling_cleaned.csv) ---
    st.markdown("### Top 10 Kategori Lowongan Pekerjaan")
    # Menggunakan 'category_from_job_data' yang sudah diolah dari 'categoriesName'
    if 'category_from_job_data' in df_job_original.columns and not df_job_original.empty:
        top10_job_categories = df_job_original['category_from_job_data'].value_counts().nlargest(10)

        if not top10_job_categories.empty: 
            fig_top_jobs, ax_top_jobs = plt.subplots(figsize=(10, 6))
            top10_job_categories.plot(kind='barh', ax=ax_top_jobs, color=sns.color_palette("crest", 10))
            ax_top_jobs.set_xlabel('Jumlah Lowongan')
            ax_top_jobs.set_ylabel('Kategori Pekerjaan')
            ax_top_jobs.set_title('Top 10 Kategori Lowongan Kerja pada Jobstreet Indonesia')
            ax_top_jobs.invert_yaxis() 
            plt.tight_layout()
            st.pyplot(fig_top_jobs)
            st.markdown("""
            Bar chart ini menampilkan 10 kategori pekerjaan dengan jumlah lowongan terbanyak yang ditemukan dari data crawling. Ini memberikan gambaran tentang sektor-sektor pekerjaan yang paling aktif.

            **Insight:**
            1.  **Penjualan / Pemasaran Mendominasi Jauh**
                Kategori “Penjualan / Pemasaran” paling banyak, menembus ~320.000 lowongan. Ini menunjukkan bahwa hampir sepertiga lowongan kerja yang ter-crawl berada di domain sales & marketing—mulai dari eksekutif penjualan, account manager, hingga digital marketing.

            2.  **Teknologi Informasi di Posisi Runner-Up**
                “Komputer/Teknologi Informasi” (~180.000 lowongan) berada di urutan kedua. Permintaan tinggi mencakup berbagai peran TI: software development, network & database admin, hingga cyber security. Hal ini menegaskan sektor IT masih sangat dibutuhkan di era digital.

            3.  **Akuntansi / Keuangan & SDM Kuat**
                * Akuntansi / Keuangan (~140.000 lowongan) menempati posisi ketiga. Fintech, digital banking, dan audit otomatis jadi pendorong utama.
                * Sumber Daya Manusia/Personalia (~90.000 lowongan) di urutan keempat, menandakan perusahaan intensif melakukan hiring, manajemen talenta, dan employer branding.

            4.  Pelayanan, Hospitality & Sektor Lainnya
                * Hotel/Restoran dan Pelayanan masing-masing ~90.000 dan ~85.000 lowongan, menunjukkan rebound industri F&B & service pasca-pandemi.
                * Manufaktur (~60.000), Seni/Media/Komunikasi (~55.000), dan Teknik (~55.000) juga masih menyumbang porsi signifikan.
                * Pendidikan/Pelatihan (~40.000) menutup Top 10—sektor ini terus mencari guru, instruktur, dan content developer.           

            """)
        else:
            st.info("Tidak ada kategori lowongan yang cukup untuk ditampilkan.")
    else:
        st.info("Data lowongan pekerjaan tidak tersedia atau tidak memiliki kolom 'categoriesName'. Visualisasi ini tidak dapat ditampilkan.")


    # --- Visualisasi: Jumlah Lowongan per Kategori Industri 4.0 ---
    st.markdown("### Jumlah Lowongan per Kategori Industri 4.0")
    if 'industri4_0' in df_job_original.columns and not df_job_original.empty:
        industry_counts = df_job_original['industri4_0'].value_counts()
        industry_4_0_counts = industry_counts[industry_counts.index != 'Other / Non-4.0']

        if not industry_4_0_counts.empty:
            # Mengurutkan kategori industri 4.0 berdasarkan jumlah lowongan
            industry_4_0_counts = industry_4_0_counts.sort_values(ascending=False)
            
            fig_industry_40, ax_industry_40 = plt.subplots(figsize=(10, 6))
            # Menggunakan bar plot untuk visualisasi
            ax_industry_40.bar(industry_4_0_counts.index, industry_4_0_counts.values) 
            ax_industry_40.set_xlabel('Kategori Industri 4.0')
            ax_industry_40.set_ylabel('Jumlah Lowongan')
            ax_industry_40.set_title('Jumlah Lowongan per Kategori Industri 4.0')
            # Perbaikan untuk tick_params ValueError
            plt.setp(ax_industry_40.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_industry_40)
            st.markdown("""
            Visualisasi ini menunjukkan distribusi jumlah lowongan pekerjaan dalam berbagai kategori yang terkait dengan Industri 4.0. Ini menyoroti area teknologi dan inovasi yang paling banyak diminati di pasar kerja.

            **Insight:**
            1. **Dominasi Software Development:** Sekitar 70.000 lowongan (±84% dari total lowongan Industry 4.0 di dataset) berada di kategori *Software Development*. Ini menegaskan bahwa pengembangan aplikasi & pemrograman masih menjadi tulang punggung kebutuhan industri 4.0—mulai dari *full-stack, backend, frontend*, hingga *mobile/web developer*.

            2. **Data & Security Menduduki Peringkat Dua dan Tiga:** *Data & Big Data*: ±7.334 lowongan (±8,8%).  
            *Cybersecurity*: ±6.708 lowongan (±8,1%).  
            Artinya organisasi juga butuh banyak *data engineer/analyst* dan spesialis keamanan siber, sejalan dengan tren pengolahan data besar dan ancaman keamanan digital.

            3. **Layanan Cloud dan Otomasi Industri:** *Cloud Computing* (3.175; ±3,8%) dan *Robotics & Automation* (3.005; ±3,6%) menunjukkan perlunya infrastruktur *cloud* modern dan otomatisasi proses manufaktur/robotik meski dalam skala lebih kecil dibanding software.

            4. **Long Tail pada Niche-Tech:** Kategori *Augmented/Virtual Reality* (1.437; ±1,7%), *IoT* (469; ±0,6%), *AI/ML* (439; ±0,5%), dan *Blockchain* (338; ±0,4%) relatif jarang. Ini bisa berarti:
            - Lowongan spesialis di bidang *emerging tech* masih terbatas.
            - Talenta *niche* (misal AR/VR developer, IoT engineer, data scientist, blockchain dev) punya daya tawar tinggi karena pasokan lowongan jauh melebihi jumlah pelamar.

            5. **Proporsi Industri 4.0 Secara Keseluruhan:** Dari total ±1,24 juta lowongan, hanya ±83.000 masuk kategori Industry 4.0—yakni sekitar 6,7% dari keseluruhan. Artinya lebih dari 93% lowongan masih di sektor “tradisional” atau non-4.0 (administrasi, *marketing*, produksi, dsb.). Untuk meningkatkan ekosistem 4.0, masih banyak ruang bagi *stakeholder* (pemerintah, perguruan tinggi, pelatihan) untuk menumbuhkan kapasitas talenta digital/industri.
            """)
        else:
            st.info("Tidak ada lowongan yang teridentifikasi dalam kategori Industri 4.0 atau data lowongan kosong.")
    else:
        st.info("Data lowongan pekerjaan tidak tersedia atau tidak memiliki kolom 'industri4_0'. Visualisasi ini tidak dapat ditampilkan.")


    # --- Visualisasi: Distribusi Lowongan Kerja Berdasarkan Lokasi ---
    st.markdown("### Distribusi Lowongan Kerja Berdasarkan Lokasi (Top 10)")
    # Asumsi kolom lokasi di df_job_original adalah 'locations'
    if 'locations' in df_job_original.columns and not df_job_original.empty:
        df_job_original['cleaned_location'] = df_job_original['locations'].apply(
            lambda x: str(x).split(',')[0].strip()
        )
        valid_locations = df_job_original['cleaned_location'][df_job_original['cleaned_location'].str.len() > 2]
        valid_locations = valid_locations[~valid_locations.str.contains(r'\d', na=False)] # Hapus yang mengandung angka
        
        # Filter lokasi yang tidak relevan
        valid_locations = valid_locations[
            ~valid_locations.isin(['Indonesia', 'Jawa Barat'])
        ]

        location_counts = valid_locations.value_counts().reset_index()
        location_counts.columns = ['location', 'job_count']

        if not location_counts.empty:
            top_locations = location_counts.head(10)
            fig_locations, ax_locations = plt.subplots(figsize=(10, 8))
            ax_locations.barh(top_locations['location'], top_locations['job_count'], color=sns.color_palette("cubehelix", 10))
            ax_locations.set_xlabel('Jumlah Lowongan Kerja')
            ax_locations.set_ylabel('Lokasi')
            ax_locations.set_title('Distribusi Lowongan Kerja Berdasarkan Lokasi (Top 10)')
            ax_locations.invert_yaxis() 
            plt.tight_layout()
            st.pyplot(fig_locations)
            st.markdown("""
            Bar chart ini menampilkan 10 lokasi dengan jumlah lowongan kerja terbanyak. Ini membantu mengidentifikasi pusat-pusat kesempatan kerja dan tren geografis dalam pasar kerja.

            **Insight:**
            1.  **Dominasi Jakarta Raya:**
                Jakarta Raya (gabungan DKI Jakarta + sekitarnya) jauh mengungguli lokasi lain, dengan lebih dari 80.000 lowongan—sekitar 3–4× lipat dari peringkat kedua (Jakarta Selatan). Artinya, peluang kerja paling banyak, Jakarta adalah *hotspot* utama.
            2.  **Konsentrasi di Pulau Jawa:**
                Setelah Jakarta, kota-kota di Pulau Jawa (Bandung, Jawa Barat, Jawa Tengah, Yogyakarta, Jawa Timur, Bekasi, Bogor) juga menempati sebagian besar daftar. Ini menunjukkan masih terpusatnya industri dan perusahaan besar di Jawa.
            3.  **Cluster Sekunder di Luar Jawa:**
                Surabaya (meski sudah masuk daftar Jawa Timur) dan Bali tampil cukup tinggi, menandakan pertumbuhan sektor pariwisata/teknologi di Bali. Kota-kota lain di luar Jawa (Tangerang—walau secara administratif Banten tapi masuk area Jabodetabek, Batam, Medan) masih relatif terbatas.
            4.  **Long Tail Distribution:**
                Setelah Jakarta menjadi Top 1, jumlah lowongan per kota langsung turun drastis (*long tail*). Artinya, banyak kota tingkat kedua/ketiga yang hanya punya sedikit lowongan—perlu *effort* ekstra bagi pelamar di kota-kota tersebut.
            """)
        else:
            st.info("Tidak ada data lokasi lowongan kerja yang valid untuk ditampilkan.")
    else:
        st.info("Data lowongan pekerjaan tidak tersedia atau tidak memiliki kolom 'locations'. Visualisasi ini tidak dapat ditampilkan.")

elif selected_tab == "Heatmap Korelasi Metrik Utama":
    st.header("Visualisasi Data")
    st.subheader("Heatmap Korelasi Metrik Utama")

    st.markdown("### Heatmap Korelasi Antar Fitur Numerik Utama")
    # Pilih metrik utama sesuai yang ada di notebook
    cols_for_heatmap = ['ratio_numeric','hor_selaras_pct_clean','mtk_selaras_pct','vert_tinggi_pct','job_count']
    
    # Filter df_clean untuk hanya menyertakan kolom yang ada
    available_cols_for_heatmap = [col for col in cols_for_heatmap if col in df_clean.columns and not df_clean[col].isnull().all()]
    
    if len(available_cols_for_heatmap) > 1:
        corr_matrix = df_clean[available_cols_for_heatmap].corr()

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(6,5)) # Sesuaikan ukuran figur sesuai notebook
        im = ax_heatmap.imshow(corr_matrix, vmin=-1, vmax=1, cmap='viridis') # Menggunakan viridis sesuai gambar
        # Perbaikan: Menggunakan fig_heatmap.colorbar() yang merupakan cara yang benar
        fig_heatmap.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
        ax_heatmap.set_xticks(range(len(available_cols_for_heatmap)))
        # Perbaikan untuk tick_params ValueError
        plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        ax_heatmap.set_yticks(range(len(available_cols_for_heatmap)))
        ax_heatmap.set_yticklabels(available_cols_for_heatmap)
        ax_heatmap.set_title('Heatmap Korelasi Metrik Utama')
        plt.tight_layout()
        st.pyplot(fig_heatmap)
        st.markdown("""
        Heatmap ini menunjukkan koefisien korelasi Pearson antara metrik numerik utama yang digunakan dalam analisis dan model. Nilai mendekati 1 menunjukkan korelasi positif kuat, -1 menunjukkan korelasi negatif kuat, dan 0 menunjukkan tidak ada korelasi linear. Ini membantu mengidentifikasi hubungan antar variabel.

        **Insight:**
        1.  **Korelasi Negatif Rasio vs Hor Selaras:**
            Rasio mahasiswa per dosen (`ratio_numeric`) menunjukkan korelasi negatif (misal sekitar –0.4) dengan persentase lulusan yang merasa selaras horizontal (`hor_selaras_pct_clean`). Artinya: semakin banyak mahasiswa per dosen, semakin rendah proporsi alumni yang menilai pekerjaannya sesuai dengan bidang studi. Ini menegaskan pentingnya rasio dosen-mahasiswa yang ideal untuk kualitas “*match*” lulusan.
        2.  **Hubungan Kuat Antara Hor Selaras & Masa Tunggu:**
            `hor_selaras_pct_clean` dan `mtk_selaras_pct` (lulusan selaras yang cepat dapat kerja) berkorelasi sangat positif (misal $> 0.7$). Program studi yang alumninya merasa sesuai bidangnya cenderung juga cepat terserap ($\leq 6$ bulan), menunjukkan keselarasan horizontal berpengaruh langsung ke kecepatan *employability*.
        3.  **Keselarasan Horizontal & Vertikal Saling Mendukung:**
            Ada korelasi sedang (sekitar 0.5) antara `hor_selaras_pct_clean` dan `vert_tinggi_pct`. Artinya: lulusan yang “*match*” secara bidang juga cenderung mendapatkan posisi dengan level tanggung jawab yang lebih tinggi (“vertikal”).
        4.  **Job Count (Peluang Industri 4.0) Relatif Independen:**
            `job_count` (jumlah lowongan kategori 4.0 per prodi) menunjukkan korelasi rendah ($< 0.2$) dengan metrik-metrik *tracer study*. Ini mengindikasikan kebutuhan pasar (`job_count`) tidak selalu selaras 1-1 dengan kualitas output PTN: ada program yang punya banyak lowongan tapi lulusan belum optimal “*match*”, dan sebaliknya.
        """)
    else:
        st.warning("Tidak cukup fitur numerik yang tersedia untuk membuat heatmap korelasi.")

elif selected_tab == "Model Machine Learning":
    st.header("Model Machine Learning: Prediksi Waktu Tunggu Lulusan")

    st.markdown("### Penjelasan Singkatan Inputan Model")
    st.write("""
    Berikut adalah penjelasan untuk singkatan yang digunakan pada inputan model prediksi:
    | Singkatan | Penjelasan |
    | :---------- | :---------- |
    | **MTK Selaras (Jumlah)** | Masa Tunggu Kerja Selaras: Jumlah alumni yang mendapatkan pekerjaan yang sesuai dengan bidang studi mereka dalam waktu yang dianggap 'cepat' (misalnya $\\leq$ 6 bulan). |
    | **MTK Tidak Selaras (Jumlah)** | Masa Tunggu Kerja Tidak Selaras: Jumlah alumni yang mendapatkan pekerjaan yang tidak sesuai dengan bidang studi mereka atau dalam waktu 'lama' (misalnya $>$ 6 bulan). |
    | **Horizontal Selaras (Jumlah)** | Jumlah lulusan yang bekerja di bidang yang selaras secara horizontal (sesuai jurusan) dengan program studinya. |
    | **Horizontal Tidak Selaras (Jumlah)** | Jumlah lulusan yang bekerja di bidang yang TIDAK selaras secara horizontal (tidak sesuai jurusan) dengan program studinya. |
    | **Vertikal Tinggi (Jumlah)** | Jumlah lulusan yang mendapatkan pekerjaan dengan tingkat tanggung jawab yang lebih tinggi dari pendidikan yang diperoleh (misalnya S1 menduduki posisi manajerial). |
    | **Vertikal Sama (Jumlah)** | Jumlah lulusan yang mendapatkan pekerjaan dengan tingkat tanggung jawab yang sesuai dengan pendidikan yang diperoleh (misalnya S1 menduduki posisi staf). |
    | **Vertikal Rendah (Jumlah)** | Jumlah lulusan yang mendapatkan pekerjaan dengan tingkat tanggung jawab yang lebih rendah dari pendidikan yang diperoleh (misalnya S1 menduduki posisi teknisi). |
    """)

    # --- Data Preprocessing and Feature Engineering ---
    df_model = df_raw.copy()

    def convert_ratio_ipynb(ratio):
        try:
            if ':' in str(ratio):
                num, denom = map(float, ratio.split(':'))
                return num / denom if denom != 0 else 0
            return float(ratio)
        except:
            return 0

    akreditasi_map = {'Unggul': 3, 'Baik Sekali': 2, 'Baik': 1, 'A': 1, 'B': 1, 'Terakreditasi Sementara': 0, '-': 0, 'Tidak Diketahui': 0}
    df_model['Akreditasi_encoded'] = df_model['Akreditasi'].map(akreditasi_map).fillna(0)

    percentage_cols = ['hor_selaras_pct', 'hor_tidak_selaras_pct', 'mtk_selaras_pct', 'mtk_tidak_selaras_pct', 'vert_tinggi_pct', 'vert_sama_pct', 'vert_rendah_pct']
    for col in percentage_cols:
        if col in df_model.columns and df_model[col].dtype == object:
            df_model[col] = df_model[col].replace('NaN', '0%').str.rstrip('%').astype(float) / 100 
        else: 
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

    # Buat 'waiting_time_label' (variabel target)
    def create_waiting_time_label(pct):
        return 'Cepat' if pct >= 0.7 else 'Lama'
    df_model['waiting_time_label'] = df_model['hor_selaras_pct'].apply(create_waiting_time_label)

    # Hitung NaN untuk semua kolom numerik menggunakan mediannya
    numeric_cols_for_fillna_median = df_model.select_dtypes(include=np.number).columns
    for col in numeric_cols_for_fillna_median:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(df_model[col].median() if not df_model[col].empty and df_model[col].median() == df_model[col].median() else 0)

    # Kolom 'kategori' One-Hot Encode
    if 'program_studi' in df_model.columns: # Pastikan kolom program_studi ada
        if 'category' not in df_model.columns:
            df_model['category'] = df_model['program_studi'].apply(get_category)
        df_model['category'] = df_model['category'].fillna('Lainnya')
        df_model = pd.get_dummies(df_model, columns=['category'], prefix='cat', dummy_na=False)
    else:
        st.warning("Kolom 'program_studi' tidak ditemukan di df_model. One-hot encoding untuk kategori tidak dapat dilakukan.")
        # Jika category_column tidak ada, pastikan tidak ada kolom 'cat_' di features
        features = [
            'Akreditasi_encoded',
            'Jumlah Pendidik Tetap',
            'Rasio dosen/mahasiswa_numeric', 
            'mtk_selaras_pct',
            'vert_tinggi_pct',
            'vert_sama_pct',
            'vert_rendah_pct'
        ]

    features = [
        'Akreditasi_encoded', 
        'Jumlah Pendidik Tetap', 
        # Pastikan 'Rasio dosen/mahasiswa_numeric' sudah ada di df_model dari load_data/prepare_data
        'Rasio dosen/mahasiswa_numeric', 
        'mtk_selaras_pct', 
        'vert_tinggi_pct', 
        'vert_sama_pct', 
        'vert_rendah_pct'
    ]
    # Tambahkan kolom kategori yang dikodekan satu-panas dari df_model
    features.extend([col for col in df_model.columns if col.startswith('cat_')])
    features = sorted(list(set(features))) 

    
    X = df_model[[col for col in features if col in df_model.columns]]
    y = df_model['waiting_time_label']

    # Kodekan variabel target
    le_label = LabelEncoder()
    y_encoded = le_label.fit_transform(y)
    
    if y_encoded.size == 0 or len(np.unique(y_encoded)) <= 1:
        st.error("Data target (waktu_tunggu_label) tidak cukup untuk melatih model atau hanya memiliki satu kelas. Model tidak dapat dilatih.")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) #
    except ValueError as e:
        st.error(f"Error saat splitting data: {e}. Pastikan data target memiliki setidaknya 2 kelas unik dan data tidak kosong.")
        st.stop()

    # Menentukan dan latih modelnya
    model = RandomForestClassifier( 
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    # rf_classifier.fit(X_train_scaled, y_train) # Cocok dengan data yang tidak berskala
    model.fit(X_train, y_train)

    st.subheader("Evaluasi Model Random Forest Classifier")

    # y_pred_train = rf_classifier.predict(X_train_scaled) -> Prediksi dengan data yang tidak berskala
    # y_pred_test = rf_classifier.predict(X_test_scaled)
    y_pred_train = model.predict(X_train) 
    y_pred_test = model.predict(X_test) 

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    if test_accuracy == 1.0 and f1_score(y_test, y_pred_test, average='macro') == 1.0:
        st.warning("PERINGATAN: Akurasi dan F1-Macro Score model adalah 1.00. Ini kemungkinan besar mengindikasikan overfitting atau data uji yang tidak representatif (misalnya, hanya satu kelas yang tersisa). Harap periksa data Anda.")
    
    report = classification_report(y_test, y_pred_test, target_names=le_label.classes_, output_dict=True)

    st.write(f"**Akurasi Model (Data Pengujian):** {test_accuracy:.2f}")
    st.write(f"**Akurasi Model (Data Pelatihan):** {train_accuracy:.2f}")
    st.write(f"**F1-Macro Score (Data Pengujian):** {f1_score(y_test, y_pred_test, average='macro'):.2f}")
    st.write(f"**Precision (Data Pengujian - Cepat):** {report['Cepat']['precision']:.2f}")
    st.write(f"**Recall (Data Pengujian - Cepat):** {report['Cepat']['recall']:.2f}")
    st.write(f"**F1-Score (Data Pengujian - Cepat):** {report['Cepat']['f1-score']:.2f}")
    
    st.write(f"**Precision (Data Pengujian - Lama):** {report['Lama']['precision']:.2f}")
    st.write(f"**Recall (Data Pengujian - Lama):** {report['Lama']['recall']:.2f}")
    st.write(f"**F1-Score (Data Pengujian - Lama):** {report['Lama']['f1-score']:.2f}")


    st.text("Classification Report:")
    st.dataframe(pd.DataFrame(report).T.round(2))

    st.subheader("Prediksi Waktu Tunggu Kerja Lulusan")
    st.write("Masukkan karakteristik program studi untuk memprediksi apakah lulusannya akan cepat atau lama mendapatkan pekerjaan.")

    with st.form("prediction_form"):
        st.write("### Informasi Program Studi")
        prodi_name = st.text_input("Nama Program Studi (contoh: S1 Teknik Informatika)", "S1 Teknik Informatika")
        
        # Pilihan untuk selectbox harus berasal dari nilai unik di df_model['Akreditasi']
        akreditasi_options = sorted(df_model['Akreditasi'].unique().tolist())
        akreditasi_input = st.selectbox("Akreditasi", akreditasi_options, index=akreditasi_options.index('Unggul') if 'Unggul' in akreditasi_options else 0)
        
        status_options = sorted(df_model['Status'].unique().tolist())
        status_input = st.selectbox("Status", status_options, index=status_options.index('Aktif') if 'Aktif' in status_options else 0)
        
        jenjang_options = sorted(df_model['Jenjang'].unique().tolist())
        jenjang_input = st.selectbox("Jenjang", jenjang_options, index=jenjang_options.index('S1') if 'S1' in jenjang_options else 0)

        st.write("### Statistik Dosen dan Mahasiswa")
        jumlah_dosen = st.number_input("Jumlah dosen", min_value=0, value=20)
        jumlah_pendidik_tetap = st.number_input("Jumlah Pendidik Tetap", min_value=0, value=15)
        jumlah_pendidik_tidak_tetap = st.number_input("Jumlah pendidik tidak tetap", min_value=0, value=5)
        total_pendidik = st.number_input("Total pendidik", min_value=0, value=20)
        jumlah_mahasiswa = st.number_input("Jumlah mahasiswa", min_value=0, value=500)
        
        st.write("### Statistik Keselarasan Horizontal dan Masa Tunggu (Persentase)")
        mtk_selaras_pct = st.number_input("Masa Tunggu Kerja Selaras (%)", min_value=0.0, max_value=100.0, value=90.0) / 100.0
        vert_tinggi_pct = st.number_input("Vertikal Tinggi (%)", min_value=0.0, max_value=100.0, value=15.0) / 100.0
        vert_sama_pct = st.number_input("Vertikal Sama (%)", min_value=0.0, max_value=100.0, value=75.0) / 100.0
        vert_rendah_pct = st.number_input("Vertikal Rendah (%)", min_value=0.0, max_value=100.0, value=10.0) / 100.0
        
        st.write("### Statistik Keselarasan Horizontal dan Masa Tunggu (Jumlah)")
        hor_selaras_jumlah = st.number_input("Horizontal Selaras (Jumlah)", min_value=0.0, value=100.0)
        hor_tidak_selaras_jumlah = st.number_input("Horizontal Tidak Selaras (Jumlah)", min_value=0.0, value=20.0)
        mtk_selaras_jumlah = st.number_input("MTK Selaras (Jumlah)", min_value=0.0, value=50.0)
        mtk_tidak_selaras_jumlah = st.number_input("MTK Tidak Selaras (Jumlah)", min_value=0.0, value=10.0)
        
        job_count_input = st.number_input("Jumlah Lowongan Pekerjaan (terkait kategori prodi)", min_value=0, value=1000, help="Catatan: Fitur ini tidak digunakan oleh model klasifikasi saat ini.")
        submitted = st.form_submit_button("Prediksi Waktu Tunggu")

        if submitted:
            # Hitung Rasio dosis/mahasiswa_numeric dari input
            if jumlah_mahasiswa > 0:
                rasio_dosen_mahasiswa_numeric_input = total_pendidik / jumlah_mahasiswa
            else:
                rasio_dosen_mahasiswa_numeric_input = 0 

            input_df_raw_format = pd.DataFrame([{
                'Prodi': prodi_name.split(' ')[1] if ' ' in prodi_name else prodi_name, 
                'program_studi': prodi_name,
                'Akreditasi': akreditasi_input,
                'Status': status_input,
                'Jenjang': jenjang_input,
                'Jumlah dosen': float(jumlah_dosen),
                'Jumlah Pendidik Tetap': float(jumlah_pendidik_tetap),
                'Jumlah pendidik tidak tetap': float(jumlah_pendidik_tidak_tetap),
                'Total pendidik': float(total_pendidik),
                'Jumlah mahasiswa ': float(jumlah_mahasiswa),
                'hor_selaras_jumlah': float(hor_selaras_jumlah),
                'hor_tidak_selaras_jumlah': float(hor_tidak_selaras_jumlah),
                'mtk_selaras_jumlah': float(mtk_selaras_jumlah),
                'mtk_tidak_selaras_jumlah': float(mtk_tidak_selaras_jumlah),
                'vert_tinggi_jumlah': df_model['vert_tinggi_jumlah'].median(), # Gunakan median jika tidak ada dalam fitur, atau dapatkan
                'vert_sama_jumlah': df_model['vert_sama_jumlah'].median(),
                'vert_rendah_jumlah': df_model['vert_rendah_jumlah'].median(),
                'hor_selaras_pct': df_model['hor_selaras_pct'].median(), # Bukan fitur, tetapi digunakan untuk target dalam df penuh
                'hor_tidak_selaras_pct': df_model['hor_tidak_selaras_pct'].median(), # Bukanfeature
                'mtk_selaras_pct': float(mtk_selaras_pct), # Feature
                'mtk_tidak_selaras_pct': float(mtk_tidak_selaras_jumlah) / (float(mtk_selaras_jumlah) + float(mtk_tidak_selaras_jumlah)) if (float(mtk_selaras_jumlah) + float(mtk_tidak_selaras_jumlah)) > 0 else 0.0, # Feature (calculated)
                'vert_tinggi_pct': float(vert_tinggi_pct), # Feature
                'vert_sama_pct': float(vert_sama_pct), # Feature
                'vert_rendah_pct': float(vert_rendah_pct), # Feature
                'job_count': float(job_count_input), 
                'Rasio dosen/mahasiswa': str(rasio_dosen_mahasiswa_numeric_input) # Tambahkan ini untuk memastikannya diproses
            }])

            # Pastikan input_df_raw_format memiliki kolom yang diperlukan
            input_df_raw_format['Rasio dosen/mahasiswa_numeric'] = input_df_raw_format['Rasio dosen/mahasiswa'].apply(convert_ratio_ipynb)
            input_df_raw_format['Akreditasi_encoded'] = input_df_raw_format['Akreditasi'].map(akreditasi_map).fillna(0)
            
            input_df_raw_format['category'] = input_df_raw_format['program_studi'].apply(get_category)
            input_df_raw_format['category'] = input_df_raw_format['category'].fillna('Lainnya')
            
            # Buat ulang kolom dummy untuk baris input tunggal
            # Pastikan semua kolom 'cat_' yang mungkin dari X_train ada di X_predict
            X_predict_data = {col: 0 for col in X_train.columns if col.startswith('cat_')}

            # Isi fitur lainnya
            for feature_col in features:
                if feature_col.startswith('cat_'):
                    # Tetapkan variabel dummy kategori tertentu ke 1
                    input_category_dummy_col = f"cat_{input_df_raw_format['category'].iloc[0]}"
                    if input_category_dummy_col in X_predict_data:
                        X_predict_data[input_category_dummy_col] = 1
                elif feature_col in input_df_raw_format.columns:
                    X_predict_data[feature_col] = input_df_raw_format[feature_col].iloc[0]
                else:
                    # Pengganti untuk fitur lain yang hilang dalam input_df_raw_format
                    X_predict_data[feature_col] = 0
            
            X_predict = pd.DataFrame([X_predict_data])
            
            # Pastikan urutan dan tipe data sesuai dengan data pelatihan
            X_predict = X_predict[X_train.columns].astype(X_train.dtypes)

           # Prediksi dengan data yang tidak berskala karena Pengklasifikasi RF 
            prediction_encoded = model.predict(X_predict)
            prediction_label = le_label.inverse_transform(prediction_encoded)

            st.success(f"**Prediksi Waktu Tunggu Lulusan:** {prediction_label[0]}")

    st.subheader("Pentingnya Fitur dalam Prediksi")
    st.write("Berikut adalah 10 fitur teratas yang paling berkontribusi pada keputusan model:")

    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.dataframe(feature_importances.reset_index().rename(columns={'index': 'Fitur', 0: 'Kontribusi'}).round(4))
    else:
        st.write("Model tidak memiliki atribut 'feature_importances_'.")