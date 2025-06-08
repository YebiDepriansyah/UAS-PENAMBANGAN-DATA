import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi DO Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    h1 {
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    h2 {
        color: #1565C0;
    }
    h3 {
        color: #1976D2;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
try:
    df = pd.read_csv('dataset_prediksi_DO_mahasiswa.csv', sep=';')
    
    # Konversi kolom numerik
    numeric_columns = ['ipk_sem1', 'ipk_sem2', 'ipk_sem3', 'ipk_sem4', 
                      'kehadiran_rata2', 'matkul_diulang', 'beban_kerja', 
                      'pendapatan_ortu']
    
    for col in numeric_columns:
        # Pastikan kolom adalah string sebelum melakukan replace
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Load model dan encoder
try:
    model = joblib.load('random_forest_model.joblib')
    encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    scaler = joblib.load('label_Sacler.pkl')
except Exception as e:
    st.error(f"Error loading model or encoders: {str(e)}")
    st.stop()

# Judul aplikasi
st.markdown("""
    <div style='text-align: center;'>
        <h1>Dashboard Prediksi Drop Out Mahasiswa</h1>
        <p style='color: #666;'>Sistem Prediksi Cerdas untuk Mencegah Drop Out Mahasiswa</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Tambahkan informasi statistik di bagian atas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Akurasi Model", value="90%", delta="5%")
with col2:
    st.metric(label="Total Data Training", value="876", delta="76")
with col3:
    st.metric(label="Fitur Prediksi", value="15", delta="3")

# Tambahkan tab untuk navigasi
tab1, tab2 = st.tabs(["Prediksi DO", "Analisis Data"])

with tab1:
    st.header("Data Mahasiswa")

    # Membagi layout menjadi 2 kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Akademik")
        ipk_sem1 = st.number_input("IPK Semester 1", min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")
        ipk_sem2 = st.number_input("IPK Semester 2", min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")
        ipk_sem3 = st.number_input("IPK Semester 3", min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")
        ipk_sem4 = st.number_input("IPK Semester 4", min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")
        
        # Tambahkan grafik IPK yang lebih besar
        ipk_data = pd.DataFrame({
            'Semester': ['Semester 1', 'Semester 2', 'Semester 3', 'Semester 4'],
            'IPK': [ipk_sem1, ipk_sem2, ipk_sem3, ipk_sem4]
        })
        fig = px.line(ipk_data, x='Semester', y='IPK', markers=True, title='Tren IPK per Semester')
        fig.update_layout(
            height=400,
            xaxis_title="Semester",
            yaxis_title="IPK",
            yaxis_range=[0, 4],
            showlegend=False
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
        
        kehadiran = st.number_input("Persentase Kehadiran (%)", min_value=0, max_value=100, value=0)
        matkul_diulang = st.number_input("Jumlah Mata Kuliah Diulang", min_value=0, value=0)
        prodi = st.selectbox("Program Studi", ["Pilih Program Studi", "Ilmu Komunikasi", "Teknik Elektro", "Hukum", "Sistem Informasi", "Teknik Informatika", "Pendidikan Matematika", "Kedokteran", "Akuntansi", "Teknik Sipil", "Manajemen"])
        aktivitas_lms = st.selectbox("Aktivitas LMS", ["Pilih Aktivitas LMS", "sangat_aktif", "aktif", "sedang", "pasif", "kurang_aktif"])

    with col2:
        st.subheader("Data Pribadi")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pilih Jenis Kelamin", "Laki-laki", "Perempuan"])
        status_pekerjaan = st.selectbox("Status Pekerjaan", ["Pilih Status Pekerjaan", "tidak_bekerja", "paruh_waktu", "penuh_waktu"])
        beban_kerja = st.number_input("Beban Kerja (jam/minggu)", min_value=0, max_value=40, value=0)
        pendapatan_ortu = st.number_input("Pendapatan Orang Tua (juta/bulan)", min_value=0.0, max_value=20.0, value=0.0, step=0.01)
        pendidikan_ortu = st.selectbox("Pendidikan Orang Tua", ["Pilih Pendidikan Orang Tua", "S1", "SMA", "SD", "SMP", "S2", "D3"])
        lokasi_tinggal = st.selectbox("Lokasi Tinggal", ["Pilih Lokasi Tinggal", "kota", "desa"])
        keterlibatan_organisasi = st.selectbox("Keterlibatan Organisasi", ["Pilih Keterlibatan Organisasi", "aktif", "tidak aktif"])

    # Tombol prediksi dengan styling
    st.markdown("---")
    if st.button("Prediksi Status DO", key="predict_button"):
        # Validasi input
        if any(value == 0.0 for value in [ipk_sem1, ipk_sem2, ipk_sem3, ipk_sem4]) or \
           any(value == "Pilih" in str(value) for value in [prodi, aktivitas_lms, jenis_kelamin, status_pekerjaan, pendidikan_ortu, lokasi_tinggal, keterlibatan_organisasi]):
            st.warning("Mohon lengkapi semua data terlebih dahulu!")
        else:
            # Membuat DataFrame dari input
            user_input = {
                "ipk_sem1": ipk_sem1,
                "ipk_sem2": ipk_sem2,
                "ipk_sem3": ipk_sem3,
                "ipk_sem4": ipk_sem4,
                "kehadiran_rata2": kehadiran,
                "matkul_diulang": matkul_diulang,
                "prodi": prodi,
                "jenis_kelamin": jenis_kelamin,
                "aktivitas_lms": aktivitas_lms,
                "status_pekerjaan": status_pekerjaan,
                "beban_kerja": beban_kerja,
                "pendapatan_ortu": pendapatan_ortu,
                "pendidikan_ortu": pendidikan_ortu,
                "lokasi_tinggal": lokasi_tinggal,
                "keterlibatan_organisasi": keterlibatan_organisasi
            }
            
            # Ubah ke DataFrame
            df_input = pd.DataFrame([user_input])
            
            # Label Encode 'prodi'
            if 'prodi' in encoders:
                df_input['prodi'] = encoders['prodi'].transform(df_input['prodi'])
            
            # One-hot encoding untuk kolom kategorikal
            categorical_cols = df_input.select_dtypes(include='object').columns.tolist()
            df_input = pd.get_dummies(df_input, columns=categorical_cols)
            
            # Pastikan fitur sama seperti saat pelatihan
            target_col = 'status_DO'
            if target_col in feature_columns:
                feature_columns.remove(target_col)
            
            # Tambahkan kolom yang hilang
            for col in feature_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            # Hapus kolom ekstra
            extra_cols = set(df_input.columns) - set(feature_columns)
            if extra_cols:
                df_input = df_input.drop(columns=list(extra_cols))
            
            # Urutkan kolom sesuai fitur pelatihan
            df_input = df_input[feature_columns]
            
            # Normalisasi data
            df_input = scaler.transform(df_input)
            
            # Prediksi
            prediction = model.predict(df_input)[0]
            prediction_proba = model.predict_proba(df_input)[0]
            
            # Mapping hasil prediksi
            label_mapping = {
                0: "Tidak Drop Out",
                1: "Drop Out"
            }
            
            # Tampilkan hasil prediksi
            st.markdown("---")
            st.subheader("Hasil Prediksi")
            
            # Buat dua kolom untuk hasil prediksi
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                # Tampilkan status DO dengan warna yang sesuai
                if prediction == 1:
                    st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #ff4b4b; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>Status: Drop Out</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #4CAF50; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>Status: Tidak Drop Out</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col_pred2:
                # Tampilkan probabilitas dengan warna yang sesuai
                prob_do = prediction_proba[1] * 100
                prob_tidak_do = prediction_proba[0] * 100
                
                if prediction == 1:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background-color: #ff4b4b; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>Probabilitas Drop Out: {prob_do:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background-color: #4CAF50; border-radius: 10px;'>
                            <h2 style='color: white; margin: 0;'>Probabilitas Tidak Drop Out: {prob_tidak_do:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)

            if prediction == 1:
                st.subheader("Rekomendasi:")
                st.markdown("""
                <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; color: #000000;'>
                    <ol>
                        <li>Tingkatkan kehadiran di kelas</li>
                        <li>Manfaatkan layanan bimbingan akademik</li>
                        <li>Kurangi beban kerja jika memungkinkan</li>
                        <li>Tingkatkan keterlibatan dalam kegiatan kampus</li>
                        <li>Manfaatkan fasilitas LMS untuk belajar</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.subheader("Rekomendasi:")
                st.markdown("""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; color: #000000;'>
                    <ol>
                        <li>Pertahankan IPK dan kehadiran</li>
                        <li>Tingkatkan keterlibatan dalam organisasi</li>
                        <li>Manfaatkan kesempatan magang</li>
                        <li>Jaga keseimbangan antara akademik dan non-akademik</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("Analisis Data")
    
    # Tampilkan statistik dasar
    st.subheader("Statistik Dasar")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Distribusi Status DO")
        status_counts = df['status_DO'].value_counts()
        fig = px.pie(values=status_counts.values, 
                    names=status_counts.index, 
                    title='Distribusi Status DO')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("Rata-rata IPK per Semester")
        ipk_means = df[['ipk_sem1', 'ipk_sem2', 'ipk_sem3', 'ipk_sem4']].mean()
        fig = px.bar(x=ipk_means.index, y=ipk_means.values, 
                    title='Rata-rata IPK per Semester',
                    labels={'x': 'Semester', 'y': 'Rata-rata IPK'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan distribusi fitur penting
    st.subheader("Distribusi Fitur Penting")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Distribusi Program Studi")
        prodi_counts = df['prodi'].value_counts()
        fig = px.bar(x=prodi_counts.index, y=prodi_counts.values,
                    title='Distribusi Program Studi',
                    labels={'x': 'Program Studi', 'y': 'Jumlah Mahasiswa'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("Distribusi Kehadiran")
        fig = px.histogram(df, x='kehadiran_rata2', 
                          title='Distribusi Kehadiran',
                          labels={'x': 'Persentase Kehadiran', 'y': 'Jumlah Mahasiswa'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan korelasi antara fitur
    st.subheader("Korelasi antara Fitur")
    numeric_cols = ['ipk_sem1', 'ipk_sem2', 'ipk_sem3', 'ipk_sem4', 
                   'kehadiran_rata2', 'matkul_diulang', 'beban_kerja', 
                   'pendapatan_ortu']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, 
                   title='Korelasi antara Fitur Numerik',
                   labels=dict(x="Fitur", y="Fitur", color="Korelasi"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(df[numeric_cols].describe())

# Footer dengan styling yang lebih menarik
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; background-color: #f5f5f5; border-radius: 10px;'>
        <h3 style='color: #1E88E5;'>© 2024 Sistem Prediksi DO Mahasiswa</h3>
        <p style='font-size: 1.2em;'>Dibuat oleh Kelompok 10:</p>
        <p style='font-size: 1.1em;'>Arya Mulahernawan (G1A022029)</p>
        <p style='font-size: 1.1em;'>Yebi Depriansyah (G1A022063)</p>
        <p style='font-size: 1.1em;'>Aisyah Amelia Zarah Juaita (G1A022075)</p>
    </div>
""", unsafe_allow_html=True)
