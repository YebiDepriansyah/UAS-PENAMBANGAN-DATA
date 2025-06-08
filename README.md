# UAS-PENAMBANGAN-DATA
Kelompok 10 :
1. Arya Mulahernawan          (G1A022029)
2. Yebi Depriansyah           (G1A022063)
3. Aisyah Amelia Zarah Juaita (G1A022075)

## Project Overview

Permasalahan mahasiswa yang mengalami Drop Out (DO) masih menjadi isu serius di berbagai perguruan tinggi, khususnya di Indonesia. Fenomena ini tidak hanya berdampak pada individu mahasiswa, tetapi juga pada institusi pendidikan tinggi dan efektivitas sistem pendidikan secara keseluruhan. Oleh karena itu, diperlukan pendekatan yang lebih sistematis dan prediktif untuk mengidentifikasi mahasiswa yang berisiko tinggi mengalami DO sejak dini.

Proyek ini dirancang untuk mengembangkan sistem prediksi Drop Out mahasiswa berbasis machine learning, yang bertujuan membantu institusi pendidikan dalam mengambil tindakan preventif. Sistem ini memanfaatkan data historis mahasiswa seperti Indeks Prestasi Kumulatif (IPK) per semester, kehadiran, jumlah mata kuliah yang diulang, latar belakang sosial-ekonomi, serta keterlibatan mahasiswa dalam organisasi. Dengan pendekatan tersebut, model dapat mengidentifikasi pola-pola tertentu yang menjadi indikator risiko DO.

Urgensi dari proyek ini didasarkan pada tiga poin utama. Pertama, sistem ini mampu memberikan peringatan dini bagi institusi terhadap mahasiswa yang membutuhkan perhatian khusus. Kedua, model ini dapat meningkatkan efektivitas intervensi akademik seperti bimbingan, konseling, atau program remedial. Ketiga, dari sisi manajemen pendidikan, sistem ini dapat menjadi alat bantu dalam pengambilan keputusan berbasis data untuk meningkatkan retensi mahasiswa.

Dari sisi akademik, penelitian mengenai sistem prediksi Drop Out telah banyak dilakukan dengan menggunakan berbagai algoritma machine learning, seperti Decision Tree, Support Vector Machine (SVM), hingga Random Forest. Salah satu buku referensi penting dalam bidang ini adalah karya Géron (2019) Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, yang membahas teknik-teknik pengolahan data dan penerapan model prediktif dalam dunia nyata Géron, 2019. Selain itu, penerapan encoding pada fitur kategorikal dan pentingnya validasi model juga menjadi bagian integral dalam proses pembangunan sistem.

Referensi
Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.). O'Reilly Media. https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

Kapila Devi & Saroj Ratnoo. (2022). Predicting student dropouts using random forest. Journal of Statistics and Management Systems, 25(7), 1301–1309.
https://doi.org/10.1080/09720510.2022.2130570

Putra, R. H. H., Suryanegara, M., & Lestari, N. D. (2022). Predicting Student Dropout Using Random Forest and XGBoost Method. Jurnal Ilmiah INTENSIF, 6(1), 11–21.
https://doi.org/10.29407/intensif.v9i1.21191

Wan Yaacob, W. F., Mat Dangi, M. S. S., Ahmad, M. A., & others. (2020). Predicting Student Drop-Out in Higher Institution Using Data Mining Techniques. Journal of Physics: Conference Series, 1496(1), 012005.
https://doi.org/10.1088/1742-6596/1496/1/012005

Andrade-Girón, R. J., Yampis-Paucar, J., Vargas-Poma, H. A., & others. (2023). Predicting Student Dropout based on Machine Learning and Deep Learning: A Systematic Review. EAI Endorsed Transactions on Smart Cities, 7(23), e6.
https://doi.org/10.4108/eetsis.3586

## Business Understanding
Tingkat dropout mahasiswa merupakan salah satu tantangan serius yang dihadapi oleh institusi pendidikan tinggi. Mahasiswa yang tidak menyelesaikan studinya tepat waktu atau memutuskan untuk berhenti kuliah dapat berdampak negatif terhadap reputasi institusi, efisiensi operasional, serta perencanaan akademik dan keuangan. Di sisi lain, bagi mahasiswa itu sendiri, dropout bisa berarti kerugian waktu, biaya, serta peluang masa depan yang tertunda.

Untuk itu, diperlukan sebuah sistem yang dapat mengidentifikasi mahasiswa yang berisiko tinggi mengalami dropout sejak dini. Sistem ini akan membantu pihak kampus atau fakultas untuk melakukan intervensi yang lebih cepat dan tepat, seperti pendampingan akademik, konseling, atau dukungan finansial.

Proyek ini bertujuan membangun model prediktif berbasis data mahasiswa, menggunakan algoritma Random Forest, guna memprediksi kemungkinan seorang mahasiswa akan mengalami dropout. Data yang digunakan mencakup IPK per semester, kehadiran kuliah, jumlah mata kuliah yang diulang, latar belakang keluarga, dan aktivitas mahasiswa. Dengan pendekatan ini, model prediksi tidak hanya berfungsi sebagai alat analisis, tetapi juga sebagai dasar pengambilan keputusan strategis dalam meningkatkan retensi dan kesuksesan studi mahasiswa.

### Problem Statements

1. Bagaimana memprediksi mahasiswa yang berisiko tinggi mengalami drop out menggunakan data akademik dan non-akademik?

2. Fitur atau faktor apa saja yang paling berpengaruh terhadap status drop out mahasiswa?

3. Bagaimana membangun model prediksi yang akurat untuk mendukung keputusan pihak kampus dalam melakukan intervensi dini?

4. Sejauh mana model yang dibangun dapat digeneralisasi ke data mahasiswa baru atau mahasiswa dari jurusan/program studi lain?

### Goals

1. Mengembangkan model machine learning untuk memprediksi status drop out mahasiswa berdasarkan data seperti IPK tiap semester, kehadiran, dan informasi demografis.

2. Mengidentifikasi fitur-fitur penting yang memiliki kontribusi signifikan terhadap kemungkinan drop out.

3. Memberikan wawasan dan rekomendasi berbasis data kepada pihak kampus atau akademik untuk mencegah drop out secara proaktif.

4. Menyediakan antarmuka prediksi (misalnya dalam bentuk dashboard atau form input) yang mudah digunakan untuk analisis calon mahasiswa berisiko.

### Solution Statements

1. Menggunakan beberapa algoritma machine learning seperti Support Vector Machine (SVM), Decision Tree, dan Random Forest untuk membandingkan performa dalam memprediksi status drop out.

2. Menerapkan teknik preprocessing seperti Label Encoding, One-Hot Encoding, dan Feature Alignment agar data dapat diproses secara konsisten.

3. Menggunakan confusion matrix, akurasi, precision, recall, dan F1-score untuk mengevaluasi performa masing-masing model.

4. Menyimpan model dan encoder dalam format .joblib agar dapat digunakan ulang dalam proses inferensi pada data baru.


## Data Understanding
Dataset yang digunakan adalah data dummy yang di buat sendiri

Dataset ini terdiri dari 600 baris dan 17 kolom yang menggambarkan berbagai atribut mahasiswa, mulai dari data akademik, informasi pribadi, hingga partisipasi dalam kegiatan non-akademik.

Tidak terdapat kolom-kolom missing values pada data

Data duplikat tidak ditemukan pada data

Outlier terdeteksi pada kolom :

1. ipk_sem1
2. ipk_sem2
3. ipk_sem3
4. ipk_sem4

Berikut adalah penjelasan singkat mengenai fitur/variabel dalam dataset prediksi mahasiswa Drop Out:

- id_mahasiswa: ID unik untuk masing-masing mahasiswa sebagai identifikasi individual dalam dataset.

- ipk_sem1: Indeks Prestasi Kumulatif (IPK) mahasiswa pada semester 1.

- ipk_sem2: Indeks Prestasi Kumulatif (IPK) mahasiswa pada semester 2.

- ipk_sem3: Indeks Prestasi Kumulatif (IPK) mahasiswa pada semester 3.

- ipk_sem4: Indeks Prestasi Kumulatif (IPK) mahasiswa pada semester 4.

- kehadiran_rata2: Persentase rata-rata kehadiran mahasiswa dalam kegiatan akademik.

- matkul_diulang: Jumlah mata kuliah yang pernah diulang oleh mahasiswa karena tidak lulus atau nilai tidak memuaskan.

- prodi: Program studi atau jurusan tempat mahasiswa terdaftar (misalnya Teknik Informatika, Ilmu Komunikasi, dll).

- jenis_kelamin: Jenis kelamin mahasiswa (Laki-laki atau Perempuan).

- aktivitas_lms: Tingkat aktivitas mahasiswa di platform Learning Management System (LMS), seperti aktif, sangat aktif, atau tidak aktif.

- status_pekerjaan: Status pekerjaan mahasiswa (bekerja, tidak bekerja).

- beban_kerja: Jumlah jam kerja per minggu jika mahasiswa bekerja (0 jika tidak bekerja).

- pendapatan_ortu: Estimasi rata-rata pendapatan orang tua mahasiswa (dalam satuan juta rupiah per bulan).

- pendidikan_ortu: Tingkat pendidikan terakhir orang tua (misalnya SMP, SMA, S1, S2).

- lokasi_tinggal: Lokasi tempat tinggal mahasiswa selama kuliah (misalnya kota, desa, atau kos/kontrak).

- keterlibatan_organisasi: Tingkat keterlibatan mahasiswa dalam organisasi kampus (aktif, tidak aktif, atau tidak ikut).

- status_DO: Label target, yaitu apakah mahasiswa termasuk kategori Drop Out (Ya) atau Tidak (Tidak).

**exploratory data analysis**:

**Boxplot Visualization**
Untuk mempermudah visualisasi data, maka di feature dibagi menjadi categorical_feature dan numerical_feature.

![Image](https://github.com/user-attachments/assets/56972c34-a26e-43b0-bff2-c89786d89754)

![Image](https://github.com/user-attachments/assets/8b65818f-28a1-405c-a335-7fc47679f59e)

![Image](https://github.com/user-attachments/assets/d58df53f-d531-4676-94a4-8152614fb4c0)

![Image](https://github.com/user-attachments/assets/890f6f52-cff4-4051-964d-ade1c19c8dc7)

Untuk rata-rata distribusi box plot dapat dilihat sebagai berikut:

- ipk_sem1: Rata-rata (median) nilai IPK pada semester 1 berada di sekitar angka 2.9 hingga 3.0. Mayoritas data (50% dari tengah) berada dalam rentang IPK sekitar 2.6 hingga 3.3.
- ipk_sem2: Rata-rata (median) nilai IPK pada semester 2 berada di sekitar angka 3.0 hingga 3.1. Mayoritas data (50% dari tengah) berada dalam rentang IPK sekitar 2.7 hingga 3.4.
- ipk_sem3: Rata-rata (median) nilai IPK pada semester 3 berada di sekitar angka 3.0 hingga 3.1. Mayoritas data (50% dari tengah) berada dalam rentang IPK sekitar 2.7 hingga 3.4.
- ipk_sem4: Rata-rata (median) nilai IPK pada semester 4 berada di sekitar angka 3.0 hingga 3.1. Mayoritas data (50% dari tengah) berada dalam rentang IPK sekitar 2.7 hingga 3.4.

### EDA - Univariate Analysis.

#### Univariate Analysis - Numerical Feature

![image](https://github.com/user-attachments/assets/8fd3ed83-c61b-437e-af93-1c8494007f56)

Gambar ini menampilkan distribusi dari fitur numerik mahasiswa dalam bentuk histogram. Setiap subplot menggambarkan sebaran nilai dari satu fitur. Berikut adalah penjelasan tiap fitur:

1. ipk_sem1, ipk_sem2, ipk_sem3, ipk_sem4
   
- Distribusi IPK dari semester 1 hingga 4.

- Sebagian besar mahasiswa memiliki IPK di antara 2.5 – 3.5.

- Distribusinya cenderung normal (berbentuk lonceng) namun sedikit condong ke kiri (left-skewed).

- Menunjukkan performa akademik cenderung stabil dari semester 1 hingga 4.

2. kehadiran_rata2
   
- Persentase kehadiran kuliah.

- Sebagian besar mahasiswa memiliki tingkat kehadiran di atas 70%, banyak yang mendekati 100%.

- Ada juga yang rendah (<60%), tetapi lebih jarang.

- Ini bisa berkorelasi positif dengan IPK.

3. matkul_diulang
   
- Jumlah mata kuliah yang diulang oleh mahasiswa.

- Banyak mahasiswa yang mengulang 1–4 matkul.

- Mahasiswa yang tidak mengulang (0) cukup banyak, dan sedikit yang mengulang hingga 5 matkul.

4. beban_kerja
   
- Sebagian besar mahasiswa memiliki beban kerja 0, artinya tidak bekerja sambil kuliah.

- Sisanya tersebar di angka 5 hingga 40 jam/minggu, tetapi jumlahnya jauh lebih kecil.

- Hal ini sejalan dengan data kategori sebelumnya (status_pekerjaan) yang menunjukkan banyak mahasiswa tidak bekerja.

5. pendapatan_ortu
   
- Data ini diskalakan (kemungkinan dalam jutaan atau rentang kategori).

- Distribusinya cukup merata dengan sedikit konsentrasi di tengah (sekitar 5–15).

- Hanya sedikit mahasiswa dari keluarga dengan pendapatan sangat rendah atau sangat tinggi.
- 

#### Univariate Analysis - Multivariate analysis

![image](https://github.com/user-attachments/assets/7f97abbd-60b7-4eb6-981d-ea557dc83237)

1. prodi (Program Studi)
   
Menunjukkan jumlah mahasiswa berdasarkan program studi.

- Tiga prodi terbanyak: Hukum, Sistem Informasi, dan Ilmu Komunikasi.

- Tiga prodi tersedikit: Matematika, Informatika, dan Kedokteran.

2. jenis_kelamin
   
- Mayoritas mahasiswa adalah Perempuan, melebihi jumlah Laki-laki.

3. aktivitas_lms (Aktivitas di Learning Management System)
   
- Kategori paling banyak: aktif

- Disusul oleh: sedang, sangat_aktif, pasif, dan paling sedikit kurang_aktif

- Menunjukkan sebagian besar mahasiswa cukup terlibat dalam LMS.

4. status_pekerjaan
   
- Mayoritas mahasiswa tidak bekerja.

- Sebagian bekerja paruh waktu, dan paling sedikit bekerja penuh waktu.

5. pendidikan_ortu (Tingkat Pendidikan Orang Tua)
   
- Mayoritas orang tua lulusan SMA, diikuti oleh SMP dan S1.

- Sangat sedikit yang hanya lulusan SD atau D3.

6. lokasi_tinggal
   
- Mahasiswa hampir seimbang antara yang tinggal di kota dan desa.

7. keterlibatan_organisasi
   
- Jumlah mahasiswa yang tidak aktif organisasi sedikit lebih banyak daripada yang aktif.

8. status_DO (Drop Out)
   
- Mayoritas mahasiswa tidak DO.

- Ada sebagian mahasiswa yang DO, namun jumlahnya lebih kecil secara signifikan.

#### Univariate Analysis - Multivariate analysis

![image](https://github.com/user-attachments/assets/258c0c82-482f-41e5-a294-bd0e3eddffeb)

Gambar ini menampilkan rata-rata jumlah mata kuliah yang diulang (matkul_diulang) terhadap berbagai variabel kategorikal. Setiap subplot merupakan hasil agregasi rata-rata untuk masing-masing kategori, yang membantu kita memahami faktor-faktor yang mungkin berkaitan dengan pengulangan mata kuliah.

Berikut adalah interpretasi tiap subplot:

1. Program Studi (prodi)
   
- Prodi seperti Teknik Elektro, Teknik Sipil, dan Sistem Informasi memiliki rata-rata matkul_diulang tertinggi (>2.5).

- Prodi seperti Matematika dan Manajemen menunjukkan rata-rata terendah.

- Ini bisa mengindikasikan tingkat kesulitan atau beban akademik berbeda antar prodi.

2. Jenis Kelamin
   
- Mahasiswa laki-laki cenderung mengulang lebih banyak mata kuliah dibanding perempuan.

- Perbedaan ini tidak terlalu besar tetapi konsisten.

3. Aktivitas LMS
   
- Mahasiswa yang sangat aktif atau aktif di LMS cenderung lebih sedikit mengulang dibanding yang pasif dan kurang aktif.

- Ini menunjukkan bahwa keterlibatan dalam pembelajaran online berkorelasi dengan performa akademik.

4. Status Pekerjaan
   
- Mahasiswa dengan pekerjaan penuh waktu mengulang paling banyak mata kuliah.

- Yang tidak bekerja mengulang paling sedikit, menunjukkan bahwa beban kerja luar kampus mempengaruhi performa.

5. Pendidikan Orang Tua
   
- Mahasiswa dari orang tua berpendidikan S1 dan S2 cenderung mengulang lebih sedikit.

- Yang berasal dari latar belakang pendidikan D3, SD, atau SMP cenderung lebih banyak mengulang.

- Ini bisa berkaitan dengan dukungan akademik atau sosial dari rumah.

6. Lokasi Tinggal
   
- Tidak ada perbedaan mencolok antara tinggal di kota atau desa.

- Rata-rata matkul_diulang hampir sama.

7. Keterlibatan Organisasi
   
- Mahasiswa aktif di organisasi sedikit lebih banyak mengulang dibanding yang tidak aktif.

- Bisa jadi karena waktu belajar terbagi, atau karena beban non-akademik.

8. Status DO
   
- Mahasiswa yang terancam atau terkena DO memiliki rata-rata matkul_diulang yang lebih tinggi.

- Hal ini sangat masuk akal, karena banyaknya pengulangan bisa menyebabkan akumulasi nilai buruk dan akhirnya DO.


#### pair plot

![image](https://github.com/user-attachments/assets/e678857c-519b-4337-b97e-138573f8ba5a)

- Diagonal: Menampilkan distribusi (bentuk kurva) untuk setiap variabel tunggal (misalnya, ipk_norm1, kehadiran_rata2, pendapatan_ortu). Ini menunjukkan bagaimana nilai-nilai untuk setiap variabel tersebar.
  
- Non-Diagonal: Menampilkan plot sebar (scatter plot) untuk setiap pasangan variabel. Ini membantu kita melihat:
Korelasi: Apakah ada hubungan linier (positif, negatif, atau tidak ada) antara dua variabel? Misalnya, ipk_norm1 dan ipk_norm2 tampak berkorelasi positif.


#### Correlation Matrix (Heatmap)

![image](https://github.com/user-attachments/assets/3f178208-c7cb-43b9-bf52-415f39bed344)


- Korelasi antar fitur numerik terlihat sangat lemah secara umum, terutama terhadap fitur target matkul_diulang.

- Beberapa nilai korelasi spesifik:

- matkul_diulang vs ipk_sem3: 0.07 (sangat lemah, hampir tidak signifikan)

- matkul_diulang vs ipk_sem2: 0.06 (lemah positif)

- matkul_diulang vs ipk_sem1: -0.03 (lemah negatif)

- matkul_diulang vs kehadiran_rata2: -0.01 (tidak ada korelasi)

- matkul_diulang vs beban_kerja: -0.01 (tidak ada korelasi)

- matkul_diulang vs pendapatan_ortu: 0.02 (tidak signifikan)

  
## Data Preparation

Teknik yang Digunakan

Berikut adalah tahapan data preparation yang dilakukan sebelum membangun model machine learning:

### a. Type Casting

Melakukan konversi tipe data dari object (string) menjadi float pada kolom-kolom numerik seperti ipk_sem1, ipk_sem2, ipk_sem3, ipk_sem4, dan pendapatan_ortu. Proses ini melibatkan:

- Mengubah isi kolom menjadi string.

- Mengganti koma (,) menjadi titik (.) sebagai pemisah desimal.

- Menghapus spasi di awal atau akhir nilai.

- Mengonversi string hasil pembersihan ke tipe data numerik (float) menggunakan pd.to_numeric() dengan errors='coerce' agar nilai tidak valid menjadi NaN.


### b. Penghapusan Kolom yang Tidak Diperlukan

Kolom-kolom yang tidak relevan atau tidak berguna dalam proses pemodelan, seperti kolom identifikasi 'id_mahasiswa', dihapus.

### c. Penanganan Outlier

Outlier hanya ditangani pada kolom :

ipk_sem1, ipk_sem2, ipk_sem3 dan ipk_sem4.

Penanganan dilakukan menggunakan Winsorizing, yaitu mengganti nilai ekstrim di luar rentang IQR dengan batas bawah atau atas.

### d. Encoding Variabel Kategorikal

Fitur yang tidak memiliki banyak kategori dilakukan One-Hot Encoding, seperti pada kolom:

1.  jenis_kelamin
   
2.  aktivitas_lms

3.  status_pekerjaan

4.  pendidikan_ortu
   
5.  lokasi_tinggal
    
6.  keterlibatan_organisasi

Fitur yang memiliki banyak kategori dan kolom label dilakukan Label Encoding, misalnya pada kolom:

1. prodi
   
2. status_DO

setelah dilakukan proses encoding, jumlah kolom meningkat menjadi 30 kolom.

### e. Penyeimbangan Data dengan SMOTE

Karena dataset memiliki ketidakseimbangan kelas (jumlah data Tidak, dan Ya) dengan jumlah data :

Tidak : 438

Ya : 162

digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan data latih. dan jumlah data nya menjadi

Tidak : 438

Ya : 438

jumlah keseluruhan data juga berubah dari 600 menjadi 876

### f. Pembagian Dataset

Data dibagi menjadi data latih (training set) dan data uji (testing set) dengan rasio 80:20, menggunakan train_test_split. Pembagian ini dilakukan agar model dapat dievaluasi secara adil menggunakan data yang belum pernah dilihat sebelumnya.

### g. Normalisasi / Scaling Fitur Numerik

Fitur numerik dinormalisasi menggunakan metode StandardScaler agar memiliki distribusi dengan rata-rata 0 dan standar deviasi 1. Ini penting agar model tidak berat sebelah terhadap fitur dengan skala yang besar.

### Alasan Dilakukannya Data Preparation

Tahapan data preparation dilakukan untuk memastikan kualitas dan kesesuaian data dengan algoritma machine learning yang digunakan. Alasan masing-masing teknik adalah sebagai berikut:

**Type Casting**: Kolom-kolom tersebut secara semantik merupakan data numerik, namun awalnya terbaca sebagai string karena format penulisan (misalnya angka desimal menggunakan koma). Untuk memungkinkan analisis statistik, visualisasi, dan pemodelan machine learning, data harus berada dalam tipe numerik. Tanpa type casting, operasi matematis tidak bisa dilakukan dan potensi kesalahan saat training model menjadi tinggi.

**Penghapusan kolom**: Mengurangi noise dari fitur yang tidak informatif atau redundan.

**Penanganan outlier**: Menghindari bias dalam model akibat nilai ekstrem.

**Encoding**: Algoritma machine learning membutuhkan data numerik, sehingga fitur kategorikal harus diubah terlebih dahulu.

**SMOTE**: Membantu model belajar dari kelas minoritas dan meningkatkan performa klasifikasi.

**Pembagian data**: Agar model bisa dievaluasi secara valid.

**Normalisasi**: Membantu mempercepat proses training dan membuat model lebih stabil, terutama untuk algoritma yang sensitif terhadap skala data.


## Modeling

### Model yang Digunakan

Dalam proyek ini, tiga algoritma machine learning digunakan untuk membangun model klasifikasi churn pelanggan, yaitu: Support Vector Machine (SVM), Decision Tree,  Random Forest, ogistic Regression, Gradient Boosting Classifier, dan MLPClassifier (Multi-layer Perceptron). Masing-masing model dilatih menggunakan data training hasil dari tahap data preparation.

### Penjelasan Cara Kerja Setiap Model

#### Support Vector Machine (SVM)

SVM bekerja dengan mencari hyperplane terbaik yang memisahkan kelas-kelas data secara optimal. Tujuannya adalah memaksimalkan margin antara dua kelas data. Dalam kasus data yang tidak dapat dipisahkan secara linear, SVM menggunakan fungsi kernel untuk memetakan data ke dimensi yang lebih tinggi agar dapat dipisahkan.

- **Parameter yang digunakan:**
  - random_state=42 — Seed random untuk reproduksi hasil.
  
  - C=1.0 — Parameter regularisasi, trade-off antara margin dan kesalahan klasifikasi.
  
  - kernel='rbf' — Fungsi kernel yang digunakan (linear, poly, rbf, sigmoid).
  
  - degree=3 — Derajat polinomial untuk kernel 'poly'.
  
  - gamma='scale' — Kernel coefficient untuk 'rbf', 'poly', dan 'sigmoid'.
  
  - coef0=0.0 — Konstanta kernel independen untuk kernel 'poly' dan 'sigmoid'.
  
  - shrinking=True — Menggunakan heuristik shrinking untuk mempercepat pelatihan.
  
  - probability=False — Mengaktifkan prediksi probabilitas (memperlambat pelatihan).
  
  - tol=0.001 — Toleransi kriteria penghentian solver.
  
  - cache_size=200 — Ukuran cache kernel matrix dalam MB.
  
  - class_weight=None — Bobot kelas untuk penanganan ketidakseimbangan kelas.
  
  - verbose=False — Menampilkan log selama pelatihan.
  
  - max_iter=-1 — Maksimal iterasi solver (-1 berarti tidak terbatas).
  
  - decision_function_shape='ovr' — Bentuk fungsi keputusan ('ovr' = one-vs-rest, 'ovo' = one-vs-one).
  
  - break_ties=False — Memecahkan ties dalam prediksi saat decision_function_shape='ovr'.

#### Decision Tree

Decision Tree membagi data berdasarkan fitur yang memberikan informasi paling tinggi (menggunakan metrik seperti gini atau entropy) untuk memisahkan kelas. Model ini membentuk struktur pohon di mana setiap node merepresentasikan keputusan berdasarkan nilai fitur tertentu.

- **Parameter yang digunakan:**
  
  - random_state=42 — Seed random untuk reproduksi hasil.
    
  - criterion='gini' — Fungsi pengukuran kualitas split ('gini' atau 'entropy').
    
  - splitter='best' — Strategi pemilihan split ('best' atau 'random').
    
  - max_depth=None — Kedalaman maksimum pohon (None berarti tidak dibatasi).
    
  - min_samples_split=2 — Minimal jumlah sampel untuk melakukan split.
    
  - min_samples_leaf=1 — Minimal jumlah sampel di daun.
    
  - min_weight_fraction_leaf=0.0 — Fraksi minimal bobot sampel di daun.
    
  - max_features=None — Jumlah fitur yang dipertimbangkan saat mencari split terbaik.
    
  - max_leaf_nodes=None — Maksimal jumlah daun (None berarti tidak dibatasi).
    
  - min_impurity_decrease=0.0 — Minimal pengurangan impuritas untuk melakukan split.
    
  - class_weight=None — Bobot kelas untuk penanganan ketidakseimbangan kelas.
    
  - ccp_alpha=0.0 — Parameter kompleksitas pruning post-pruning.

#### Random Forest

Random Forest merupakan ensemble model yang terdiri dari banyak pohon keputusan (Decision Trees). Setiap pohon dilatih menggunakan subset acak dari data dan fitur. Hasil akhir diperoleh dengan majority voting dari seluruh pohon. Teknik ini membantu mengurangi overfitting dan meningkatkan akurasi.

- **Parameter yang digunakan:**
  
  - random_state=42 — Seed random untuk reproduksi hasil.
    
  - n_estimators=100 — Jumlah pohon dalam hutan.
    
  - criterion='gini' — Fungsi pengukuran kualitas split ('gini' atau 'entropy').
    
  - max_depth=None — Kedalaman maksimum pohon (None berarti tidak dibatasi).
    
  - min_samples_split=2 — Minimal jumlah sampel untuk melakukan split.
    
  - min_samples_leaf=1 — Minimal jumlah sampel di daun.
    
  - min_weight_fraction_leaf=0.0 — Fraksi minimal bobot sampel di daun.
    
  - max_features='auto' — Jumlah fitur yang dipertimbangkan saat mencari split terbaik ('auto', 'sqrt', 'log2', atau int/float).
    
  - max_leaf_nodes=None — Maksimal jumlah daun (None berarti tidak dibatasi).
    
  - min_impurity_decrease=0.0 — Minimal pengurangan impuritas untuk melakukan split.
    
  - bootstrap=True — Menggunakan bootstrap sampling untuk membuat pohon.
    
  - oob_score=False — Menggunakan out-of-bag samples untuk estimasi akurasi.
    
  - n_jobs=None — Jumlah core CPU untuk paralelisasi (None berarti 1).
    
  - verbose=0 — Kontrol keluaran log selama pelatihan.
    
  - warm_start=False — Melanjutkan pelatihan dari model sebelumnya.
    
  - class_weight=None — Bobot kelas untuk penanganan ketidakseimbangan kelas.
    
  - ccp_alpha=0.0 — Parameter kompleksitas pruning post-pruning.
    
  - max_samples=None — Jumlah sampel untuk bootstrap (None = semua data).


**Logistic Regression**
Logistic Regression adalah model linier yang digunakan untuk klasifikasi. Ia memprediksi probabilitas suatu sampel termasuk ke dalam suatu kelas berdasarkan fungsi logit (sigmoid) dari kombinasi linier fitur input. Cocok untuk klasifikasi biner dan multikelas.

- **Parameter yang digunakan:**

- random_state=42 — Untuk memastikan hasil yang konsisten.

- solver='lbfgs' — Algoritma optimisasi, cocok untuk dataset kecil hingga menengah.

- multi_class='auto' — Menyesuaikan strategi klasifikasi multikelas secara otomatis.

- max_iter=1000 — Batas maksimum iterasi selama pelatihan.

**Gradient Boosting Classifier**
Gradient Boosting adalah teknik ensemble yang membangun model secara bertahap, di mana setiap model baru berusaha mengoreksi kesalahan model sebelumnya. Model dasarnya adalah Decision Tree, dan tiap iterasi mengoptimalkan fungsi loss dengan pendekatan gradient descent.

- **Parameter yang digunakan:**

- n_estimators=100 — Jumlah pohon dalam boosting.

- learning_rate=0.1 — Menentukan kontribusi masing-masing pohon terhadap model akhir.

- max_depth=3 — Kedalaman maksimum tiap pohon.

- random_state=42 — Untuk reproduksibilitas hasil.

- subsample=1.0 — Fraksi sampel yang digunakan di setiap iterasi boosting.

- loss='deviance' — Fungsi loss yang digunakan (log loss untuk klasifikasi).

**MLPClassifier (Multi-Layer Perceptron)**
MLPClassifier adalah neural network feed-forward yang terdiri dari satu atau lebih lapisan tersembunyi. Model ini belajar menggunakan backpropagation dan sangat cocok untuk memodelkan relasi non-linear dalam data.

- **Parameter yang digunakan:**

- hidden_layer_sizes=(100,) — Ukuran lapisan tersembunyi (1 layer dengan 100 neuron).

- activation='relu' — Fungsi aktivasi antar neuron.

- solver='adam' — Optimizer berbasis stochastic gradient descent.

- max_iter=300 — Jumlah iterasi maksimum.

- random_state=42 — Untuk hasil yang dapat direproduksi.

- early_stopping=True — Menghentikan pelatihan saat validasi tidak membaik.



### Kelebihan dan Kekurangan Model

#### Support Vector Machine (SVM)

**Kelebihan:**

1. Akurat untuk margin yang jelas – SVM bekerja sangat baik jika terdapat batas pemisah (margin) yang jelas antara dua kelas.
   
2. Efektif di ruang berdimensi tinggi – SVM tetap bekerja baik meski jumlah fitur sangat banyak.
   
3. Bisa digunakan untuk data non-linear – Dengan penggunaan kernel trick, SVM mampu memisahkan data yang tidak linear.
   
4. Robust terhadap overfitting – Terutama jika jumlah fitur lebih banyak dari jumlah sampel.

**Kekurangan:**

1. Kurang cocok untuk dataset besar – Proses training-nya sangat lambat ketika jumlah data besar.
   
2. Butuh tuning parameter yang hati-hati – Pemilihan kernel, nilai C, dan gamma sangat memengaruhi performa.
   
3. Kurang interpretatif – Tidak sejelas Decision Tree dalam menunjukkan logika keputusan.

#### Decision Tree

**Kelebihan:**

1. Mudah dipahami dan divisualisasikan – Model ini menyerupai pohon keputusan yang bisa dengan mudah dipahami oleh manusia.
   
2. Tidak perlu banyak pra-pemrosesan data – Tidak memerlukan normalisasi atau scaling.
   
3. Bisa menangani data numerik dan kategorikal – Tanpa perlakuan khusus.
   
4. Cepat dilatih – Umumnya memiliki waktu pelatihan yang cepat.

**Kekurangan:**

1. Rentan terhadap overfitting – Apalagi jika pohon terlalu dalam atau tidak dipangkas.
   
2. Kurang stabil – Perubahan kecil pada data dapat menghasilkan struktur pohon yang sangat berbeda.
   
3. Cenderung bias terhadap fitur dengan banyak level/kategori.

#### Random Forest

**Kelebihan:**

1. Lebih akurat dari Decision Tree tunggal – Karena merupakan kumpulan dari banyak Decision Tree (ensemble).
   
2. Kurang rentan overfitting – Menggunakan metode bagging untuk mengurangi variansi.
   
3. Dapat menangani data besar dan fitur banyak – Cocok untuk dataset kompleks.
   
4. Memberikan estimasi pentingnya fitur (feature importance).

**Kekurangan:**

1. Kurang interpretatif – Sulit memahami keseluruhan logika karena banyak pohon.
   
2. Waktu pelatihan dan prediksi lebih lama – Dibandingkan dengan model sederhana seperti Decision Tree.
   
3. Model lebih besar – Membutuhkan lebih banyak memori dan ruang penyimpanan.

#### Logistic Regression

**Kelebihan:**

1. Mudah dipahami dan diinterpretasikan – Termasuk model linier yang transparan.

2. Cepat dilatih – Cocok untuk baseline model.

3. Bisa digunakan untuk probabilistik output – Berguna untuk prediksi peluang.

Kekurangan:

1. Tidak bisa menangani relasi non-linear secara langsung – Performa menurun jika fitur tidak linear terhadap target.

2. Sensitif terhadap multikolinearitas – Fitur yang saling berkorelasi tinggi memengaruhi performa.

3. Perlu fitur numerik – Harus dilakukan encoding untuk fitur kategorikal.

#### Gradient Boosting Classifier

**Kelebihan:**

1. Sangat akurat – Salah satu metode terbaik untuk banyak masalah klasifikasi.

2. Mampu menangani fitur numerik dan kategorikal.

3. Bisa menangani data tidak seimbang dengan baik.

**Kekurangan:**

1. Proses pelatihan lambat – Karena model dibangun secara bertahap.

2. Mudah overfitting – Jika tidak disetel dengan benar (terlalu banyak pohon atau terlalu dalam).

3. Perlu tuning parameter secara hati-hati – Seperti learning_rate, n_estimators, dll.

#### MLPClassifier (Neural Network)

**Kelebihan:**

1. Mampu memodelkan relasi non-linear yang kompleks.

2. Cocok untuk berbagai jenis data – Termasuk data numerik dan hasil embedding.

3. Bisa digunakan untuk multi-class classification dengan probabilitas prediksi.

**Kekurangan:**

1. Memerlukan lebih banyak data dan waktu pelatihan.

2. Kurang interpretatif – Model seperti “black box”.

3. Sensitif terhadap parameter dan skala fitur – Harus distandardisasi terlebih dahulu.


## Evaluation

### Metrik Evaluasi yang Digunakan

Dalam proyek ini, digunakan empat metrik utama untuk mengukur performa model klasifikasi dalam memprediksi Drop Out pada mahasiswa (Tidak, atau Ya), yaitu:

1. Accuracy (Akurasi)
   
   Definisi: Mengukur seberapa banyak prediksi model yang benar dibandingkan dengan seluruh data.

   Cara Kerja: Jumlah prediksi benar dibagi total prediksi.
   
   Catatan: Cocok untuk data seimbang, tapi bisa menyesatkan jika data tidak seimbang.

3. Precision (Presisi)
   
   Definisi: Seberapa tepat model dalam memprediksi suatu kelas tertentu — dari semua prediksi positif, berapa yang benar.
   
   Cara Kerja: Jumlah prediksi benar untuk kelas dibagi total prediksi kelas tersebut.
   
   Catatan: Penting jika false positives berdampak besar, misal mengira pelanggan churn padahal tidak.

5. Recall (Sensitivitas)
   
   Definisi: Seberapa baik model menemukan seluruh data yang benar-benar termasuk suatu kelas.
   
   Cara Kerja: Jumlah prediksi benar untuk kelas dibagi total data sebenarnya dalam kelas tersebut.
   
   Catatan: Penting jika false negatives harus dihindari, misal tidak ingin melewatkan pelanggan churn.

7. F1-Score
   
   Definisi: Rata-rata harmonis dari precision dan recall, memberi keseimbangan antara keduanya.
   
   Cara Kerja: Tinggi jika precision dan recall sama-sama tinggi.
   
   Catatan: Cocok untuk data tidak seimbang.

9. Confusion Matrix (Matriks Kebingungan)
   
Definisi: Matriks yang menampilkan jumlah prediksi benar dan salah untuk setiap kelas.

Contoh Singkat Cara Kerja Metrik

### Perbandingan Hasil Evaluasi Model:

- Model Support Vector Machine (SVM)
  
  Metrik Nilai
  
  Akurasi: 0.8750
  
  Precision (weighted): 0.8799
  
  Recall (weighted): 0.8750
  
  F1 Score (weighted): 0.8746

  ![image](https://github.com/user-attachments/assets/80d02922-3911-403e-be6a-b5040471b6fd)


- Model Decision Tree
  
  Metrik Nilai
  
  Akurasi: 0.8125
  
  Precision (weighted): 0.8129
  
  Recall (weighted): 0.8125
  
  F1 Score (weighted): 0.8124

  ![image](https://github.com/user-attachments/assets/9ea03ca3-44fe-40f3-902d-2e70775f5c1d)


- Random Forest
  
  Metrik Nilai
  
  Akurasi: 0.9034
  
  Precision (weighted): 0.9035
  
  Recall (weighted): 0.9034
  
  F1 Score (weighted): 0.9034

![image](https://github.com/user-attachments/assets/492bac74-1cb2-424b-b3b7-90dd24ad8b06)

- Logistic Regression
  
  Metrik Nilai
  
  Akurasi: 0.8523
  
  Precision (weighted): 0.8569
  
  Recall (weighted): 0.8523
  
  F1 Score (weighted): 0.8518

![image](https://github.com/user-attachments/assets/506bdd67-db1d-4aa3-8a64-30e5763f4780)

- Gradient Boosting Classifier
  
  Metrik Nilai
  
  Akurasi: 0.8864
  
  Precision (weighted): 0.8866
  
  Recall (weighted): 0.8864
  
  F1 Score (weighted): 0.8863


![image](https://github.com/user-attachments/assets/673995e4-1751-485c-8cfe-5eb9d73dedd7)

- MLPClassifier (Multi-Layer Perceptron)
  
  Metrik Nilai
  
  Akurasi: 0.8750
  
  Precision (weighted): 0.8758
  
  Recall (weighted): 0.8750
  
  F1 Score (weighted): 0.8749

![image](https://github.com/user-attachments/assets/5d1426cf-be8b-46f5-b8b1-a6527e1407f1)





**Kesimpulan Evaluasi**

Model Random Forest menunjukkan performa terbaik dengan hasil rata-rata antar kelas:

Metrik Nilai

  Akurasi: 0.9034
  
  Precision (weighted): 0.9035
  
  Recall (weighted): 0.9034
  
  F1 Score (weighted): 0.9034

Model ini mampu melakukan klasifikasi status mahasiswa dengan cukup baik. Akurasi yang tinggi menunjukkan bahwa prediksi model secara umum tepat, sementara nilai presisi dan recall yang seimbang mengindikasikan bahwa model mampu mengidentifikasi mahasiswa dropout maupun non-dropout secara proporsional. F1-score yang stabil menunjukkan bahwa model cukup andal dalam menangani ketidakseimbangan data antar kelas.

Dengan performa seperti ini, model dapat dijadikan alat bantu yang andal bagi institusi pendidikan dalam merancang strategi pencegahan dropout, seperti memberikan intervensi dini bagi mahasiswa yang berisiko tinggi untuk tidak melanjutkan studi.




