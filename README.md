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
1. Bagaimana cara membuat sistem rekomendasi aplikasi yang memberikan saran kepada pengguna berdasarkan kemiripan fitur aplikasi (seperti kategori)?

2. Bagaimana cara mengukur performa model sistem rekomendasi aplikasi secara kuantitatif?

### Goals
1. Menghasilkan rekomendasi aplikasi sebanyak Top-N rekomendasi yang relevan untuk pengguna berdasarkan kemiripan fitur aplikasi.

2. Mengukur performa model sistem rekomendasi menggunakan metrik evaluasi seperti precision@k, diversity score dan Intra-list Similarity (ILS) untuk menilai relevansi dan keberagaman rekomendasi.


 ### Solution statements
Solution Approach
Untuk mencapai tujuan yang telah ditetapkan, digunakan dua pendekatan sistem rekomendasi berikut:

1. Content-Based Filtering
Pendekatan ini merekomendasikan aplikasi berdasarkan kemiripan fitur konten antar aplikasi. Fitur-fitur yang digunakan antara lain: Category (kategori aplikasi), Rating dan Rating Count, Installs, Free atau Price, Size, Content Rating

Fitur-fitur tersebut diolah menjadi vektor fitur aplikasi, lalu dihitung tingkat kemiripannya menggunakan metode seperti cosine similarity. Sistem ini akan menyarankan aplikasi yang mirip dengan aplikasi yang sudah populer atau telah dipilih pengguna sebelumnya.

2. Popularity-Based Filtering (Sebagai Alternatif Collaborative Filtering)
Karena tidak tersedia data interaksi pengguna (seperti ID pengguna, histori klik, atau rating per pengguna), maka pendekatan collaborative filtering klasik tidak dapat diterapkan. Sebagai gantinya, digunakan pendekatan berbasis popularitas sebagai baseline:

Menampilkan aplikasi dengan jumlah instalasi, rating count, atau rating tertinggi sebagai rekomendasi umum.


## Data Understanding
Pada tahap ini, dilakukan pemahaman awal terhadap dataset yang digunakan dalam proyek sistem rekomendasi aplikasi. Informasi yang disampaikan mencakup jumlah data, kondisi data, serta penjelasan setiap fitur dalam dataset.
erdapat kolom-kolom yang memiliki missing values, antara lain:

App Name : 5

Rating : 22883

Rating Count : 22883

Installs : 107

Size : 196

Minimum Android : 2499

Data duplikat tidak ditemukan pada data

Outlier terdeteksi pada kolom

1. Rating
2. Rating Count
3. Price      

Dataset yang digunakan diperoleh dari sumber berikut: [Kaggle](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps)

Dataset ini terdiri dari 2.312.944 baris dan 24 kolom. Setiap baris merepresentasikan satu aplikasi yang tersedia di Google Play Store, sementara kolom-kolomnya berisi informasi terkait atribut aplikasi tersebut.

Berikut adalah penjelasan singkat mengenai fitur/variabel dalam dataset:
- App Name	       : Nama aplikasi seperti yang ditampilkan di Play Store.
- App Id	       : ID unik aplikasi di Play Store (biasanya berupa nama paket, seperti com.example.app).
- Category	       : Kategori aplikasi, seperti Games, Education, Shopping, dll.
- Rating	       : Nilai rata-rata penilaian pengguna (biasanya dari 1.0 hingga 5.0).
- Rating Count     : Jumlah total penilaian atau rating yang diberikan oleh pengguna.
- Installs	       : Jumlah pemasangan aplikasi (dalam format seperti "1,000+").
- Minimum Installs : Estimasi jumlah minimum instalasi berdasarkan data.
- Maximum Installs : Estimasi jumlah maksimum instalasi berdasarkan data.
- Free	           : Apakah aplikasi gratis (True) atau berbayar (False).
- Price	           : Harga aplikasi jika tidak gratis.
- Currency         : Mata uang yang digunakan untuk harga (misalnya USD, IDR).
- Size	           : Ukuran file aplikasi.
- Minimum Android  : Versi minimum Android yang dibutuhkan untuk menjalankan aplikasi.
- Developer Id     : ID unik pengembang di Play Store.
- Developer Website : Situs web resmi dari pengembang aplikasi (jika tersedia).
- Developer Email	: Alamat email kontak pengembang.
- Released	        : Tanggal pertama kali aplikasi dirilis di Play Store.
- Last Updated	    : Tanggal terakhir aplikasi diperbarui.
- Content Rating	: Rating konten aplikasi berdasarkan umur pengguna (seperti "Everyone", "Teen", dll).
- Privacy Policy	: URL ke kebijakan privasi aplikasi.
- Ad Supported	    : Menunjukkan apakah aplikasi mendukung iklan (True/False).
- In App Purchases	: Menunjukkan apakah aplikasi memiliki pembelian dalam aplikasi (True/False).
- Editors Choice	: Apakah aplikasi mendapat label “Pilihan Editor” dari Google Play.
- Scraped Time      : Waktu saat data aplikasi ini dikumpulkan/scraped.

  Tetapi karena beberapa pertimbangan sebagai berikut:

1. Kolom-kolom tersebut memiliki nilai informatif yang tinggi dalam merepresentasikan popularitas, kualitas, serta segmentasi aplikasi.
2. Kolom yang dipilih cenderung memiliki data yang lebih lengkap dan bersih, sehingga dapat meminimalkan risiko ketidaksesuaian atau missing value pada tahap pemrosesan data.
3. Fitur-fitur tersebut lebih mudah untuk diolah dan dianalisis, baik dalam analisis statistik deskriptif maupun dalam proses modeling seperti sistem rekomendasi

maka Dalam proses pemilihan fitur, dipilih 11 kolom utama dari dataset
- App Name	       : Nama aplikasi seperti yang ditampilkan di Play Store.
- App Id	       : ID unik aplikasi di Play Store (biasanya berupa nama paket, seperti com.example.app).
- Category	       : Kategori aplikasi, seperti Games, Education, Shopping, dll.
- Rating	       : Nilai rata-rata penilaian pengguna (biasanya dari 1.0 hingga 5.0).
- Rating Count     : Jumlah total penilaian atau rating yang diberikan oleh pengguna.
- Installs	       : Jumlah pemasangan aplikasi (dalam format seperti "1,000+").
- Free	           : Apakah aplikasi gratis (True) atau berbayar (False).
- Price	           : Harga aplikasi jika tidak gratis.
- Size	           : Ukuran file aplikasi.
- Minimum Android  : Versi minimum Android yang dibutuhkan untuk menjalankan aplikasi.
jadi Dataset nya terdiri dari 2.312.944 baris dan 11 kolom.


**exploratory data analysis**:
- Ini beberapa tahapan yang dilakukan untuk memahami data :
![Image](https://github.com/user-attachments/assets/2c34c81b-27bd-4021-b344-5ae14dd39fc9)
- **Histogram**
- Rating: Terlihat mayoritas nilai Rating berkisar di antara 4 dan 5, artinya sebagian besar aplikasi mendapatkan penilaian bagus.
- Rating Count: Distribusinya sangat condong ke kiri (right-skewed), sebagian besar aplikasi hanya memiliki sedikit rating.
- Price: Hampir semua aplikasi gratis (harga = 0), hanya sebagian kecil yang memiliki harga tinggi, dan ini menunjukkan adanya outlier.

![Image](https://github.com/user-attachments/assets/f362fb1c-7329-4316-af05-848f92ee6a1b)
- **Pairplot**
- airplot menunjukkan hubungan antar fitur numerik secara visual.
- Hampir semua kombinasi fitur memiliki pola menyebar luas (tidak linear), dengan beberapa titik ekstrem (outlier).
- Rating vs Rating Count menunjukkan bahwa meski banyak aplikasi dengan rating bagus, tidak semua mendapat banyak ulasan.
- Price vs fitur lain terlihat tidak memiliki hubungan yang kuat dan dipenuhi nilai nol.
![Image](https://github.com/user-attachments/assets/370fbd33-e2d2-44e6-afb1-cb14456db1d6)
- **Correlation Matrix (Heatmap)**
- Korelasi antar fitur sangat lemah:
a. Rating vs Rating Count: 0.01 (hampir tidak ada korelasi)
b. Rating vs Price: -0.00 (sangat lemah negatif)
c. Rating Count vs Price: -0.00 juga.
- Artinya, fitur-fitur numerik ini tidak saling bergantung secara linear, dan bisa dianggap sebagai fitur independen dalam model sederhana.
- Ini juga memberi sinyal bahwa kamu perlu teknik feature engineering atau transformasi lain jika ingin membangun model yang lebih baik.
  
## Data Preparation
1. Feature Selection
   
**Proses:** Memilih 11 kolom penting dari dataset asli Google-Playstore.csv yang relevan untuk analisis dan sistem rekomendasi, antara lain App Name, App Id, Category, Rating, Rating Count, Installs, Free, Price, Size, dan Minimum Android.

**Alasan:** Kolom-kolom ini dipilih karena memiliki nilai informatif tinggi terhadap popularitas dan segmentasi aplikasi, serta tersedia secara konsisten dan mudah diolah untuk modeling.

2. Missing Value Handling

**Proses:**
- Menghapus baris dengan nilai 0 pada kolom Rating dan Rating Count, karena dianggap sebagai data tidak valid.
- Menghapus baris yang memiliki nilai null pada kolom-kolom kunci seperti App Name, Rating, Rating Count, Size, dan Minimum Android.
  
**Alasan:** Data kosong atau nol pada kolom penting dapat menyebabkan bias atau kesalahan pada analisis dan model, sehingga perlu dihapus untuk meningkatkan kualitas data.

3. Duplicate Handling
   
**Proses:** Mengecek jumlah data duplikat pada dataset dan di data tidak ditemukan data yang duplikat.

**Alasan:** Mencegah model terpengaruh oleh entri ganda yang bisa menyebabkan informasi berulang.

4. Outlier Handling (Winsorizing)
   
**Proses:** Menggunakan teknik winsorizing pada kolom Rating dan Rating Count dengan menggantikan nilai ekstrem di luar batas IQR dengan nilai batas atas/bawah.

**Alasan:** Mengurangi pengaruh outlier ekstrem yang dapat merusak distribusi data dan menyebabkan model bias.

5. Data Visualization & Exploratory Analysis
   
**Proses:**
Visualisasi boxplot untuk kolom numerik guna melihat sebaran dan outlier.
Visualisasi histogram dan pairplot untuk melihat distribusi data.
Matriks korelasi untuk mengetahui hubungan antar fitur numerik.

**Alasan:** Untuk memahami pola data, mendeteksi potensi masalah, dan menyesuaikan strategi pemodelan.

6. Sampling
    
**Proses:** Mengambil sampel acak sebanyak 10.000 baris dari dataset bersih.

**Alasan:** Untuk mempercepat proses komputasi, terutama saat menghitung similarity matrix dan pelatihan model.

7. Text Vectorization
    
**Proses:** Menggunakan TF-IDF Vectorizer pada kolom Category untuk mengubah data teks menjadi representasi numerik.

**Alasan:** Karena sistem rekomendasi ini berbasis content-based filtering, diperlukan representasi numerik dari kategori aplikasi agar kemiripan antar aplikasi dapat dihitung.

## Modeling
Tahapan ini membahas pembangunan model sistem rekomendasi yang dirancang untuk membantu pengguna menemukan aplikasi yang relevan berdasarkan preferensi atau kesamaan konten.

1. Pendekatan 1: Content-Based Filtering dengan TF-IDF + Cosine Similarity
Pada pendekatan pertama, digunakan teknik content-based filtering yang memanfaatkan informasi dari fitur kategori (Category) setiap aplikasi. Pendekatan ini didasarkan pada asumsi bahwa aplikasi yang mirip dari segi kategori akan relevan satu sama lain.

**Langkah-Langkah:**
- Representasi kategori diubah menjadi vektor numerik menggunakan TF-IDF Vectorizer.

- Kemudian dihitung cosine similarity antar aplikasi berdasarkan hasil vektorisasi.

- Untuk setiap aplikasi yang dimasukkan, sistem akan merekomendasikan Top-10 aplikasi serupa berdasarkan skor kemiripan tertinggi.

a. Kelebihan:
- Tidak bergantung pada data pengguna atau interaksi historis.
- Bisa merekomendasikan item baru (cold-start friendly, selama deskripsinya tersedia).
- Mudah diimplementasikan dan ditafsirkan.

b. Kekurangan:
- Rekomendasi terbatas hanya pada konten yang tersedia (hanya berdasarkan kategori).
- Tidak bisa menangkap preferensi pengguna secara personal.

2. Pendekatan 2: Popularity-Based Filtering
Sebagai pembanding, juga dibuat pendekatan berbasis popularitas. Pendekatan ini menyarankan aplikasi dengan jumlah pemasangan (Installs), rating tertinggi (Rating), dan jumlah rating (Rating Count) terbanyak.

**Langkah-Langkah:**
- Aplikasi diurutkan berdasarkan gabungan tiga metrik: Installs, Rating Count, dan Rating.
- Diambil Top-10 aplikasi terpopuler sebagai rekomendasi universal untuk semua pengguna.

a. Kelebihan:
- Sederhana dan efektif dalam banyak kasus.
- Relevan bagi pengguna baru yang belum memiliki preferensi khusus.

b. Kekurangan:
- Tidak personalisasi (semua pengguna mendapat rekomendasi yang sama).
- Rentan bias terhadap aplikasi lama dan terkenal.
- 
**Top-N Recommendation Output**
Berikut adalah contoh hasil rekomendasi untuk beberapa aplikasi:
![Image](https://github.com/user-attachments/assets/78d826c6-6a7f-41e8-864b-9d1337b788b5)

Sistem merekomendasikan beberapa aplikasi lain yang berada dalam kategori Shopping dan memiliki kemiripan fitur dengan Supermarket Deal Calculator. Beberapa aplikasi yang direkomendasikan antara lain Shopping List Barcode Scanner, FidMe Loyalty Cards & Deals at Grocery Supermarkets, hingga Toy Store App. Rekomendasi ini relevan karena memiliki fungsi serupa dalam membantu aktivitas berbelanja dan pengelolaan produk.

Rekomendasi untuk aplikasi Happy birth:
Untuk aplikasi bertema hiburan seperti Happy birth, sistem menghasilkan rekomendasi dari kategori Entertainment seperti LAVA TV, Pelet Online Prank, dan Among us mod MCPE 2021. Aplikasi-aplikasi ini memiliki konten hiburan yang bervariasi, dari video lucu hingga mod game, yang dinilai sesuai dengan selera pengguna aplikasi Happy birth.

Rekomendasi untuk aplikasi Fire Truck Simulator 3D:
Aplikasi ini termasuk dalam kategori Simulation, sehingga sistem memberikan rekomendasi aplikasi simulasi lainnya seperti Car Driving, Armed Air Forces, dan Real Sports Car Game. Aplikasi-aplikasi ini menawarkan pengalaman interaktif dan simulasi kendaraan atau aktivitas serupa yang sesuai dengan konsep dari Fire Truck Simulator 3D.

![Image](https://github.com/user-attachments/assets/4051d424-ad3e-48da-b72d-16d69360f044)

Hasil yang ditampilkan menunjukkan 10 aplikasi yang dianggap paling populer, seperti:
- Contacts dari Google, dengan lebih dari 500 juta pemasangan.
- Книга Вслух. Аудиокниги, aplikasi audiobook dengan rating tinggi (4.9).
- Lose Belly Fat Workouts dan Taiwan Drivers License Test, yang meskipun jumlah installs lebih kecil, tetap masuk karena rating dan jumlah ratingnya tinggi.
Aplikasi-aplikasi ini berasal dari berbagai kategori, menunjukkan bahwa popularitas tidak hanya bergantung pada satu jenis aplikasi, tetapi juga kualitas.

## Evaluation
1. ### Evaluasi Pendekatan Content-Based Filtering
Evaluasi sistem rekomendasi dilakukan untuk mengukur sejauh mana hasil rekomendasi yang dihasilkan oleh model dapat dianggap relevan, bervariasi, dan tidak terlalu seragam. Dalam proyek ini, digunakan tiga metrik evaluasi utama yang umum digunakan dalam sistem rekomendasi:

#### 1. Precision@K
**Definisi:**
Precision@K mengukur proporsi item yang relevan dari total item yang direkomendasikan sebanyak K.
**Formula:**
![Image](https://github.com/user-attachments/assets/3f821a9b-69bd-459e-9494-b27e886af4bc)
**Implementasi:**
Sebuah rekomendasi dianggap relevan jika aplikasi yang direkomendasikan memiliki kategori yang sama dengan aplikasi input.

**Hasil:**

![Image](https://github.com/user-attachments/assets/720af300-3f7d-4326-ac10-44ced8b6bc1b)

Untuk beberapa aplikasi seperti:
- Supermarket Deal Calculator: Precision@10 =  1.00
- Happy birth: Precision@10 =  1.00
- 40 Hadist Peristiwa Akhir Zaman: Precision@10 = 1.00

Kesimpulan:
Sistem berhasil merekomendasikan aplikasi yang sangat relevan (semua rekomendasi sesuai kategori aplikasi awal), terbukti dari nilai precision yang mencapai 100%.



#### 2. Diversity Score
**Definisi:**
Diversity score mengukur seberapa beragam kategori dari aplikasi yang direkomendasikan. Semakin banyak kategori berbeda dalam daftar rekomendasi, semakin tinggi nilai diversitas.

![Image](https://github.com/user-attachments/assets/8ec0e5e9-c8f1-448e-9cd0-00e71d81c62f)

**Hasil:**

![Image](https://github.com/user-attachments/assets/e0e5ec79-db4e-4908-909d-0a758be93b37)

Supermarket Deal Calculator → Diversity Score: 0.10
Happy birth → Diversity Score: 0.10
40 Hadist Peristiwa Akhir Zaman → Diversity Score: 0.20

Kesimpulan:
Skor diversity tergolong rendah, yang artinya sistem cenderung merekomendasikan aplikasi dari kategori yang serupa. Hal ini bisa disebabkan oleh pendekatan content-based filtering yang mengutamakan kemiripan fitur.

#### 3. Intra-list Similarity (ILS)
**Definisi:**
ILS mengukur seberapa mirip aplikasi-aplikasi dalam daftar rekomendasi satu sama lain. Semakin rendah ILS, maka semakin beragam dan tidak terlalu seragam item rekomendasinya.

![Image](https://github.com/user-attachments/assets/2a2ad68d-20c5-4378-acce-bad03219cbf4)

**Hasil:**

![Image](https://github.com/user-attachments/assets/42e80aa9-b2c4-4a75-a0de-f42da840d7d0)

Supermarket Deal Calculator → ILS: 1.00
Happy birth → ILS: 1.00
40 Hadist Peristiwa Akhir Zaman → ILS: 1.00

Kesimpulan:
Nilai ILS yang tinggi menunjukkan bahwa aplikasi yang direkomendasikan sangat mirip satu sama lain. Ini bisa jadi menguntungkan dalam konteks pencarian aplikasi sejenis, namun bisa juga menjadi kelemahan jika pengguna menginginkan variasi yang lebih luas.

2. ### Evaluasi Popularity-Based Recommendation
   
![Image](https://github.com/user-attachments/assets/fbd9b68e-3c29-4e7b-a0a5-3ba546bf076d)

Pendekatan ini memberikan rekomendasi berdasarkan aplikasi yang memiliki jumlah pemasangan tertinggi, jumlah rating terbanyak, dan rating rata-rata tertinggi.

Statistik deskriptif untuk 10 aplikasi terpopuler menunjukkan bahwa:
- Rata-rata jumlah rating adalah 427 untuk setiap aplikasi, tanpa variasi.
- Nilai rating bervariasi dari 4.3 hingga 4.9, dengan rata-rata 4.77, dan standar deviasi 0.17, menandakan semua aplikasi sangat populer dan disukai.
Analisis korelasi antar metrik menunjukkan bahwa terdapat korelasi negatif yang kuat antara jumlah pemasangan dan rating (nilai korelasi sekitar -0.97). Artinya, aplikasi yang paling banyak diunduh tidak selalu mendapatkan rating tertinggi. Data ini memberikan wawasan bahwa popularitas berdasarkan jumlah pemasangan tidak selalu mencerminkan kualitas aplikasi berdasarkan penilaian pengguna.

Pendekatan ini tidak personal karena memberikan rekomendasi umum kepada semua pengguna, namun keunggulannya adalah cepat, sederhana, dan efektif dalam menampilkan aplikasi yang dikenal banyak orang.


**Kesimpulan Evaluasi**

Pendekatan content-based filtering menghasilkan rekomendasi yang sangat relevan dan sesuai kategori, dengan precision sempurna, tetapi rekomendasi cenderung kurang beragam karena semua aplikasi sangat mirip. Pendekatan ini sangat cocok untuk pengguna yang mencari aplikasi serupa berdasarkan minat atau kategori tertentu.

Sementara itu, pendekatan popularity-based recommendation memberikan daftar aplikasi yang secara umum paling populer dan memiliki rating tinggi. Meskipun kurang personal, pendekatan ini efektif digunakan untuk pengguna baru yang belum memiliki preferensi.

Jika dimungkinkan di masa mendatang, sistem hybrid yang menggabungkan kedua pendekatan dapat digunakan untuk menghasilkan rekomendasi yang lebih kuat, baik dari segi relevansi maupun keberagaman.



