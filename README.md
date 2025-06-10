# Laporan Proyek Machine Learning - Muhammad Elfikry

## Project Overview

Sistem rekomendasi merupakan komponen penting dalam dunia digital saat ini, terutama dalam industri media dan berita. Dengan banyaknya konten berita yang tersedia setiap harinya, pengguna kerap mengalami kesulitan menemukan berita yang relevan dengan minat mereka. Oleh karena itu, dibutuhkan sistem yang mampu memberikan rekomendasi berita yang sesuai dengan preferensi pengguna. Sistem rekomendasi memiliki peran penting dalam meningkatkan pengalaman pengguna. Sebagaimana dijelaskan oleh Ricci, Rokach, dan Shapira (2015), sistem rekomendasi dapat meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang menarik dan relevan, didukung dengan desain interaksi manusia-komputer yang baik sehingga pengguna menikmati penggunaan sistem tersebut.

Salah satu pendekatan sistem rekomendasi adalah *content-based filtering*, yang memanfaatkan kemiripan konten antar item (dalam hal ini, berita) untuk memberikan rekomendasi. Upreti, Sengar, Goel, dan Bahl (2025), menjelaskan bahwa sistem rekomendasi berbasis konten bekerja dengan menganalisis atribut-atribut seperti genre, penulis, dan kata kunci, kemudian mencocokkannya dengan riwayat bacaan atau preferensi yang dinyatakan. Proyek ini bertujuan membangun sistem rekomendasi berita berbasis *content-based filtering* dengan menggunakan *cosine similarity* terhadap fitur deskripsi berita pada dataset *BBC News*.

**Mengapa Masalah Ini Penting?**
- Pengguna cenderung kehilangan minat ketika tidak segera menemukan konten yang relevan.
- Rekomendasi yang personal dapat meningkatkan waktu keterlibatan pengguna pada platform berita.
- Meningkatkan kepuasan pengguna serta potensi monetisasi melalui iklan yang lebih tepat sasaran.

**Referensi**:
1. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
2. Upreti, M., Sengar, N., Goel, A., & Bahl, V. (2025). A book tracking and recommender system using machine learning algorithms. International Journal for Research in Applied Science & Engineering Technology (IJRASET), 13(5), 2377.

## Business Understanding

### Problem Statements
- Bagaimana memberikan rekomendasi berita yang serupa berdasarkan konten deskripsi berita?
- Bagaimana memastikan bahwa rekomendasi yang diberikan relevan terhadap berita yang sedang dibaca pengguna?

### Goals
- Membangun sistem rekomendasi item-to-item berdasarkan kemiripan deskripsi berita menggunakan cosine similarity.
- Menyajikan rekomendasi top-N artikel yang paling relevan untuk setiap artikel.

### Solution Statements
- Menggunakan pendekatan *content-based filtering* dengan transformasi teks menggunakan *TF-IDF*.
- Menghitung kemiripan antar berita menggunakan *cosine similarity matrix*.
- Menyediakan fungsi rekomendasi yang menerima input berupa judul artikel dan mengembalikan daftar artikel dengan konten serupa.

## Data Understanding

Dataset yang digunakan adalah [*BBC News dataset on Kaggle*](https://www.kaggle.com/datasets/gpreda/bbc-news), yang berisi kumpulan berita dari berbagai kategori seperti bisnis, hiburan, politik, olahraga, dan teknologi.

**Jumlah dan Kondisi Data**:
- Jumlah data: 42115 baris.
- Format: CSV.

**Variabel pada dataset BBC News**.
- *title*: Judul Berita.
- *pubDate*: Tanggal publikasi berita.
- *guid*: Link panduan umpan.
- *link*: Link berita.
- *description*: Ringkasan isi berita.

Pemeriksaan dilakukan menggunakan fungsi ```info()``` terhadap fitur-fitur dalam dataset, yang menunjukkan adanya ketidaksesuaian tipe data pada fitur *pubDate*. Tipe data pada fitur tersebut perlu diubah menjadi tipe data *datetime*.

```python
news["pubDate"] = pd.to_datetime(news["pubDate"])
```

Output:
```text
datetime64[ns]
```

Pemeriksaan terhadap *missing values* dan duplikasi data pada dataset dilakukan menggunakan kode berikut. Hasil analisis menunjukkan bahwa tidak terdapat missing value maupun data yang terduplikasi, sehingga tahap pembersihan data tidak diperlukan.

```python
print("Total missing value: ", news.isnull().sum())
print("Total duplicates: ", news.duplicated().sum())
```

Output:

```text
Total missing value:  title          0
pubDate        0
guid           0
link           0
description    0
dtype: int64
Total duplicates:  0
```

### Exploratory Data Analysis
- Menampilkan tanggal pertama dan terakhir berita dipublikasikan dalam dataset.
  
  ```python
  print("Start Publish : ", news["pubDate"].min())
  print("End Publish : ", news["pubDate"].max())
  ```

  Output:
  
  ```text
  Start Publish :  2013-08-30 01:01:55
  End Publish :  2024-12-04 00:05:52
  ```
  
  Berdasarkan hasil analisis, diketahui bahwa rentang waktu publikasi berita tercatat dimulai pada tanggal 30 Agustus 2013 pukul 01.01.55 dan berakhir pada tanggal 4 Desember 2024 pukul 00.05.52.
  
- Mengelompokan data berita berdasarkan tahun publikasi untuk memperoleh jumlah berita yang dipublikasikan pada setiap tahun.

  ```python
  total_news_per_year = news["pubDate"].dt.year.value_counts().sort_index()
  total_news_per_year
  ```

  Output:

  Tabel 1. Jumlah berita yang diterbitkan per tahun.
  
  | pubDate | Count  |
  |---------|--------|
  | 2013    | 1      |
  | 2017    | 1      |
  | 2018    | 1      |
  | 2019    | 1      |
  | 2021    | 6      |
  | 2022    | 12301  |
  | 2023    | 15043  |
  | 2024    | 14761  |

  Data menunjukkan bahwa publikasi berita didominasi pada tahun 2022, 2023, dan 2024, dengan jumlah yang jauh lebih tinggi dibandingkan tahun-tahun sebelumnya.

## Data Preparation
Beberapa tahapan yang dilakukan dalam proses data preparation:
- **Seleksi Data Berita**: Pada tahap ini dilakukan seleksi data berita yang berasal dari tahun 2024, dengan total sebanyak 14.761 entri. Data tersebut akan digunakan sebagai dasar dalam pembangunan model sistem rekomendasi.
  
  ```python
  news_2024 = news[news["pubDate"].dt.year == 2024]
  ```

- ***Text Processing***: Proses ini meliputi beberapa langkah penting seperti pembersihan teks (menghapus karakter khusus, angka, dan tanda baca yang tidak relevan), normalisasi (seperti mengubah huruf menjadi huruf kecil.
  
  ```python
  def cleaningText(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text)
  text = re.sub(r'#[A-Za-z0-9]+', '', text)
  text = re.sub(r'RT[\s]', '', text)
  text = re.sub(r"http\S+", '', text)
  text = re.sub(r'[0-9]+', '', text)
  text = re.sub(r'[^\w\s]', '', text)

  text = text.replace('\n', ' ')
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = text.strip(' ')
  return text

  def casefoldingText(text):
    text = text.lower()
    return text

  news_to_clean = news_2024[["description"]].copy()

  news_to_clean["description_clean"] = news_to_clean["description"].apply(cleaningText)
  news_to_clean["description_casefolding"] = news_to_clean["description_clean"].apply(casefoldingText)
  ```

- **Pembuatan dataset**: Pada tahap ini, dibuat dataset akhir yang berisi dua kolom utama dari data berita tahun 2024: *title* dan *description* (yang telah melalui proses pembersihan).

  ```python
  dataset = pd.DataFrame({
      "title": news_2024["title"],
      "description": news_to_clean["description_casefolding"]
  })
  ```

- ***Feature Extraction*** **(TF-IDF)**: Pada tahap ini, teks yang telah diproses diubah menjadi representasi numerik menggunakan metode *TF-IDF*. Metode ini memberikan bobot pada kata berdasarkan frekuensi kemunculannya dalam dokumen dan kelangkaannya di seluruh kumpulan dokumen, sehingga kata-kata yang lebih penting memiliki nilai lebih tinggi. Hasil ekstraksi berupa vektor fitur dengan dimensi (14761, 1000).

  ```python
  tfidf = TfidfVectorizer(
      stop_words = "english",
      max_df=0.8,
      min_df=10,
      max_features=1000,
  )
  
  tfidf_matrix = tfidf.fit_transform(dataset["description"])
  ```

  Output:

  ```text
  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]
  ```

- ***Cosine Similarity***: Pada tahapan ini cosine Similarity digunakan untuk mengukur kemiripan antar dokumen. Hasilnya dipakai dalam sistem rekomendasi untuk menemukan dokumen dengan konten paling mirip.

  ```python
  cosine_sim = cosine_similarity(tfidf_matrix)
  ```

  Output:
  
  ```text
  array([[1., 0., 0., ..., 0., 0., 0.],
         [0., 1., 0., ..., 0., 0., 0.],
         [0., 0., 1., ..., 0., 0., 0.],
         ...,
         [0., 0., 0., ..., 1., 0., 0.],
         [0., 0., 0., ..., 0., 1., 0.],
         [0., 0., 0., ..., 0., 0., 1.]])
  ```

## Modeling

Pada tahap pengembangan model (*model development*), dibangun sebuah fungsi utama bernama ```news_recommendation()``` yang bertanggung jawab dalam menghasilkan rekomendasi berita berdasarkan kemiripan konten. Fungsi ini merupakan inti dari sistem CBF dan dikembangkan dengan tahapan sebagai berikut:
- Pengecekan Judul Masukan
  Fungsi pertama-tama memeriksa apakah judul berita yang dimasukkan pengguna (*news_title*) terdapat dalam data kemiripan (*similarity_data*). Jika tidak ditemukan, fungsi akan mengembalikan pesan kesalahan.

- Pengurutan Berdasarkan Nilai Kemiripan
  Jika judul tersedia, fungsi akan mengambil nilai kemiripan antar berita dalam bentuk vektor, kemudian mengurutkannya berdasarkan nilai tertinggi. Proses ini dilakukan menggunakan fungsi ```argsort()``` untuk mendapatkan indeks dari k berita teratas yang paling mirip dengan judul input.

- Penyaringan Berita yang Sama
  Judul berita input akan dikeluarkan dari hasil rekomendasi agar tidak direkomendasikan kepada dirinya sendiri. Proses ini dilakukan dengan ```drop(news_title, errors="ignore")```.

- Penggabungan dengan Dataset Asli
  Judul-judul hasil rekomendasi akan digabungkan dengan dataset asli (*items*) agar deskripsi berita juga ditampilkan. Hasil akhir berupa DataFrame berisi k berita yang paling relevan secara konten dengan judul input.

```python
def news_recommendation(news_title, similarity_data=cosine_sim_df, items=dataset[["title", "description"]], k=5):
  if news_title not in similarity_data.columns:
    return f"News title '{news_title}' is not found in the dataset."

  index = similarity_data[news_title].to_numpy().argsort()[-k-1:][::-1]

  closest = similarity_data.columns[index]

  closest = closest.drop(news_title, errors="ignore")

  return pd.DataFrame(closest, columns=["title"]).merge(items, on="title").head(k)
```

Untuk menghasilkan daftar rekomendasi berita yang relevan berdasarkan judul tertentu, jalankan fungsi ```news_recommendation()``` dengan memberikan parameter ```news_title```, yaitu judul berita yang ingin dijadikan acuan.

```python
news_recommendation("Tom Cruise abseils off stadium roof in daring Olympic finale")
```
Berikut adalah output dari *top-k* hasil rekomendasi:

Tabel 2. Hasil rekomendasi berita.
| No | Title                                                          | Description                                                    |
|----|----------------------------------------------------------------|----------------------------------------------------------------|
| 1  | Actor Chance Perdomo dies in motorcycle accident               | the ukus star was known for playing ambrose sp...              |
| 2  | Queen Margrethe: Will abdication cause a ripple effect?        | nordic monarchies are known to embrace moderni...              |
| 3  | Sebastián Piñera: Former president of Chile dies               | sebastián piñera became known abroad for overs...              |
| 4  | Fans fume over Jason Donovan Rocky Horror no-show              | fans said they would not have booked if they h...              |
| 5  | Coronation Street's John Savident - who played Fred Elliott... | the star played fred elliott a character best ...              |

## Evaluation

Dalam studi kasus ini, *ground truth* label tidak tersedia secara eksplisit, sehingga anotasi dilakukan secara manual dengan memilih sejumlah item dari hasil rekomendasi. Pemilihan dilakukan berdasarkan asumsi bahwa model belum memiliki informasi preferensi pengguna (*cold-start*). Evaluasi performa sistem dilakukan menggunakan metrik *Precision@K* dan *Average Precision* untuk mengukur relevansi item yang direkomendasikan.

### Precision@K
metrik evaluasi ini digunakan untuk mengukur seberapa relevan item-item yang direkomendasikan oleh sistem dalam K posisi teratas. Metrik ini menghitung proporsi item relevan di antara K rekomendasi teratas yang diberikan oleh model. Nilai Precision@K berkisar antara 0 hingga 1, di mana semakin tinggi nilainya menunjukkan semakin baik kualitas rekomendasi.

**Precision@K** dihitung dengan rumus berikut:

```text
Precision@K = (Jumlah item relevan dalam K rekomendasi teratas) / K
```

Implementasi dalam kode Python adalah sebagai berikut:

```python
def precision_at_k(user_history, recommended_titles, k=5):
    user_history_set = set([title.strip().lower() for title in user_history])
    recommended_top_k = [title.strip().lower() for title in recommended_titles[:k]]

    hits = sum(1 for title in recommended_top_k if title in user_history_set)
    return hits / k

precision = precision_at_k(ground_truth, recommendations)
```

Output:

```text
Precision@5: 0.40
```

### Average Precision
*Average Precision* (AP) adalah metrik evaluasi yang mengukur kualitas urutan hasil rekomendasi dengan menghitung rata-rata precision pada setiap posisi di mana item relevan ditemukan. AP mempertimbangkan urutan hasil dan memberikan bobot lebih pada item relevan yang muncul lebih awal dalam daftar rekomendasi. Nilai AP berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan performa rekomendasi yang lebih baik.

**Average Precision** dihitung dengan rumus berikut:

```text
AP = (1 / jumlah item relevan) × Σ (Precision@k pada posisi k di mana item relevan ditemukan)
```

Implementasi dalam kode Python adalah sebagai berikut:

```python
def average_precision(user_history, recommended_titles):
  hits = 0
  sum_precision = 0

  for i, title in enumerate(recommended_titles):
    if title in user_history:
      hits += 1
      precision_at_i = hits / (i + 1)
      sum_precision += precision_at_i

    if hits == 0:
      return 0.0

  return sum_precision / len(user_history)

ap = average_precision(ground_truth, recommendations)
```

Output:

```text
Average Precision: 0.83
```

**Kesimpulan Evaluasi**:

Dari evaluasi model menggunakan kedua metrik tersebut, diperoleh nilai Precision@5 sebesar 0,40, yang berarti rata-rata 2 dari 5 rekomendasi teratas merupakan item yang relevan dengan preferensi pengguna. Sementara itu, nilai Average Precision sebesar 0,83 menunjukkan bahwa secara keseluruhan urutan rekomendasi memiliki tingkat relevansi yang tinggi, di mana item relevan cenderung muncul di posisi teratas daftar rekomendasi.

Tabel 3. Hasil evaluasi model.

| Metrik           | Nilai |
|------------------|-------|
| Precision@5      | 0,40  |
| Average Precision| 0,83  |
