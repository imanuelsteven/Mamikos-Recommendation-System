#  Machine Learning Project Report - Steven Graciano

## 📌 Project Overview
## Kos Reccomendation System🏘️
![Mamikos](Asset/mamikos.png)
This project aims to design and compare two distinct recommendation models—Content-Based Filtering and Cluster-Based Filtering to assist users in identifying suitable boarding houses (kos) in Salatiga, Indonesia. The objective is to improve the accommodation selection process by providing personalized recommendations aligned with user preferences and previously liked properties. Rather than combining models into a hybrid system, each will be developed and evaluated separately to determine which offers higher accuracy, relevance, and efficiency.

## ❔ Business Understanding

The motivation for this project arises from the common difficulty users face when searching for rental accommodations that match their specific preferences. Traditional listing platforms often lack personalization features, requiring users to manually browse numerous irrelevant listings.

To address this, the project investigates two machine learning-based techniques:

- Content-Based Filtering (CBF) recommends properties that share similar features—such as location, facilities, and type—with those the user has previously shown interest in.

- Cluster-Based Filtering (CLF) utilizes unsupervised learning (e.g., K-Means clustering) to group properties with similar characteristics and recommends items from the same cluster as the user’s selected reference.

By developing both models separately, the project aims to evaluate which technique provides more relevant recommendations, better performance, and interpretability.

### ❌ Problem Statements

- How can a content-based filtering model be constructed to provide personalized boarding house recommendations based on explicit features?

- How effective is a cluster-based filtering model in delivering recommendations when items are grouped using shared characteristics?

- Between content-based and cluster-based filtering, which model yields more accurate and relevant suggestions for users in the Salatiga region?

### 🎯 Goals
- To build a content-based filtering model that recommends boarding houses by calculating the similarity between user-specified preferences and property features using appropriate similarity metrics.

- To construct a cluster-based filtering model that segments the property dataset and recommends items from the same cluster as a given reference.

- To evaluate both models using defined performance metrics such as Precision, Recall, and Silhouette Score, and determine which better reflects user preferences and usability.


### 🟢 Solution Statements
**Solution 1: Content-Based Filtering Model**

Develop a similarity-based recommendation system using cosine similarity to measure the likeness between properties based on their feature vectors.

Key features may include:
1. location
2. Type
3. Facility

**Solution 2: Cluster-Based Filtering Model**

Apply K-Means clustering to group properties into homogeneous clusters based on the same set of features.

When a user inputs a preferred boarding house, the system identifies its cluster and recommends other properties from within the same cluster, assuming semantic similarity.

Both models will be assessed independently through comparative analysis covering recommendation quality, computational performance, and system usability.

--- 

## Data Understanding

![First Distribution](Asset/first_distri.png)

Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
