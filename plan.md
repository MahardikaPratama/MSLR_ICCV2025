# Plan: Perancangan APE (Aplikasi Pendukung Eksperimen)
## Tugas Akhir – CSLR BISINDO (GCN–Conv1D–BiLSTM–CTC)

> **Tujuan dokumen ini:** Menjadi panduan bagi AI agent untuk melakukan *reverse engineering* dari repository kode yang sudah ada, lalu menyusun dokumentasi Perancangan APE (Bab IV) mengikuti struktur laporan TA 302 sebagai referensi utama.

---

## Konteks Penelitian

| Item | Detail |
|---|---|
| **Topik TA** | Continuous Sign Language Recognition (CSLR) untuk BISINDO (varian Bandung) |
| **Arsitektur model** | GCN → Conv1D → BiLSTM → CTC |
| **Skenario** | Signer-independent |
| **Variabel bebas** | Konfigurasi pipeline pre-processing |
| **Metrik evaluasi** | WER (Word Error Rate) dan inference speed |
| **APE** | Aplikasi berbasis web lokal untuk mendukung eksperimen & evaluasi model |

---

## Referensi Utama: Struktur APE dari TA 302

Laporan TA 302 (YOLO11 – deteksi buah jeruk dekopon) menggunakan struktur berikut untuk Perancangan APE (Subbab IV.8):

```
IV.8   Perancangan APE
IV.8.1 Analisis Kebutuhan Fungsional
IV.8.2 Alur Proses APE
IV.8.3 Spesifikasi Tools Pengembangan APE
IV.8.4 Rancangan Struktur Proses (per fungsi)
IV.8.5 Skenario Pengujian APE
```

**Pola penulisan yang wajib diikuti:**
- Setiap subseksi diawali dengan definisi/tujuan, lalu tabel/gambar, lalu narasi detail.
- Kebutuhan fungsional ditulis dalam tabel dengan kolom `IDE` dan `Kebutuhan Fungsional`.
- Struktur proses per fungsi ditulis dalam format tabel: Nama State, Nama Fungsi, Deskripsi, Input, Output, Proses (langkah bernomor), Tools/Library, Algoritma (pseudocode).
- Skenario pengujian ditulis dalam tabel dengan kolom: Case IDE, IDE (FR), Item Case, Test Data, Expected Result, Actual Result, Status.

---

## Tahap 1 — Eksplorasi Repository

### 1.1 Identifikasi Struktur Repository

Baca seluruh struktur direktori repository. Catat:
- Entry point utama aplikasi (misal: `app.py`, `main.py`, atau file Streamlit lainnya).
- Direktori untuk model (`.pt` atau format lain untuk CSLR).
- Direktori untuk data input (video/skeleton sequence).
- Direktori untuk output/hasil inferensi.
- File konfigurasi (misal: `config.py`, `.env`, `requirements.txt`).
- Modul atau file yang berkaitan dengan pre-processing pipeline.
- Modul atau file yang berkaitan dengan inferensi dan evaluasi (WER, inference speed).

**Output yang dihasilkan:**
```
docs/repo_structure.md   ← peta direktori dan penjelasan singkat tiap file penting
```

### 1.2 Identifikasi Fitur Aplikasi

Dari entry point dan kode UI, identifikasi fitur-fitur apa saja yang sudah diimplementasikan. Petakan ke kategori berikut (sesuaikan jika ada fitur tambahan atau berbeda):

| Kode Fitur | Nama Fitur | Deskripsi Singkat |
|---|---|---|
| F01 | Inferensi Video/Sequence | Upload video atau file skeleton, pilih model, jalankan inferensi, tampilkan hasil recognition |
| F02 | Evaluasi WER | Hitung dan tampilkan WER antara prediksi gloss sequence vs ground truth |
| F03 | Perbandingan Model | Bandingkan performa (WER, inference speed) antarmodel eksperimen |
| F04 | Rekap Hasil Inferensi | Lihat histori inferensi yang sudah dilakukan |
| F05 | Visualisasi Pre-processing | (jika ada) Tampilkan perbandingan output pre-processing per konfigurasi |

> **Instruksi agent:** Sesuaikan tabel di atas berdasarkan kode aktual. Hapus fitur yang tidak ada, tambahkan yang ada tapi belum tercantum.

---

## Tahap 2 — Penyusunan Kebutuhan Fungsional (FR)

### 2.1 Format Tabel

Ikuti format Tabel IV.12 dari TA 302. Gunakan prefix `FR` dengan nomor urut dua digit.

```
| IDE  | Kebutuhan Fungsional                                     |
|------|----------------------------------------------------------|
| FR01 | Sistem dapat menerima input ...                          |
| FR02 | Sistem dapat melakukan validasi bahwa ...                |
| ...  | ...                                                      |
```

### 2.2 Panduan Pemetaan Fitur → FR

Untuk setiap fitur yang ditemukan di Tahap 1, uraikan ke dalam FR atomik mengikuti pola:

**Fitur Inferensi (setara Fitur Inferensi Video di TA 302):**
- FR untuk menerima input (file video MP4 atau file skeleton CSV/NPY).
- FR untuk validasi format dan ukuran file.
- FR untuk menyimpan file ke direktori sementara.
- FR untuk memilih model hasil eksperimen dari daftar.
- FR untuk memilih konfigurasi pre-processing (variabel bebas penelitian).
- FR untuk menjalankan inferensi.
- FR untuk menampilkan hasil berupa gloss sequence, WER (jika ground truth ada), dan inference speed.
- FR untuk menyimpan hasil inferensi ke database/log.
- FR untuk mencegah inferensi ulang pada kombinasi input + model yang sama.

**Fitur Evaluasi WER:**
- FR untuk menerima atau memilih ground truth (label gloss sequence).
- FR untuk menghitung WER dari hasil prediksi.
- FR untuk menampilkan tabel hasil WER per model.

**Fitur Perbandingan Model:**
- FR untuk memilih dua atau lebih model untuk dibandingkan.
- FR untuk menampilkan tabel perbandingan metrik (WER, inference speed, jumlah parameter jika relevan).
- FR untuk menampilkan visualisasi (bar chart atau line chart) perbandingan metrik.

**Fitur Rekap Inferensi:**
- FR untuk membaca data rekap dari database/log.
- FR untuk menampilkan tabel rekap hasil seluruh inferensi.

**Output yang dihasilkan:**
```
docs/functional_requirements.md   ← daftar lengkap FR dengan deskripsi
```

---

## Tahap 3 — Penyusunan Alur Proses APE

### 3.1 Flowchart

Buat flowchart keseluruhan alur proses APE mengikuti pola Gambar IV.24 dari TA 302. Flowchart harus mencakup:

- Titik mulai (Mulai).
- Percabangan pilihan fitur (Inferensi / Perbandingan Model / Rekap / dll.).
- Untuk setiap cabang: alur validasi, proses utama, dan output.
- Titik selesai (Selesai).

Gunakan simbol standar flowchart:
- Oval: Mulai/Selesai.
- Jajaran genjang: Input/Output.
- Persegi panjang: Proses.
- Belah ketupat: Keputusan (True/False).

**Output yang dihasilkan:**
```
docs/flowchart_APE.drawio   ← atau format gambar yang sesuai
docs/flowchart_APE.png      ← versi gambar untuk dimasukkan ke laporan
```

### 3.2 Narasi Fitur

Untuk setiap fitur utama, tulis narasi singkat (3–6 kalimat) yang menjelaskan alur kerja fitur tersebut dari perspektif pengguna. Ikuti gaya penulisan poin 1–4 di Subbab IV.8.2 TA 302.

---

## Tahap 4 — Spesifikasi Tools Pengembangan APE

### 4.1 Format Tabel

Ikuti format Tabel IV.13 dari TA 302:

```
| Komponen            | Spesifikasi                        |
|---------------------|------------------------------------|
| Platform            | Personal Computer (spesifikasi ...) |
| Bahasa Pemrograman  | Python                             |
| Framework           | [nama framework UI]                |
| Library Pendukung   | [daftar library utama]             |
| Metode Eksekusi     | Secara lokal melalui terminal      |
| Lokasi Penyimpanan  | Lokal                              |
```

### 4.2 Panduan Identifikasi Tools

Baca `requirements.txt` (atau `pyproject.toml`, `environment.yml`) dari repository. Petakan ke kategori:

| Kategori | Library yang Dicari |
|---|---|
| Framework UI | Streamlit, Gradio, Flask, FastAPI |
| Pemrosesan skeleton/pose | MediaPipe, OpenPose output parser |
| Pemrosesan video | OpenCV (`cv2`) |
| Deep learning / inferensi | PyTorch, TensorFlow, ONNX Runtime |
| Evaluasi CSLR | library WER (jiwer, atau custom) |
| Pengelolaan data | Pandas, NumPy |
| Visualisasi | Matplotlib, Plotly |
| Graph neural network | PyTorch Geometric, DGL |

---

## Tahap 5 — Rancangan Struktur Proses

### 5.1 Format Tabel per Fungsi

Untuk setiap fungsi utama yang ditemukan dari kode, buat tabel dengan format berikut (ikuti Tabel IV.14–IV.16 dari TA 302):

```
Nama State    : [nama state/kondisi]
Nama Fungsi   : [nama fungsi aktual dari kode]
Deskripsi     : [penjelasan singkat fungsi]
Input         : [parameter input]
Output        : [nilai kembalian / efek]
Proses        : 1. ...
                2. ...
Tools/Library : [library yang digunakan]
Algoritma     : [pseudocode dalam format CALL/IF/WHILE/RETURN]
```

### 5.2 Fungsi Prioritas untuk Didokumentasikan

Berdasarkan FR yang telah disusun, prioritaskan dokumentasi untuk fungsi-fungsi berikut:

1. **`getAndValidateInput`** – menerima dan memvalidasi file input (video/skeleton).
2. **`selectModel`** – menampilkan dan memilih model eksperimen.
3. **`selectPreprocessingConfig`** – memilih konfigurasi pre-processing (variabel bebas).
4. **`runInference`** – menjalankan inferensi pada input dengan model terpilih.
5. **`computeWER`** – menghitung WER dari prediksi dan ground truth.
6. **`displayInferenceResult`** – menampilkan hasil (gloss sequence, WER, inference speed).
7. **`compareModels`** – mengambil dan menampilkan perbandingan metrik antarmodel.
8. **`getInferenceSummary`** – membaca dan menampilkan rekap seluruh inferensi.

> **Instruksi agent:** Sesuaikan nama fungsi dengan nama aktual di kode. Jika satu fungsi besar dipecah menjadi beberapa sub-fungsi, dokumentasikan masing-masing secara terpisah.

**Output yang dihasilkan:**
```
docs/struktur_proses.md   ← semua tabel struktur proses
```

---

## Tahap 6 — Skenario Pengujian APE

### 6.1 Format Tabel

Ikuti format Tabel IV.25 dari TA 302:

```
| Case IDE | IDE  | Item Case                          | Test Data          | Expected Result          | Actual Result | Status |
|----------|------|------------------------------------|--------------------|--------------------------|---------------|--------|
| 1.1      | FR01 | Menerima input video/skeleton ...  | File valid ...     | Sistem menampilkan ...   |               |        |
| 1.2      | FR02 | Validasi format file               | File .txt tidak valid | Sistem menampilkan error |           |        |
```

### 6.2 Panduan Pembuatan Skenario

Untuk setiap FR, buat minimal satu skenario *happy path* dan satu skenario *error/edge case*:

| FR Kategori | Skenario Happy Path | Skenario Error/Edge Case |
|---|---|---|
| Input video/skeleton | File valid, ukuran dalam batas | File melebihi ukuran, format salah |
| Pemilihan model | Model tersedia dan dipilih | Folder model kosong |
| Pemilihan pre-processing config | Config valid dipilih | Tidak ada config tersedia |
| Inferensi | Inferensi berhasil, output gloss ditampilkan | Inferensi pada kombinasi input+model yang sudah pernah dilakukan |
| Kalkulasi WER | Ground truth tersedia, WER dihitung | Ground truth tidak diunggah |
| Perbandingan model | Dua model berbeda dipilih | Dua model sama dipilih |
| Rekap inferensi | Data rekap tersedia | Data rekap kosong |

**Output yang dihasilkan:**
```
docs/skenario_pengujian.md   ← tabel skenario pengujian lengkap
```

---

## Tahap 7 — Kompilasi Dokumen Perancangan APE

Setelah semua tahap selesai, kompilasi seluruh output menjadi satu dokumen narasi siap pakai untuk Bab IV laporan TA. Ikuti struktur berikut:

```
IV.8   Perancangan Aplikasi Pendukung Eksperimen (APE)
       [Paragraf pengantar: tujuan, ruang lingkup, pengguna APE]

IV.8.1 Analisis Kebutuhan Fungsional
       [Paragraf pengantar kebutuhan fungsional]
       [Tabel FR01–FRnn]
       [Paragraf penutup: fokus kebutuhan]

IV.8.2 Alur Proses APE
       [Paragraf pengantar alur]
       [Gambar flowchart APE]
       [Narasi fitur 1–n (bernomor)]

IV.8.3 Spesifikasi Tools Pengembangan APE
       [Paragraf pengantar]
       [Tabel spesifikasi tools]
       [Narasi penjelasan tiap library]

IV.8.4 Rancangan Struktur Proses
       [Paragraf pengantar]
       [Tabel struktur proses per fungsi (IV.8.4.1 dst)]

IV.8.5 Skenario Pengujian APE
       [Paragraf pengantar]
       [Tabel skenario pengujian]
```

**Output akhir:**
```
docs/bab4_perancangan_APE.md   ← narasi lengkap siap dipindahkan ke dokumen Word
```

---

## Catatan Penting untuk AI Agent

1. **Domain berbeda, pola sama.** TA 302 membahas deteksi objek (YOLO11), sedangkan TA ini membahas CSLR (GCN–BiLSTM–CTC). Ikuti *struktur dan pola penulisan* TA 302, bukan konten teknisnya.

2. **Sesuaikan terminologi CSLR:**
   - "video buah jeruk" → "video rekaman tangan/tubuh penanda"
   - "bounding box" → "skeleton keypoint overlay" (jika relevan)
   - "FPS" → tetap FPS (inference speed)
   - "jumlah objek terdeteksi" → "gloss sequence yang dikenali"
   - "Ultralytics / YOLO11" → framework deep learning yang digunakan (PyTorch, dll.)
   - "model .pt" → file checkpoint model (`.pt`, `.pth`, atau format lain)
   - "mAP50, precision, recall" → WER, CER (jika digunakan)

3. **Variabel bebas adalah konfigurasi pre-processing pipeline**, bukan arsitektur model. APE harus bisa memilih/menampilkan konfigurasi ini saat inferensi.

4. **Signer-independent testing:** APE mungkin memiliki fitur khusus untuk memuat data dari 30 signer tambahan yang dipisahkan untuk uji signer-independent. Identifikasi ini di kode dan masukkan ke FR jika ada.

5. **Gunakan bahasa Indonesia formal** dalam seluruh dokumen output, konsisten dengan gaya penulisan TA akademik Politeknik Negeri Bandung.

6. **Jangan mendahului Bab IV dengan konten Bab V.** Kolom "Actual Result" dan "Status" di tabel skenario pengujian dikosongkan — diisi saat implementasi (Bab V).

7. **Algoritma ditulis dalam pseudocode** mengikuti gaya TA 302: `CALL`, `IF/THEN/ENDIF`, `WHILE`, `RETURN`, `SET`, `DISPLAY`. Bukan Python syntax, bukan flowchart.

---

## Urutan Eksekusi yang Disarankan

```
Tahap 1 → Baca repo, petakan struktur dan fitur
Tahap 2 → Susun daftar FR berdasarkan fitur aktual
Tahap 3 → Buat flowchart dan narasi alur
Tahap 4 → Identifikasi tools dari requirements
Tahap 5 → Dokumentasikan fungsi-fungsi utama
Tahap 6 → Buat skenario pengujian per FR
Tahap 7 → Kompilasi jadi narasi Bab IV
```

Jika ada fitur di kode yang tidak terpetakan ke kategori manapun dalam plan ini, **dokumentasikan dulu** di `docs/repo_structure.md` dan tanyakan ke pengguna sebelum melanjutkan ke Tahap 2.
