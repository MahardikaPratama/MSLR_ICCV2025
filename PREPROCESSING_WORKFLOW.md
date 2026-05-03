
# Dokumentasi Proses Preprocessing Data

## Tujuan
Menyiapkan data mentah dan anotasi menjadi format yang siap digunakan untuk training dan evaluasi model. Proses ini menghasilkan kamus gloss, file info dataset (train/dev/test), dan groundtruth untuk evaluasi.

## Input
- File anotasi:
   - `si_train_list.txt`, `si_dev_list.txt`, `us_train_list.txt`, `us_dev_list.txt`
   - Format: Teks (bukan pickle), dipisahkan dengan tanda `|`
      - Kolom: `id|gloss|text`
      - Contoh baris:
         ```
         00_0001|سوال هو|من هو
         ```
- File/folder tujuan output:
   - `datasets/mslr2025/` (output akan disimpan di sini)

## Output
- File info dataset (JSON):
   - `si_train_info.json`, `si_dev_info.json`, `us_train_info.json`, `us_dev_info.json`
   - Berisi list dictionary: signer, video_id, gloss_sequence, sentence_id, original_info

- File ground truth STM:
   - `mslr-si-groundtruth-train.stm`, `mslr-si-groundtruth-dev.stm`, dst
   - Format per baris:
      ```
      [video_id] 1 [signer] 0.0 1.79769e+308 [gloss_sequence]
      ```
   - Contoh:
      ```
      00_0001 1 00 0.0 1.79769e+308 سوال هو
      00_0002 1 00 0.0 1.79769e+308 هو معلم لغه اشاره
      ```
   - Penjelasan detail proses dan alasan format:
      - Data mentah diambil dari file seperti `si_train_list.txt` yang berisi baris: `id|gloss|text`.
      - Script membaca setiap baris, memisahkan kolom berdasarkan tanda `|`.
      - Kolom `id` (misal: 00_0001) dipecah menjadi `signer` (00) dan `sentence_id` (0001).
      - Kolom `gloss` (misal: سوال هو) adalah urutan label/gloss untuk video tersebut.
      - Untuk setiap data, script menulis satu baris ke file STM dengan format:
         - `[video_id]` = id video, misal 00_0001
         - `1` = channel (standar STM, biasanya 1)
         - `[signer]` = id signer, misal 00
         - `0.0` = start time (dummy, tidak dipakai di SLR)
         - `1.79769e+308` = end time (dummy, nilai float terbesar, artinya tidak terbatas)
         - `[gloss_sequence]` = urutan gloss/label, misal سوال هو
      - Format STM ini dipilih karena kompatibel dengan tool evaluasi WER (Word Error Rate) yang umum digunakan di bidang pengenalan bahasa isyarat dan pengenalan ujaran.
      - Dengan format ini, setiap baris merepresentasikan satu kalimat/video, siapa penandanya, dan urutan gloss yang menjadi ground truth untuk evaluasi.

- Kamus gloss (JSON):
   - `si_gloss_dict.json`, `us_gloss_dict.json`
   - Berisi mapping gloss ke index dan frekuensi
   - Contoh struktur:
      ```json
      {
         "gloss2id": {"سوال": {"index": 1, "frequency": 10}, ...},
         "id2gloss": {"1": {"gloss": "سوال", "frequency": 10}, ...}
      }
      ```
   - **Catatan:**
     Jika gloss berupa karakter non-ASCII (misal: Arab), pada file JSON bisa muncul dalam format Unicode escape (misal: "\u0627"). Ini hanya tampilan di file, saat dibaca Python nilainya tetap karakter asli.
     Untuk menampilkan karakter asli di file JSON, script harus menggunakan `json.dump(..., ensure_ascii=False)`.

---

## Alur Kerja Preprocessing

1. **Mulai**
2. **Persiapan Data**
   - Download dataset dan letakkan di folder `datasets/`
   - Download file anotasi dan letakkan di folder `preprocess/mslr2025/`
3. **Jalankan Script Preprocessing**
   - Masuk ke direktori `preprocess/mslr2025/`
   - Jalankan perintah:
     ```bash
     python mslr_process.py
     ```
4. **Proses dalam Script mslr_process.py**

   4.1. Membaca file anotasi (misal: train.csv, dev.csv, test.csv)

   4.2. Mengubah file anotasi menjadi format dictionary terstruktur
   
   4.3. Membuat kamus gloss (gloss dictionary) berdasarkan seluruh label pada data
   
   4.4. Membuat file info dataset (train_info.json, dev_info.json, test_info.json) berisi informasi video, label, dsb
   
   4.5. Membuat groundtruth STM (mslr-si-groundtruth-train.stm, dst) untuk evaluasi WER
   
   4.6. Menyimpan hasil-hasil tersebut ke folder `datasets/mslr2025/`
5. **Selesai Preprocessing**

---

## Rincian Step-by-Step

### 1. Membaca File Anotasi
- Script membaca file CSV anotasi (train.csv, dev.csv, test.csv)
- Setiap baris berisi informasi: video_id, signer, gloss_sequence, dll

### 2. Konversi ke Dictionary
- Setiap baris diubah menjadi dictionary Python
- Disimpan dalam list untuk setiap split (train/dev/test)

### 3. Membuat Kamus Gloss
- Menghitung frekuensi kemunculan setiap gloss
- Membuat mapping gloss ke index (gloss2id)
- Menyimpan kamus gloss ke file JSON (misal: si_gloss_dict.json)

### 4. Membuat Info Dataset
- Menyimpan list dictionary info untuk setiap split ke file JSON (train_info.json, dev_info.json, test_info.json)

### 5. Membuat Groundtruth STM
- Membuat file STM untuk setiap split, format: `[video_id] 1 [signer] 0.0 1.79769e+308 [gloss_sequence]`
- File STM digunakan untuk evaluasi WER

### 6. Output
- Semua file hasil preprocessing disimpan di `datasets/mslr2025/`
  - Kamus gloss: `*_gloss_dict.json`
  - Info dataset: `*_info.json`
  - Groundtruth STM: `mslr-*-groundtruth-*.stm`

---

## Catatan Penting
- Pastikan struktur folder sudah sesuai sebelum menjalankan script
- Proses ini wajib dilakukan sebelum training atau evaluasi model
- Jika ada perubahan pada data mentah/anotasi, ulangi proses preprocessing

---

## Contoh Perintah
```bash
cd preprocess/mslr2025
python mslr_process.py
```

Setelah proses selesai, data siap digunakan untuk training dan evaluasi.
