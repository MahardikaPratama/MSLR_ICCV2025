# Dokumentasi Proses Testing Model

## Tujuan
Mengevaluasi performa model SLR terlatih pada data dev/test menggunakan metrik Word Error Rate (WER) dan menghasilkan file prediksi untuk keperluan evaluasi atau submission.

## Input
- Model terlatih (file `.pt` hasil training)
- File konfigurasi YAML (misal: `configs/Double_Cosign_si.yaml`)
- Data hasil preprocessing:
  - Kamus gloss: `*_gloss_dict.json`
  - Info dataset: `*_info.json`
  - Groundtruth STM: `mslr-*-groundtruth-*.stm` (untuk evaluasi)

## Output
- Nilai WER pada data dev/test
- File prediksi hasil model:
  - output-hypothesis-fusion-*.ctm
  - output-hypothesis-conv-fusion-*.ctm
  - test.csv (untuk submission)
- Log evaluasi

## Alur Kerja Testing

1. **Inisialisasi**
   - Membaca file konfigurasi YAML dan argumen testing.
   - Memuat model terlatih dan bobotnya.
   - Memuat kamus gloss dan info dataset.
   - Membuat DataLoader untuk dev/test.

2. **Evaluasi Model**
   - Model di-set ke mode eval.
   - Untuk setiap batch data:
     - Data dipindahkan ke device.
     - Model melakukan forward pass (tanpa grad).
     - Simpan hasil prediksi (recognized_sents_fusion, conv_sents_fusion).
   - Hasil prediksi ditulis ke file .ctm.

3. **Perhitungan WER**
   - Jika mode dev:
     - Hitung WER menggunakan groundtruth STM dan fungsi evaluate.
     - Logging hasil WER (Conv1D dan BiLSTM).
   - Jika mode test:
     - File .ctm diubah ke test.csv (id, gloss) untuk submission.

## Rincian Step-by-Step

### 1. Menjalankan Testing
- Contoh perintah:
  ```bash
  python main.py --config ./configs/Double_Cosign_si.yaml --phase test --load-weights PATH_TO_TRAINED_MODEL
  ```

### 2. Proses Testing (Fungsi Utama)
- Testing dijalankan oleh kelas `SLRProcessor` (lihat main.py), fungsi utama:
  - `test()`: evaluasi model pada dev/test, hitung WER, simpan hasil prediksi.
  - `seq_eval()`: proses evaluasi batch, tulis file .ctm, hitung WER atau generate test.csv.

### 3. Output
- Nilai WER pada dev/test dicatat di log.
- File prediksi untuk evaluasi/submission dihasilkan di work_dir.

---
