# Dokumentasi Proses Training Model

## Tujuan
Melatih model pengenalan bahasa isyarat berkelanjutan (SLR) menggunakan data hasil preprocessing, menghasilkan model terlatih yang siap dievaluasi dan digunakan untuk inferensi.

## Input
- File konfigurasi YAML (misal: `configs/Double_Cosign_si.yaml`)
- Data hasil preprocessing:
  - Kamus gloss: `*_gloss_dict.json`
  - Info dataset: `*_info.json`
  - Groundtruth STM: `mslr-*-groundtruth-*.stm` (untuk evaluasi)
- (Opsional) Pretrained weights untuk transfer learning

## Output
- Model terlatih (file `.pt`)
- Log training dan evaluasi (loss, WER per epoch)
- Model checkpoint terbaik (best_dev_*.pt) dan terakhir (cur_dev_*.pt)

## Alur Kerja Training

1. **Inisialisasi**
   - Membaca file konfigurasi YAML.
   - Menyiapkan argumen training (device, batch size, epoch, optimizer, dsb).
   - Memuat kamus gloss dan info dataset.
   - Membuat DataLoader untuk train/dev/test.

2. **Membangun Model**
   - Membuat arsitektur model sesuai argumen (`model_args`).
   - Memuat bobot pretrained jika ada.
   - Inisialisasi optimizer dan scheduler.

3. **Training Loop**
   - Untuk setiap epoch:
     - Melatih model pada data training (`seq_train`).
     - Logging loss dan learning rate.
     - Setiap interval tertentu:
       - Evaluasi model pada data dev (`seq_eval`), hitung WER.
       - Simpan checkpoint model (current dan best jika WER membaik).

4. **Evaluasi**
   - Setelah training selesai, model terbaik dapat dievaluasi pada data test untuk mendapatkan WER akhir.

## Rincian Step-by-Step

### 1. Menjalankan Training
- Contoh perintah:
  ```bash
  python main.py --config ./configs/Double_Cosign_si.yaml
  ```
- Untuk transfer learning:
  ```bash
  python main.py --config ./configs/Double_Cosign_us.yaml --load-weights PATH_TO_PRETRAINED_MODEL --ignore-weights classifier_static.weight classifier_motion.weight classifier_fusion.weight
  ```

### 2. Proses Training (Fungsi Utama)
- Training dijalankan oleh kelas `SLRProcessor` (lihat main.py).
- Fungsi utama:
  - `train()`: loop epoch, training, evaluasi, simpan model.
  - `seq_train()`: training satu epoch (forward, loss, backward, optimizer step).
  - `seq_eval()`: evaluasi model, hitung WER.

### 3. Logging & Checkpoint
- Loss dan WER dicatat setiap epoch.
- Model disimpan secara berkala dan saat WER dev membaik.

### 4. Output Akhir
- Model terbaik (`best_dev_*.pt`) dan log training.
- Siap untuk inferensi atau evaluasi lebih lanjut.

---
