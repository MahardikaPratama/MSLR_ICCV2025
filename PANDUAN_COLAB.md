# Panduan Menjalankan MSLR ICCV2025 di Google Colab

Panduan ini akan membantu Anda mempersiapkan lingkungan, melakukan proses *training*, dan *testing* untuk proyek **A Closer Look at Skeleton-based Continuous Sign Language Recognition** menggunakan Google Colab.

## 1. Persiapan Awal
Buat *notebook* baru di [Google Colab](https://colab.research.google.com/) dan pastikan Anda menggunakan runtime GPU (**Runtime > Change runtime type > Hardware accelerator > T4 GPU** atau lebih tinggi).

Gunakan *cell* di Colab untuk menghubungkan Google Drive Anda. Hal ini penting agar dataset dan model yang Anda *train* tidak hilang ketika sesi Colab berakhir.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Selanjutnya, *clone* repositori project ini ke Google Drive Anda:

```bash
# Pindah ke direktori Google Drive Anda
%cd /content/drive/MyDrive/

# Clone repositorinya (ganti link berikut jika Anda punya repositori Github sendiri)
!git clone https://github.com/MahardikaPratama/MSLR_ICCV2025.git

# Masuk ke folder project
%cd MSLR_ICCV2025
```

## 2. Instalasi Dependensi

Google Colab sudah ter-install PyTorch (biasanya versi terbaru), yang secara umum sudah cukup, tapi kita perlu menginstal `ctcdecode` dan tool `sclite`.

Jalankan *cell* berikut untuk menginstal **ctcdecode**:
```bash
!git clone --recursive https://github.com/WayenVan/ctcdecode.git
%cd ctcdecode
!pip install .
%cd /content/drive/MyDrive/MSLR_ICCV2025
```

Jalankan *cell* berikut untuk menginstal dan setup **sclite (SCTK)**:
```bash
!mkdir -p ./software
!git clone https://github.com/usnistgov/SCTK.git
%cd SCTK
!make config
!make all
!make check
!make install
!make doc
%cd ..
!ln -s $(pwd)/SCTK/bin/sclite ./software/sclite
```

## 3. Persiapan Dataset

Ikuti langkah-langkah di bawah untuk menyiapkan dataset.

1. **Unduh Dataset BISINDO**:
   Unduh file *pickle* dataset (`pose_bisindo_test.pkl` dan `pose_bisindo_train_dev.pkl`) menggunakan `gdown` dan simpan di folder `./datasets`:
   
   ```bash
   !pip install gdown
   !mkdir -p ./datasets
   
   # Unduh pose_bisindo_test.pkl
   !gdown "183QsX05JjyJ8IrgSQIPjHdLnPL9fvdK2" -O ./datasets/pose_bisindo_test.pkl
   
   # Unduh pose_bisindo_train_dev.pkl
   !gdown "1wXL2AwZGTFiOpvvcSSAG-xlwGaFotMoU" -O ./datasets/pose_bisindo_train_dev.pkl
   ```

2. **Preprocess Dataset**:
   Buat *gloss dict*, *dataset info*, dan *groundtruth* untuk evaluasi dengan command ini:
   ```bash
   %cd preprocess/mslr2025
   !python mslr_process.py
   %cd ../../
   ```

## 4. Proses Training

Pastikan Anda telah mengatur strategi *data augmentation* di file `./datasets/skeleton_feeder.py` pada baris ke-194 sesuai dengan kebutuhan Anda.

Jalankan *cell* ini untuk memulai *training* pada *task* Signer Dependent:
```bash
!python main.py --config ./configs/bisindo.yaml
```

## 5. Proses Testing

Untuk melakukan testing, Anda membutuhkan file model hasil *training* atau *pretrained model* (.pth / .pt).

Jalankan *cell* ini untuk melakukan evaluasi model pada *task* Signer Dependent:
```bash
!python main.py \
    --config ./configs/bisindo.yaml \
    --phase test \
    --load-weights PATH_TO_PRETRAINED_MODEL
```
*(Catatan: Ganti `PATH_TO_PRETRAINED_MODEL` dengan nama/path file model yang ingin Anda uji)*

---
## 💡 Tips Penting di Google Colab:
- **`!` vs `%`**: Gunakan `!` untuk menjalankan command terminal biasa (misal: `!pip install`), dan gunakan `%cd` untuk berpindah folder secara permanen di dalam sel.
- **Mencegah Disconnect**: Google Colab dapat terputus jika tidak ada aktivitas. Menyimpan *checkpoints* model ke dalam folder Google Drive Anda secara berkala sangat disarankan agar kerja berjam-jam tidak hilang.
- **Error pada PyTorch dan ctcdecode**: Jika `ctcdecode` mengalami error, pertimbangkan untuk men-downgrade versi PyTorch di Colab (misalnya ke versi `2.0.0`) sesuai saran di `README.md`.
