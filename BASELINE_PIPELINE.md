# 1. Gambaran Umum Baseline

Baseline ini merupakan sistem Continuous Sign Language Recognition (CSLR) berbasis skeleton. Tujuan utamanya adalah mengenali urutan kata (gloss) dari video isyarat secara otomatis dan kontinu, tanpa segmentasi manual. Model ini menerima input berupa data skeleton (koordinat pose tubuh dan tangan) yang diekstrak dari video, dan menghasilkan output berupa urutan prediksi gloss (kata isyarat) beserta evaluasi akurasi menggunakan metrik Word Error Rate (WER).

- **Tujuan:** Mengubah video isyarat menjadi urutan kata (gloss) secara otomatis.
- **Input:** Data skeleton (pose tubuh & tangan per frame).
- **Output:** Urutan prediksi gloss (teks) dan nilai WER.
- **Task:** Continuous Sign Language Recognition (CSLR).

# 2. Struktur Project

```
MSLR_ICCV2025/
├── datasets/
│   ├── pose_bisindo_train_dev_sd.pkl
│   ├── pose_bisindo_test_sd.pkl
│   ├── pose_bisindo_test_si-maj.pkl
│   └── pose_bisindo_test_si-min.pkl
├── modules/
│   ├── stgcn_layers/
│   │   ├── gcn_utils.py
│   │   └── stgcn_block.py
│   ├── temporal_layers/
│   │   ├── BiLSTM.py
│   │   └── tconv.py
│   ├── criterion/
│   │   └── radialctc.py
│   └── visual_extractor.py
├── evaluation/
│   └── slr_eval/
│       ├── mergectmstm.py
│       ├── preprocess.sh
│       ├── python_wer_evaluation.py
│       └── wer_calculation.py
├── preprocess/
│   └── mslr2025/
│       ├── mslr_process.py
│       ├── sd_dev_list.txt
│       ├── sd_test_list.txt
│       ├── sd_train_list.txt
│       └── SD/
│           ├── dev.csv
│           ├── test.csv
│           └── train.csv
├── configs/
│   ├── Double_Cosign_sd.yaml
│   └── dataset_configs/
│       └── bisindo.yaml
├── utils/
│   ├── __init__.py
│   ├── decode.py
│   ├── device.py
│   ├── optimizer.py
│   ├── pack_code.py
│   ├── parameters.py
│   ├── random_state.py
│   ├── record.py
│   └── skeleton_augmentation.py
├── main.py
├── seq_scripts.py
└── slr_network.py
```

**Penjelasan folder dan file utama:**
**Penjelasan folder dan file utama:**
- **datasets/**: Loader, pre-processing, dan file data skeleton.
      - **mslr2025/**: Data skeleton per subject/split.
      - **pose_bisindo_train_dev_sd.pkl, pose_bisindo_test_*.pkl**: File pickle data skeleton hasil ekstraksi pose dengan pembagian SD, SI-Maj, dan SI-Min.
      - **skeleton_feeder.py**: Loader utama untuk data skeleton.
- **modules/**: Implementasi model utama.
      - **stgcn_layers/**: Layer Graph Convolution (ST-GCN).
            - **gcn_utils.py**: Utilitas graph dan adjacency matrix.
            - **stgcn_block.py**: Blok utama ST-GCN.
      - **temporal_layers/**: Layer temporal.
            - **BiLSTM.py**: Layer BiLSTM untuk modeling urutan.
            - **tconv.py**: Temporal convolution.
      - **criterion/**: Loss function.
            - **radialctc.py**: Implementasi CTC loss dan decoding.
      - **visual_extractor.py**: Wrapper ekstraksi fitur skeleton.
- **evaluation/**: Script evaluasi dan perhitungan WER.
      - **slr_eval/**: Submodul evaluasi.
            - **mergectmstm.py**: Utility merge hasil decoding.
            - **preprocess.sh**: Script preprocessing untuk evaluasi.
            - **python_wer_evaluation.py**: Evaluasi WER berbasis Python.
            - **wer_calculation.py**: Perhitungan Word Error Rate.
- **preprocess/**: Script untuk menyiapkan data mentah menjadi skeleton.
      - **mslr2025/**: Script dan list data split.
            - **mslr_process.py**: Script utama preprocessing skeleton.
            - **sd_dev_list.txt, sd_test_list.txt, sd_train_list.txt**: List file untuk split data.
            - **SD/**: Folder CSV hasil split.
                  - **dev.csv, test.csv, train.csv**: Data skeleton per split.
- **configs/**: File konfigurasi YAML eksperimen.
      - **Double_Cosign_sd.yaml**: Contoh file konfigurasi eksperimen.
      - **dataset_configs/**: Konfigurasi dataset.
            - **bisindo.yaml**: Konfigurasi dataset Bisindo SD.
- **utils/**: Fungsi utilitas umum (helper, logging, dsb).
      - **decode.py, device.py, optimizer.py, pack_code.py, parameters.py, random_state.py, record.py, skeleton_augmentation.py**: Utility training dan data.
      - **__init__.py**: Inisialisasi modul utils.
- **main.py**: Entry point training/testing.
- **seq_scripts.py**: Script training dan evaluasi per-epoch.
- **slr_network.py**: Definisi arsitektur model utama.

# 3. Pipeline Sistem

```
Raw Skeleton Data
      ↓
Data Loader (Feeder)
      ↓
ST-GCN Feature Extraction
      ↓
Temporal Modeling (BiLSTM/TConv)
      ↓
CTC Decoding
      ↓
Gloss Prediction
      ↓
WER Evaluation
```

**Penjelasan tiap tahap:**
- **Raw Skeleton Data:** Data pose tubuh & tangan per frame hasil ekstraksi dari video.
- **Data Loader:** Membaca, mengolah, dan membentuk batch data skeleton menjadi tensor.
- **ST-GCN Feature Extraction:** Ekstraksi fitur spasial-temporal dari skeleton menggunakan Graph Convolution.
- **Temporal Modeling:** Menangkap dependensi urutan waktu menggunakan BiLSTM atau Temporal Convolution.
- **CTC Decoding:** Mengubah output frame-wise menjadi urutan gloss menggunakan Connectionist Temporal Classification.
- **Gloss Prediction:** Menghasilkan urutan kata isyarat (gloss) sebagai output.
- **WER Evaluation:** Menghitung akurasi prediksi dengan Word Error Rate.

# 4. Penjelasan Modul Inti

| Modul                | File                        | Fungsi                                                                 | Input                | Output               |
|----------------------|-----------------------------|------------------------------------------------------------------------|----------------------|----------------------|
| Skeleton Feeder      | datasets/skeleton_feeder.py | Membaca & memproses data skeleton, membentuk batch tensor              | File skeleton        | Tensor batch         |
| ST-GCN Block         | modules/stgcn_layers/stgcn_block.py | Layer graph convolution untuk ekstraksi fitur spasial-temporal | Tensor skeleton      | Fitur spasial        |
| Graph Utils          | modules/stgcn_layers/gcn_utils.py   | Membuat adjacency matrix & graph skeleton                             | Layout skeleton      | Matriks adjacency    |
| BiLSTM               | modules/temporal_layers/BiLSTM.py   | Model temporal berbasis LSTM dua arah                                 | Fitur spasial        | Fitur temporal       |
| Temporal Conv        | modules/temporal_layers/tconv.py    | Model temporal berbasis convolution                                   | Fitur spasial        | Fitur temporal       |
| Visual Extractor     | modules/visual_extractor.py         | Wrapper ekstraksi fitur (ST-GCN + temporal)                           | Tensor skeleton      | Fitur akhir          |
| CTC Loss             | modules/criterion/radialctc.py      | Loss function & decoding CTC                                          | Fitur temporal       | Prediksi urutan gloss|
| WER Calculation      | evaluation/slr_eval/wer_calculation.py | Menghitung Word Error Rate (WER)                                   | Prediksi & label     | Nilai WER            |

**Hubungan antar modul:**
- Data skeleton → Feeder → ST-GCN (gcn_utils + stgcn_block) → Temporal (BiLSTM/tconv) → Visual Extractor → CTC Loss/Decoding → WER Calculation

# 5. Aliran Data Antar Modul

1. **Raw skeleton** (CSV/JSON) dibaca oleh Feeder →
2. **Tensor input** (batch, frame, joint, dimensi) →
3. **Feature extraction** oleh ST-GCN (spasial-temporal) →
4. **Temporal modeling** oleh BiLSTM/tconv (urutan waktu) →
5. **Sequence prediction** oleh CTC (frame-to-gloss) →
6. **Evaluation**: Prediksi dibandingkan label, dihitung WER.

Setiap tahap mengubah representasi data dari bentuk mentah menjadi semakin abstrak dan informatif untuk prediksi urutan gloss.


# 6. Kesimpulan

**Asumsi:**
- Data skeleton sudah tersedia dari proses ekstraksi pose.
- Pipeline ini dapat dikembangkan untuk menambah fitur visual lain jika diperlukan.
