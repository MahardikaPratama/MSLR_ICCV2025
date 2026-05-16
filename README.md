<div align="center">

# 🤟 BISINDO-CSLR

### Skeleton-based Continuous Sign Language Recognition
### for BISINDO — Bandung Variant · Signer-Independent

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7?style=flat-square)](https://google.github.io/mediapipe/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green?style=flat-square)](LICENSE)
[![Based on](https://img.shields.io/badge/Based%20on-ICCV%202025%20Winner-FFD700?style=flat-square)](https://openaccess.thecvf.com/)

<br/>

**Undergraduate Thesis (Tugas Akhir) · Politeknik Negeri Bandung · 2026**

[Mahardika Pratama](mailto:) (221524044) &nbsp;·&nbsp; [Sarah](mailto:) (221524059)

*D-IV Teknik Informatika — Jurusan Teknik Komputer dan Informatika*

<br/>

[📄 Thesis Report](#-citation) &nbsp;|&nbsp; [📦 Dataset](#-dataset) &nbsp;|&nbsp; [🚀 Quick Start](#-quick-start) &nbsp;|&nbsp; [⚙️ Configuration](#️-configuration--augmentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
  - [1. Installation](#1-installation)
  - [2. Dataset Setup](#2-dataset-setup)
  - [3. Preprocessing](#3-preprocessing)
- [Configuration & Augmentation](#️-configuration--augmentation)
- [Training & Evaluation](#️-training--evaluation)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## 🔍 Overview

This repository is the official implementation for the undergraduate thesis:

> **"Analisis Konfigurasi Pipeline Pre-Processing pada Model GCN-1DCNN-BiLSTM untuk Continuous Sign Language Recognition BISINDO Variasi Bandung dalam Skenario Signer-Independent"**
>
> *Analysis of Pre-Processing Pipeline Configuration on a GCN-1DCNN-BiLSTM Model for Continuous Sign Language Recognition of the Bandung Variant of BISINDO in a Signer-Independent Scenario*

The central research question is: **how do pre-processing pipeline configurations affect model performance on skeleton-based CSLR?** Three independent variables are systematically analyzed:

| Variable | Options Explored |
|---|---|
| **Input Signal Selection** | Skeleton component combinations (GL, GR, GP, GM) |
| **Downsampling Activation** | Enable/disable frame subsampling before augmentation/normalization |
| **Downsampling Ratio** | Frame subsampling rate used when downsampling is enabled |
| **Data Augmentation** | Spatial & temporal augmentation strategies |
| **Normalization Pipeline** | Spatial normalization, missing keypoint reconstruction, temporal normalization |

All experiments are conducted under a **signer-independent** scenario — signers in the test set are entirely unseen during training — on a self-collected **BISINDO (Bahasa Isyarat Indonesia)** dataset in its **Bandung regional variant**.

The implementation adapts:
- 🏆 **[Min et al., ICCV Workshop 2025]** — *A Closer Look at Skeleton-based CSLR* (SignEval 2025 winner)
- **[CoSign — Jiao et al., ICCV 2023]** — *Exploring Co-occurrence Signals in Skeleton-based CSLR*

---

## 🧠 Model Architecture

<!-- 
  TODO: Replace this placeholder with your architecture diagram.
  Recommended: export from draw.io or a similar tool as a high-resolution PNG,
  place it at docs/architecture.png, then uncomment the line below.
-->

> 📌 **Architecture diagram coming soon.**
<!-- > *(Add your figure at `docs/architecture.png` and uncomment the line below.)* -->

![Architecture Diagram](docs/architecture.png)

The model follows a skeleton-to-gloss pipeline:

**MediaPipe Holistic** → **Skeleton Keypoints** (GL · GR · GP · GM) → **Pre-Processing** *(Input Selection → Downsampling → Augmentation)* → **GCN** → **1D-CNN** → **BiLSTM** → **CTC Decoder** → **Gloss Sequence**

Evaluation is performed using **WER (Word Error Rate)** and **Inference Speed (seq/s)**.

**Skeleton components extracted via MediaPipe Holistic:**

| Symbol | Component | Keypoints |
|---|---|---|
| `GP` | Pose / Body | 33 landmarks |
| `GL` | Left Hand | 21 landmarks |
| `GR` | Right Hand | 21 landmarks |
| `GM` | Face Mesh | 468 landmarks |

---

## 📦 Dataset

The BISINDO dataset used in this research was **independently collected and curated** as part of this thesis. It captures the **Bandung regional variant** of BISINDO and is structured for signer-independent evaluation, with annotation performed using the **ELAN** tool and ground truth files generated via custom scripts.

<div align="center">

| Property | Detail |
|---|---|
| Language | BISINDO (Bandung Variant) |
| Scenario | Signer-Independent |
| Format | Pre-extracted skeleton (`.pkl`) |
| Annotation Tool | ELAN |
| Keypoint Extractor | MediaPipe Holistic |

</div>

### ⬇️ Download

The pre-extracted skeleton dataset is publicly available on Google Drive:

<div align="center">

**[📥 Download BISINDO Skeleton Dataset (Google Drive)](https://drive.google.com/drive/folders/1jxAJ7VIvrL2X4WpvqhDdmZH6lUXGQrnF?usp=drive_link)**

</div>

Download the entire Google Drive folder and place all of its contents directly inside the `datasets/` directory:

```
datasets/
├── pose_bisindo_train_dev_sd.pkl
├── pose_bisindo_test_sd.pkl
├── pose_bisindo_test_si-maj.pkl
└── pose_bisindo_test_si-min.pkl
```


## 🛠 Prerequisites

Ensure the following are installed before proceeding:

- **Python** 3.8+
- **SciPy** — used for interpolation and downsampling helpers
- **PyTorch `==2.0.0`** — required for `ctcdecode` compatibility → [pytorch.org](https://pytorch.org/get-started/locally/)
- **`ctcdecode==0.4`** — beam search decoder → [[WayenVan/ctcdecode]](https://github.com/WayenVan/ctcdecode)
- **`sclite`** (via Kaldi) — evaluation scoring tool → [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi)

After installing Kaldi, create a soft link to `sclite`:

```bash
mkdir ./software
ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/bisindo-cslr.git
cd bisindo-cslr
pip install -r requirements.txt
```

### 2. Dataset Setup

[Download the dataset](#️-download) from Google Drive and place the files as follows:

```
datasets/
├── pose_bisindo_train_dev_sd.pkl
├── pose_bisindo_test_sd.pkl
├── pose_bisindo_test_si-maj.pkl
└── pose_bisindo_test_si-min.pkl
```

### 3. Preprocessing

Generate the gloss dictionary, dataset info, and ground truth `.stm` files required for evaluation:

```bash
cd preprocess/mslr2025
python mslr_process.py
cd ../../
```

---

## ⚙️ Configuration & Augmentation

All model and training hyperparameters are controlled via YAML files in:

```
configs/experiment_configs/normalization/
```

### Experiment Scenarios

Each scenario uses the same base model, but different normalization and downsampling settings:

| Scenario | `downsampling` | `downsampling_ratio` | `normalization_types` |
|---|---:|---:|---|
| `Baseline` | `true` | `0.5` | `[]` |
| `Baseline+SN` | `true` | `0.5` | `['spatial']` |
| `Baseline+MKR` | `true` | `0.5` | `['missing_kp']` |
| `Baseline+TN` | `true` | `0.5` | `['temporal']` |
| `Baseline+SN+MKR` | `true` | `0.5` | `['spatial', 'missing_kp']` |
| `Baseline+SN+TN` | `true` | `0.5` | `['spatial', 'temporal']` |
| `Baseline+MKR+TN` | `true` | `0.5` | `['missing_kp', 'temporal']` |
| `Baseline+SN+MKR+TN` | `true` | `0.5` | `['spatial', 'missing_kp', 'temporal']` |

### Data Augmentation

Toggle augmentation strategies under `feeder_args`:

```yaml
feeder_args:
  augmentation_types: []
  # Options: ['SpatialJitter', 'SpatialScale', 'TemporalDrop', 'TemporalRescale']
  downsampling: true
  downsampling_ratio: 0.5
  normalization_types: ['spatial', 'missing_kp', 'temporal']
```

The four augmentation types investigated in this thesis:

| Augmentation | Domain | Description |
|---|---|---|
| `SpatialJitter` | Spatial | Adds Gaussian noise (σ-controlled) to keypoint coordinates — simulates natural hand tremor and sensor noise |
| `SpatialScale` | Spatial | Randomly scales keypoint coordinates — simulates signer distance variation |
| `TemporalDrop` | Temporal | Randomly drops frames — simulates frame loss and variable recording conditions |
| `TemporalRescale` | Temporal | Randomly rescales temporal sequence length — simulates signing speed variation |

> Multiple types can be combined, e.g., `['SpatialJitter', 'TemporalDrop']`

### Normalization Order

When enabled in `feeder_args`, the preprocessing pipeline runs in this order:

1. Downsampling
2. Data augmentation
3. Spatial normalization
4. Missing keypoint reconstruction
5. Temporal normalization

---

## 🏋️ Training & Evaluation

### Train

```bash
python main.py --config ./configs/experiment_configs/normalization/Baseline.yaml
```

### Evaluate

```bash
python main.py --config ./configs/experiment_configs/normalization/Baseline.yaml \
               --phase test \
               --load-weights PATH_TO_PRETRAINED_MODEL
```

> Replace `PATH_TO_PRETRAINED_MODEL` with the path to your trained `.pt` checkpoint file.

### Output Layout

This experimental framework is designed for a **single unified training phase followed by parallel evaluation across three distinct testing splits (Signer-Dependent, Signer-Independent Majority, and Signer-Independent Minority)**.

For each experimental scenario, training logs and model checkpoints are stored in the root of the working directory (`work_dir/`), while the prediction outputs (CSV) and quantitative WER performance metrics (TXT) are isolated within respective subdirectories:

```text
work_dir/{scenario_name}/train/
work_dir/{scenario_name}/test/
├── test_sd/
│   ├── test_sd.csv
│   └── test_sd_wer.txt
├── test_si_major/
│   ├── test_si_major.csv
│   └── test_si_major_wer.txt
└── test_si_minor/
    ├── test_si_minor.csv
    └── test_si_minor_wer.txt
```

**Interpreting Final Results:**
- **`*_wer.txt` files**: Serve as the primary quantitative evaluation metric. These files contain the final **Word Error Rate (WER)** percentages computed for both the Conv1D and BiLSTM decoding modules. Lower WER values indicate superior model accuracy for the respective testing split (SD, SI-Major, or SI-Minor).
- **`*.csv` files**: Contain the discrete sequential mapping of video IDs to the raw, predicted gloss sequences. These are provided for qualitative analysis and direct observation of the generated sign language tokens.
- **`*.ctm` files**: Intermediate alignment timestamp records generated and parsed by the Kaldi `sclite` scoring toolkit. These are utilized internally for word-level sequence evaluation and can generally be bypassed unless granular token-level alignment analysis is required.

### Metrics

| Metric | Description | Direction |
|---|---|---|
| **WER** (Word Error Rate) | Edit distance between predicted and reference gloss sequences | ↓ Lower is better |
| **Inference Speed** (seq/s) | Number of sequences processed per second | ↑ Higher is better |

---

## 📁 Repository Structure

```
MSLR_ICCV2025/
│
├── configs/
│   ├── Double_Cosign_sd.yaml
│   ├── dataset_configs/
│   │   └── bisindo.yaml
│   └── experiment_configs/
│       └── normalization/
│           ├── Baseline.yaml
│           ├── Baseline+MKR.yaml
│           ├── Baseline+MKR+TN.yaml
│           ├── Baseline+SN.yaml
│           ├── Baseline+SN+MKR.yaml
│           ├── Baseline+SN+MKR+TN.yaml
│           ├── Baseline+SN+TN.yaml
│           └── Baseline+TN.yaml
│
├── datasets/
│   ├── pose_bisindo_train_dev_sd.pkl
│   ├── pose_bisindo_test_sd.pkl
│   ├── pose_bisindo_test_si-maj.pkl
│   ├── pose_bisindo_test_si-min.pkl
│   ├── skeleton_feeder.py
│   ├── downsample_skeleton.py
│   └── mslr2025/
│       ├── sd_test_list.txt
│       ├── sd_train_list.txt
│       ├── mslr_process.py
│       └── SD/
│           ├── dev.csv
│           ├── test.csv
│           └── train.csv
│
├── docs/
│   └── architecture.png
│
├── evaluation/
│   └── slr_eval/
│       ├── mergectmstm.py
│       ├── preprocess.sh
│       ├── python_wer_evaluation.py
│       ├── wer_calculation.py
│       └── __init__.py
│
├── modules/
│   ├── __init__.py
│   ├── visual_extractor.py
│   ├── criterion/
│   │   └── radialctc.py
│   ├── stgcn_layers/
│   │   ├── __init__.py
│   │   ├── gcn_utils.py
│   │   └── stgcn_block.py
│   └── temporal_layers/
│       ├── __init__.py
│       ├── BiLSTM.py
│       └── tconv.py
│
├── preprocess/
│   └── mslr2025/
│       ├── sd_dev_list.txt
│       ├── sd_test_list.txt
│       ├── sd_train_list.txt
│       └── SD/
│           ├── dev.csv
│           ├── test.csv
│           └── train.csv
│
├── results/
│   ├── log_hidden_state_1024.txt
│   ├── log_hidden_state_256.txt
│   └── log_hidden_state_512.txt
│
├── work_dir/
│   ├── Baseline/
│   ├── Baseline+MKR/
│   ├── Baseline+MKR+TN/
│   ├── Baseline+SN/
│   ├── Baseline+SN+MKR/
│   ├── Baseline+SN+MKR+TN/
│   ├── Baseline+SN+TN/
│   └── Baseline+TN/
│
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
│
├── BASELINE_PIPELINE.md
├── identifikasi_data_hilang.ipynb
├── main.py
├── PANDUAN_COLAB.md
├── PREPROCESSING_WORKFLOW.md
├── README.md
├── requirements.txt
├── seq_scripts.py
├── skenario-eksperimen-01.ipynb
├── slr_network.py
├── TESTING_WORKFLOW.md
├── TRAINING_WORKFLOW.md
└── .gitignore
```

---

## 📚 Citation

If you use this code, dataset, or findings in your research, please cite:

```bibtex
@thesis{pratama2026bisindo,
  title   = {Analisis Konfigurasi Pipeline Pre-Processing pada Model GCN-1DCNN-BiLSTM
             untuk Continuous Sign Language Recognition BISINDO Variasi Bandung
             dalam Skenario Signer-Independent},
  author  = {Pratama, Mahardika and Sarah},
  year    = {2026},
  school  = {Politeknik Negeri Bandung},
  type    = {Laporan Tugas Akhir},
  program = {D-IV Teknik Informatika, Jurusan Teknik Komputer dan Informatika}
}
```

This work builds upon the following:

```bibtex
@inproceedings{min2025closer,
  title     = {A Closer Look at Skeleton-based Continuous Sign Language Recognition},
  author    = {Min, Yuecong and Yang, Yifan and Jiao, Peiqi and Nan, Zixi and Chen, Xilin},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year      = {2025}
}

@inproceedings{jiao2023cosign,
  title     = {CoSign: Exploring Co-occurrence Signals in Skeleton-based Continuous Sign Language Recognition},
  author    = {Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and Chen, Xilin},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     = {20676--20686},
  year      = {2023}
}
```

---

## 🙏 Acknowledgements

This work was conducted as part of an undergraduate thesis at **Politeknik Negeri Bandung**. We gratefully acknowledge the base framework provided by Min et al. (ICCV Workshop 2025) and the CoSign architecture by Jiao et al. (ICCV 2023). We also thank all signers and contributors who participated in the BISINDO dataset collection process.

---

## 📬 Contact

For questions about the research, dataset, or implementation:

| Name | NIM | Institution |
|---|---|---|
| Mahardika Pratama | 221524044 | Politeknik Negeri Bandung |
| Sarah | 221524059 | Politeknik Negeri Bandung |

*Jurusan Teknik Komputer dan Informatika · D-IV Teknik Informatika*

---

<div align="center">
<sub>Made with ❤️ for the Indonesian Deaf community · Politeknik Negeri Bandung · 2026</sub>
</div>
