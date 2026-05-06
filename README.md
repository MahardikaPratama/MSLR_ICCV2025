# Skeleton-based Continuous Sign Language Recognition for BISINDO

🏆 This repository is an adaptation of the official repository for *A Closer Look at Skeleton-based Continuous Sign Language Recognition* (winner of ICCV 2025 SignEval 2025). This version has been specifically tailored to support **Signer-Dependent (SD)** tasks using the **BISINDO (Indonesian Sign Language)** dataset.

The core implementation is built upon [VAC](https://github.com/VIPL-SLP/VAC_CSLR) and [CoSign](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html) frameworks.

## Prerequisites

- This project is implemented in Pytorch (recommended `==2.0.0` to be compatible with `ctcdecode` and prevent errors). Thus, please install Pytorch first.
- `ctcdecode==0.4` [[WayenVan/ctcdecode]](https://github.com/WayenVan/ctcdecode), for beam search decode.
- `sclite` [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install the kaldi tool to get sclite for evaluation. After installation, create a soft link to the sclite:  

```bash
mkdir ./software
ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
```

## Setup Instructions

1. **Download the BISINDO dataset**. Download the pre-extracted skeleton pickle files and place them in the `./datasets` folder.
   - `pose_bisindo_test.pkl`
   - `pose_bisindo_train_dev.pkl`

2. **Preprocess the dataset**. Run the command to generate the gloss dict, dataset info, and groundtruth (`.stm` files) for evaluation.

```bash
cd ./preprocess/mslr2025
python mslr_process.py
cd ../../
```

## Configuration & Augmentation

The model uses a configuration file located at `configs/bisindo_sd.yaml`.
You can configure dynamic data augmentation during training directly from this YAML file by modifying `augmentation_types` under `feeder_args`:
```yaml
feeder_args:
  augmentation_types: [] # Options: ['SpatialJitter', 'SpatialScale', 'TemporalDrop', 'TemporalRescale']
```

## Running the Model

### Signer Dependent (BISINDO)

- **Train:** Run the following command to start training the model:

```bash
python main.py --config ./configs/bisindo_sd.yaml
```

- **Test:** Run the following command for evaluation (testing):

```bash
python main.py --config ./configs/bisindo_sd.yaml --phase test --load-weights PATH_TO_PRETRAINED_MODEL
```

*(Note: Replace `PATH_TO_PRETRAINED_MODEL` with your trained `.pt` model file path).*

## Citation

If you find the base architectures useful in your research works, please consider citing:

```latex
@inproceedings{min2025closer,
  title={A Closer Look at Skeleton-based Continuous Sign Language Recognition},
  author={Min, Yuecong and Yang, Yifan and Jiao, Peiqi and Nan, Zixi and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  year={2025}
}

@inproceedings{jiao2023cosign,
  title={Cosign: Exploring co-occurrence signals in skeleton-based continuous sign language recognition},
  author={Jiao, Peiqi and Min, Yuecong and Li, Yanan and Wang, Xiaotao and Lei, Lei and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20676--20686},
  year={2023}
}
```
