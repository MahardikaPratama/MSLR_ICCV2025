import os
import sys
import json
import torch
import pickle
import warnings
import itertools
import random

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import torch.utils.data as data
from utils import skeleton_augmentation
from utils import skeleton_transform
from itertools import chain
from scipy.interpolate import interp1d

sys.path.append("..")


class SkeletonFeeder(data.Dataset):
    """Dataset feeder untuk data skeleton BISINDO.

    Kelas ini membaca metadata video, memuat pose dari file pickle, memfilter
    sample yang valid, lalu menyiapkan augmentasi, normalisasi, dan collate
    function untuk dipakai oleh DataLoader.
    """

    def __init__(
        self,
        gloss_dict,
        mode="train",
        setting="sd",
        transform_mode=True,
        datatype="lmdb",
        dataset="bisindo",
        dataset_root="./datasets/mslr2025",
        si_signer=None,
        split=None,
        norm_point=None,
        used_part=None,
        augmentation_types=None,
        normalization_types=None,
        downsampling=False,
        downsampling_ratio=0.5,
        temporal_length=194,
    ):
        """
        Skeleton dataset feeder.

        Parameters
        ----------
        gloss_dict : dict
            Mapping gloss -> index.

        mode : str
            Dataset split:
            train, dev, test_sd, test_si_major, test_si_minor.

        setting : str
            Experimental setting.

        transform_mode : bool
            True for training mode, False for evaluation mode.

        datatype : str
            Input data type.

        dataset : str
            Dataset name.

        dataset_root : str
            Dataset root directory.

        split : list[int], optional
            Cumulative keypoint split positions used by normalization.

        norm_point : list[int], optional
            Reference keypoints used by normalization.

        used_part : list[str], optional
            Selected skeleton parts.

        augmentation_types : list[str], optional
            Online augmentation pipeline.

        normalization_types : list[str], optional
            Offline normalization pipeline.

        downsampling : bool
            Enable temporal downsampling.

        downsampling_ratio : float
            Downsampling ratio.

        temporal_length : int
            Target sequence length for temporal normalization.
        """

        # --------------------------------------------------
        # Basic configuration
        # --------------------------------------------------
        self.mode = mode
        self.dict = gloss_dict
        self.setting = setting
        self.data_type = datatype
        self.transform_mode = "train" if transform_mode else "test"

        self.dataset = dataset
        self.dataset_root = dataset_root

        self.used_part = used_part or []

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------
        info_file = os.path.join(
            self.dataset_root,
            f"{mode}_info.json"
        )

        with open(info_file, "r") as f:
            inputs_list = json.load(f)

        # --------------------------------------------------
        # Load skeleton dictionary
        # --------------------------------------------------
        self.kps_global = self.load_kps_global(mode)

        # --------------------------------------------------
        # Keep only samples with available pose data
        # --------------------------------------------------
        self.inputs_list = [
            item
            for item in inputs_list
            if item["video_id"] in self.kps_global
        ]

        print(
            f"[SkeletonFeeder] "
            f"Loaded {len(self.inputs_list)} valid samples "
            f"for mode='{mode}'"
        )

        # --------------------------------------------------
        # Coordinate normalization divisor
        # --------------------------------------------------
        # Used to map coordinates approximately to [-1, 1]
        # following the original implementation.
        self.norm_div = (10240 - 1) / 2

        # --------------------------------------------------
        # Selected keypoint indices
        # --------------------------------------------------
        self.pose_idx = []

        if self.data_type == "skeleton":

            for part in self.used_part:

                if part == "left_hand":
                    self.pose_idx.extend(range(0, 21))

                elif part == "right_hand":
                    self.pose_idx.extend(range(21, 42))

                elif part == "mouth_8":
                    self.pose_idx.extend(range(42, 61))

                elif part == "body":
                    self.pose_idx.extend(range(61, 86))

                elif part == "upper_limb":
                    self.pose_idx.extend(range(72, 84))

                else:
                    raise ValueError(
                        f"Unknown skeleton part: {part}"
                    )

        # --------------------------------------------------
        # Normalization configuration
        # --------------------------------------------------
        self.split = split
        self.norm_point = norm_point

        if self.split is not None:
            assert len(self.split) == len(self.used_part), (
                "split length must match number of used_part."
            )

        if self.norm_point is None:
            print("[SkeletonFeeder] No centralization point specified.")

        # --------------------------------------------------
        # Augmentation configuration
        # --------------------------------------------------
        self.augmentation_types = (
            augmentation_types
            if augmentation_types is not None
            else []
        )

        self.data_aug = self.pose_transform()

        # --------------------------------------------------
        # Normalization configuration
        # --------------------------------------------------
        self.temporal_length = temporal_length

        self.normalization_types = (
            normalization_types
            if normalization_types is not None
            else []
        )

        print(
            f"[SkeletonFeeder] "
            f"Normalization pipeline: {self.normalization_types}"
        )

        # --------------------------------------------------
        # Downsampling configuration
        # --------------------------------------------------
        self.downsampling = downsampling
        self.downsampling_ratio = downsampling_ratio

        if self.downsampling:
            print(
                f"[SkeletonFeeder] "
                f"Downsampling enabled "
                f"(ratio={self.downsampling_ratio})"
            )

        # --------------------------------------------------
        # Build preprocessed samples
        # --------------------------------------------------
        self.samples = self.build_samples()

    def __getitem__(self, idx):
        """
        Retrieve one sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            (
                input_data,
                label,
                original_info
            )
        """
        if self.data_type != "skeleton":
            raise NotImplementedError(
                f"Unsupported data type: {self.data_type}"
            )

        input_data, label, fi = self.samples[idx]

        input_data = self.apply_online_transform(input_data)

        return (
            input_data,
            label,
            fi,
        )

    def read_pose(self, index):
        """
        Read pose sequence and gloss labels.
        """

        fi = self.inputs_list[index]

        pose_data = self.kps_global[
            fi["video_id"]
        ]["keypoints"]

        gloss_sequence = fi["gloss_sequence"]

        label_list = []

        for gloss in gloss_sequence.split():

            if gloss not in self.dict:
                raise KeyError(
                    f"Unknown gloss: {gloss}"
                )

            label_list.append(
                self.dict[gloss]
            )

        return (
            pose_data,
            label_list,
            fi,
        )


    def load_kps_global(self, mode):
        """
        Load skeleton dictionary for the selected dataset split.

        Parameters
        ----------
        mode : str
            Dataset split.

        Returns
        -------
        dict
            Skeleton dictionary indexed by video_id.
        """

        mode_to_file = {
            "train": "pose_bisindo_train_dev_sd.pkl",
            "dev": "pose_bisindo_train_dev_sd.pkl",
            "test_sd": "pose_bisindo_test_sd.pkl",
            "test_si_major": "pose_bisindo_test_si-maj.pkl",
            "test_si_minor": "pose_bisindo_test_si-min.pkl",
        }

        if mode not in mode_to_file:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        pkl_file = os.path.join(
            self.dataset_root,
            mode_to_file[mode]
        )

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(
                f"Pose file not found: {pkl_file}. "
                f"Expected file under dataset_root: {self.dataset_root}"
            )

        with open(pkl_file, "rb") as f:
            return pickle.load(f)


    def build_samples(self):
        """
        Build preprocessed sample cache.

        Returns
        -------
        list
            List of tuples:
            (
                video,
                label,
                original_info
            )
        """
        if self.data_type != "skeleton":
            return []

        samples = []

        for index in range(len(self.inputs_list)):

            pose_data, label, fi = self.read_pose(index)

            # --------------------------------------------------
            # Select keypoints and xy coordinates
            # --------------------------------------------------
            pose_data = pose_data[:, self.pose_idx, :2]

            pose_data = torch.from_numpy(
                pose_data
            ).float()

            # --------------------------------------------------
            # Deterministic preprocessing
            # --------------------------------------------------
            pose_data = self.normalize(
                pose_data
            )

            # --------------------------------------------------
            # Confidence channel
            # Placeholder channel (not used)
            # --------------------------------------------------
            conf = torch.zeros(
                pose_data.shape[:2],
                dtype=pose_data.dtype,
            )

            # --------------------------------------------------
            # Forward / backward motion
            # --------------------------------------------------
            total_motion = torch.zeros(
                pose_data.shape[:2] + (4,),
                dtype=pose_data.dtype,
            )

            total_motion[1:, :, 0:2] = (
                pose_data[1:, :, 0:2]
                - pose_data[:-1, :, 0:2]
            )

            total_motion[:-1, :, 2:4] = (
                pose_data[:-1, :, 0:2]
                - pose_data[1:, :, 0:2]
            )

            # --------------------------------------------------
            # Final feature tensor
            # [x, y, dx+, dy+, dx-, dy-, conf]
            # --------------------------------------------------
            final = torch.cat(
                [
                    pose_data,
                    total_motion,
                    conf.unsqueeze(-1),
                ],
                dim=-1,
            )

            samples.append(
                (
                    final,
                    torch.LongTensor(label),
                    fi["original_info"],
                )
            )

        return samples


    def normalize(self, video):
        """
        Apply deterministic preprocessing pipeline.

        Pipeline order:
            1. Missing Keypoint Reconstruction (MKR)
            2. Spatial Normalization (SN)
            3. Temporal Normalization (TN)
            4. Temporal Downsampling (optional)

        Parameters
        ----------
        video : Tensor
            Skeleton sequence with shape (T, K, C).

        Returns
        -------
        Tensor
            Preprocessed skeleton sequence.
        """
        if self.data_type != "skeleton":
            return video

        output = video

        # --------------------------------------------------
        # 1. Missing Keypoint Reconstruction
        # --------------------------------------------------
        if "missing_kp" in self.normalization_types:
            output = skeleton_transform.missing_keypoint_reconstruction(
                output
            )

        # --------------------------------------------------
        # 2. Spatial Normalization
        # --------------------------------------------------
        if "spatial" in self.normalization_types:
            output = skeleton_transform.spatial_normalize(
                output,
                norm_div=self.norm_div,
                norm_point=self.norm_point,
                split=self.split,
                used_part=self.used_part,
            )

        # --------------------------------------------------
        # 3. Temporal Normalization
        # --------------------------------------------------
        if "temporal" in self.normalization_types:
            output = skeleton_transform.temporal_normalize(
                output,
                target_length=self.temporal_length,
            )

        # --------------------------------------------------
        # 4. Optional Temporal Downsampling
        # --------------------------------------------------
        if self.downsampling:
            output = skeleton_transform.downsample(
                output,
                ratio=self.downsampling_ratio,
            )

        return output


    def apply_online_transform(self, input_data):
        """
        Apply online augmentation during training.
        """

        if self.transform_mode != "train":
            return input_data

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        output = self.data_aug(input_data)

        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(
                output,
                dtype=torch.float32,
            )

        return output




    def pose_transform(self):
        """
        Build online augmentation pipeline.
        """

        transforms = []

        if self.transform_mode == "train":

            print(
                f"[SkeletonFeeder] "
                f"Training augmentations: "
                f"{self.augmentation_types}"
            )

            if "TemporalDrop" in self.augmentation_types:
                transforms.append(
                    skeleton_augmentation.TemporalDropout(
                        max_dp=0.25
                    )
                )

            if "TemporalRescale" in self.augmentation_types:
                transforms.append(
                    skeleton_augmentation.TemporalRescale(
                        temp_scaling=0.2
                    )
                )

            if "SpatialScale" in self.augmentation_types:
                transforms.append(
                    skeleton_augmentation.Scale(
                        scale_range=(0.8, 1.2)
                    )
                )

            if "SpatialJitter" in self.augmentation_types:
                transforms.append(
                    skeleton_augmentation.Jitter(
                        sigma=0.003
                    )
                )

        else:

            print(
                "[SkeletonFeeder] "
                "Test transform: ToTensor only."
            )

        transforms.append(
            skeleton_augmentation.ToTensor()
        )

        return skeleton_augmentation.Compose(
            transforms
        )


    def __len__(self):
        """Return the number of available samples."""
        return len(self.samples)


    @staticmethod
    def collate_fn(batch):
        """
        Merge a list of samples into a mini-batch.

        This function performs:
            1. Sorting by sequence length (descending).
            2. Temporal padding at the beginning and end.
            3. Alignment of sequence length to a multiple of 4.
            4. Construction of batched video tensors.
            5. Flattening of label sequences for CTC-based training.

        Parameters
        ----------
        batch : list
            List of samples returned by __getitem__():

            (
                video,
                label,
                original_info
            )

        Returns
        -------
        dict
            {
                'x'           : padded video tensor,
                'len_x'       : padded sequence lengths,
                'label'       : flattened label tensor,
                'label_lgt'   : label lengths,
                'origin_info' : original sample metadata
            }
        """

        # --------------------------------------------------
        # Sort sequences by descending length
        # --------------------------------------------------
        batch = sorted(
            batch,
            key=lambda x: len(x[0]),
            reverse=True
        )

        video, label, info = zip(*batch)

        length = [len(vid) for vid in video]
        max_len = max(length)

        # --------------------------------------------------
        # Temporal length used by the original implementation.
        # Lengths are aligned to a multiple of 4 and include
        # left/right temporal context padding (+12 frames).
        # --------------------------------------------------
        video_length = torch.LongTensor([
            np.ceil(len(vid) / 4.0) * 4 + 12
            for vid in video
        ])

        # --------------------------------------------------
        # Temporal padding
        # --------------------------------------------------
        left_pad = 6
        right_pad = (
            int(np.ceil(max_len / 4.0)) * 4
            - max_len
            + 6
        )

        max_len = max_len + left_pad + right_pad

        # --------------------------------------------------
        # Replicate padding using first and last frames
        # --------------------------------------------------
        padded_video = [
            torch.cat(
                (
                    vid[0][None].expand(
                        left_pad,
                        -1,
                        -1,
                    ),
                    vid,
                    vid[-1][None].expand(
                        max_len - len(vid) - left_pad,
                        -1,
                        -1,
                    ),
                ),
                dim=0,
            )
            for vid in video
        ]

        padded_video = torch.stack(padded_video)

        # --------------------------------------------------
        # Label lengths
        # --------------------------------------------------
        label_length = torch.LongTensor(
            [len(lab) for lab in label]
        )

        # --------------------------------------------------
        # Flatten labels for CTC training
        # --------------------------------------------------
        padded_label = []

        for lab in label:
            padded_label.extend(lab)

        padded_label = torch.LongTensor(padded_label)

        return {
            "x": padded_video,
            "len_x": video_length,
            "label": padded_label,
            "label_lgt": label_length,
            "origin_info": info,
        }