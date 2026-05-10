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
from itertools import chain
from scipy.interpolate import interp1d

sys.path.append("..")


# Kelas utama untuk memproses dan menyediakan data skeleton untuk training/testing
class SkeletonFeeder(data.Dataset):
    def __init__(
        self,
        gloss_dict,
        mode="train",
        setting="sd",
        transform_mode=True,
        datatype="lmdb",
        dataset='bisindo',
        si_signer=None,
        split=None,
        norm_point=None,
        used_part=None,
        augmentation_types=None,
    ):
        self.mode = mode  # Mode data (train/dev/test)
        self.mode_list = mode.split("_")  # Untuk mode gabungan (misal: train_dev)
        self.dict = gloss_dict  # Kamus gloss (gloss ke index)
        self.setting = setting  # Setting eksperimen (si/us)
        self.data_type = datatype  # Jenis data (skeleton/lmdb)
        self.transform_mode = "train" if transform_mode else "test"  # Mode augmentasi
        self.dataset = dataset  # Nama dataset
        self.used_part = used_part  # Bagian skeleton yang digunakan

        # Memuat data pose dan info video
        # Untuk mode train/dev/test
        if len(self.mode_list) == 2:
            # Jika mode gabungan (misal: train_dev), gabungkan dua file info
            inputs_list = []
            for mode_type in self.mode_list:
                with open(f"./datasets/mslr2025/{self.setting}_{mode_type}_info.json", 'r') as f:
                    inputs_list_temp = json.load(f)
                    inputs_list.extend(inputs_list_temp)
        else:
            # Jika mode tunggal, load satu file info
            with open(f"./datasets/mslr2025/{self.setting}_{mode}_info.json", 'r') as f:
                inputs_list = json.load(f)
                
        # Load file pickle pose sesuai mode
        pkl_file = "./datasets/pose_bisindo_test.pkl" if mode == 'test' else "./datasets/pose_bisindo_train_dev.pkl"
        with open(pkl_file, "rb") as f:
            self.kps_global = pickle.load(f)

        # Filter hanya video yang ada di pose
        self.inputs_list = list()
        for item in inputs_list:
            if item['video_id'] in self.kps_global.keys():
                self.inputs_list.append(item)
            else:
                pass  # Abaikan video yang tidak ditemukan

        self.norm_div = (10240 - 1) / 2  # Nilai normalisasi skeleton
        print(mode, len(self))  # Print info jumlah data

        # Menentukan index bagian pose yang digunakan
        if self.data_type == 'skeleton':
            self.pose_idx = []
            for part in self.used_part:
                if part == 'body':
                    self.pose_idx += [i for i in range(61, 86)]  # Index body
                elif part == 'hand21':
                    self.pose_idx += [i for i in range(0, 21)]  # Index tangan kiri
                    self.pose_idx += [i for i in range(21, 42)]  # Index tangan kanan
                elif part == 'mouth_8':
                    self.pose_idx += [i for i in range(42, 61)]  # Index mulut

        self.split = split  # Untuk normalisasi per bagian
        self.norm_point = norm_point  # Titik pusat normalisasi
        if norm_point is None:
            print('no centeralization')
        
        self.augmentation_types = augmentation_types if augmentation_types else []
        self.data_aug = self.pose_transform()  # Pipeline augmentasi diaktifkan lewat config

    # Mengambil satu sample data (dipanggil oleh DataLoader)
    def __getitem__(self, idx):
        if self.data_type == 'skeleton':
            input_data, label, fi = self.read_pose(idx)  # Ambil pose dan label
            input_data = input_data[:, self.pose_idx, :2]  # Ambil bagian pose yang dipilih
            conf = np.zeros_like(input_data)[:, :, 0]  # Confidence dummy

            # Hitung fitur gerak (motion)
            total_motion = np.zeros(input_data.shape[0:2] + (4,))
            total_motion[1:, :, 0:2] = input_data[1:, :, 0:2] - input_data[0:-1, :, 0:2]  # Delta maju
            total_motion[0:-1, :, 2:4] = input_data[:-1, :, 0:2] - input_data[1:, :, 0:2]  # Delta mundur

            # Gabungkan pose, motion, dan confidence
            final = np.concatenate([input_data, total_motion, conf[:,:,None]], axis=-1)

            input_data = self.normalize(final)  # Normalisasi dan augmentasi
            return (
                input_data,
                torch.LongTensor(label),
                self.inputs_list[idx]['original_info'],
            )


    # Fungsi opsional untuk menghapus data tidak valid (tidak dipakai utama)
    def deleteInvalidInputs(self):
        new_list = []
        for index in range(len(self.inputs_list)-1):
            fi = self.inputs_list[index]
            signer = fi['signer']
            if not signer == 'Signer05':
                new_list.append(fi)
        new_list.append(self.inputs_list['prefix'])
        return new_list


    # Membaca pose dan label untuk satu video
    def read_pose(self, index, num_glosses=-1):
        fi = self.inputs_list[index]  # Info video
        pose_data = self.kps_global[fi['video_id']]['keypoints']  # Pose
        label = fi['gloss_sequence']  # Label gloss
        label_list = []
        for phase in label.split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase])  # Konversi gloss ke index
        return (
            pose_data,
            label_list,
            fi,
        )


    # Normalisasi dan augmentasi data skeleton
    def normalize(self, video, label=None, file_id=None):
        if self.data_type == 'skeleton':
            input_data = self.data_aug(video)  # Selalu panggil data_aug (minimal ToTensor)
            input_data = self.spatial_normalize(input_data)  # Normalisasi range
            return input_data


    # Normalisasi skeleton ke rentang [-1, 1] dan sentralisasi
    def spatial_normalize(self, origin_input_data):
        conf = origin_input_data[:,:,6]  # Ambil confidence
        origin_input_data = origin_input_data / self.norm_div - 1  # Normalisasi range

        input_data = origin_input_data[:, :, 0:2]  # Ambil koordinat xy
        if self.norm_point is not None:
            index = 0
            for part in self.used_part:
                if index == 0:
                    start, end = 0, self.split[0]
                else:
                    start, end = self.split[index-1], self.split[index]
                if part == 'body':
                    # Sentralisasi body
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[0,self.norm_point[index]:self.norm_point[index]+2].mean(0)[None,None]
                    )
                elif part == 'hand21':
                    # Sentralisasi tangan kiri
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                    index += 1
                    start, end = self.split[index-1], self.split[index]
                    # Sentralisasi tangan kanan
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                else:
                    # Sentralisasi bagian lain
                    input_data[:, start:end] = (
                        input_data[:, start:end] - input_data[:,self.norm_point[index]][:,None,:]
                    )
                index += 1
        # Gabungkan hasil normalisasi dan fitur lain
        return torch.cat(
            [input_data, origin_input_data[:, :, 2:6], conf.unsqueeze(-1)], dim=-1
        )
    
    # Rekonstruksi keypoint hilang menggunakan interpolasi linier temporal
    def missing_keypoint_reconstruction(self, origin_input_data):
        """
        origin_input_data : torch.Tensor (T, K, C)

        Input  :
            channel 0: x
            channel 1: y
            channel lain tetap dipertahankan

        Output :
            torch.Tensor (T, K, C)
        """

        result = origin_input_data.clone()

        # Ambil koordinat xy
        kp_xy = result[:, :, 0:2].cpu().numpy().astype(float)

        T, K, _ = kp_xy.shape

        for k in range(K):

            coords = kp_xy[:, k, :]  # (T, 2)

            # Keypoint dianggap missing jika x == 0 dan y == 0
            valid_mask = ~(
                (coords[:, 0] == 0) &
                (coords[:, 1] == 0)
            )

            valid_idx = np.where(valid_mask)[0]

            # Semua frame missing
            if len(valid_idx) == 0:
                continue

            for t in range(T):

                # Skip jika valid
                if valid_mask[t]:
                    continue

                prev_arr = valid_idx[valid_idx < t]
                next_arr = valid_idx[valid_idx > t]

                # Interpolasi linier
                if len(prev_arr) and len(next_arr):

                    p = prev_arr[-1]
                    n = next_arr[0]

                    alpha = (t - p) / (n - p)

                    coords[t] = (
                        (1 - alpha) * coords[p] +
                        alpha * coords[n]
                    )

                # Gunakan frame sebelumnya
                elif len(prev_arr):

                    coords[t] = coords[prev_arr[-1]]

                # Gunakan frame berikutnya
                elif len(next_arr):

                    coords[t] = coords[next_arr[0]]

            kp_xy[:, k, :] = coords

        # Masukkan kembali hasil rekonstruksi
        result[:, :, 0:2] = torch.from_numpy(kp_xy).to(result.device)

        return result


    # Normalisasi temporal dengan resampling interpolasi linier
    def temporal_normalize(self, origin_input_data, target_length):
        """
        origin_input_data : torch.Tensor (T, K, C)
        target_length     : int

        Output :
            torch.Tensor (target_length, K, C)
        """

        T, K, C = origin_input_data.shape

        # Jika panjang sudah sesuai
        if T == target_length:
            return origin_input_data.clone()

        data = origin_input_data.cpu().numpy()

        orig_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, target_length)

        result = np.zeros(
            (target_length, K, C),
            dtype=data.dtype
        )

        # Interpolasi setiap keypoint dan channel
        for k in range(K):
            for c in range(C):

                fn = interp1d(
                    orig_idx,
                    data[:, k, c],
                    kind='linear'
                )

                result[:, k, c] = fn(new_idx)

        return torch.from_numpy(result).to(origin_input_data.device)

    

    # Membuat pipeline augmentasi (training/test)
    def pose_transform(self):
        if self.transform_mode == "train":
            print(f"Apply training transform: {self.augmentation_types}")
            transforms = []
            if "TemporalDrop" in self.augmentation_types:
                transforms.append(skeleton_augmentation.TemporalDropout(0.25))
            if "TemporalRescale" in self.augmentation_types:
                transforms.append(skeleton_augmentation.TemporalRescale(0.2))
            if "SpatialScale" in self.augmentation_types:
                transforms.append(skeleton_augmentation.Scale((0.8, 1.2)))
            if "SpatialJitter" in self.augmentation_types:
                transforms.append(skeleton_augmentation.Jitter(0.003))
                
            transforms.append(skeleton_augmentation.ToTensor())
            return skeleton_augmentation.Compose(transforms)
        else:
            print("Apply test transform.")
            return skeleton_augmentation.Compose([skeleton_augmentation.ToTensor()])


    # Mengembalikan jumlah data
    def __len__(self):
        return len(self.inputs_list) - 1


    # Fungsi untuk menggabungkan batch (custom collate)
    @staticmethod
    def collate_fn(batch):
        # Urutkan batch berdasarkan panjang video (descending)
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))  # Unzip
        length = [len(vid) for vid in video]
        max_len = max(length)
        # Hitung panjang video setelah padding
        video_length = torch.LongTensor(
            [np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video]
        )
        left_pad = 6
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
        max_len = max_len + left_pad + right_pad
        # Padding awal dan akhir
        padded_video = [
            torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1),  # Padding awal
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1),  # Padding akhir
                ),
                dim=0,
            )
            for vid in video
        ]
        padded_video = torch.stack(padded_video)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            # Jika tidak ada label, return tuple kosong
            return padded_video, video_length, [], [], info
        else:
            # Padding label
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return {
                'x': padded_video,
                'len_x': video_length,
                'label': padded_label,
                'label_lgt': label_length,
                'origin_info': info
            }