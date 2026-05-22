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
        dataset='bisindo',
        dataset_root="./datasets/mslr2025",
        si_signer=None,
        split=None,
        norm_point=None,
        used_part=None,
        augmentation_types=None,
        normalization_types=None,
        downsampling=False,
        downsampling_ratio=0.5,
        temporal_length=194):
        """Inisialisasi feeder dan seluruh pipeline data.

        Input:
        1. gloss_dict: kamus gloss ke index.
        2. mode: split dataset yang dipakai.
        3. setting: konfigurasi eksperimen.
        4. transform_mode: penanda mode train atau test.
        5. datatype: jenis data yang diproses.
        6. dataset: nama dataset.
        7. dataset_root: folder utama file dataset.
        8. split, norm_point, used_part: parameter normalisasi skeleton.
        9. augmentation_types: daftar augmentasi yang diaktifkan.
        10. normalization_types: daftar normalisasi yang diaktifkan.
        11. downsampling dan downsampling_ratio: opsi pemendekan urutan video.
        12. temporal_length: panjang frame target untuk normalisasi temporal (default: 194).

        Proses:
        1. Menyimpan argumen dasar ke atribut class.
        2. Membaca metadata video dari file JSON.
        3. Memuat pose global sesuai mode.
        4. Memfilter input yang benar-benar punya pose.
        5. Menyiapkan indeks keypoint, augmentasi, dan normalisasi.

        Output:
        1. Instance SkeletonFeeder yang siap digunakan DataLoader.
        """
        self.mode = mode  # Mode data (train/dev/test)
        self.mode_list = mode.split("_")  # Untuk mode gabungan (misal: train_dev)
        self.dict = gloss_dict  # Kamus gloss (gloss ke index)
        self.setting = setting  # Setting eksperimen (sd/si)
        self.data_type = datatype  # Jenis data (skeleton/lmdb)
        self.transform_mode = "train" if transform_mode else "test"  # Mode augmentasi
        self.dataset = dataset  # Nama dataset
        self.dataset_root = dataset_root
        self.used_part = used_part  # Bagian skeleton yang digunakan

        # Load file info sesuai mode
        # Mode yang didukung: train, dev, test_sd, test_si_major, test_si_minor
        info_file = os.path.join(self.dataset_root, f"{mode}_info.json")
        with open(info_file, 'r') as f:
            inputs_list = json.load(f)

        # Load file pose (pickle) sesuai mode                
        self.kps_global = self.load_kps_global(mode)

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

        # Panjang target untuk normalisasi temporal (resampling)
        self.temporal_length = temporal_length

        # Jenis normalisasi yang dapat diaktifkan via config:
        # 'spatial'    : Normalisasi rentang dan sentralisasi skeleton
        # 'missing_kp' : Rekonstruksi keypoint hilang (interpolasi)
        # 'temporal'   : Normalisasi panjang urutan (resampling)
        self.normalization_types = normalization_types if normalization_types else []
        print(f"[SkeletonFeeder] Normalization pipeline: {self.normalization_types}")

        # Downsampling config
        self.downsampling = downsampling
        self.downsampling_ratio = downsampling_ratio
        if self.downsampling:
            print(f"[SkeletonFeeder] Downsampling enabled, ratio: {self.downsampling_ratio}")

    # Mengambil satu sample data (dipanggil oleh DataLoader)
    def __getitem__(self, idx):
        """Mengambil satu sample data berdasarkan indeks.

        Input:
        1. idx: indeks sample di dalam self.inputs_list.

        Proses:
        1. Membaca pose dan label dari sample terpilih.
        2. Memilih bagian skeleton sesuai pose_idx.
        3. Menghitung fitur motion antar frame.
        4. Menggabungkan pose, motion, dan confidence.
        5. Menjalankan pipeline normalisasi dan augmentasi.

        Output:
        1. Tensor input yang sudah diproses.
        2. Label dalam bentuk LongTensor.
        3. Informasi asli sample untuk logging/evaluasi.
        """
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
        """Membentuk ulang daftar input dengan membuang signer tertentu.

        Proses:
        1. Menelusuri seluruh item pada inputs_list.
        2. Mengabaikan data dengan signer 'Signer05'.
        3. Menambahkan elemen terakhir yang dipilih manual.

        Output:
        1. List baru yang sudah disaring.
        """
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
        """Membaca pose mentah dan label gloss untuk satu sample.

        Input:
        1. index: indeks sample yang akan dibaca.
        2. num_glosses: parameter opsional untuk membatasi jumlah gloss.

        Proses:
        1. Mengambil metadata video dari self.inputs_list.
        2. Mengambil keypoints pose dari self.kps_global berdasarkan video_id.
        3. Mengubah string gloss_sequence menjadi daftar index gloss.

        Output:
        1. pose_data: array keypoints mentah.
        2. label_list: daftar index label gloss.
        3. fi: metadata lengkap sample.
        """
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


    # Memuat file pickle pose berdasarkan mode
    def load_kps_global(self, mode):
        """Memuat dictionary pose global dari file pickle sesuai mode.

        Input:
        1. mode: split dataset yang sedang dipakai.

        Proses:
        1. Menentukan nama file pickle berdasarkan mode.
        2. Mengecek apakah file tersebut tersedia.
        3. Membaca isi pickle dan mengembalikannya sebagai dictionary.

        Output:
        1. Dictionary pose global yang dipetakan berdasarkan video_id.
        """
        if mode == 'train' or mode == 'dev':
            pkl_file = os.path.join(self.dataset_root, "pose_bisindo_train_dev_sd.pkl")
        elif mode == 'test_sd':
            pkl_file = os.path.join(self.dataset_root, "pose_bisindo_test_sd.pkl")
        elif mode == 'test_si_major':
            pkl_file = os.path.join(self.dataset_root, "pose_bisindo_test_si-maj.pkl")
        elif mode == 'test_si_minor':
            pkl_file = os.path.join(self.dataset_root, "pose_bisindo_test_si-min.pkl")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(
                f"Pose file tidak ditemukan: {pkl_file}. "
                f"Pastikan file .pkl berada di dataset_root: {self.dataset_root}"
            )

        with open(pkl_file, "rb") as f:
            return pickle.load(f)


    # Pipeline normalisasi dan augmentasi skeleton
    def downsample(self, video, ratio=0.5):
        """Melakukan downsampling temporal pada urutan video.

        Input:
        1. video: array atau tensor dengan bentuk (T, K, C).
        2. ratio: rasio sampling ulang, bernilai antara 0 dan 1.

        Proses:
        1. Jika rasio tidak valid, video dikembalikan apa adanya.
        2. Menghitung jumlah frame baru berdasarkan ratio.
        3. Mengambil indeks frame secara merata sepanjang video.

        Output:
        1. Video yang sudah dipendekkan secara temporal.
        """
        if ratio >= 1.0 or ratio <= 0.0:
            return video
        T = video.shape[0]
        new_len = max(1, int(T * ratio))
        idx = np.linspace(0, T - 1, new_len).astype(int)
        if isinstance(video, torch.Tensor):
            return video[idx]
        else:
            return video[idx, ...]

    def normalize(self, video, label=None, file_id=None):
        """Menjalankan pipeline normalisasi dan augmentasi.

        Input:
        1. video: tensor hasil gabungan pose dan motion.
        2. label: label opsional.
        3. file_id: identitas sample opsional.

        Proses:
        1. Downsampling jika diaktifkan.
        2. Menjalankan transformasi augmentasi dari self.data_aug.
        3. Menjalankan spatial normalization bila dipilih.
        4. Menjalankan rekonstruksi keypoint hilang bila dipilih.
        5. Menjalankan temporal normalization bila dipilih.

        Output:
        1. Tensor video yang sudah diproses.
        """
        if self.data_type != 'skeleton':
            return video

        # 0. Augmentasi (ToTensor wajib)
        input_data = self.data_aug(video)

        # 1. Spatial normalization
        if 'spatial' in self.normalization_types:
            input_data = self.spatial_normalize(input_data)

        # 2. Missing keypoint reconstruction
        if 'missing_kp' in self.normalization_types:
            input_data = self.missing_keypoint_reconstruction(input_data)

        # 3. Temporal normalization (resample ke panjang target yang dapat disetel)
        if 'temporal' in self.normalization_types:
            input_data = self.temporal_normalize(input_data, target_length=self.temporal_length)

        # 4. Downsampling paling akhir supaya target temporal tetap mengacu ke data original
        if self.downsampling:
            input_data = self.downsample(input_data, self.downsampling_ratio)

        return input_data


    # Normalisasi skeleton ke rentang [-1, 1] dan sentralisasi
    def spatial_normalize(self, origin_input_data):
        """Menormalkan koordinat skeleton ke rentang yang lebih kecil.

        Input:
        1. origin_input_data: tensor dengan channel koordinat dan fitur tambahan.

        Proses:
        1. Mengambil confidence dari channel terakhir.
        2. Menskalakan nilai mentah menggunakan self.norm_div.
        3. Mengambil koordinat xy untuk disentralisasi per bagian tubuh.
        4. Mengembalikan tensor gabungan koordinat ter-normalisasi dan fitur lain.

        Output:
        1. Tensor skeleton yang sudah dinormalisasi secara spasial.
        """
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
        """Mengisi koordinat keypoint yang hilang dengan interpolasi temporal.

        Input:
        1. origin_input_data: tensor (T, K, C) yang sudah menjadi tensor.

        Proses:
        1. Menyalin input ke tensor hasil agar aman dimodifikasi.
        2. Mengambil koordinat xy ke NumPy untuk interpolasi.
        3. Mendeteksi frame yang missing pada tiap keypoint.
        4. Mengisi nilai kosong dengan interpolasi atau frame terdekat.
        5. Menulis kembali hasil ke tensor output.

        Output:
        1. Tensor dengan koordinat keypoint yang sudah direkonstruksi.
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
        """Menyesuaikan panjang urutan video ke jumlah frame target.

        Input:
        1. origin_input_data: tensor (T, K, C) sebagai input awal.
        2. target_length: panjang frame yang ingin dihasilkan.

        Proses:
        1. Jika panjang sudah sama, data dikembalikan tanpa perubahan.
        2. Mengonversi data ke NumPy untuk interpolasi.
        3. Membuat grid indeks lama dan baru.
        4. Melakukan interpolasi linear untuk setiap keypoint dan channel.

        Output:
        1. Tensor dengan panjang temporal sesuai target_length.
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
        """Membangun pipeline augmentasi sesuai mode training atau testing.

        Proses:
        1. Jika mode train, menyusun daftar augmentasi sesuai augmentation_types.
        2. Menambahkan ToTensor sebagai transform terakhir.
        3. Jika mode test, hanya memakai ToTensor.

        Output:
        1. Objek Compose yang siap dipanggil pada data skeleton.
        """
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
        """Mengembalikan jumlah sample yang tersedia setelah filtering."""
        return len(self.inputs_list)


    # Fungsi untuk menggabungkan batch (custom collate)
    @staticmethod
    def collate_fn(batch):
        """Menggabungkan daftar sample menjadi satu batch untuk DataLoader.

        Input:
        1. batch: list sample hasil __getitem__.

        Proses:
        1. Mengurutkan sample berdasarkan panjang video.
        2. Menghitung panjang asli dan panjang padding.
        3. Melakukan padding frame depan dan belakang.
        4. Menyusun label dan metadata ke format batch.

        Output:
        1. Dict batch yang berisi video, panjang video, label, dan info asal.
        """
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