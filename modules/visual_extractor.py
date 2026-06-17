import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

from .stgcn_layers import Graph, STGCN_block


def generate_mask(shape, part_num, clip_length, ratio, dim):
    """
    Generate two complementary masks for Consistency Regularization (CR).

    Parameters
    ----------
    shape : tuple
        Shape of the feature tensor (B, T, C).
    part_num : int
        Number of spatial parts.
    clip_length : int
        Temporal segment length for mask granularity.
    ratio : float
        Proportion of elements masked per view (0 to 0.5).
    dim : int
        Channel width per part.

    Returns
    -------
    tuple
        mask_cat_q and mask_cat_k, complementary mask tensors of shape (B, T, C).
    """
    # unpack dimensi tensor fitur: batch, time, channel
    B, T, C = shape
    # hitung jumlah klip temporal berdasarkan panjang klip
    clips = T // clip_length
    # buat random mask boolean (B, clips, part_num):
    # True di posisi yang akan dimask di salah satu view
    # probabilitas True = 2*ratio karena nanti dibagi dua ke q dan k
    random_mask = np.random.rand(B, clips, part_num) > (1 - 2 * ratio)
    # inisialisasi dua mask kosong dengan shape yang sama
    mask_q, mask_k = np.zeros_like(random_mask), np.zeros_like(random_mask)
    # ambil semua posisi (b, clip, part) yang True di random_mask
    position = np.where(random_mask)
    # hitung setengah dari total posisi aktif untuk dibagi ke q dan k
    half_num = int(len(position[0]) / 2)

    # pilih secara acak setengah indeks untuk view_q; sisanya untuk view_k
    index = np.random.choice(len(position[0]), half_num, replace=False).tolist()
    for i in range(len(position[0])):
        if i in index:
            # posisi ini masuk ke mask_q
            mask_q[position[0][i], position[1][i], position[2][i]] = 1
        else:
            # posisi ini masuk ke mask_k (komplementer)
            mask_k[position[0][i], position[1][i], position[2][i]] = 1
    # konversi ke boolean untuk dipakai sebagai kondisi pengisian mask
    mask_q = mask_q.astype(bool)
    mask_k = mask_k.astype(bool)

    # inisialisasi output mask sebagai tensor satu (belum ada yang dimask)
    mask_cat_q = torch.ones(shape)
    mask_cat_k = torch.ones(shape)
    for i in range(B):
        for k in range(clips):
            if k == clips - 1:
                # klip terakhir: ambil sisa frame dari clip_length*k sampai akhir
                for j in range(part_num):
                    if mask_q[i, k, j]:
                        # nolkan channel part_j dari frame clip_length*k sampai akhir
                        mask_cat_q[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        # nolkan channel part_j yang sama di view_k
                        mask_cat_k[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
            else:
                # klip normal: ambil frame dari clip_length*k sampai clip_length*(k+1)
                for j in range(part_num):
                    if mask_q[i, k, j]:
                        # nolkan channel part_j dalam rentang klip ini di view_q
                        mask_cat_q[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        # nolkan channel part_j dalam rentang klip ini di view_k
                        mask_cat_k[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
    # kembalikan dua mask komplementer siap dipakai di apply_masks
    return mask_cat_q, mask_cat_k


class CoSign1s_block(nn.Module):
    """
    Single-stream ST-GCN block processing skeleton features per spatial-part.

    Parameters
    ----------
    modes : list of str
        Names of modes/parts (e.g., ['hand21', 'body']).
    indims : int
        Input channels per node.
    outdims : int
        Output channels per node after GCN.
    A : list of Tensor
        Adjacency matrices per mode.
    split : list of int
        Indices separating channels per part.
    temporal_kernel : int
        Temporal kernel size for ST-GCN.
    adaptive : bool
        Whether the adjacency matrix is adaptive.
    """

    def __init__(self, modes, indims, outdims, A, split, temporal_kernel, adaptive):
        # panggil constructor nn.Module
        super(CoSign1s_block, self).__init__()
        # simpan nama mode/part untuk iterasi di forward
        self.modes = modes
        # simpan dimensi input channel per-node
        self.indims = indims
        # simpan dimensi output channel per-node
        self.outdims = outdims
        # simpan list adjacency matrix per-mode
        self.A = A
        # simpan indeks pemisah channel per-part
        self.split = split
        # simpan ukuran kernel temporal
        self.temporal_kernel = temporal_kernel
        # inisialisasi dict kosong untuk modul GCN per-mode
        self.gcn_modules = {}
        # ambil K (jumlah subset adjacency) dari dimensi pertama A[0]
        self.spatial_kernel_size = A[0].size(0)
        # simpan flag adaptive
        self.adaptive = adaptive
        for index, mode in enumerate(self.modes):
            # buat satu STGCN_block per mode dengan adjacency matrix-nya sendiri
            # clone A[index] agar tiap modul punya salinan parameter terpisah
            self.gcn_modules[mode] = STGCN_block(
                indims, outdims,
                (self.temporal_kernel, self.spatial_kernel_size),
                A[index].clone(),
                self.adaptive
            )
        # bungkus dict biasa menjadi nn.ModuleDict agar parameter terdaftar
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)

    def forward(self, feature):
        """
        Forward pass for single-stream GCN per part.

        Parameters
        ----------
        feature : Tensor
            Features of all parts concatenated along the channel dimension, shape (N, C_in, T, V_total).

        Returns
        -------
        Tensor
            Concatenated output of all parts, shape (N, C_out_concat, T, V).
        """
        # indeks pointer ke split, maju sesuai jumlah part yang sudah diproses
        index = 0
        # list untuk mengumpulkan output tiap part sebelum digabung
        feat_list = []
        for mode in self.modes:
            # tentukan rentang channel untuk part ini
            if index == 0:
                # part pertama: mulai dari channel 0
                start, end = 0, self.split[0]
            else:
                # part selanjutnya: mulai dari akhir part sebelumnya
                start, end = self.split[index-1], self.split[index]

            if mode == 'hand21':
                # kedua tangan (left & right) berbagi satu GCN yang sama
                # gabungkan left (start:end) dan right (end:split[index+1])
                # pada dim batch (dim 0) agar diproses sekaligus dalam satu forward
                hand = self.gcn_modules[mode](
                    torch.cat([
                        feature[:, :, :, start:end],
                        feature[:, :, :, end:self.split[index+1]]
                    ])
                )
                # pisah kembali hasil menjadi left dan right berdasarkan dim batch
                left, right = torch.chunk(hand, 2, dim=0)
                # tambahkan keduanya ke feat_list secara terpisah
                feat_list.append(left)
                feat_list.append(right)
                # maju dua indeks karena hand21 mengkonsumsi dua part sekaligus
                index += 2
            else:
                # mode biasa (body, mouth, dll.): proses satu part lewat GCN-nya
                feat_list.append(self.gcn_modules[mode](feature[:, :, :, start:end]))
                # maju satu indeks
                index += 1
        # gabungkan semua output part pada dim channel
        return torch.cat(feat_list, dim=-1)


class CoSign2s(nn.Module):
    """
    Two-stream extractor module producing static, motion, and fusion features.

    Parameters
    ----------
    in_channels : int
        Input channels per joint for the static stream.
    split : list of int
        Indices separating channels per spatial-part.
    temporal_kernel : int
        Temporal kernel size for ST-GCN.
    hidden_size : int
        Final feature output dimension (fusion hidden dim).
    modes : list of str
        Names of spatial groups/modes.
    level : str
        Architecture depth level ('0' for shallow, '1' for deep).
    adaptive : bool, optional
        Whether the adjacency matrix is adaptive. Default is True.
    CR_args : dict, optional
        Arguments for Consistency Regularization. Default is None.
    """

    def __init__(self, in_channels, split, temporal_kernel, hidden_size, modes, level, adaptive=True, CR_args=None) -> None:
        # panggil constructor nn.Module
        super().__init__()
        # simpan indeks pemisah spatial part
        self.split = split
        # inisialisasi dict graph dan list adjacency matrix kosong
        self.graph, A = {}, []
        # hitung jumlah part dari panjang split
        self.part_num = len(self.split)
        # simpan jumlah channel input untuk static stream
        self.in_channels = in_channels
        # simpan nama mode/spatial group
        self.modes = modes
        # simpan argumen CR; None berarti CR dinonaktifkan
        self.CR_args = CR_args
        # simpan level arsitektur ('0' atau '1')
        self.level = level

        for mode in self.modes:
            # buat graph skeleton per mode dengan strategi distance partitioning
            self.graph[mode] = Graph(layout=f'custom_{mode}', strategy='distance', max_hop=1)
            # konversi adjacency matrix ke tensor float, tidak ikut backprop
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))

        # proyeksi awal static stream: in_channels → 64 via Linear+ReLU
        self.static_linear = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True)
        )
        # proyeksi awal motion stream: in_channels*2 → 64 via Linear+ReLU
        # input *2 karena motion = concat(frame_t, frame_{t-1})
        self.motion_linear = nn.Sequential(
            nn.Linear(in_channels*2, 64),
            nn.ReLU(inplace=True)
        )

        # definisikan konfigurasi channel per layer untuk level '0' dan '1'
        # setiap entry (in, out) adalah dimensi input/output satu CoSign1s_block
        self.layer_configs = {
            '0': {
                # level dangkal: 3 layer per stream
                'static': [(64, 64), (64, 128), (128, 256)],
                'motion': [(64, 64), (64, 128), (128, 256)],
                # fusion mulai dari 128 karena input = cat(static, motion) = 64+64
                'fusion': [(128, 128), (256, 256), (512, 512)]
            },
            '1': {
                # level dalam: 5 layer per stream, transisi lebih gradual
                'static': [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                'motion': [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                'fusion': [(128, 128), (128, 128), (256, 256), (256, 256), (512, 512)]
            }
        }

        # bangun semua layer CoSign1s berdasarkan layer_configs dan level
        self.create_layers(A, temporal_kernel, adaptive)

        # layer agregasi akhir: gabungkan semua part (512*part_num) → hidden_size
        self.fusion_fusion = nn.Sequential(
            nn.Linear(512 * self.part_num, hidden_size),
            nn.ReLU(inplace=True)
        )

        # fungsi pooling spasial: avg_pool2d untuk agregasi joint per-part
        self.pool_func = F.avg_pool2d
        # simpan ukuran output akhir sebagai atribut publik untuk modul luar
        self.out_size = hidden_size
        # dimensi output akhir static stream setelah semua layer
        self.final_dim_static = 256
        # dimensi output akhir motion stream setelah semua layer
        self.final_dim_motion = 256
        # dimensi output akhir fusion stream setelah semua layer
        self.final_dim_fusion = 512

    def create_layers(self, A, temporal_kernel, adaptive):
        """
        Build CoSign1s_block layers for static, motion, and fusion streams.

        Parameters
        ----------
        A : list of Tensor
            Adjacency matrices per mode.
        temporal_kernel : int
            Temporal kernel size for ST-GCN.
        adaptive : bool
            Whether the adjacency is adaptive.
        """
        # ambil konfigurasi layer sesuai level arsitektur yang dipilih
        config = self.layer_configs[self.level]

        for layer_type, layer_dims in config.items():
            # inisialisasi ModuleList kosong untuk stream ini
            layers = nn.ModuleList()

            for i, (in_dim, out_dim) in enumerate(layer_dims):
                # dapatkan nama atribut sesuai konvensi penamaan level
                layer_name = self.get_layer_name(layer_type, i)
                # buat satu blok CoSign1s dengan dimensi yang sesuai
                layer = CoSign1s_block(
                    self.modes, in_dim, out_dim, A,
                    self.split, temporal_kernel, adaptive
                )
                # tambahkan ke ModuleList agar terlacak sebagai submodul
                layers.append(layer)
                # daftarkan juga sebagai atribut bernama untuk akses langsung
                setattr(self, layer_name, layer)

            # simpan ModuleList stream ini sebagai atribut, mis. self.static_layers
            setattr(self, f'{layer_type}_layers', layers)

    def get_layer_name(self, layer_type, index):
        """
        Generate layer attribute name based on architecture level.

        Parameters
        ----------
        layer_type : str
            Stream type ('static', 'motion', 'fusion').
        index : int
            Zero-based index of the layer.

        Returns
        -------
        str
            Attribute name for the layer.
        """
        if self.level == '0':
            # penamaan sederhana: nomor layer 1-based
            return f'{layer_type}_layer{index + 1}'
        else:
            if index < 4:
                # 4 layer pertama: format layer{group}_{sub}, mis. layer1_1, layer2_2
                return f'{layer_type}_layer{index // 2 + 1}_{index % 2 + 1}'
            else:
                # layer terakhir (index=4): cukup layer3
                return f'{layer_type}_layer3'

    def pooling_stage(self, feature):
        """
        Perform per-part spatial average pooling on ST-GCN output.

        Parameters
        ----------
        feature : Tensor
            Output of ST-GCN, shape (N, C, T, V_total).

        Returns
        -------
        Tensor
            Pooled features of all parts, shape (N, C_total, T).
        """
        # list untuk mengumpulkan hasil pooling tiap part
        feature_list = []
        for i in range(len(self.split)):
            if i == 0:
                # part pertama: ambil dari joint 0 sampai split[0]
                start, end = 0, self.split[0]
            else:
                # part selanjutnya: dari split[i-1] sampai split[i]
                start, end = self.split[i-1], self.split[i]
            # avg_pool2d dengan kernel (1, jumlah joint part) → agregasi spatial
            # squeeze(-1) menghilangkan dimensi V yang kini = 1
            feature_list.append(
                self.pool_func(
                    feature[:, :, :, start:end],
                    (1, end - start)
                ).squeeze(-1)
            )
        # gabungkan semua part pada dim channel
        return torch.cat(feature_list, dim=1)

    def process_static_motion(self, static, motion):
        """
        Process static, motion, and fusion streams sequentially based on architecture level.

        Parameters
        ----------
        static : Tensor
            Static features after linear projection, shape (N, C, T, V).
        motion : Tensor
            Motion features after linear projection, shape (N, C, T, V).

        Returns
        -------
        tuple
            Final static, motion, and fusion tensors, each shape (N, C_out, T, V).
        """
        if self.level == '0':
            # level dangkal: 3 tahap, masing-masing 1 layer per stream
            processing_steps = [
                # tahap 1: gabungkan static+motion langsung (belum ada fusion sebelumnya)
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat'},
                # tahap 2: tambahkan static+motion ke fusion sebelumnya
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'},
                # tahap 3: sama seperti tahap 2
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'}
            ]
        else:
            # level dalam: 3 tahap, tahap 1-2 pakai 2 layer, tahap 3 pakai 1 layer
            processing_steps = [
                # tahap 1: concat langsung, 2 layer per stream
                {'static_steps': [1, 1], 'motion_steps': [1, 1], 'fusion_steps': [1, 1], 'fusion_input': 'concat'},
                # tahap 2: concat_sum, 2 layer per stream
                {'static_steps': [1, 1], 'motion_steps': [1, 1], 'fusion_steps': [1, 1], 'fusion_input': 'concat_sum'},
                # tahap 3: concat_sum, 1 layer per stream
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'}
            ]

        # pointer indeks ke layer saat ini untuk masing-masing stream
        static_idx = 0
        motion_idx = 0
        fusion_idx = 0

        for step in processing_steps:
            # jalankan sejumlah static layer sesuai step ini
            for _ in step['static_steps']:
                static = self.static_layers[static_idx](static)
                static_idx += 1

            # jalankan sejumlah motion layer sesuai step ini
            for _ in step['motion_steps']:
                motion = self.motion_layers[motion_idx](motion)
                motion_idx += 1

            if step['fusion_input'] == 'concat':
                # tahap pertama: belum ada fusion, gabungkan static dan motion langsung
                fusion_input = torch.cat([static, motion], dim=1)
            else:
                # tahap berikutnya: tambahkan residual static+motion ke fusion sebelumnya
                # dim 1 karena format (N, C, T, V) → channel ada di dim 1
                fusion_input = torch.cat([fusion, static + motion], dim=1)

            # jalankan sejumlah fusion layer sesuai step ini
            for _ in step['fusion_steps']:
                fusion = self.fusion_layers[fusion_idx](fusion_input)
                # output layer ini menjadi input layer fusion berikutnya dalam step
                fusion_input = fusion
                fusion_idx += 1

        # kembalikan output akhir ketiga stream
        return static, motion, fusion

    def apply_masks(self, cat_feat_static, cat_feat_motion, cat_feat_fusion):
        """
        Apply complementary masking to produce two views per stream for Consistency Regularization.

        Parameters
        ----------
        cat_feat_static : Tensor
            Static features after pooling and transpose, shape (B, T, C).
        cat_feat_motion : Tensor
            Motion features after pooling and transpose, shape (B, T, C).
        cat_feat_fusion : Tensor
            Fusion features after pooling and transpose, shape (B, T, C).

        Returns
        -------
        dict
            Contains view1 and view2 for static, motion, and fusion streams.
        """
        # definisikan tiga stream beserta fiturnya dan dimensi final masing-masing
        stream_configs = [
            ('static', cat_feat_static, self.final_dim_static),
            ('motion', cat_feat_motion, self.final_dim_motion),
            ('fusion', cat_feat_fusion, self.final_dim_fusion)
        ]
        # dict untuk mengumpulkan hasil semua view
        results = {}
        for stream_type, cat_feat, final_dim in stream_configs:
            # buat dua mask komplementer sesuai parameter CR
            mask_view1, mask_view2 = generate_mask(
                cat_feat.shape,
                self.part_num,
                self.CR_args['clip_length'],
                self.CR_args['ratio'],
                final_dim
            )
            # pindahkan mask ke device fitur (GPU jika training di GPU)
            # lalu terapkan mask: nol di posisi dimask, tetap di posisi lain
            view1 = mask_view1.to(cat_feat.device) * cat_feat
            view2 = mask_view2.to(cat_feat.device) * cat_feat

            if stream_type == 'fusion':
                # hanya fusion yang perlu transform tambahan untuk menyesuaikan
                # dimensi ke hidden_size sebelum dikirim ke downstream
                view1 = self.fusion_fusion(view1)
                view2 = self.fusion_fusion(view2)

            # simpan kedua view ke dict hasil dengan nama stream sebagai suffix
            results[f'view1_{stream_type}'] = view1
            results[f'view2_{stream_type}'] = view2

        return results

    def forward(self, x, len_x):
        """
        Forward pass separating static and motion channels to process through two-stream ST-GCN.

        Parameters
        ----------
        x : Tensor
            Input skeleton/pose coordinates, shape (N, T, V, C_in).
        len_x : Tensor or list
            Valid lengths for each sample.

        Returns
        -------
        dict
            Dict containing 'fusion' tensor during evaluation, or 6 complementary views during training with CR.
        """
        if x.shape[3] == 7:
            # format 7-channel: gabungkan x,y (0:2) dan confidence (6) untuk static
            static = torch.cat([x[:, :, :, 0:2], x[:, :, :, 6].unsqueeze(-1)], dim=-1)
        else:
            # format lain: gunakan seluruh input sebagai static
            static = x
        # potong static sesuai in_channels yang dikonfigurasi
        static = static[:, :, :, :self.in_channels]
        # ambil channel 2:6 sebagai motion features (frame difference)
        motion = x[:, :, :, 2:6]

        # proyeksi static ke 64 dim, ubah ke (N, C, T, V) untuk ST-GCN
        static = self.static_linear(static).permute(0, 3, 1, 2)
        # proyeksi motion ke 64 dim, ubah ke (N, C, T, V) untuk ST-GCN
        motion = self.motion_linear(motion).permute(0, 3, 1, 2)

        # jalankan semua layer ST-GCN berlapis untuk ketiga stream
        static, motion, fusion = self.process_static_motion(static, motion)

        # pooling spasial per-part lalu transpose ke (B, T, C) untuk downstream
        cat_feat_static = self.pooling_stage(static).transpose(1, 2)
        cat_feat_motion = self.pooling_stage(motion).transpose(1, 2)
        cat_feat_fusion = self.pooling_stage(fusion).transpose(1, 2)

        if self.CR_args is not None and self.training:
            # mode training dengan CR: hasilkan dua view komplementer per-stream
            return self.apply_masks(cat_feat_static, cat_feat_motion, cat_feat_fusion)
        else:
            # mode evaluasi atau tanpa CR: agregasi fusion ke hidden_size dan kembalikan
            fusion_feat_fusion = self.fusion_fusion(cat_feat_fusion)
            return {'fusion': fusion_feat_fusion}