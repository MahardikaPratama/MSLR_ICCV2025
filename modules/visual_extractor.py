import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

from .stgcn_layers import Graph, STGCN_block


def generate_mask(shape, part_num, clip_length, ratio, dim):
    """
    Deskripsi:
        Menghasilkan dua mask komplementer untuk Consistency Regularization (CR).
        Mask dibuat secara acak per-klip temporal dan per-part, lalu dibagi menjadi
        dua set yang saling melengkapi (view_q dan view_k) sehingga informasi yang
        disembunyikan di view_q ada di view_k dan sebaliknya.

    Input:
        - shape      (tuple)      : (B, T, C) bentuk tensor fitur yang akan dimask.
        - part_num   (int)        : jumlah spatial part sesuai `split`.
        - clip_length(int)        : panjang segmen temporal untuk granularitas mask.
        - ratio      (float, 0..0.5): proporsi elemen yang dimask per view.
        - dim        (int)        : lebar channel per-part pada axis C.

    Proses:
        1. Hitung jumlah klip: clips = T // clip_length.
        2. Buat random_mask boolean (B, clips, part_num) dengan probabilitas aktif
           sebesar 2*ratio (total elemen yang akan dibagi ke dua view).
        3. Kumpulkan posisi aktif, acak, lalu bagi setengah ke mask_q dan setengah
           ke mask_k sehingga keduanya komplementer.
        4. Ekspansi mask ke (B, T, C): setiap bit part_j dipetakan ke rentang
           channel dim*j : dim*(j+1) selama durasi klip tersebut.
        5. Kembalikan dua tensor float berisi 0 (dimask) atau 1 (tidak dimask).

    Output:
        - mask_cat_q (Tensor float, B×T×C): mask untuk view pertama (query).
        - mask_cat_k (Tensor float, B×T×C): mask untuk view kedua (key),
          komplementer terhadap mask_cat_q.
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
    CoSign1s_block

    Deskripsi:
        Blok ST-GCN satu stream yang memproses fitur skeleton per spatial-part
        secara terpisah menggunakan graph masing-masing, lalu menggabungkan
        seluruh output part menjadi satu tensor terkoncatenasi. Blok ini adalah
        unit dasar yang disusun berlapis di CoSign2s.

    Input (constructor):
        - modes          (list of str) : nama mode/part, mis. ['hand21', 'body'].
        - indims         (int)         : jumlah channel input per-node.
        - outdims        (int)         : jumlah channel output per-node setelah GCN.
        - A              (list Tensor) : adjacency matrices per-mode.
        - split          (list of int) : indeks pemisah channel per-part.
        - temporal_kernel(int)         : ukuran kernel temporal ST-GCN.
        - adaptive       (bool)        : apakah adjacency matrix bersifat adaptif.

    Output (forward):
        - Tensor (N, C_out_concat, T, V) hasil gabungan semua part pada dim channel.
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
        Deskripsi:
            Jalankan satu forward pass blok CoSign (single-stream GCN per part).
            Setiap part diproses oleh GCN-nya sendiri, lalu semua output
            digabungkan kembali pada dimensi channel.

        Input:
            - feature (Tensor, N×C_in×T×V_total): fitur semua part yang sudah
              terkoncatenasi pada dimensi channel terakhir.

        Proses:
            1. Iterasi tiap mode; tentukan rentang channel [start, end] dari split.
            2. Jika mode == 'hand21': gabungkan left+right hand secara batch (cat
               pada dim 0), proses lewat satu GCN bersama, lalu pisah kembali.
               Ini menghemat parameter karena kedua tangan berbagi bobot GCN.
            3. Untuk mode lain: jalankan GCN sesuai mode dan kumpulkan hasil.
            4. Gabungkan semua feat_list pada dim channel menjadi satu tensor.

        Output:
            - Tensor (N, C_out_concat, T, V) gabungan semua part.
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
    CoSign2s

    Deskripsi:
        Modul ekstraktor dua-stream yang menghasilkan fitur `static`, `motion`,
        dan `fusion` menggunakan beberapa lapis CoSign1s_block dan pooling
        per-part. Mendukung Consistency Regularization (CR) dengan complementary
        masking untuk menghasilkan dua view.

    Input (constructor):
        - in_channels   (int)       : jumlah channel input per-joint untuk static.
        - split         (list int)  : indeks pemisah channel per-spatial-part.
        - temporal_kernel(int)      : ukuran kernel temporal ST-GCN.
        - hidden_size   (int)       : dimensi output fitur akhir (fusion hidden dim).
        - modes         (list str)  : nama mode/spatial groups.
        - level         (str)       : kedalaman arsitektur, '0' (dangkal) atau '1' (dalam).
        - adaptive      (bool)      : apakah adjacency matrix bersifat adaptif.
        - CR_args       (dict|None) : argumen Consistency Regularization (opsional).

    Output (forward):
        - dict fitur per kondisi training/CR seperti dijelaskan di modul-level.
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
        Deskripsi:
            Membangun semua layer CoSign1s_block untuk ketiga stream (static,
            motion, fusion) berdasarkan konfigurasi layer_configs[level].
            Setiap layer juga didaftarkan sebagai atribut bernama agar mudah
            diakses dan terlacak oleh PyTorch.

        Input:
            - A               (list Tensor): adjacency matrices per-mode.
            - temporal_kernel (int)        : ukuran kernel temporal ST-GCN.
            - adaptive        (bool)       : apakah adjacency adaptif (learnable).

        Proses:
            1. Ambil konfigurasi layer sesuai self.level dari layer_configs.
            2. Untuk tiap stream ('static','motion','fusion'), iterasi pasangan
               (in_dim, out_dim) dan buat CoSign1s_block.
            3. Simpan ke nn.ModuleList sebagai `{stream}_layers`.
            4. Daftarkan juga tiap layer sebagai atribut bernama via setattr
               (nama dari get_layer_name) untuk akses langsung jika diperlukan.

        Output:
            - (tidak return) Mendaftarkan atribut:
              self.static_layers, self.motion_layers, self.fusion_layers
              sebagai nn.ModuleList, serta atribut per-layer bernama.
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
        Deskripsi:
            Menghasilkan nama atribut layer mengikuti konvensi penamaan
            berdasarkan level arsitektur. Dipanggil oleh create_layers untuk
            mendaftarkan setiap layer sebagai atribut bernama di modul.

        Input:
            - layer_type (str): tipe stream, salah satu dari
              ['static', 'motion', 'fusion'].
            - index      (int): indeks layer dalam konfigurasi (0-based).

        Proses:
            - Level '0': nama sederhana '{layer_type}_layer{index+1}'.
              Contoh: index=0 → 'static_layer1', index=2 → 'static_layer3'.
            - Level '1': nama dua-digit untuk 4 layer pertama, satu digit
              untuk layer terakhir.
              Contoh: index=0 → 'static_layer1_1', index=3 → 'static_layer2_2',
              index=4 → 'static_layer3'.

        Output:
            - layer_name (str): nama atribut yang akan digunakan oleh setattr.
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
        Deskripsi:
            Melakukan average pooling spasial per-part pada output ST-GCN,
            menghasilkan satu vektor fitur per-part per-frame dengan
            mengagregasi seluruh joint dalam setiap part.

        Input:
            - feature (Tensor, N×C×T×V_total): output ST-GCN dengan semua
              joint terkoncatenasi pada dimensi V terakhir.

        Proses:
            1. Iterasi tiap part berdasarkan self.split.
            2. Iris feature pada dim V untuk mendapatkan joint part tersebut.
            3. Terapkan avg_pool2d dengan kernel (1, end-start) sehingga
               semua joint dalam satu part diagregasi menjadi satu vektor.
            4. squeeze(-1) untuk menghilangkan dim V yang kini bernilai 1.
            5. Gabungkan semua part pada dim channel (dim 1).

        Output:
            - Tensor (N, C_total, T): fitur terpooling semua part,
              di mana C_total = C_per_part × part_num.
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
        Deskripsi:
            Menjalankan urutan pemrosesan berlapis untuk ketiga stream (static,
            motion, fusion) sesuai konfigurasi level arsitektur. Mengatur aliran
            data antar layer dan cara menggabungkan static+motion menjadi input
            fusion di tiap tahap.

        Input:
            - static (Tensor, N×C×T×V): fitur static setelah proyeksi linear.
            - motion (Tensor, N×C×T×V): fitur motion setelah proyeksi linear.

        Proses:
            - Tentukan processing_steps sesuai level:
                Level '0': 3 tahap, tiap tahap 1 layer per stream.
                Level '1': 3 tahap, tahap 1-2 masing-masing 2 layer, tahap 3 satu layer.
            - Pada tiap tahap:
                1. Jalankan sejumlah static_layers → update static.
                2. Jalankan sejumlah motion_layers → update motion.
                3. Bentuk fusion_input:
                   'concat'     → cat(static, motion) untuk tahap pertama.
                   'concat_sum' → cat(fusion, static+motion) untuk tahap berikutnya.
                4. Jalankan sejumlah fusion_layers → update fusion.

        Output:
            - tuple (static, motion, fusion): tensor output akhir ketiga stream
              setelah semua tahap selesai, shape masing-masing (N, C_out, T, V).
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
        Deskripsi:
            Menerapkan complementary masking untuk menghasilkan dua view per-stream
            sebagai bagian dari Consistency Regularization (CR). Setiap stream
            menghasilkan view1 dan view2 yang saling melengkapi.

        Input:
            - cat_feat_static (Tensor, B×T×C): fitur static setelah pooling+transpose.
            - cat_feat_motion (Tensor, B×T×C): fitur motion setelah pooling+transpose.
            - cat_feat_fusion (Tensor, B×T×C): fitur fusion setelah pooling+transpose.

        Proses:
            1. Iterasi tiga stream dengan dimensi final masing-masing.
            2. Untuk tiap stream, panggil generate_mask dengan parameter dari CR_args.
            3. Pindahkan mask ke device yang sama dengan fitur (CPU/GPU).
            4. Terapkan mask (perkalian elementwise) ke fitur untuk menghasilkan view1 dan view2.
            5. Khusus stream fusion: jalankan fusion_fusion (linear) agar dimensi
               sesuai dengan hidden_size yang diharapkan downstream.

        Output:
            - dict dengan 6 kunci: 'view1_static', 'view2_static',
              'view1_motion', 'view2_motion', 'view1_fusion', 'view2_fusion'.
              Tiap nilai adalah Tensor (B, T, feat_dim) di device yang sama dengan input.
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
        Deskripsi:
            Forward pass utama CoSign2s. Menerima tensor skeleton, memisahkan
            channel menjadi static dan motion, memproses lewat dua stream ST-GCN
            berlapis, melakukan pooling spasial, lalu mengembalikan fitur sesuai
            mode (evaluasi vs training dengan CR).

        Input:
            - x     (Tensor, N×T×V×C_in): input skeleton/koordinat pose.
              Jika C_in==7: channel 0:2=koordinat x,y; 6=confidence (static);
              channel 2:6=optical flow / motion features (motion).
              Jika C_in!=7: seluruh channel dipakai sebagai static.
            - len_x (Tensor|list): panjang valid tiap sample dalam batch
              (dipakai oleh pipeline atas, tidak langsung dipakai di sini).

        Proses:
            1. Pisahkan channel menjadi `static` (koordinat+confidence) dan
               `motion` (frame difference / optical flow), slice sesuai in_channels.
            2. Proyeksi linear ke 64 channel, permutasi ke (N, C, T, V).
            3. Jalankan process_static_motion: semua layer CoSign1s bertingkat.
            4. pooling_stage: average pooling per-part → (N, C_total, T).
            5. transpose(1,2) → (B, T, C_total) untuk format downstream.
            6a. Jika CR aktif dan mode training: apply_masks → 6 view.
            6b. Jika evaluasi atau tanpa CR: fusion_fusion → dict {'fusion'}.

        Output:
            - Saat evaluasi/tanpa CR: {'fusion': Tensor(B, T, hidden_size)}.
            - Saat training dengan CR: dict berisi 6 Tensor view komplementer.
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