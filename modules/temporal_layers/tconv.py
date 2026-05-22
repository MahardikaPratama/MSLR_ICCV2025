import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """
    TemporalConv

    Deskripsi:
        Modul konvolusi temporal 1D yang memproses fitur per-frame menjadi
        representasi temporal yang lebih ringkas. Digunakan sebagai jembatan
        antara ekstraktor fitur visual (CoSign2s) dan modul sequence modeling
        (BiLSTM-CTC) dengan cara mengurangi panjang sequence dan mengekstrak
        pola temporal lokal.

        Arsitektur dibangun secara dinamis dari string `conv_type` yang
        mendefinisikan urutan layer dalam format 'K{kernel}-P{pool}-K{kernel}-...',
        di mana:
            'K{n}' → Conv1d dengan kernel size n, diikuti BatchNorm1d dan ReLU.
            'P{n}' → MaxPool1d dengan kernel size n (mengurangi panjang sequence).

    Input (constructor):
        - input_size  (int) : dimensi fitur input per frame (channel masuk Conv1d).
        - hidden_size (int) : dimensi fitur output per frame (channel keluar Conv1d).
        - conv_type   (str) : string konfigurasi layer, mis. 'K5-P2-K5' artinya
          Conv1d kernel-5, MaxPool1d stride-2, Conv1d kernel-5.

    Proses:
        - Parse conv_type menjadi list token kernel/pool.
        - Bangun nn.Sequential dari token tersebut secara dinamis.
        - update_lgt menghitung ulang panjang sequence setelah pooling.

    Output (forward):
        - dict dengan dua kunci:
            'visual_feat' (Tensor, T×B×C): fitur temporal dalam format time-first
                                            siap masuk BiLSTM.
            'feat_len'    (Tensor, B)     : panjang valid tiap sample setelah pooling.
    """

    def __init__(self, input_size, hidden_size, conv_type=2):
        # panggil constructor nn.Module
        super(TemporalConv, self).__init__()

        # simpan dimensi input channel untuk dipakai di layer pertama
        self.input_size = input_size

        # simpan dimensi hidden/output channel untuk semua layer Conv1d
        self.hidden_size = hidden_size

        # simpan string konfigurasi conv_type untuk referensi dan update_lgt
        self.conv_type = conv_type

        # parse string conv_type menjadi list token dengan memisahkan berdasarkan '-'
        # contoh: 'K5-P2-K5' → ['K5', 'P2', 'K5']
        self.kernel_size = conv_type.split('-')

        # list untuk mengumpulkan modul layer sebelum dibungkus nn.Sequential
        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            # tentukan input channel: input_size untuk layer pertama,
            # hidden_size untuk layer berikutnya (output layer sebelumnya)
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size

            if ks[0] == 'P':
                # token 'P{n}': tambahkan MaxPool1d dengan kernel size n
                # ceil_mode=False: frame sisa yang tidak cukup satu kernel dibuang
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))

            elif ks[0] == 'K':
                # token 'K{n}': tambahkan Conv1d dengan kernel size n
                # stride=1: tidak ada downsampling di conv, hanya di pool
                # padding=kernel_size//2: same padding agar panjang tidak berubah
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=int(ks[1]),
                        stride=1,
                        padding=int(ks[1]) // 2
                    )
                )
                # BatchNorm1d untuk normalisasi aktivasi per-channel setelah conv
                modules.append(nn.BatchNorm1d(self.hidden_size))
                # ReLU sebagai fungsi aktivasi non-linear; inplace menghemat memori
                modules.append(nn.ReLU(inplace=True))

        # bungkus semua layer menjadi satu nn.Sequential untuk forward pass bersih
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        """
        Deskripsi:
            Menghitung ulang panjang sequence yang valid setelah operasi pooling.
            Hanya token 'P' yang mengubah panjang sequence; token 'K' dengan
            same padding tidak mengubah panjang.

        Input:
            - lgt (Tensor, B): panjang valid tiap sample sebelum pooling,
              dalam satuan jumlah frame.

        Proses:
            - Deep copy lgt agar tidak memodifikasi tensor asli.
            - Iterasi tiap token di kernel_size; jika token 'P{n}',
              bagi feat_len dengan n menggunakan integer division.
            - Operasi ini mencerminkan perilaku MaxPool1d dengan ceil_mode=False:
              frame sisa yang tidak cukup satu window dibuang (floor division).

        Output:
            - feat_len (Tensor, B, dtype long): panjang valid yang sudah
              disesuaikan dengan downsampling pooling, siap dipakai CTC loss
              dan decoder.
        """
        # deep copy untuk menghindari modifikasi in-place pada tensor lgt asli
        feat_len = copy.deepcopy(lgt)

        for ks in self.kernel_size:
            if ks[0] == 'P':
                # bagi panjang sequence dengan faktor pooling (integer division)
                # torch.div dengan .long() setara floor division untuk int tensor
                feat_len = torch.div(feat_len, int(ks[1])).long()

        # kembalikan panjang sequence yang sudah disesuaikan
        return feat_len

    def forward(self, frame_feat, lgt):
        """
        Deskripsi:
            Forward pass TemporalConv. Memproses fitur frame melalui semua layer
            konvolusi dan pooling, lalu menghitung ulang panjang sequence yang valid.

        Input:
            - frame_feat (Tensor, B×C×T): fitur per-frame dalam format batch-first
              dengan channel di dim 1, sesuai konvensi Conv1d PyTorch.
            - lgt        (Tensor, B)    : panjang valid tiap sample sebelum conv,
              dalam satuan jumlah frame.

        Proses:
            1. Jalankan frame_feat melalui self.temporal_conv (semua layer K dan P).
               Output shape: (B, hidden_size, T') di mana T' ≤ T akibat pooling.
            2. Hitung feat_len baru via update_lgt untuk mencerminkan downsampling.
            3. Permutasi visual_feat dari (B, C, T') ke (T', B, C) karena BiLSTM
               mengharapkan format time-first (sequence length di dim 0).

        Output:
            - dict dengan dua kunci:
                'visual_feat' (Tensor, T'×B×hidden_size): fitur temporal format
                    time-first siap dimasukkan ke nn.LSTM atau nn.GRU.
                'feat_len'    (Tensor, B, di CPU)        : panjang valid setelah
                    pooling, dipindahkan ke CPU karena CTC loss membutuhkannya
                    di CPU.
        """
        # jalankan seluruh pipeline conv/pool pada fitur frame
        visual_feat = self.temporal_conv(frame_feat)

        # hitung ulang panjang sequence valid setelah downsampling pooling
        lgt = self.update_lgt(lgt)

        return {
            # permutasi (B, C, T') → (T', B, C): format time-first untuk BiLSTM
            "visual_feat": visual_feat.permute(2, 0, 1),
            # pindahkan feat_len ke CPU: dibutuhkan oleh CTCLoss dan decoder
            "feat_len": lgt.cpu(),
        }