import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """
    1D temporal convolution module to process per-frame features.

    Parameters
    ----------
    input_size : int
        Input feature dimension per frame.
    hidden_size : int
        Output feature dimension per frame.
    conv_type : str or int, optional
        Layer configuration string (e.g., 'K5-P2-K5'). Default is 2.
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
        Recalculate valid sequence lengths after pooling operations.

        Parameters
        ----------
        lgt : Tensor
            Valid sequence lengths before pooling, shape (B,).

        Returns
        -------
        Tensor
            Adjusted sequence lengths after pooling, shape (B,).
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
        Forward pass for processing frame features through convolutions and pooling.

        Parameters
        ----------
        frame_feat : Tensor
            Per-frame features in batch-first format, shape (B, C, T).
        lgt : Tensor
            Valid sequence lengths before convolution, shape (B,).

        Returns
        -------
        dict
            Contains 'visual_feat' tensor of shape (T', B, hidden_size) and 'feat_len' tensor on CPU.
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