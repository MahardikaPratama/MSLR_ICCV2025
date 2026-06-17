import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    """
    Bidirectional RNN (LSTM/GRU) layer handling variable-length sequences via packed padding.

    Parameters
    ----------
    input_size : int
        Input feature dimension per timestep.
    debug : bool, optional
        Debug flag, unused in forward. Default is False.
    hidden_size : int, optional
        Total hidden state output dimension. Default is 512.
    num_layers : int, optional
        Number of stacked RNN layers. Default is 1.
    dropout : float, optional
        Dropout probability between RNN layers. Default is 0.3.
    bidirectional : bool, optional
        Whether to use BiLSTM/BiGRU. Default is True.
    rnn_type : str, optional
        Type of RNN, 'LSTM' or 'GRU'. Default is 'LSTM'.
    num_classes : int, optional
        Unused placeholder for pipeline compatibility. Default is -1.
    """

    def __init__(self, input_size, debug=False, hidden_size=512, num_layers=1, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM', num_classes=-1):
        # panggil constructor nn.Module
        super(BiLSTMLayer, self).__init__()

        # simpan probabilitas dropout antar layer
        self.dropout = dropout

        # simpan jumlah layer RNN yang ditumpuk
        self.num_layers = num_layers

        # simpan dimensi fitur input per timestep
        self.input_size = input_size

        # simpan flag bidirectional
        self.bidirectional = bidirectional

        # tentukan jumlah direction: 2 untuk BiLSTM, 1 untuk unidirectional
        self.num_directions = 2 if bidirectional else 1

        # bagi hidden_size dengan num_directions agar output RNN setelah concat
        # tetap berdimensi hidden_size (bukan hidden_size * 2)
        self.hidden_size = int(hidden_size / self.num_directions)

        # simpan tipe RNN sebagai string untuk getattr di bawah
        self.rnn_type = rnn_type

        # simpan flag debug untuk keperluan inspeksi opsional
        self.debug = debug

        if num_layers == 1:
            # dropout tidak berlaku pada single-layer RNN di PyTorch
            # (dropout hanya diterapkan antar layer, bukan setelah layer terakhir)
            # paksa ke 0 untuk menghindari warning dari PyTorch
            self.dropout = 0

        # buat modul RNN secara dinamis berdasarkan rnn_type ('LSTM' atau 'GRU')
        # getattr(nn, 'LSTM') setara nn.LSTM, getattr(nn, 'GRU') setara nn.GRU
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,       # per-direction hidden size
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def forward(self, src_feats, src_lens, hidden=None):
        """
        Forward pass for processing padded sequences through RNN.

        Parameters
        ----------
        src_feats : Tensor
            Input features in time-first format, shape (T, B, D).
        src_lens : Tensor
            Valid sequence lengths for each sample in the batch, shape (B,).
        hidden : Tensor or tuple, optional
            Initial hidden state. Default is None.

        Returns
        -------
        dict
            Contains 'predictions' tensor of shape (T, B, hidden_size) and 'hidden' state tensor.
        """
        # defragmentasi parameter RNN di memori untuk performa CUDNN optimal
        # wajib dipanggil sebelum forward jika menggunakan DataParallel
        self.rnn.flatten_parameters()

        # kompres sequence berpadding menjadi packed sequence
        # enforce_sorted=False: tidak perlu mengurutkan batch berdasarkan panjang
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            src_feats, src_lens, enforce_sorted=False
        )

        if hidden is not None and self.rnn_type == 'LSTM':
            # LSTM butuh hidden state dalam bentuk tuple (h_0, c_0)
            # asumsi: hidden diberikan sebagai satu tensor dengan h dan c
            # yang digabung pada dim 0, jadi perlu dipisah setengah-setengah
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])

        # jalankan RNN pada packed sequence
        # packed_outputs: packed sequence berisi output per timestep
        # hidden: state akhir setelah memproses seluruh sequence
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # kembalikan packed outputs ke tensor berpadding
        # _ adalah tensor panjang sequence (sudah kita punya dari src_lens)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            # untuk BiRNN, hidden shape: (num_layers*num_directions, B, hidden_size)
            # perlu diubah ke: (num_layers, B, hidden_size*num_directions)
            # dengan cara mengkonkatenasi hidden forward dan backward tiap layer
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            # LSTM menyimpan dua state: hidden state (h) dan cell state (c)
            # gabungkan keduanya pada dim 0 menjadi satu tensor untuk
            # memudahkan penyimpanan dan passing ke modul lain
            hidden = torch.cat(hidden, 0)

        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }

    def _cat_directions(self, hidden):
        """
        Concatenate forward and backward hidden states for bidirectional RNN.

        Parameters
        ----------
        hidden : Tensor or tuple of Tensor
            Input hidden state of shape (num_layers * num_directions, B, hidden_size).

        Returns
        -------
        Tensor or tuple of Tensor
            Concatenated hidden state of shape (num_layers, B, hidden_size * num_directions).
        """
        def _cat(h):
            # ambil semua even index (forward directions: 0, 2, 4, ...)
            # dan semua odd index (backward directions: 1, 3, 5, ...)
            # concat pada dim 2 (hidden_size) → menggabungkan kedua direction
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM: terapkan _cat pada hidden state (h_n) dan cell state (c_n)
            # secara terpisah, hasilkan tuple baru dengan shape yang sudah diubah
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU: hanya satu tensor hidden state, langsung terapkan _cat
            hidden = _cat(hidden)

        return hidden