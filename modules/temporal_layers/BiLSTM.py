import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    """
    BiLSTMLayer

    Deskripsi:
        Wrapper modul RNN (LSTM atau GRU) bidirectional yang menangani
        variable-length sequence via packed padding. Digunakan sebagai
        sequence modeling layer setelah TemporalConv untuk memodelkan
        dependensi temporal jangka panjang antar frame sebelum klasifikasi CTC.

        Mendukung konfigurasi fleksibel: jumlah layer, dropout, bidirectional
        atau unidirectional, dan pilihan tipe RNN (LSTM/GRU).

    Input (constructor):
        - input_size   (int)  : dimensi fitur input per timestep (dari TemporalConv).
        - debug        (bool) : flag debug, tidak dipakai langsung di forward.
        - hidden_size  (int)  : total dimensi hidden state output. Jika bidirectional,
          dibagi dua secara internal agar output tetap hidden_size setelah concat.
        - num_layers   (int)  : jumlah layer RNN yang ditumpuk.
        - dropout      (float): probabilitas dropout antar layer RNN (dinonaktifkan
          otomatis jika num_layers==1 karena dropout tidak berlaku pada layer tunggal).
        - bidirectional(bool) : True untuk BiLSTM/BiGRU, False untuk unidirectional.
        - rnn_type     (str)  : tipe RNN, 'LSTM' atau 'GRU'.
        - num_classes  (int)  : tidak dipakai di kelas ini (placeholder untuk
          kompatibilitas interface pipeline).

    Output (forward):
        - dict dengan dua kunci:
            'predictions' (Tensor, T×B×hidden_size): output per-timestep siap
                                                      masuk classifier CTC.
            'hidden'      (Tensor, num_layers×B×hidden_size): hidden state akhir
                          setelah forward pass, dengan forward dan backward
                          direction sudah dikoncatenasi per layer.
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
        Deskripsi:
            Forward pass BiLSTMLayer. Memproses sequence berpadding melalui RNN
            menggunakan packed sequence untuk efisiensi komputasi, lalu
            mengembalikan output per-timestep dan hidden state akhir.

        Input:
            - src_feats (Tensor, T×B×D)  : fitur input dalam format time-first,
              di mana T=panjang sequence maksimum, B=batch size, D=input_size.
            - src_lens  (Tensor, B)       : panjang valid tiap sequence dalam batch
              (tanpa padding), diperlukan oleh pack_padded_sequence.
            - hidden    (Tensor|None)     : hidden state awal opsional.
              Jika None, RNN menggunakan zero initialization.
              Jika LSTM dan hidden diberikan, diasumsikan format
              (num_layers*num_directions*2, B, hidden_size) yang perlu dipisah.

        Proses:
            1. flatten_parameters(): defragmentasi parameter RNN di memori GPU
               untuk efisiensi CUDNN.
            2. pack_padded_sequence: kompres sequence berpadding agar RNN tidak
               memproses frame padding — lebih efisien dan akurat secara gradien.
            3. Jika hidden diberikan untuk LSTM: pisah menjadi tuple (h, c)
               karena LSTM butuh dua state terpisah.
            4. Jalankan RNN → packed_outputs dan hidden state akhir.
            5. pad_packed_sequence: kembalikan ke format tensor berpadding.
            6. Jika bidirectional: gabungkan hidden forward dan backward per layer
               via _cat_directions.
            7. Jika LSTM: concat hidden state (h) dan cell state (c) menjadi
               satu tensor untuk kemudahan passing ke modul berikutnya.

        Output:
            - dict dengan dua kunci:
                'predictions' (Tensor, T×B×hidden_size*num_directions):
                    output RNN per timestep, siap masuk linear classifier CTC.
                'hidden'      (Tensor, num_layers*(1 atau 2)×B×hidden_size):
                    hidden state akhir; untuk LSTM berisi concat h dan c
                    sehingga dim 0 = num_layers * 2.
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
        Deskripsi:
            Mengubah hidden state bidirectional RNN dari format per-direction
            menjadi format per-layer dengan forward dan backward direction
            terkoncatenasi pada dimensi hidden.

            Transformasi ini diperlukan agar hidden state dapat dipakai sebagai
            inisialisasi decoder atau diteruskan ke layer berikutnya dengan
            dimensi yang konsisten.

        Input:
            - hidden (Tensor atau tuple of Tensor):
              Format masuk: (num_layers * num_directions, B, hidden_size)
              Untuk LSTM: tuple (h_n, c_n) masing-masing dengan shape di atas.
              Untuk GRU : tensor tunggal dengan shape di atas.

              Contoh untuk num_layers=2, num_directions=2 (dim 0 berisi):
                index 0: forward  layer 1
                index 1: backward layer 1
                index 2: forward  layer 2
                index 3: backward layer 2

        Proses:
            - Fungsi _cat(h) mengambil semua even index (forward: 0,2,4,...)
              dan odd index (backward: 1,3,5,...) lalu mengkonkatenasi pada dim 2.
            - Untuk LSTM: terapkan _cat pada h_n dan c_n secara terpisah.
            - Untuk GRU : terapkan _cat langsung pada hidden tensor.

        Output:
            - hidden (Tensor atau tuple of Tensor):
              Format keluar: (num_layers, B, hidden_size * num_directions)
              Contoh untuk num_layers=2:
                index 0: concat(forward layer 1, backward layer 1)
                index 1: concat(forward layer 2, backward layer 2)
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