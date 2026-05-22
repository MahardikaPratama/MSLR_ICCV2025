"""Definisi model untuk arsitektur Two-Stream CoSign pada CSLR.

Modul ini menyusun alur utama model yang dipakai saat training dan evaluasi.
Data yang masuk berasal dari `inputs_dict['x']` dan `inputs_dict['len_x']`
hasil dari dataset dan `collate_fn`. Setelah itu, data diproses oleh visual
extractor, temporal convolution, BiLSTM kontekstual, lalu classifier.
Saat training model memakai loss CTC dan KL, sedangkan saat evaluasi model
melakukan decoding untuk menghasilkan prediksi gloss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from modules.temporal_layers import BiLSTMLayer, TemporalConv
from modules.visual_extractor import CoSign2s

class KLdis(nn.Module):
    """Loss distilasi KL-divergence antara dua set logits.

    Deskripsi:
    Modul ini dipakai untuk membuat dua view dari input yang sama saling
    mendekati distribusi prediksinya. Di `get_loss()`, loss ini dipanggil
    dua arah agar hasilnya simetris.

    Input:
    1. `view1_logits`: logits dari view pertama.
    2. `view2_logits`: logits dari view kedua.
    3. `use_blank`: penanda apakah kelas blank CTC ikut dihitung.

    Output:
    1. Nilai loss KL berupa tensor skalar.
    """

    def __init__(self, T=1):
        super().__init__()
        # KLDivLoss menerima log-probability sebagai input dan probability sebagai target.
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        # Temperatur dipakai untuk membuat distribusi lebih halus saat distilasi.
        self.T = T

    def forward(self, view1_logits, view2_logits, use_blank=True):
        # Jika perlu, kelas blank CTC (indeks 0) diabaikan saat membandingkan view.
        start_idx = 0 if use_blank else 1

        # Ubah view pertama menjadi log-probability.
        view1_logits = F.log_softmax(view1_logits[:, :, start_idx:] / self.T, dim=-1) \
            .view(-1, view2_logits.shape[2] - start_idx)

        # View kedua dipakai sebagai distribusi target yang lebih lembut.
        ref_probs = F.softmax(view2_logits[:, :, start_idx:] / self.T, dim=-1) \
            .view(-1, view2_logits.shape[2] - start_idx)

        # Dikalikan T^2 sesuai rumus umum distilasi berbasis temperatur.
        loss = self.kdloss(view1_logits, ref_probs) * self.T * self.T
        return loss

class NormBothLinear(nn.Module):
    """Classifier yang menormalkan fitur dan bobot sebelum perkalian matriks.

    Deskripsi:
    Layer ini bekerja seperti classifier berbasis cosine similarity. Vektor
    fitur dinormalisasi, bobot juga dinormalisasi, lalu keduanya dikalikan
    untuk menghasilkan skor kelas.

    Input:
    1. `x`: tensor fitur dengan dimensi terakhir sebagai channel fitur.

    Output:
    1. Logits kelas untuk tiap frame/sequence.
    """

    def __init__(self, in_dim, out_dim):
        super(NormBothLinear, self).__init__()
        # Bentuk parameter adalah (in_dim, out_dim) agar hasil matmul menjadi skor kelas.
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        # Inisialisasi Xavier agar training lebih stabil.
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Normalisasi fitur dan bobot sebelum proyeksi.
        outputs = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.weight, dim=0))
        return outputs

class TwoStream_Cosign(nn.Module):
    """Model utama two-stream untuk CSLR.

    Deskripsi:
    Model ini menerima batch skeleton, mengekstrak fitur visual, lalu
    meneruskan fitur tersebut ke temporal convolution, BiLSTM, dan classifier.
    Saat training dengan consistency regularization, model memproses dua view
    untuk setiap stream. Saat evaluasi, model memakai representasi fusion
    untuk menghasilkan prediksi hasil decoding.

    Input:
    1. `visual_args`: konfigurasi untuk `CoSign2s`.
    2. `gloss_dict`: kamus gloss untuk decoder.
    3. `conv_type`: jenis temporal convolution.
    4. `loss_weights`: bobot loss yang akan dihitung saat training.
    5. `norm_scale`: faktor skala logits sebelum decoding atau CTC.

    Output:
    1. Model PyTorch siap dipakai untuk training dan evaluasi.
    """

    def __init__(self, visual_args, gloss_dict, conv_type, loss_weights, norm_scale=32) -> None:
        super().__init__()
        # Jika `CR_args` ada, visual module akan menghasilkan pasangan view.
        self.apply_CR = True if 'CR_args' in visual_args else False

        # Backbone visual yang mengubah input skeleton mentah menjadi fitur stream.
        self.visual_module = CoSign2s(**visual_args)
        hidden_size = self.visual_module.out_size

        # Jumlah kelas sudah termasuk token blank untuk CTC.
        self.num_classes = len(gloss_dict['id2gloss']) + 1
        # Decoder untuk mengubah logits menjadi prediksi gloss saat evaluasi.
        self.decoder = utils.Decode(gloss_dict, self.num_classes, 'beam')

        # Setiap bagian menyumbang fitur 256 dimensi pada stream static/motion.
        part_num = len(visual_args['split'])
        self.stream_configs = {
            'static': {'input_dim': 256 * part_num},
            'motion': {'input_dim': 256 * part_num}, 
            'fusion': {'input_dim': hidden_size}
        }

        # Bangun modul lanjutan untuk setiap stream.
        for name, config in self.stream_configs.items():
            #  Temporal convolution untuk menangkap pola temporal lokal.
            conv1d = TemporalConv(config['input_dim'], hidden_size, conv_type)
            # BiLSTM untuk menangkap konteks urutan yang lebih panjang.
            contextual_module = BiLSTMLayer(
                rnn_type='LSTM',
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
            )
            #  Classifier yang menormalkan fitur dan bobot untuk menghasilkan skor kelas.
            classifier = NormBothLinear(hidden_size, self.num_classes)

            # Daftarkan modul sebagai atribut, misalnya `conv1d_static`.
            setattr(self, f'conv1d_{name}', conv1d)
            setattr(self, f'contextual_module_{name}', contextual_module)
            setattr(self, f'classifier_{name}', classifier)

        # Objek loss yang dipakai di `get_loss()`.
        self.loss = {
            'ctc': torch.nn.CTCLoss(reduction='none', zero_infinity=False),
            'kl': KLdis(),
        }
        self.loss_weights = loss_weights
        self.norm_scale = norm_scale

    def backward_hook(self, module, grad_input, grad_output):
        # Mencegah gradien NaN menyebar ke proses backward.
        for g in grad_input:
            g[g != g] = 0

    def forward_contextual(self, framewise, len_x, conv1d_module, contextual_module, classifier):
        """Memproses satu stream melalui conv temporal, BiLSTM, dan classifier.

        Deskripsi:
        Fungsi ini menjadi jalur umum untuk stream static, motion, dan fusion.
        Fitur framewise diproses dulu oleh temporal convolution, lalu dilanjutkan
        ke BiLSTM kontekstual, dan akhirnya dikonversi menjadi logits kelas.

        Input:
        1. `framewise`: tensor fitur stream dengan bentuk `(B, T, C_in)`.
        2. `len_x`: panjang asli setiap sequence dalam batch.
        3. `conv1d_module`: modul temporal convolution untuk stream ini.
        4. `contextual_module`: modul BiLSTM untuk konteks sequence.
        5. `classifier`: classifier akhir.

        Proses:
        1. Ubah format tensor agar cocok dengan `TemporalConv`.
        2. Ambil fitur hasil conv dan panjang fitur (`feat_len`).
        3. Ubah fitur ke format yang sesuai untuk BiLSTM.
        4. Hitung logits dari jalur conv dan jalur kontekstual.

        Output:
        1. `conv1d_logits`: logits dari jalur temporal conv.
        2. `seq_logits`: logits dari jalur BiLSTM kontekstual.
        3. `feat_len`: panjang sequence setelah downsampling temporal.
        """

        # `TemporalConv` mengharapkan bentuk `(B, C_in, T)`.
        conv1d_ret = conv1d_module(framewise.transpose(1, 2), len_x)

        # `visual_feat` biasanya kembali dalam bentuk `(T_feat, B, C)`.
        conv1d_feat = conv1d_ret['visual_feat'].transpose(0, 1)
        feat_len = conv1d_ret['feat_len']

        # BiLSTM menerima `(T_feat, B, C)` dan mengembalikan prediksi dalam dictionary.
        contextual_feat = contextual_module(conv1d_feat.transpose(0, 1), feat_len)['predictions']
        contextual_feat = contextual_feat.transpose(0, 1)

        # Ubah kedua representasi menjadi logits kelas.
        conv1d_logits = classifier(conv1d_feat.transpose(0, 1))
        seq_logits = classifier(contextual_feat.transpose(0, 1))

        return conv1d_logits, seq_logits, feat_len

    def forward(self, inputs_dict):
        """Melakukan forward pass saat `model(data)` dipanggil.

        Deskripsi:
        Fungsi ini adalah pintu masuk utama model. Data batch dari DataLoader
        dibongkar, lalu diproses oleh visual module. Jika training memakai
        consistency regularization, model akan memproses dua view untuk setiap
        stream. Jika tidak, model hanya memakai stream fusion untuk decoding.

        Input:
        1. `inputs_dict`: dictionary batch dari DataLoader.
        2. Di dalamnya wajib ada key `x` dan `len_x`.

        Proses:
        1. Ambil `x` dan `len_x` dari batch.
        2. Jalankan visual extractor untuk menghasilkan fitur stream.
        3. Jika training dengan CR, proses `view1` dan `view2` untuk tiap stream.
        4. Jika evaluasi, proses stream fusion lalu decode hasilnya.

        Output:
        1. Saat training CR: dictionary berisi output view1/view2 tiap stream.
        2. Saat evaluasi: dictionary berisi hasil decoding fusion.
        """

        # Ambil batch yang dikirim dari DataLoader.
        x, len_x = inputs_dict['x'], inputs_dict['len_x']

        # Visual module adalah tahap pertama setelah batch masuk ke model.
        visual_ret = self.visual_module(x, len_x)

        # Training dengan consistency regularization: proses dua view sekaligus.
        if self.apply_CR and self.training:
            results = {}
            for stream_type in self.stream_configs.keys():
                # Setiap stream memiliki dua view hasil augmentasi: view1_* dan view2_*.
                view1, view2 = visual_ret[f'view1_{stream_type}'], visual_ret[f'view2_{stream_type}']

                # Ambil modul yang sesuai dengan stream saat ini.
                conv1d_module = getattr(self, f'conv1d_{stream_type}')
                contextual_module = getattr(self, f'contextual_module_{stream_type}')
                classifier = getattr(self, f'classifier_{stream_type}')

                # Proses kedua view lewat pipeline yang sama.
                results[f'view1_{stream_type}'] = self.forward_contextual(
                    view1, len_x, conv1d_module, contextual_module, classifier
                )
                results[f'view2_{stream_type}'] = self.forward_contextual(
                    view2, len_x, conv1d_module, contextual_module, classifier
                )

            # Panjang fitur utama diambil dari stream static sebagai referensi.
            results['feat_len'] = results['view1_static'][-1]
            return results

        # Evaluasi atau training tanpa CR: gunakan representasi fusion saja.
        fusion = visual_ret['fusion']
        conv1d_logits_fusion, seq_logits_fusion, feat_len = self.forward_contextual(
            fusion,
            len_x,
            self.conv1d_fusion,
            self.contextual_module_fusion,
            self.classifier_fusion,
        )

        def decode_if_not_training(logits):
            # Saat training, hasil decoding dikosongkan agar tidak dihitung.
            return None if self.training else self.decoder.decode(
                logits * self.norm_scale, feat_len, batch_first=False, probs=False
            )

        return {
            'conv_sents_fusion': decode_if_not_training(conv1d_logits_fusion),
            'recognized_sents_fusion': decode_if_not_training(seq_logits_fusion),
        }

    def get_ctc_loss(self, no_scale_logits, label, feat_len, label_len):
        """Menghitung loss CTC rata-rata untuk satu batch.

        Deskripsi:
        Fungsi ini dipakai untuk menghitung CTC loss dari logits yang belum
        diskalakan. Output loss masih per-sample, lalu dirata-ratakan.

        Input:
        1. `no_scale_logits`: logits sebelum skala akhir, bentuk `(T, B, C)`.
        2. `label`: label target yang sudah digabung menjadi satu tensor.
        3. `feat_len`: panjang fitur hasil downsampling, bentuk `(B,)`.
        4. `label_len`: panjang label per sample, bentuk `(B,)`.

        Proses:
        1. Kalikan logits dengan `norm_scale`.
        2. Ubah ke `log_softmax`.
        3. Hitung `CTCLoss` menggunakan panjang input dan label.

        Output:
        1. Nilai loss CTC rata-rata dalam bentuk tensor skalar.
        """

        ctc_loss = self.loss['ctc'](
            (no_scale_logits * self.norm_scale).log_softmax(-1),
            label.cpu().int(),
            feat_len.cpu().int(),
            label_len.cpu().int(),
        )

        # Criterion mengembalikan loss per-sample karena `reduction='none'`.
        return ctc_loss.mean()

    def get_loss(self, ret_dict, inputs_dict):
        """Menghitung semua loss yang sedang diaktifkan pada model.

        Deskripsi:
        Fungsi ini dipanggil setelah `ret_dict = model(data)` pada loop training.
        Label diambil dari `inputs_dict`, lalu model menghitung loss sesuai isi
        `self.loss_weights`. Loss yang dihitung bisa berupa CTC loss atau KL loss.

        Input:
        1. `ret_dict`: output dari `forward()`.
        2. `inputs_dict`: batch asli dari DataLoader yang berisi label.

        Proses:
        1. Ambil `label` dan `label_lgt` dari batch.
        2. Iterasi semua key loss pada `self.loss_weights`.
        3. Jika loss bertipe CTC, hitung loss untuk view1 dan view2.
        4. Jika loss bertipe KL, hitung loss simetris antara dua view.
        5. Jumlahkan semua loss menjadi total loss.

        Output:
        1. `loss`: total loss skalar.
        2. `loss_dict`: dictionary berisi komponen loss per item.
        """

        loss, loss_dict = 0, {}
        label, label_lgt = inputs_dict['label'], inputs_dict['label_lgt']

        # Setiap key di `loss_weights` menyimpan jenis loss dan nama stream.
        for k, weight in self.loss_weights.items():
            temp_loss = 0

            # Contoh format key: `sesuatu_ConvCTC_static` atau `sesuatu_Conv_motion`.
            parts = k.split('_')
            loss_type = parts[1]
            stream_type = parts[2]

            if loss_type in ['ConvCTC', 'SeqCTC']:
                # Pilih logits conv (idx=0) atau logits sequence (idx=1).
                idx = 0 if loss_type == 'ConvCTC' else 1

                # Setiap entry di `ret_dict` adalah tuple: (conv_logits, seq_logits, feat_len).
                view1_loss = self.get_ctc_loss(
                    ret_dict[f'view1_{stream_type}'][idx],
                    label,
                    ret_dict['feat_len'],
                    label_lgt,
                )
                view2_loss = self.get_ctc_loss(
                    ret_dict[f'view2_{stream_type}'][idx],
                    label,
                    ret_dict['feat_len'],
                    label_lgt,
                )

                # Rata-ratakan kedua view lalu kalikan dengan bobot loss.
                temp_loss = (view1_loss + view2_loss) * 0.5 * weight

            else:
                # Loss non-CTC memakai KL divergence antara dua view.
                idx = 0 if loss_type == 'Conv' else 1
                view1_logits = ret_dict[f'view1_{stream_type}'][idx] * self.norm_scale
                view2_logits = ret_dict[f'view2_{stream_type}'][idx] * self.norm_scale

                # KL simetris: view1 -> view2 dan view2 -> view1.
                kl_loss1 = self.loss['kl'](view1_logits, view2_logits)
                kl_loss2 = self.loss['kl'](view2_logits, view1_logits)
                temp_loss = (kl_loss1 + kl_loss2) * 0.5 * weight

            # Tambahkan ke total loss dan simpan untuk logging.
            loss += temp_loss
            loss_dict[k] = temp_loss

        return loss, loss_dict