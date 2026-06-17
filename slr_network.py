"""
Model definition for the Two-Stream CoSign architecture in CSLR.

This module composes the main model pipeline used during training and evaluation.
The input data comes from `inputs_dict['x']` and `inputs_dict['len_x']` produced
by the dataset and `collate_fn`. The data is then processed by the visual
extractor, temporal convolution, contextual BiLSTM, and the classifier.
During training, the model uses CTC and KL loss, while during evaluation,
the model performs decoding to generate gloss predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from modules.temporal_layers import BiLSTMLayer, TemporalConv
from modules.visual_extractor import CoSign2s

class KLdis(nn.Module):
    """
    KL-divergence distillation loss between two sets of logits.

    This module is used to make the prediction distributions of two views
    from the same input approach each other. In `get_loss()`, this loss is
    called bidirectionally so that the result is symmetric.

    Parameters
    ----------
    view1_logits : torch.Tensor
        Logits from the first view.
    view2_logits : torch.Tensor
        Logits from the second view.
    use_blank : bool, optional
        Whether the CTC blank class is included in the calculation.
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
    """
    Classifier that normalizes features and weights before matrix multiplication.

    This layer works like a cosine similarity-based classifier. Feature vectors
    are normalized, weights are also normalized, and then they are multiplied
    to produce class scores.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output class dimension.
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
    """
    Main two-stream model for CSLR.

    This model receives a skeleton batch, extracts visual features, and passes
    them to temporal convolution, BiLSTM, and classifier. During training with
    consistency regularization, the model processes two views for each stream.
    During evaluation, the model uses the fusion representation to produce
    predictions.

    Parameters
    ----------
    visual_args : dict
        Configuration for `CoSign2s`.
    gloss_dict : dict
        Gloss dictionary for the decoder.
    conv_type : int
        Type of temporal convolution.
    loss_weights : dict
        Loss weights calculated during training.
    norm_scale : int, optional
        Scale factor for logits before decoding or CTC. Default is 32.
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
        """
        Process a single stream through temporal conv, BiLSTM, and classifier.

        This function serves as a common path for static, motion, and fusion streams.
        Framewise features are first processed by temporal convolution, then
        passed to the contextual BiLSTM, and finally converted into class logits.

        Parameters
        ----------
        framewise : torch.Tensor
            Stream feature tensor with shape `(B, T, C_in)`.
        len_x : torch.Tensor
            Original length of each sequence in the batch.
        conv1d_module : torch.nn.Module
            Temporal convolution module for this stream.
        contextual_module : torch.nn.Module
            BiLSTM module for sequence context.
        classifier : torch.nn.Module
            Final classifier.

        Returns
        -------
        tuple
            (conv1d_logits, seq_logits, feat_len)
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
        """
        Perform a forward pass.

        This function is the main entry point of the model. Batch data from
        DataLoader is unpacked and processed by the visual module. If training
        uses consistency regularization, the model processes two views for each
        stream. Otherwise, the model only uses the fusion stream for decoding.

        Parameters
        ----------
        inputs_dict : dict
            Batch dictionary from DataLoader. Must contain `x` and `len_x`.

        Returns
        -------
        dict
            Outputs of view1/view2 for each stream during CR training,
            or decoding results of fusion during evaluation.
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
            if self.training or inputs_dict.get('skip_decoding', False):
                return None
            return self.decoder.decode(
                logits * self.norm_scale, feat_len, batch_first=False, probs=False
            )

        return {
            'conv_logits_fusion': conv1d_logits_fusion,
            'seq_logits_fusion': seq_logits_fusion,
            'feat_len': feat_len,
            'conv_sents_fusion': decode_if_not_training(conv1d_logits_fusion),
            'recognized_sents_fusion': decode_if_not_training(seq_logits_fusion),
        }

    def get_ctc_loss(self, no_scale_logits, label, feat_len, label_len):
        """
        Calculate the average CTC loss for a single batch.

        This function calculates the CTC loss from unscaled logits.
        The output loss is per-sample, then averaged.

        Parameters
        ----------
        no_scale_logits : torch.Tensor
            Logits before final scaling, shape `(T, B, C)`.
        label : torch.Tensor
            Target labels concatenated into a single tensor.
        feat_len : torch.Tensor
            Feature lengths after downsampling, shape `(B,)`.
        label_len : torch.Tensor
            Label lengths per sample, shape `(B,)`.

        Returns
        -------
        torch.Tensor
            Average CTC loss as a scalar tensor.
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
        """
        Calculate all active losses on the model.

        This function is called after `ret_dict = model(data)` in the training loop.
        Labels are taken from `inputs_dict`, then the model calculates the loss
        according to `self.loss_weights`. The calculated loss can be CTC or KL.

        Parameters
        ----------
        ret_dict : dict
            Output from `forward()`.
        inputs_dict : dict
            Original batch from DataLoader containing labels.

        Returns
        -------
        tuple
            Total scalar loss and a dictionary of individual loss components.
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