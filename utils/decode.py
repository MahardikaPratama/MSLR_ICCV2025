import os
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    """
    Decode

    Deskripsi:
        Kelas decoder untuk mengubah output probabilitas jaringan CTC menjadi
        urutan gloss yang dapat dibaca. Mendukung dua mode decoding:
            - 'max'  : greedy decoding via argmax per frame (cepat, akurasi lebih rendah).
            - 'beam' : beam search via CTCBeamDecoder (lebih akurat, lebih lambat).

        Digunakan setelah tahap klasifikasi CTC untuk menghasilkan prediksi
        urutan tanda (gloss sequence) dari output BiLSTM-CTC.

    Input (constructor):
        - gloss_dict  (dict) : kamus dengan dua subkunci:
              'id2gloss'  → {str_id: {'gloss': nama_gloss, ...}}
              'gloss2id'  → {nama_gloss: {'index': int_id, ...}}
        - num_classes (int)  : jumlah kelas gloss termasuk token blank.
        - search_mode (str)  : mode decoding, 'max' atau selain itu (beam search).
        - blank_id    (int)  : indeks token blank CTC (default 0).

    Proses:
        - Bangun lookup table dua arah: id→gloss (i2g_dict) dan gloss→id (g2i_dict).
        - Inisialisasi CTCBeamDecoder dengan vocab karakter Unicode sintetis
          (bukan karakter nyata, hanya sebagai placeholder indeks untuk decoder).

    Output (atribut publik):
        - self.i2g_dict    (dict int→str) : mapping indeks integer ke nama gloss.
        - self.g2i_dict    (dict str→int) : mapping nama gloss ke indeks integer.
        - self.ctc_decoder               : instance CTCBeamDecoder siap pakai.
    """

    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        # bangun mapping id (int) → nama gloss dari subkunci 'id2gloss'
        # kunci di gloss_dict berupa string, dikonversi ke int sebagai key dict
        self.i2g_dict = {int(k): v['gloss'] for k, v in gloss_dict['id2gloss'].items()}

        # bangun mapping nama gloss → id (int) dari subkunci 'gloss2id'
        # dipakai saat perlu mengkonversi prediksi gloss kembali ke indeks
        self.g2i_dict = {k: int(v['index']) for k, v in gloss_dict['gloss2id'].items()}

        # simpan jumlah kelas (termasuk blank) untuk inisialisasi decoder
        self.num_classes = num_classes

        # simpan mode pencarian: 'max' untuk greedy, selainnya untuk beam search
        self.search_mode = search_mode

        # simpan indeks token blank CTC; default 0 mengikuti konvensi PyTorch CTC
        self.blank_id = blank_id

        # buat vocab sintetis: num_classes karakter Unicode mulai dari U+4E20
        # CTCBeamDecoder butuh list karakter sebagai vocab, tapi nilainya tidak
        # penting karena kita decode ulang via i2g_dict — ini hanya placeholder
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]

        # inisialisasi CTCBeamDecoder dengan vocab sintetis di atas
        # beam_width=10: pertahankan 10 hipotesis terbaik di setiap langkah
        # blank_id: posisi token blank di vocab
        # num_processes=10: jumlah thread paralel untuk mempercepat decoding
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            vocab,
            beam_width=10,
            blank_id=blank_id,
            num_processes=10
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """
        Deskripsi:
            Entry point decoding. Menerima output jaringan dan mendelegasikan
            ke MaxDecode atau BeamSearch sesuai search_mode yang dikonfigurasi.
            Menangani permutasi tensor jika format bukan batch-first.

        Input:
            - nn_output  (Tensor, B×T×N atau T×B×N): output logit/prob jaringan.
            - vid_lgt    (Tensor, B)                : panjang valid (frame count)
                                                      tiap sample dalam batch.
            - batch_first(bool, default True)        : True jika dim 0 adalah batch.
              Jika False (format T×B×N), tensor dipermutasi ke B×T×N terlebih dulu.
            - probs      (bool, default False)       : True jika nn_output sudah
              berupa probabilitas (sudah melalui softmax); False jika masih logit.

        Proses:
            - Jika tidak batch_first: permutasi (T,B,N) → (B,T,N).
            - Pilih decoder berdasarkan self.search_mode.

        Output:
            - ret_list (list of list of tuple): hasil decode per sample dalam batch.
              Tiap elemen adalah list pasangan (nama_gloss, indeks_posisi).
        """
        if not batch_first:
            # permutasi dari format (T, B, N) ke (B, T, N) yang dibutuhkan decoder
            nn_output = nn_output.permute(1, 0, 2)

        if self.search_mode == "max":
            # gunakan greedy decoding: cepat tapi tidak optimal
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            # gunakan beam search: lebih akurat, cocok untuk evaluasi final
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        Deskripsi:
            Melakukan CTC beam search decoding menggunakan CTCBeamDecoder.
            Mempertahankan beam_width hipotesis terbaik di setiap langkah
            waktu dan memilih hipotesis dengan skor tertinggi sebagai output.

        Input:
            - nn_output (Tensor, B×T×N): output jaringan dalam format batch-first.
              Harus sudah dipermutasi sebelum dipanggil.
            - vid_lgt   (Tensor, B)    : panjang valid tiap sequence dalam batch.
            - probs     (bool)         : True jika sudah probabilitas (post-softmax);
              False jika masih logit (akan di-softmax di dalam fungsi ini).

        Proses:
            1. Jika belum prob: terapkan softmax pada dim -1 (per-frame per-class)
               lalu pindahkan ke CPU (CTCBeamDecoder hanya berjalan di CPU).
            2. Pindahkan vid_lgt ke CPU.
            3. Jalankan CTCBeamDecoder.decode → beam_result, beam_scores,
               timesteps, out_seq_len.
            4. Untuk tiap sample di batch:
               a. Ambil hipotesis terbaik (beam index 0).
               b. Potong sesuai panjang valid out_seq_len[batch_idx][0].
               c. Hilangkan duplikat berurutan via groupby (CTC collapse).
               d. Konversi indeks gloss ke nama gloss via i2g_dict.

        Output:
            - ret_list (list of list of tuple): satu list per sample.
              Tiap tuple: (nama_gloss: str, posisi: int).
              Posisi adalah indeks urutan dalam hasil decode (bukan frame index).
        """
        if not probs:
            # terapkan softmax agar output menjadi distribusi probabilitas valid
            # pindahkan ke CPU karena CTCBeamDecoder tidak mendukung GPU tensor
            nn_output = nn_output.softmax(-1).cpu()

        # pindahkan panjang sequence ke CPU untuk konsistensi dengan nn_output
        vid_lgt = vid_lgt.cpu()

        # jalankan beam search decoding
        # beam_result : (B, N_beams, T)  — indeks gloss per hipotesis per frame
        # beam_scores : (B, N_beams)     — log-prob tiap hipotesis (makin kecil makin baik)
        # timesteps   : (B, N_beams)     — posisi frame tiap token
        # out_seq_len : (B, N_beams)     — panjang valid tiap hipotesis
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(
            nn_output, vid_lgt
        )

        # list untuk mengumpulkan hasil decode tiap sample
        ret_list = []
        for batch_idx in range(len(nn_output)):
            # ambil hipotesis terbaik (beam index 0) dan potong sampai panjang valid
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]

            if len(first_result) != 0:
                # hilangkan token duplikat berurutan menggunakan groupby
                # (CTC collapse: "A A B B A" → "A B A")
                # x[0] mengambil nilai unik pertama dari tiap grup
                first_result = torch.stack([x[0] for x in groupby(first_result)])

            # konversi indeks integer ke nama gloss dan buat list tuple (gloss, posisi)
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(first_result)
            ])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        """
        Deskripsi:
            Melakukan greedy CTC decoding dengan mengambil argmax per frame,
            lalu menerapkan CTC collapsing rules: hapus duplikat berurutan
            dan hapus token blank.

        Input:
            - nn_output (Tensor, B×T×N): output logit jaringan (belum softmax).
              Softmax tidak diperlukan karena argmax tidak terpengaruh oleh monotonic
              transformation — posisi maksimum sama sebelum dan sesudah softmax.
            - vid_lgt   (Tensor, B)    : panjang valid tiap sequence dalam batch.

        Proses:
            1. Ambil argmax pada dim 2 (per frame) → index_list (B, T).
            2. Untuk tiap sample dalam batch:
               a. Potong sequence sesuai panjang valid vid_lgt[batch_idx].
               b. Hapus duplikat berurutan via groupby (CTC collapse step 1).
               c. Filter token blank (CTC collapse step 2).
               d. Jika masih ada token tersisa, hapus duplikat sekali lagi
                  setelah filtering (groupby kedua).
               e. Konversi indeks ke nama gloss via i2g_dict.

        Output:
            - ret_list (list of list of tuple): satu list per sample.
              Tiap tuple: (nama_gloss: str, posisi: int).
        """
        # ambil indeks kelas dengan probabilitas tertinggi untuk setiap frame
        # axis=2 karena format (B, T, N): N adalah dimensi kelas
        index_list = torch.argmax(nn_output, axis=2)

        # ambil ukuran batch dan panjang sequence maksimum
        batchsize, lgt = index_list.shape

        # list untuk mengumpulkan hasil decode tiap sample
        ret_list = []
        for batch_idx in range(batchsize):
            # potong sequence sesuai panjang valid sample ini (buang padding)
            # lalu hapus duplikat berurutan: "A A B B A" → "A B A" (CTC step 1)
            group_result = [
                x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])
            ]

            # hapus semua token blank dari hasil collapse (CTC step 2)
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]

            if len(filtered) > 0:
                # stack list tensor menjadi satu tensor untuk operasi berikutnya
                max_result = torch.stack(filtered)
                # groupby kedua: setelah blank dihapus, mungkin muncul duplikat
                # baru yang tadinya dipisahkan blank, mis. "A blank A" → setelah
                # hapus blank jadi "A A" → collapse lagi → "A"
                max_result = [x[0] for x in groupby(max_result)]
            else:
                # tidak ada token selain blank → hasil kosong
                max_result = filtered

            # konversi indeks integer ke nama gloss dan buat list tuple (gloss, posisi)
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(max_result)
            ])
        return ret_list