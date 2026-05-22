import os
# import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        """Menyiapkan konteiner informasi device yang akan dipakai program.

        Proses:
        1. Menginisialisasi daftar GPU yang dipakai.
        2. Menginisialisasi output_device yang akan menjadi target utama pemindahan data dan model.
        """
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        """Menentukan GPU yang dipakai program dari string konfigurasi device.

        Input:
        1. device: string berisi id GPU, misalnya '0' atau '0,1'.

        Proses:
        1. Mengubah input menjadi string.
        2. Jika device tidak bernilai 'None', menyusun daftar GPU aktif.
        3. Mengatur CUDA_VISIBLE_DEVICES sesuai konfigurasi.
        4. Memanggil occupy_gpu() agar GPU terlihat terpakai oleh proses ini.
        5. Menetapkan output_device sebagai GPU pertama atau cpu jika tidak ada GPU.
        """
        device = str(device)
        if device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        """Memindahkan model ke device utama dan membungkusnya dengan DataParallel bila perlu.

        Proses:
        1. Memindahkan model ke output_device.
        2. Jika lebih dari satu GPU tersedia, membungkus model dengan nn.DataParallel.

        Output:
        1. Model yang sudah siap dipakai di device target.
        """
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        """Memindahkan tensor data ke device target dengan menyesuaikan tipe datanya.

        Proses:
        1. Memeriksa tipe tensor input.
        2. Memindahkan FloatTensor, DoubleTensor, ByteTensor, dan LongTensor ke output_device.
        3. Jika input berupa list atau tuple, memproses setiap elemen secara rekursif.

        Output:
        1. Data yang sudah berada di device yang benar.
        """
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))
    
    def dict_data_to_device(self, data_dict):
        """Memindahkan isi dictionary data ke device target secara selektif.

        Proses:
        1. Membuat dictionary baru untuk data yang sudah dipindahkan.
        2. Membiarkan key yang mengandung 'origin' atau 'datasets' tetap apa adanya.
        3. Memindahkan nilai lain ke device target menggunakan data_to_device().

        Output:
        1. Dictionary baru dengan tensor yang sudah dipindahkan ke device.
        """
        cuda_dict = {}
        for k, v in data_dict.items():
            if 'origin' in k or 'datasets' in k:
                cuda_dict[k] = v
            else:
                cuda_dict[k] = self.data_to_device(v)
        return cuda_dict

    def criterion_to_device(self, loss):
        """Memindahkan objek loss atau criterion ke device target.

        Input:
        1. loss: objek loss/criterion yang mendukung method to().

        Output:
        1. Loss yang sudah dipindahkan ke output_device.
        """
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """Membuat GPU terlihat aktif di nvidia-smi dengan alokasi tensor kecil.

        Input:
        1. gpus: daftar GPU atau satu indeks GPU yang ingin ditempati.

        Proses:
        1. Jika gpus kosong, mengalokasikan tensor kecil di CUDA default.
        2. Jika gpus berisi indeks GPU, membuat tensor kecil pada setiap GPU tersebut.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
