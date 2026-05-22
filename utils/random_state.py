# import pdb
import torch
import random
import numpy as np


class RandomState(object):
    def __init__(self, seed):
        """Mengatur seluruh sumber random agar hasil eksperimen reproducible.

        Input:
        1. seed: nilai seed yang digunakan untuk Torch, CUDA, NumPy, dan random.

        Proses:
        1. Membatasi jumlah thread Torch ke 1.
        2. Mengaktifkan mode deterministik pada cuDNN.
        3. Menonaktifkan benchmark cuDNN agar hasil tetap konsisten.
        4. Menetapkan seed untuk Torch CPU dan CUDA.
        5. Menetapkan seed untuk NumPy dan modul random Python.
        """
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def save_rng_state(self):
        """Menyimpan state random dari Torch, CUDA, NumPy, dan random.

        Proses:
        1. Mengambil state RNG Torch CPU.
        2. Mengambil state RNG semua device CUDA.
        3. Mengambil state NumPy dan random Python.

        Output:
        1. Dictionary berisi seluruh state random saat ini.
        """
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    def set_rng_state(self, rng_dict):
        """Memulihkan state random yang sebelumnya sudah disimpan.

        Input:
        1. rng_dict: dictionary state hasil save_rng_state().

        Proses:
        1. Memulihkan state Torch CPU.
        2. Memulihkan state semua device CUDA.
        3. Memulihkan state NumPy dan random Python.
        """
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        np.random.set_state(rng_dict["numpy"])
        random.setstate(rng_dict["random"])
