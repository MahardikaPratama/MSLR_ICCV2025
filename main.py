

## 1. Import libraries & set CUDA order
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import shutil
import utils
import numpy as np
import modules
import torch
import torch.nn as nn
import datasets
import yaml
import json
import faulthandler
faulthandler.enable()
from seq_scripts import seq_train, seq_eval
import slr_network



class SLRProcessor(object):
    # 2. Inisialisasi objek SLRProcessor dengan memuat parameter, dataset, model, dan optimizer
    def __init__(self, arg):
        """Inisialisasi processor, konfigurasi, dataset, model, dan optimizer.

        Input:
        1. arg: objek argumen hasil parsing konfigurasi program.

        Proses:
        1. Memanggil konstruktor parent class.
        2. Menyimpan argumen ke self.arg.
        3. Menyimpan salinan konfigurasi ke file work_dir/config.yaml.
        4. Membuat random state jika random_fix aktif.
        5. Menyiapkan device GPU dan recorder log.
        6. Menginisialisasi container dataset dan data loader.
        7. Memuat informasi dataset dari file konfigurasi dataset.
        8. Membaca dictionary gloss dari path yang didefinisikan dataset.
        9. Memuat model dan optimizer lewat loading().
        10. Menentukan nilai awal best_dev_wer dan nama task dataset.

        Output:
        1. Seluruh komponen utama processor siap digunakan.
        """
        super().__init__()
        self.arg = arg
        self.save_arg()  
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.load_dataset_info()  # 7
        with open(self.arg.dataset_info['dict_path'], 'r') as f:
            self.gloss_dict = json.load(f)
        self.model, self.optimizer = self.loading()  # 9
        self.best_dev_wer = 1000
        self.tasks = self.arg.dataset[-2:]

    def save_arg(self):
        """Menyimpan argumen runtime ke file konfigurasi di work_dir.

        Proses:
        1. Mengambil seluruh atribut self.arg menjadi dictionary.
        2. Membuat work_dir jika belum ada.
        3. Menulis konfigurasi ke file config.yaml.
        """
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        """Membangun model, optimizer, lalu memuat bobot jika diperlukan.

        Proses:
        1. Menetapkan device aktif.
        2. Membangun model dari argumen model.
        3. Membuat optimizer yang membungkus model.
        4. Memuat bobot atau checkpoint bila diminta oleh konfigurasi.
        5. Memindahkan model ke device utama.
        6. Memuat dataset dan data loader.

        Output:
        1. Model yang sudah siap dipakai.
        2. Optimizer yang sudah dikonfigurasi.
        """
        self.device.set_device(self.arg.device)
        print("Loading model")
        model = self.build_module(self.arg.model_args)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)
        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        """Memindahkan model ke device output yang dipakai proses training.

        Proses:
        1. Mengirim model ke output_device.
        2. Memastikan model berada di CUDA.
        """
        model = model.to(self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        """Memuat bobot model dari file checkpoint.

        Proses:
        1. Membaca state_dict model dari file weight_path.
        2. Menghapus bobot yang tercantum di ignore_weights bila ada.
        3. Memasang state_dict ke model dengan strict=False.
        """
        state_dict = torch.load(weight_path, weights_only=False)['model_state_dict']
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        model.load_state_dict(state_dict, strict=False)

    def build_dataloader(self, dataset, mode, train_flag):
        """Membuat DataLoader untuk satu split dataset.

        Input:
        1. dataset: objek dataset yang akan dibungkus DataLoader.
        2. mode: nama split dataset.
        3. train_flag: penanda apakah split dipakai untuk training.

        Proses:
        1. Menentukan batch size berdasarkan mode.
        2. Mengatur shuffle dan drop_last sesuai kebutuhan training.
        3. Memakai collate_fn dari feeder.

        Output:
        1. torch.utils.data.DataLoader untuk split tersebut.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,
            collate_fn=self.feeder.collate_fn,
        )

    def build_module(self, args):
        """Membuat instance model dari modul slr_network.

        Proses:
        1. Mengambil class model berdasarkan nama di self.arg.model.
        2. Menginisialisasi model dengan argumen model dan gloss_dict.

        Output:
        1. Objek model siap dipakai.
        """
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(
            **args,
            gloss_dict=self.gloss_dict,
        )
        return model

    def load_data(self):
        """Memuat seluruh split data dan membangun DataLoader-nya.

        Proses:
        1. Mengambil class feeder dari modul datasets.
        2. Menentukan split train, dev, dan seluruh test set.
        3. Membuat mapping gloss ke index.
        4. Menyiapkan argumen feeder untuk tiap split.
        5. Membuat dataset dan DataLoader untuk setiap split.

        Output:
        1. self.dataset dan self.data_loader terisi untuk semua split.
        """
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)
        dataset_list = zip(
            ["train", "dev", "test_sd", "test_si_major", "test_si_minor"],
            [True, False, False, False, False]
        )
        # Membuat mapping gloss ke index untuk digunakan oleh feeder
        g2i_dict = {k: v['index'] for k, v in self.gloss_dict['gloss2id'].items()}
        # Memuat dataset dan data loader untuk setiap split
        # iterasi melalui dataset_list untuk membuat dataset dan data loader sesuai mode dan train_flag
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["mode"] = mode
            arg["transform_mode"] = train_flag
            arg["dataset"] = self.arg.dataset
            arg["dataset_root"] = self.arg.dataset_info.get("dataset_root", "./datasets")
            self.dataset[mode] = self.feeder(gloss_dict=g2i_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def load_dataset_info(self):
        """Memuat metadata dataset dari file konfigurasi YAML.

        Proses:
        1. Menentukan file konfigurasi berdasarkan nama dataset.
        2. Membaca file YAML.
        3. Menyimpan hasilnya ke self.arg.dataset_info.
        """
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    def judge_save_eval(self, epoch):
        """Menentukan apakah model perlu disimpan dan dievaluasi.

        Proses:
        1. Mengecek interval penyimpanan model.
        2. Mengecek interval evaluasi model.
        3. Mengembalikan dua status tersebut.
        """
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        """Menyimpan checkpoint model, optimizer, scheduler, dan RNG state.

        Proses:
        1. Mengambil state dari model.
        2. Mengambil state optimizer dan scheduler.
        3. Menyimpan semuanya ke save_path.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def custom_save_model(self, dev_wer, epoch, save_dir):
        """Mengelola file model cur dan best pada folder penyimpanan.

        Proses:
        1. Mencari file .pt yang sudah ada di save_dir.
        2. Menghapus checkpoint cur lama jika ada.
        3. Menyimpan checkpoint cur baru.
        4. Jika dev_wer membaik, mengganti checkpoint best.
        5. Memperbarui best_dev_wer.
        """
        dirs = os.listdir(save_dir)
        dirs = list(filter(lambda x: x.endswith('.pt'), dirs))
        assert len(dirs) <= 2
        best_path, cur_path = None, None
        for item in dirs:
            if 'best' in item:
                best_path = os.path.join(save_dir, item)
            if 'cur' in item:
                cur_path = os.path.join(save_dir, item)
        if cur_path is not None:
            os.remove(cur_path)
        model_path = "{}cur_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
        self.save_model(epoch, model_path)
        if best_path is not None:
            if dev_wer <= self.best_dev_wer:
                os.remove(best_path)
                model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
                self.save_model(epoch, model_path)
                self.best_dev_wer = dev_wer
        else:
            model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
            self.save_model(epoch, model_path)
            self.best_dev_wer = dev_wer

    def finalize_model_artifacts(self, dev_wer, epoch, save_dir):
        """Membersihkan checkpoint lama dan menyimpan artifact final model.

        Proses:
        1. Menghapus file cur atau best yang masih tersisa di save_dir.
        2. Menentukan nilai WER final untuk penamaan file.
        3. Menyimpan model final sebagai best_dev_..._model.pt.
        4. Menulis log bahwa model final sudah disimpan.
        """
        dirs = os.listdir(save_dir)
        pt_files = [os.path.join(save_dir, item) for item in dirs if item.endswith('.pt')]

        for path in pt_files:
            name = os.path.basename(path)
            if 'cur' in name or 'best' in name:
                os.remove(path)

        final_wer = 999.99 if dev_wer is None else dev_wer
        model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, final_wer, epoch)
        self.save_model(epoch, model_path)
        self.recoder.print_log(
            "Final model saved from last epoch: {}".format(model_path)
        )

    def sync_workdir_to_google_drive(self):
        """Menyalin work_dir ke folder Google Drive bila dikonfigurasi.

        Proses:
        1. Membaca target Google Drive dari argumen.
        2. Mengabaikan sinkronisasi jika target tidak diset.
        3. Membuat folder target bila perlu.
        4. Menghapus salinan lama dengan nama folder yang sama.
        5. Menyalin seluruh work_dir ke target.
        """
        target_root = getattr(self.arg, 'google_drive_dir', None)
        if not target_root:
            return

        src_dir = os.path.abspath(self.arg.work_dir)
        target_root = os.path.abspath(os.path.expanduser(target_root))
        os.makedirs(target_root, exist_ok=True)

        dst_dir = os.path.join(target_root, os.path.basename(os.path.normpath(src_dir)))
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        self.recoder.print_log(
            "Work dir synced to Google Drive: {}".format(dst_dir)
        )

    def train(self):
        """Menjalankan loop training lengkap untuk semua epoch.

        Proses:
        1. Mencetak parameter training ke log.
        2. Melatih model pada split train untuk setiap epoch.
        3. Mengevaluasi dev set saat interval evaluasi atau penyimpanan tercapai.
        4. Menyimpan checkpoint model sesuai aturan save interval.
        5. Menyinkronkan work_dir ke Google Drive setelah training selesai.
        """
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        # Loop utama training untuk setiap epoch
        for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
            # Menentukan apakah model perlu disimpan dan dievaluasi pada epoch ini
            save_model, eval_model = self.judge_save_eval(epoch)
            # Melatih model pada split train untuk epoch ini
            seq_train(
                self.data_loader['train'], self.model, self.optimizer, self.device,
                epoch, self.recoder, **self.arg.train_args
            )
            # Inisialisasi dev_error untuk menyimpan hasil evaluasi dev jika eval_model True
            dev_error = None
            # Mengevaluasi dev set saat interval evaluasi atau penyimpanan tercapai
            if eval_model or save_model or (epoch == self.arg.num_epoch - 1):
                dev_error = self.test('dev', epoch)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_error))
            if save_model:
                self.custom_save_model(dev_error, epoch, self.arg.work_dir)

        # Langsung sync — best_ model sudah ada dari custom_save_model
        self.sync_workdir_to_google_drive()

    def test(self, mode, epoch):
        """Menjalankan evaluasi pada split data tertentu.

        Input:
        1. mode: nama split data yang akan diuji.
        2. epoch: penanda epoch untuk logging dan penamaan hasil.

        Proses:
        1. Memanggil seq_eval dengan parameter evaluasi lengkap.

        Output:
        1. Nilai WER hasil evaluasi.
        """
        wer = seq_eval(
            self.arg,
            self.data_loader[mode],
            self.model,
            self.device,
            mode,
            epoch,
            self.arg.work_dir,
            self.recoder,
            self.tasks,
            self.arg.evaluate_tool
        )
        return wer

    def start(self):
        """Menjalankan alur utama training atau testing berdasarkan phase.

        Input:
        1. self.arg.phase: penentu mode proses, yaitu train atau test.

        Proses:
        1. Jika phase bernilai train, memanggil train().
        2. Jika phase bernilai test, menampilkan informasi model dan bobot.
        3. Pada mode test, menjalankan evaluasi untuk dev, test_sd, test_si_major, dan test_si_minor.
        4. Menyinkronkan work_dir ke Google Drive jika dikonfigurasi.

        Output:
        1. Training atau evaluasi model dijalankan sesuai konfigurasi.
        """
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.recoder.print_log('--- Testing on Dev ---')
            self.test('dev', 6667)
            self.recoder.print_log('--- Testing on Test SD ---')
            self.test('test_sd', 6667)
            self.recoder.print_log('--- Testing on Test SI-Major ---')
            self.test('test_si_major', 6667)
            self.recoder.print_log('--- Testing on Test SI-Minor ---')
            self.test('test_si_minor', 6667)
            self.recoder.print_log('Evaluation Done.\n')
            # Sync test results to Google Drive if configured
            self.sync_workdir_to_google_drive()

# 1. Blok utama program untuk menjalankan CSLR
if __name__ == '__main__':
    """
    Deskripsi:
    Blok utama program untuk membaca konfigurasi argument, memuat parameter dari file konfigurasi, 
    kemudian menjalankan proses Continuous Sign Language Recognition (CSLR).

    Input:
    1. Argument command line dari terminal.
    2. File konfigurasi (.yaml) apabila parameter config diberikan.

    Proses:
    1. Membuat/mengambil parser untuk mendefinisikan argument yang bisa digunakan saat program dijalankan.
    2. Membaca argument dari terminal lalu menyimpannya ke variabel p.
    3. Mengecek apakah parameter config diberikan.
       3a. Jika p.config tidak bernilai None, maka file konfigurasi YAML dibuka dan dibaca.
       3b. Jika parameter pada file konfigurasi tidak sesuai dengan parser argument, maka program menampilkan pesan error.
       3c. Jika parameter valid, maka nilai parameter dari file konfigurasi dijadikan default argument.
    4. Membaca ulang seluruh argument dan menyimpannya ke variabel args.
    5. Membuat objek SLRProcessor menggunakan argument yang telah diproses.
    6. Menjalankan proses utama CSLR melalui method start().

    Output:
    Program CSLR dijalankan sesuai konfigurasi argument dan file konfigurasi yang diberikan.
    """

    # 1. Membuat/mengambil parser untuk mendefinisikan argument yang bisa digunakan saat program dijalankan.    
    sparser = utils.get_parser()
    # 2. Membaca argument dari terminal lalu menyimpannya ke variabel p.
    p = sparser.parse_args()
    # 3. Mengecek apakah parameter config diberikan.
    if p.config is not None:
        # 3a. Jika p.config tidak bernilai None, maka file konfigurasi YAML dibuka dan dibaca.
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            # 3b. Jika parameter pada file konfigurasi tidak sesuai dengan parser argument, maka program menampilkan pesan error.
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert k in key
        # 3c. Jika parameter valid, maka nilai parameter dari file konfigurasi dijadikan default argument.
        sparser.set_defaults(**default_arg)
    # 4. Membaca ulang seluruh argument dan menyimpannya ke variabel args.
    args = sparser.parse_args()
    # 5. Membuat objek SLRProcessor menggunakan argument yang telah diproses.
    main_processor = SLRProcessor(args)
    # 6. Menjalankan proses utama CSLR melalui method start().
    main_processor.start()