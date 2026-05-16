

## 1. Import libraries & set CUDA order
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
        """
        Deskripsi:
        Fungsi inisialisasi untuk objek SLRProcessor yang memuat parameter konfigurasi, dataset, model, dan optimizer.

        Input:
        1. arg → objek yang berisi parameter konfigurasi untuk proses training dan testing.

        Proses:
        1. Memanggil konstruktor parent class menggunakan super().__init__().
        2. Menyimpan parameter konfigurasi ke dalam self.arg.
        3. Memanggil fungsi save_arg() untuk menyimpan konfigurasi argument.
        4. Mengecek apakah random_fix bernilai True.
        4a. Jika bernilai True, maka membuat random state menggunakan random_seed.
        5. Membuat device GPU menggunakan utils.GpuDataParallel().
        6. Membuat objek recorder untuk pencatatan log proses.
        7. Menginisialisasi dictionary dataset dan data_loader.
        8. Memanggil fungsi load_dataset_info() untuk memuat informasi dataset.
        9. Membuka file dictionary gloss berdasarkan path dataset_info['dict_path'].
        10. Memuat dictionary gloss menggunakan json.load().
        11. Memanggil fungsi loading() untuk memuat model dan optimizer.
        12. Menginisialisasi best_dev_wer dengan nilai 1000.
        13. Mengambil task dataset dari dua karakter terakhir nama dataset.

        Output:
        1. Parameter konfigurasi tersimpan.
        2. Dataset, model, optimizer, dan recorder berhasil diinisialisasi.
        3. Dictionary gloss berhasil dimuat.
        """
        super().__init__()
        self.arg = arg
        self.save_arg()  # 4
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
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
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
        model = model.to(self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path, weights_only=False)['model_state_dict']
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        model.load_state_dict(state_dict, strict=False)

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,
            collate_fn=self.feeder.collate_fn,
        )

    def build_module(self, args):
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(
            **args,
            gloss_dict=self.gloss_dict,
        )
        return model

    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)
        dataset_list = zip(["train", "dev", "test"], [True, False, False])
        g2i_dict = {k: v['index'] for k, v in self.gloss_dict['gloss2id'].items()}
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["mode"] = mode
            arg["transform_mode"] = train_flag
            arg["dataset"] = self.arg.dataset
            self.dataset[mode] = self.feeder(gloss_dict=g2i_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def load_dataset_info(self):
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    def judge_save_eval(self, epoch):
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def custom_save_model(self, dev_wer, epoch, save_dir):
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
            os.system(f'rm {cur_path}')
        model_path = "{}cur_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
        self.save_model(epoch, model_path)
        if best_path is not None:
            if dev_wer <= self.best_dev_wer:
                os.system(f'rm {best_path}')
                model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
                self.save_model(epoch, model_path)
                self.best_dev_wer = dev_wer
        else:
            model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(save_dir, dev_wer, epoch)
            self.save_model(epoch, model_path)
            self.best_dev_wer = dev_wer

    def train(self):
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
            save_model, eval_model = self.judge_save_eval(epoch)
            seq_train(
                self.data_loader['train'], self.model, self.optimizer, self.device,
                epoch, self.recoder, **self.arg.train_args
            )
            if eval_model:
                dev_error = self.test('dev', epoch)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_error))
            if save_model:
                self.custom_save_model(dev_error, epoch, self.arg.work_dir)

    def test(self, mode, epoch):
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
        """
        Deskripsi:
        Fungsi utama untuk menjalankan proses training atau testing model CSLR.

        Input:
        1. self.arg.phase → mode proses ('train' atau 'test').
        2. Parameter model dan weight hasil konfigurasi.

        Proses:
        1. Mengecek nilai self.arg.phase.
        1a. Jika bernilai 'train', maka memanggil fungsi train() untuk melakukan proses training.
        1b. Jika bernilai 'test', maka:
                - menampilkan informasi model,
                - menampilkan informasi weight model,
                - memanggil fungsi test() menggunakan mode 'dev',
                - memanggil fungsi test() menggunakan mode 'test',
                - menampilkan log bahwa evaluasi selesai dilakukan.

        Output:
        1. Proses training model dijalankan.
        2. Proses evaluasi model dijalankan.
        """
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.test('dev', 6667)
            self.test('test', 6667)
            self.recoder.print_log('Evaluation Done.\n')

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
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    main_processor = SLRProcessor(args)
    main_processor.start()