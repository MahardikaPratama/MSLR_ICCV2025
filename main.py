

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
from utils.class_weighting import calculate_class_weights, log_class_weights


 # Kelas utama untuk memproses training dan evaluasi SLR
 # Tugas utama kelas ini adalah mengelola seluruh proses training dan evaluasi,
 # termasuk memuat data, membangun model, menyimpan model, dan menjalankan loop training/evaluasi.
 # Kelas ini juga menangani konfigurasi dan logging selama proses berlangsung.
class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()
        # 3. Simpan argumen & set seed
        self.arg = arg
        self.save_arg()  # 4
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        # 5. Inisialisasi device & logger
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        # 6. Muat info dataset & kamus gloss
        self.load_dataset_info()  # 7
        with open(self.arg.dataset_info['dict_path'], 'r') as f:
            self.gloss_dict = json.load(f)
        # 6.5. Hitung class weights jika diaktifkan
        self.class_weights = None
        if hasattr(self.arg, 'enable_class_weighting') and self.arg.enable_class_weighting:
            weighting_method = getattr(self.arg, 'weighting_method', 'inverse_frequency')
            weighting_gamma = getattr(self.arg, 'weighting_gamma', 0.8)
            train_info_path = f"./datasets/{self.arg.dataset.split('_')[0]}/{self.arg.dataset}_train_info.json"
            self.class_weights = calculate_class_weights(
                self.gloss_dict, 
                train_info_path, 
                method=weighting_method,
                gamma=weighting_gamma
            )
            self.class_weights = self.class_weights.cuda()
            log_class_weights(self.gloss_dict, train_info_path, weighting_method, top_k=15)
        # 8. Inisialisasi model & optimizer
        self.model, self.optimizer = self.loading()  # 9
        self.best_dev_wer = 1000
        self.tasks = self.arg.dataset[-2:]

    # Menyimpan argumen konfigurasi ke file yaml
    def save_arg(self):
        # 4. Simpan argumen ke file config
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    # Memuat model dan optimizer
    def loading(self):
        # 9. Set device, bangun model & optimizer, load bobot jika ada, load data
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

    # Memindahkan model ke device (GPU)
    def model_to_device(self, model):
        # 9.1. Pindahkan model ke device
        model = model.to(self.device.output_device)
        model.cuda()
        return model

    # Memuat bobot model dari file
    def load_model_weights(self, model, weight_path):
        # 9.0. Load bobot model dari file
        state_dict = torch.load(weight_path, weights_only=False)['model_state_dict']
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        model.load_state_dict(state_dict, strict=False)

    # Membuat DataLoader untuk dataset
    def build_dataloader(self, dataset, mode, train_flag):
        # 12. Buat DataLoader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,
            collate_fn=self.feeder.collate_fn,
        )

    # Membuat model dari argumen
    def build_module(self, args):
        # 10. Ambil kelas model & inisialisasi
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(
            **args,
            gloss_dict=self.gloss_dict,
            class_weights=self.class_weights
        )
        return model

    # Memuat data dan membuat DataLoader
    def load_data(self):
        # 11. Muat feeder, dataset, dan DataLoader
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

    # Memuat info dataset dari file yaml
    def load_dataset_info(self):
        # 7. Muat info dataset dari YAML
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # Menentukan kapan model disimpan dan dievaluasi
    def judge_save_eval(self, epoch):
        # 15. Tentukan kapan simpan/evaluasi model
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    # Menyimpan model ke file
    def save_model(self, epoch, save_path):
        # 18. Simpan model ke file
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    # Menyimpan model dengan format custom (best dan current)
    def custom_save_model(self, dev_wer, epoch, save_dir):
        # 19. Simpan model best & current
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

    # Proses training model
    def train(self):
        # 16. Training loop utama
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

    # Proses evaluasi model
    def test(self, mode, epoch):
        # 17. Evaluasi model (dev/test)
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

    # Fungsi utama untuk menjalankan training atau testing
    def start(self):
        # 13. Mulai training atau testing
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.test('dev', 6667)
            self.test('test', 6667)
            self.recoder.print_log('Evaluation Done.\n')


## 2. Entry point utama
if __name__ == '__main__':
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