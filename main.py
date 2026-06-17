

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
        """
        Initialize processor, configurations, dataset, model, and optimizer.

        Parameters
        ----------
        arg : argparse.Namespace
            Parsed command-line arguments.
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
        """
        Save runtime arguments to the configuration file in work_dir.
        """
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        """
        Build the model and optimizer, and load weights if necessary.

        Returns
        -------
        tuple
            Ready-to-use model and configured optimizer.
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
        """
        Move the model to the output device used for training.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be moved.

        Returns
        -------
        torch.nn.Module
            Model on the target device.
        """
        model = model.to(self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        """
        Load model weights from a checkpoint file.

        Parameters
        ----------
        model : torch.nn.Module
            Model to update.
        weight_path : str
            Path to the checkpoint file.
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
        """
        Create a DataLoader for a specific dataset split.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to wrap.
        mode : str
            Dataset split name.
        train_flag : bool
            True if the split is used for training.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for the split.
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
        """
        Create a model instance from the slr_network module.

        Parameters
        ----------
        args : dict
            Model arguments.

        Returns
        -------
        torch.nn.Module
            Instantiated model.
        """
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(
            **args,
            gloss_dict=self.gloss_dict,
        )
        return model

    def load_data(self):
        """
        Load all data splits and build their corresponding DataLoaders.
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
        """
        Load dataset metadata from the YAML configuration file.
        """
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    def judge_save_eval(self, epoch):
        """
        Determine if the model should be saved and evaluated at the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        tuple
            Boolean flags (save_model, eval_model).
        """
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        """
        Save checkpoint containing model, optimizer, scheduler, and RNG state.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        save_path : str
            Path to save the checkpoint.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def custom_save_model(self, dev_wer, epoch, save_dir):
        """
        Manage current and best model files in the save directory.

        Parameters
        ----------
        dev_wer : float
            Current development set WER.
        epoch : int
            Current epoch number.
        save_dir : str
            Directory to save models.
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
        """
        Clean up old checkpoints and save the final model artifact.

        Parameters
        ----------
        dev_wer : float
            Final development set WER.
        epoch : int
            Final epoch number.
        save_dir : str
            Directory to save the final model.
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
        """
        Copy the work directory to a Google Drive folder if configured.
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
        """
        Run the complete training loop for all epochs.
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
        """
        Run evaluation on a specific data split.

        Parameters
        ----------
        mode : str
            Name of the data split to evaluate.
        epoch : int
            Current epoch indicator for logging.

        Returns
        -------
        float
            Word Error Rate (WER).
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
        """
        Execute the main workflow based on the selected phase.
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
    Main entry point for Continuous Sign Language Recognition (CSLR).
    
    Reads command-line arguments and configuration files, then initializes
    and starts the SLRProcessor.
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