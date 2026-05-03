
# Mengimpor modul os untuk operasi sistem
import os

# Mengatur urutan device CUDA agar sesuai dengan urutan PCI_BUS_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Mengimpor berbagai modul yang dibutuhkan
import utils  # Modul utilitas (fungsi pembantu) dari direktori utils
import numpy as np  # Untuk operasi numerik
import modules  # Modul-modul model dari direktori modules
import torch  # Library utama deep learning
import torch.nn as nn  # Neural network PyTorch
import datasets  # Modul dataset dari direktori datasets
import yaml  # Untuk membaca file konfigurasi YAML
import json  # Untuk membaca file JSON
import faulthandler  # Untuk debugging error
faulthandler.enable()  # Mengaktifkan faulthandler

# Mengimpor fungsi training dan evaluasi dari seq_scripts
from seq_scripts import seq_train, seq_eval
import slr_network  # Modul arsitektur model dari direktori slr_network


 # Kelas utama untuk memproses training dan evaluasi SLR
 # Tugas utama kelas ini adalah mengelola seluruh proses training dan evaluasi,
 # termasuk memuat data, membangun model, menyimpan model, dan menjalankan loop training/evaluasi.
 # Kelas ini juga menangani konfigurasi dan logging selama proses berlangsung.
class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()  # Memanggil konstruktor parent

        self.arg = arg  # Menyimpan argumen konfigurasi
        self.save_arg()  # Menyimpan argumen ke file config
        if self.arg.random_fix:
            # Jika random_fix diaktifkan, set semua seed untuk reproducibility
            self.rng = utils.RandomState(seed=self.arg.random_seed)  # Set random seed
        self.device = utils.GpuDataParallel()  # Inisialisasi device (GPU)
        self.recoder = utils.Recorder( 
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )  # Logger/tracker training
        self.dataset = {}  # Dictionary untuk dataset
        self.data_loader = {}  # Dictionary untuk data loader

        self.load_dataset_info()  # Memuat info dataset dari file yaml
        with open(self.arg.dataset_info['dict_path'], 'r') as f:
            self.gloss_dict = json.load(f)  # Memuat kamus gloss
        self.model, self.optimizer = self.loading()  # Inisialisasi model dan optimizer
        self.best_dev_wer = 1000  # Inisialisasi nilai WER terbaik
        self.tasks = self.arg.dataset[-2:]  # Menyimpan info task dari nama dataset untuk evaluasi

    # Menyimpan argumen konfigurasi ke file yaml
    def save_arg(self):
        arg_dict = vars(self.arg) # Mengubah argumen menjadi dictionary
        if not os.path.exists(self.arg.work_dir):
            # Jika direktori kerja belum ada, buat direktori tersebut
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            # Simpan argumen ke file config.yaml di direktori kerja
            yaml.dump(arg_dict, f)

    # Memuat model dan optimizer
    def loading(self):
        self.device.set_device(self.arg.device)  # Set device GPU
        print("Loading model")
        model = self.build_module(self.arg.model_args)  # Bangun model dari argumen model_args
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)  # Bangun optimizer dari argumen optimizer_args

        if self.arg.load_weights:
            # Jika ada bobot model yang ditentukan, muat bobot tersebut ke model
            self.load_model_weights(model, self.arg.load_weights)  # Load bobot model
        elif self.arg.load_checkpoints:
            # Jika ada checkpoint yang ditentukan, muat bobot model dan optimizer dari checkpoint tersebut
            self.load_checkpoint_weights(model, optimizer)  # Load checkpoint
        model = self.model_to_device(model)  # Pindahkan model ke device
        print("Loading model finished.")
        self.load_data()  # Muat data dan buat DataLoader
        # Kembalikan model dan optimizer yang sudah dimuat dan siap digunakan
        return model, optimizer

    # Memindahkan model ke device (GPU)
    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        # Pindahkan model ke GPU yang ditentukan
        model.cuda()  # Aktifkan mode CUDA untuk model
        return model  # Kembalikan model yang sudah dipindahkan ke device

    # Memuat bobot model dari file
    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)['model_state_dict']
        # Muat state_dict model dari file yaitu weight_path
        if len(self.arg.ignore_weights):
            # Jika ada bobot yang ingin diabaikan, hapus bobot tersebut dari state_dict sebelum memuat ke model
            for w in self.arg.ignore_weights:
                # Iterasi melalui daftar bobot yang ingin diabaikan
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        # Muat state_dict ke model, dengan strict=False untuk mengabaikan bobot yang tidak cocok
        model.load_state_dict(state_dict, strict=False)

    # Membuat DataLoader untuk dataset
    def build_dataloader(self, dataset, mode, train_flag):
        # DataLoader digunakan untuk mengelola batch data selama training dan evaluasi.
        # Fungsi ini membuat DataLoader dengan parameter yang sesuai berdasarkan mode (train/dev/test)
        # dan apakah data tersebut digunakan untuk training atau tidak.
        return torch.utils.data.DataLoader( 
            dataset, # Dataset yang akan dimuat oleh DataLoader
            batch_size=self.arg.batch_size
            # Ukuran batch yang digunakan untuk training jika train_flag True,
            # atau ukuran batch untuk testing jika train_flag False
            if mode == "train"
            else self.arg.test_batch_size,
            shuffle=train_flag,  # Jika train_flag True, data akan diacak setiap epoch
            # untuk meningkatkan generalisasi model. Jika False, data tidak akan diacak
            # untuk memastikan evaluasi yang konsisten.
            drop_last=train_flag,  # Jika train_flag True, batch terakhir yang tidak lengkap
            # akan dibuang untuk memastikan semua batch memiliki ukuran yang sama.
            # Jika False, batch terakhir akan tetap digunakan meskipun ukurannya tidak lengkap.
            num_workers=self.arg.num_worker,  # Jumlah worker untuk DataLoader
            collate_fn=self.feeder.collate_fn,  # Fungsi penggabungan batch
        )

    # Membuat model dari argumen
    def build_module(self, args):
        model_class = getattr(slr_network, self.arg.model)  # Ambil kelas model
        model = model_class(
            # Inisialisasi model dengan argumen dan kamus gloss
            **args,
            gloss_dict = self.gloss_dict
            # Kamus gloss yang digunakan untuk model, biasanya berisi mapping antara gloss dan indeksnya,
            # serta informasi lain yang relevan untuk pemrosesan data dan pelatihan model.
        )
        return model  # Kembalikan model yang sudah dibuat

    # Memuat data dan membuat DataLoader
    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)
        # Ambil feeder dataset. Feeder ini bertanggung jawab untuk memproses data mentah
        # dan mengubahnya menjadi format yang dapat digunakan oleh model. Feeder biasanya mencakup
        # fungsi untuk membaca data, melakukan augmentasi, dan menggabungkan batch data.
        # Nama feeder diambil dari argumen yang diberikan, dan kelas feeder yang sesuai akan diinstansiasi
        # untuk digunakan dalam memuat data. Feeder ini juga akan menggunakan kamus gloss untuk memetakan
        # label gloss ke indeks yang sesuai selama proses pelatihan dan evaluasi.
        # dataset_list = zip(
        #     ["train_dev", "test"], [True, False]
        # )
        dataset_list = zip(
            # Daftar mode data dan flag training yang sesuai. Mode "train" akan memiliki flag True
            # untuk mengaktifkan pengacakan dan pembentukan batch yang sesuai untuk training,
            # sedangkan mode "dev" dan "test" akan memiliki flag False untuk memastikan evaluasi
            # yang konsisten tanpa pengacakan data.
            ["train", "dev", "test"], [True, False, False]
        )  # Daftar mode data
        g2i_dict = {k: v['index'] for k, v in self.gloss_dict['gloss2id'].items()}
        # Membuat dictionary mapping dari gloss ke indeksnya, yang akan digunakan oleh feeder
        # untuk memproses label gloss menjadi format yang dapat digunakan oleh model selama pelatihan dan evaluasi.
        # Dictionary ini diambil dari bagian 'gloss2id' dalam kamus gloss yang sudah dimuat sebelumnya,
        # dan hanya menyimpan mapping antara nama gloss (k) dan indeksnya (v['index']).
        for idx, (mode, train_flag) in enumerate(dataset_list):
            # Iterasi melalui daftar mode data dan flag training untuk memuat dataset dan membuat DataLoader untuk setiap mode.
            # Proses ini mencakup memproses data mentah menggunakan feeder, membuat dataset yang sesuai untuk setiap mode,
            # dan kemudian membuat DataLoader yang akan digunakan selama pelatihan dan evaluasi model.
            arg = self.arg.feeder_args  # Mengambil argumen feeder dari konfigurasi.
            # Argumen ini akan digunakan untuk menginisialisasi dataset yang dibuat oleh feeder,
            # dan biasanya mencakup informasi seperti path ke data, parameter pemrosesan data,
            # dan informasi lain yang relevan untuk memuat dan memproses data sesuai dengan kebutuhan model.
            arg["mode"] = mode  # Menambahkan mode (train/dev/test) ke argumen feeder untuk digunakan dalam proses pemuatan data.
            # Mode ini akan memberitahu feeder bagaimana memproses data, apakah untuk training (dengan pengacakan dan pembentukan batch yang sesuai)
            # atau untuk evaluasi (dengan pengaturan yang memastikan konsistensi data).
            arg["transform_mode"] = train_flag  # Menambahkan flag training ke argumen feeder untuk digunakan dalam proses pemuatan data.
            # Flag ini akan memberitahu feeder apakah data yang dimuat akan digunakan untuk training (True) atau untuk evaluasi (False),
            # sehingga feeder dapat menyesuaikan cara memproses data, seperti apakah akan mengacak data, membentuk batch dengan ukuran tertentu,
            # atau melakukan augmentasi data selama pelatihan.
            arg["dataset"] = self.arg.dataset
            self.dataset[mode] = self.feeder(gloss_dict=g2i_dict, **arg)
            # Membuat dataset untuk mode tertentu menggunakan feeder dengan argumen yang sudah disiapkan,
            # termasuk kamus gloss yang digunakan untuk memetakan label gloss ke indeksnya.
            # Dataset ini akan berisi data yang sudah diproses dan siap digunakan untuk pelatihan atau evaluasi model,
            # tergantung pada mode yang sedang diproses.
            self.data_loader[mode] = self.build_dataloader(
                self.dataset[mode], mode, train_flag
            )
            # Buat DataLoader untuk dataset yang sudah dibuat, dengan parameter yang sesuai berdasarkan mode dan flag training.
            # DataLoader ini akan digunakan untuk mengelola batch data selama pelatihan dan evaluasi model,
            # termasuk pengacakan data untuk training dan memastikan konsistensi data untuk evaluasi.
        print("Loading data finished.")

    # Memuat info dataset dari file yaml
    def load_dataset_info(self):
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
            # Buka file YAML yang berisi informasi tentang dataset yang akan digunakan,
            # dengan nama file yang diambil dari argumen dataset yang diberikan.
            # File YAML ini biasanya berisi informasi seperti path ke data, parameter pemrosesan data,
            # dan informasi lain yang relevan untuk memuat dan memproses data sesuai dengan kebutuhan model.
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
            # Muat informasi dataset dari file YAML ke dalam argumen dataset_info.
            # Informasi ini akan digunakan selama proses pemuatan data dan pembuatan dataset,
            # serta dapat mencakup berbagai parameter yang diperlukan untuk memproses data dengan benar sesuai dengan kebutuhan model.

    # Menentukan kapan model disimpan dan dievaluasi
    def judge_save_eval(self, epoch):
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.5 * self.arg.num_epoch)
        # Model akan disimpan setiap save_interval epoch, tetapi hanya setelah mencapai setengah dari total epoch yang ditentukan.
        # Ini memungkinkan model untuk menyimpan checkpoint pada titik-titik penting selama pelatihan,
        # terutama setelah model mulai menunjukkan kinerja yang lebih stabil dan menghindari penyimpanan terlalu sering
        # pada awal pelatihan ketika model mungkin masih sangat tidak stabil.
        # save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        # Model akan dievaluasi setiap eval_interval epoch, tetapi hanya setelah mencapai epoch ke-0.
        # Ini memungkinkan evaluasi model secara berkala selama pelatihan untuk memantau kinerja model pada data dev,
        # tetapi tidak membatasi evaluasi hanya pada bagian akhir pelatihan, sehingga memberikan wawasan tentang bagaimana
        # kinerja model berkembang dari awal hingga akhir pelatihan.
        return save_model, eval_model
        # Kembalikan dua boolean yang menunjukkan apakah model harus disimpan dan dievaluasi pada epoch saat ini,
        # berdasarkan interval yang ditentukan dalam argumen konfigurasi

    # Menyimpan model ke file
    def save_model(self, epoch, save_path):
        torch.save(
            {
                'epoch': epoch,  # Menyimpan nomor epoch saat model disimpan, yang dapat digunakan untuk melanjutkan pelatihan dari checkpoint ini di masa depan.
                'model_state_dict': self.model.state_dict(),  # Menyimpan state_dict model, yang berisi bobot dan parameter model yang diperlukan untuk memuat kembali model dengan performa yang sama di masa depan.
                'optimizer_state_dict': self.optimizer.state_dict(),  # Menyimpan state_dict optimizer, yang berisi informasi tentang status optimizer, termasuk momentum, learning rate, dan parameter lain yang diperlukan untuk melanjutkan pelatihan dengan kondisi yang sama di masa depan.
                'scheduler_state_dict': self.optimizer.scheduler.state_dict(),  # Menyimpan state_dict scheduler dari optimizer, yang berisi informasi tentang status scheduler learning rate, termasuk epoch saat ini, langkah pembaruan, dan parameter lain yang diperlukan untuk melanjutkan penjadwalan learning rate dengan kondisi yang sama di masa depan.
                'rng_state': self.rng.save_rng_state(),  # Menyimpan state RNG (Random Number Generator) untuk memastikan bahwa jika pelatihan dilanjutkan dari checkpoint ini, urutan acak yang sama akan dihasilkan,
                # yang penting untuk reproducibility dan konsistensi dalam pelatihan model.
            },
            save_path,  # Menyimpan model ke path yang ditentukan, yang biasanya berupa file dengan ekstensi .pt atau .pth
            # yang berisi semua informasi yang diperlukan untuk memuat kembali model, optimizer, scheduler, dan state RNG di masa depan.
        )

    # Menyimpan model dengan format custom (best dan current)
    def custom_save_model(self, dev_wer, epoch, save_dir):
        dirs = os.listdir(save_dir)
        # List semua file di direktori penyimpanan model untuk memeriksa apakah sudah ada model yang disimpan sebelumnya,
        # baik model "best" dengan WER terbaik maupun model "current" yang baru saja disimpan.
        # Proses ini memungkinkan manajemen model yang lebih baik dengan menyimpan model terbaik berdasarkan WER
        # dan menghapus model current sebelumnya untuk menghemat ruang penyimpanan.
        dirs = list(filter(lambda x: x.endswith('.pt'), dirs))
        # Filter file yang hanya berakhir dengan ekstensi .pt, yang biasanya digunakan untuk menyimpan model PyTorch.
        # Ini memastikan bahwa hanya file model yang dipertimbangkan dalam proses manajemen model,
        # dan file lain yang mungkin ada di direktori penyimpanan tidak akan mempengaruhi logika penyimpanan model.
        assert len(dirs) <= 2
        # Pastikan bahwa tidak ada lebih dari 2 file model yang disimpan di direktori, yaitu satu untuk model "best" dan satu untuk model "current".
        # Jika ada lebih dari 2 file, ini mungkin menunjukkan masalah dalam manajemen model, seperti tidak menghapus model current sebelumnya
        # atau menyimpan terlalu banyak checkpoint, sehingga perlu diperiksa dan diperbaiki untuk memastikan manajemen model yang efisien.
        best_path, cur_path = None, None
        # Inisialisasi path untuk model "best" dan "current" sebagai None, yang akan diupdate jika ditemukan file model yang sesuai dalam direktori penyimpanan.
        # Ini memungkinkan logika selanjutnya untuk menentukan apakah model current perlu dihapus dan apakah model best perlu diperbarui berdasarkan WER yang baru dihitung.
        for item in dirs:
            # Iterasi melalui file model yang ditemukan di direktori penyimpanan untuk menentukan mana yang merupakan model "best"
            # dan mana yang merupakan model "current" berdasarkan nama file.
            # Proses ini memungkinkan manajemen model yang lebih baik dengan memastikan bahwa model current yang baru saja disimpan
            # akan menggantikan model current sebelumnya, dan model best akan diperbarui jika WER baru lebih baik dari WER terbaik sebelumnya.
            if 'best' in item: 
                best_path = os.path.join(save_dir, item)
            if 'cur' in item:
                cur_path = os.path.join(save_dir, item)
        if cur_path is not None:
            # Jika sudah ada model current yang disimpan sebelumnya, hapus model tersebut untuk menghemat ruang penyimpanan,
            # karena model current yang baru akan segera disimpan setelah ini.
            # Proses ini memastikan bahwa hanya satu model current yang disimpan pada satu waktu,
            # sehingga manajemen model menjadi lebih efisien dan tidak membingungkan dengan banyak checkpoint yang tersimpan.
            os.system(f'rm {cur_path}')  # Hapus model current sebelumnya
        model_path = "{}cur_dev_{:05.2f}_epoch{}_model.pt".format(
            # Simpan model current dengan format nama yang mencakup WER dev saat ini dan nomor epoch,
            # sehingga mudah untuk mengidentifikasi model berdasarkan performa dan epochnya.
            # Model ini akan segera disimpan setelah ini, dan akan menggantikan model current sebelumnya jika ada.
            save_dir, dev_wer, epoch
        )
        self.save_model(epoch, model_path)
        # Simpan model current ke path yang sudah ditentukan, yang mencakup informasi tentang WER dev saat ini dan nomor epoch,
        # sehingga memudahkan manajemen model dan identifikasi performa model berdasarkan nama file.
        # Model ini akan segera disimpan setelah ini, dan akan menggantikan model current sebelumnya jika ada.
        if best_path is not None:
            # Jika sudah ada model best yang disimpan sebelumnya, bandingkan WER dev saat ini dengan WER terbaik sebelumnya
            # untuk menentukan apakah model best perlu diperbarui. Proses ini memastikan bahwa model best selalu mencerminkan
            # performa terbaik yang dicapai selama pelatihan, dan jika WER baru lebih baik, model best akan diperbarui dengan
            # model current yang baru saja disimpan.
            if dev_wer <= self.best_dev_wer:
                # Jika WER dev saat ini lebih baik atau sama dengan WER terbaik sebelumnya, perbarui model best dengan model current
                # yang baru saja disimpan, dan hapus model best sebelumnya untuk menghemat ruang penyimpanan. Proses ini memastikan
                # bahwa model best selalu mencerminkan performa terbaik yang dicapai selama pelatihan, dan jika WER baru lebih baik,
                # model best akan diperbarui dengan model current yang baru saja disimpan.
                os.system(f'rm {best_path}')  # Hapus model best sebelumnya
                model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(
                    save_dir, dev_wer, epoch
                )
                self.save_model(epoch, model_path)
                self.best_dev_wer = dev_wer
        else:
            # Jika belum ada model best yang disimpan sebelumnya, simpan model current yang baru saja disimpan sebagai model best,
            # karena ini adalah model pertama yang disimpan dan secara otomatis menjadi model terbaik saat ini. Proses ini memastikan
            # bahwa model best selalu mencerminkan performa terbaik yang dicapai selama pelatihan, dan jika WER baru lebih baik,
            # model best akan diperbarui dengan model current yang baru saja disimpan.
            model_path = "{}best_dev_{:05.2f}_epoch{}_model.pt".format(
                save_dir, dev_wer, epoch
            )
            self.save_model(epoch, model_path)
            self.best_dev_wer = dev_wer

    # Proses training model
    def train(self):
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        # Log parameter konfigurasi yang digunakan untuk pelatihan, sehingga memudahkan untuk melacak konfigurasi yang digunakan
        # dalam setiap sesi pelatihan dan memungkinkan reproduksi hasil di masa depan dengan menggunakan konfigurasi yang sama.
        for epoch in range(
            # Untuk setiap epoch dalam rentang yang ditentukan oleh start_epoch dan num_epoch, jalankan proses training untuk satu epoch,
            # dan kemudian evaluasi model pada data dev jika eval_model True, serta simpan model jika save_model True.
            # Proses ini memungkinkan pelatihan model secara iteratif dengan evaluasi berkala untuk memantau kinerja model pada data dev,
            # dan manajemen model yang efisien dengan menyimpan checkpoint pada interval yang ditentukan.
            self.arg.optimizer_args['start_epoch'], self.arg.num_epoch
            # Rentang epoch untuk pelatihan, dimulai dari start_epoch yang ditentukan dalam argumen optimizer_args
            # hingga num_epoch yang ditentukan dalam argumen konfigurasi. Proses ini memungkinkan pelatihan model secara iteratif
            # dengan evaluasi berkala untuk memantau kinerja model pada data dev, dan manajemen model yang efisien dengan menyimpan
            # checkpoint pada interval yang ditentukan.
        ):
            save_model, eval_model = self.judge_save_eval(epoch)
            # Cek apakah model harus disimpan dan dievaluasi pada epoch saat ini berdasarkan interval yang ditentukan dalam argumen konfigurasi
            # seq_train(self.data_loader['train_dev'], self.model, self.optimizer, self.device,
            #     epoch, self.recoder, **self.arg.train_args
            # )
            seq_train(
                self.data_loader['train'], self.model, self.optimizer, self.device,
                # Jalankan proses training untuk satu epoch menggunakan data loader untuk mode "train", model, optimizer, device,
                # nomor epoch saat ini, recorder untuk logging, dan argumen pelatihan tambahan yang ditentukan dalam argumen konfigurasi.
                # Proses ini akan melatih model pada data training untuk satu epoch, dengan logging yang sesuai untuk memantau kemajuan pelatihan.
                epoch, self.recoder, **self.arg.train_args
            )  # Training satu epoch

            if eval_model:
                # Jika model perlu dievaluasi pada epoch saat ini, jalankan proses evaluasi pada data dev untuk menghitung WER dev saat ini,
                # dan log hasil evaluasi tersebut. Proses ini memungkinkan pemantauan kinerja model pada data dev secara berkala selama pelatihan,
                # sehingga memberikan wawasan tentang bagaimana kinerja model berkembang dari epoch ke epoch.
                dev_error = self.test('dev', epoch)  # Evaluasi pada data dev
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_error))
            
            if save_model:
                # Jika model perlu disimpan pada epoch saat ini, jalankan proses penyimpanan model dengan format custom yang mencakup WER dev saat ini dan nomor epoch,
                # serta manajemen model untuk checkpoint.
                self.custom_save_model(dev_error, epoch, self.arg.work_dir)  # Simpan model

    # Proses evaluasi model
    def test(self, mode, epoch):
        # Jalankan proses evaluasi pada mode tertentu (dev/test) untuk menghitung WER dev saat ini, dan kembalikan nilai WER tersebut. Proses ini memungkinkan evaluasi model pada data dev atau test untuk memantau kinerja model dan memberikan wawasan tentang bagaimana model berkinerja pada data yang tidak terlihat selama pelatihan.
        wer = seq_eval(
            # Jalankan proses evaluasi menggunakan data loader untuk mode yang ditentukan, model, device, nomor epoch saat ini,
            # recorder untuk logging, task yang sedang dievaluasi, dan alat evaluasi yang ditentukan dalam argumen konfigurasi.
            # Proses ini akan menghitung WER untuk mode yang ditentukan (dev atau test) dan mengembalikan nilai WER tersebut.
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
        )  # Hitung WER
        return wer  # Kembalikan nilai WER yang dihitung selama proses evaluasi

    # Fungsi utama untuk menjalankan training atau testing
    def start(self):
        # Jalankan proses training jika phase adalah 'train', atau proses testing jika phase adalah 'test'.
        # Proses ini memungkinkan menjalankan pelatihan model atau evaluasi model berdasarkan konfigurasi yang ditentukan dalam argumen,
        # sehingga memberikan fleksibilitas untuk menggunakan kelas ini untuk kedua tujuan tersebut.
        if self.arg.phase == 'train':
            self.train()  # Training
        elif self.arg.phase == 'test':
            # if self.arg.load_weights is None and self.arg.load_checkpoints is None:
            #     raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            # Log nama model yang digunakan untuk evaluasi, sehingga memudahkan untuk melacak model yang dievaluasi
            # dan membandingkan hasil evaluasi antara model yang berbeda jika diperlukan.
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # Log path ke bobot model yang digunakan untuk evaluasi, sehingga memudahkan untuk melacak bobot model yang dievaluasi
            # dan membandingkan hasil evaluasi antara bobot model yang berbeda jika diperlukan.
            self.test('dev', 6667)
            # Evaluasi pada data dev untuk mendapatkan WER dev saat ini, yang dapat digunakan untuk membandingkan dengan WER dev terbaik
            # yang disimpan selama pelatihan, serta memberikan wawasan tentang bagaimana model berkinerja pada data dev sebelum evaluasi pada data test.
            # Proses ini memungkinkan evaluasi model pada data dev untuk memantau kinerja model dan memberikan wawasan tentang bagaimana model berkinerja
            # pada data yang tidak terlihat selama pelatihan, sebelum melakukan evaluasi akhir pada data test.
            self.test('test', 6667)
            # Evaluasi pada data test untuk mendapatkan WER test saat ini, yang merupakan metrik utama untuk menilai kinerja model pada data
            # yang benar-benar tidak terlihat selama pelatihan, dan memberikan wawasan tentang bagaimana model berkinerja dalam situasi dunia nyata.
            # Proses ini memungkinkan evaluasi model pada data test untuk memberikan gambaran akhir tentang kinerja model setelah pelatihan selesai,
            # dan hasil evaluasi ini dapat digunakan untuk membandingkan dengan model lain atau untuk melaporkan hasil dalam publikasi atau presentasi.
            self.recoder.print_log('Evaluation Done.\n')


# Entry point program utama
if __name__ == '__main__':
    sparser = utils.get_parser()  # Ambil argument parser
    p = sparser.parse_args()  # Parse argumen command line
    if p.config is not None:
        # Jika argumen config diberikan, baca file YAML yang berisi konfigurasi default, periksa argumen yang valid,
        # dan set default argumen dari file YAML. Proses ini memungkinkan penggunaan file konfigurasi untuk menyimpan pengaturan default
        # yang dapat dengan mudah digunakan kembali di masa depan, serta memastikan bahwa hanya argumen yang valid yang digunakan dalam program.
        with open(p.config, 'r') as f:
            # Buka file YAML yang berisi konfigurasi default untuk program, dengan nama file yang diambil dari argumen config yang diberikan.
            # File YAML ini biasanya berisi pengaturan default untuk berbagai parameter yang digunakan dalam program, seperti parameter model,
            # parameter pelatihan, dan parameter evaluasi, sehingga memudahkan penggunaan kembali konfigurasi yang sama di masa depan tanpa perlu
            # menyebutkan semua argumen secara manual setiap kali menjalankan program.
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)  # Load config yaml
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        # Ambil semua kunci argumen yang sudah diparse dari command line, yang akan digunakan untuk memeriksa apakah semua argumen yang diberikan
        # dalam file YAML valid dan sesuai dengan argumen yang diharapkan oleh program. Proses ini memastikan bahwa hanya argumen yang valid yang digunakan
        # dalam program, dan jika ada argumen yang tidak dikenali, program akan memberikan peringatan atau error untuk membantu memperbaiki konfigurasi.
        for k in default_arg.keys():
            # Iterasi melalui semua kunci argumen yang diberikan dalam file YAML untuk memeriksa apakah setiap kunci tersebut valid dan sesuai dengan argumen
            # yang diharapkan oleh program. Proses ini memastikan bahwa hanya argumen yang valid yang digunakan dalam program, dan jika ada argumen yang tidak dikenali,
            # program akan memberikan peringatan atau error untuk membantu memperbaiki konfigurasi.
            if k not in key:
                # Jika ada kunci argumen dalam file YAML yang tidak dikenali (tidak ada dalam argumen yang sudah diparse dari command line),
                # cetak peringatan untuk menunjukkan bahwa ada argumen yang salah atau tidak valid dalam file YAML, sehingga membantu memperbaiki konfigurasi
                # dengan memastikan bahwa hanya argumen yang valid yang digunakan dalam program.
                print('WRONG ARG: {}'.format(k))  # Cek argumen yang salah
                assert k in key  # Pastikan semua argumen dalam file YAML valid
        sparser.set_defaults(**default_arg)
        # Set default argumen dari config yaml, sehingga argumen yang diberikan dalam file YAML akan digunakan sebagai default untuk program,
        # dan jika ada argumen yang juga diberikan melalui command line, argumen tersebut akan menggantikan default dari file YAML.
        # Proses ini memungkinkan penggunaan file konfigurasi untuk menyimpan pengaturan default yang dapat dengan mudah digunakan kembali di masa depan,
        # serta memberikan fleksibilitas untuk menimpa default tersebut dengan argumen yang diberikan melalui command line jika diperlukan.
    args = sparser.parse_args()  # Parse ulang argumen

    main_processor = SLRProcessor(args)
    # Inisialisasi processor utama dengan argumen yang sudah diparse, yang akan mempersiapkan semua komponen yang diperlukan untuk pelatihan dan evaluasi model,
    # termasuk memuat data, membangun model, dan menyiapkan logging. Proses ini memungkinkan program untuk siap menjalankan proses training atau testing
    # berdasarkan konfigurasi yang diberikan dalam argumen.
    main_processor.start()  # Jalankan proses (train/test)