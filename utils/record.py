# import pdb
import time
from tqdm import tqdm


class Recorder(object):
    def __init__(self, work_dir, print_log, log_interval):
        """Menyiapkan utilitas untuk logging dan pencatatan waktu proses.

        Input:
        1. work_dir: folder tujuan untuk file log.
        2. print_log: penanda apakah log juga disimpan ke file.
        3. log_interval: interval pencatatan log waktu, dipakai oleh pemanggil eksternal.

        Proses:
        1. Menyimpan waktu awal sebagai referensi pengukuran durasi.
        2. Menyimpan flag logging dan interval logging.
        3. Menentukan path file log.
        4. Menyiapkan dictionary timer untuk beberapa tahap proses.
        """
        self.cur_time = time.time()
        self.print_log_flag = print_log
        self.log_interval = log_interval
        self.log_path = '{}/log.txt'.format(work_dir)
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def print_time(self):
        """Mencetak waktu lokal saat ini ke log.

        Proses:
        1. Mengambil waktu lokal sistem.
        2. Menulisnya melalui print_log().
        """
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, path=None, print_time=True):
        """Menulis pesan log ke terminal dan, bila aktif, ke file log.

        Input:
        1. str: pesan yang akan ditulis.
        2. path: path file tujuan. Jika None, memakai log_path bawaan.
        3. print_time: jika True, menambahkan timestamp di awal pesan.

        Proses:
        1. Menentukan path log tujuan.
        2. Menambahkan timestamp jika diminta.
        3. Mencetak pesan menggunakan tqdm.write() agar aman di progress bar.
        4. Menyimpan pesan ke file jika print_log_flag aktif.
        """
        if path is None:
            path = self.log_path
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        tqdm.write(str)
        if self.print_log_flag:
            with open(path, 'a') as f:
                f.writelines(str)
                f.writelines("\n")

    def record_time(self):
        """Menyimpan timestamp saat ini sebagai acuan pengukuran durasi.

        Output:
        1. Nilai waktu saat ini dalam detik.
        """
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """Menghitung selisih waktu sejak record_time() terakhir dipanggil.

        Proses:
        1. Menghitung waktu yang telah berlalu dari cur_time.
        2. Memperbarui cur_time ke waktu sekarang.

        Output:
        1. Durasi sejak pengukuran terakhir.
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def timer_reset(self):
        """Mereset seluruh pencacah waktu ke nilai awal.

        Proses:
        1. Mengatur ulang cur_time ke waktu sekarang.
        2. Mengembalikan semua komponen timer ke nilai awal kecil.
        """
        self.cur_time = time.time()
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def record_timer(self, key):
        """Menambahkan durasi interval terakhir ke kategori timer tertentu.

        Input:
        1. key: nama kategori timer, misalnya dataloader, device, forward, atau backward.

        Proses:
        1. Menghitung durasi sejak record terakhir.
        2. Menambahkan durasi itu ke timer[key].
        """
        self.timer[key] += self.split_time()

    def print_time_statistics(self):
        """Mencetak persentase penggunaan waktu untuk setiap tahap proses.

        Proses:
        1. Menghitung proporsi waktu tiap kategori terhadap total waktu.
        2. Menulis ringkasan distribusi waktu ke log.
        """
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(self.timer.values()))))
            for k, v in self.timer.items()}
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [GPU]{device}, [Forward]{forward}, [Backward]{backward}'.format(
                **proportion))
