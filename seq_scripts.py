
"""Fungsi training dan evaluasi sequence untuk pipeline CSLR.

File ini berisi helper utama untuk menjalankan satu epoch training,
melakukan evaluasi pada split validasi/test, serta menulis hasil prediksi
ke file CTM yang dipakai oleh evaluator eksternal.
"""

# Import berbagai library yang dibutuhkan untuk training, evaluasi, dan utilitas
import os
import csv
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
from evaluation.slr_eval.wer_calculation import evaluate


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    """Menjalankan training untuk satu epoch penuh.

    Input:
    1. loader: DataLoader untuk data training.
    2. model: model yang akan dilatih.
    3. optimizer: wrapper optimizer beserta scheduler.
    4. device: utilitas untuk memindahkan data ke device aktif.
    5. epoch_idx: indeks epoch saat ini.
    6. recoder: objek logger untuk mencatat progres training.

    Proses:
    1. Mengubah model ke mode training.
    2. Mengambil batch satu per satu dari loader.
    3. Memindahkan batch ke device.
    4. Melakukan forward pass dan menghitung loss.
    5. Mengabaikan batch yang menghasilkan NaN atau inf.
    6. Melakukan backward pass, clipping gradien, dan update optimizer.
    7. Mencatat loss per batch dan rata-rata loss epoch.

    Output:
    1. List nilai loss untuk semua batch yang valid.
    """
    model.train()  # Set model ke mode training
    loss_value = []  # List untuk menyimpan nilai loss tiap batch
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]  # Ambil learning rate saat ini

    # Iterasi setiap batch data
    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)  # Pindahkan data ke device (CPU/GPU)
        ret_dict = model(data)  # Forward pass, dapatkan output model

        loss, loss_details = model.get_loss(ret_dict, data)  # Hitung loss dan detail loss
        # Skip batch jika loss tidak valid
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data['origin_info'])
            continue
        optimizer.zero_grad()  # Reset gradien
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Clip gradients to prevent exploding gradients
        optimizer.step()  # Update parameter model

        loss_value.append(loss.item())  # Simpan nilai loss
        # Logging setiap beberapa batch
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.2f}  lr:{clr[0]:.6f}'
            )
            recoder.print_log(
                "\t"
                + ", ".join([f"{k}: {v.item():.2f}" for k, v in loss_details.items()])
            )
    optimizer.scheduler.step()  # Update learning rate scheduler
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))  # Log rata-rata loss
    return loss_value  # Kembalikan list loss


def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"
):
    """Menjalankan evaluasi model pada split tertentu.

    Input:
    1. cfg: objek konfigurasi utama yang memuat info dataset.
    2. loader: DataLoader untuk split yang dievaluasi.
    3. model: model yang akan dievaluasi.
    4. device: utilitas pemindahan data ke device aktif.
    5. mode: nama split, misalnya train, dev, atau test.
    6. epoch: penanda epoch untuk logging hasil.
    7. work_dir: folder kerja untuk menyimpan output evaluasi.
    8. recoder: objek logger.
    9. task: nama task atau suffix dataset.
    10. evaluate_tool: nama evaluator, python atau eksternal.

    Proses:
    1. Mengubah model ke mode evaluasi.
    2. Mengiterasi seluruh batch tanpa gradien.
    3. Mencatat waktu inferensi dan jumlah frame/sequence.
    4. Mengumpulkan prediksi hasil decoding.
    5. Menulis file CTM dan CSV hasil prediksi.
    6. Menjalankan evaluator untuk menghitung WER.

    Output:
    1. Nilai WER terbaik dari dua jalur prediksi yang dievaluasi.
    """
    model.eval()  # Set model ke mode evaluasi
    total_info = []  # List untuk menyimpan info file
    total_sent_fusion = []  # List hasil prediksi BiLSTM
    total_sent_conv_fusion = []  # List hasil prediksi Conv1D
    
    total_inference_time = 0.0
    total_frames = 0
    total_sequences = 0
    
    # Iterasi setiap batch data
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")  # Catat waktu pemindahan ke device
        data = device.dict_data_to_device(data)  # Pindahkan data ke device
        
        # Hitung ukuran batch untuk kecepatan
        if torch.is_tensor(data['len_x']):
            batch_frames = data['len_x'].sum().item()
        else:
            batch_frames = sum(data['len_x'])
        # Hitung jumlah sequence dalam batch untuk kecepatan
        batch_sequences = len(data['origin_info'])
        
        with torch.no_grad():
            start_time = time.time()
            ret_dict = model(data)  # Forward pass tanpa gradien
            end_time = time.time()

        # Update total waktu inferensi dan jumlah frame/sequence 
        total_inference_time += (end_time - start_time)
        total_frames += batch_frames
        total_sequences += batch_sequences

        # Simpan info file dan hasil prediksi
        total_info += [file_name.split("|")[0] for file_name in data['origin_info']]
        total_sent_fusion += ret_dict['recognized_sents_fusion']
        total_sent_conv_fusion += ret_dict['conv_sents_fusion']

    # Hitung kecepatan inferensi
    fps = total_frames / total_inference_time if total_inference_time > 0 else 0
    sps = total_sequences / total_inference_time if total_inference_time > 0 else 0
    
    # Log waktu inferensi dan kecepatan
    recoder.print_log(f"[{mode.upper()} EVAL] Total Inference Time: {total_inference_time:.2f}s")
    recoder.print_log(f"[{mode.upper()} EVAL] Inference Speed: {fps:.2f} Frames/s, {sps:.2f} Sequences/s")

    # Pilih mode evaluasi (python atau eksternal)
    python_eval = True if evaluate_tool == "python" else False


    # Penentuan direktori hasil sesuai mode dan task
    if mode.startswith('test'):
        results_dir = os.path.join(work_dir, 'test', mode)
    else:
        results_dir = os.path.join(work_dir, 'train', mode)

    # Buat direktori hasil jika belum ada
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #  Tulis hasil prediksi ke file CTM untuk kedua jalur (BiLSTM dan Conv1D)
    write2file(
        os.path.join(results_dir, "output-hypothesis-fusion-{}.ctm".format(mode)), total_info, total_sent_fusion
    )
    write2file(
        os.path.join(results_dir, "output-hypothesis-conv-fusion-{}.ctm".format(mode)), total_info, total_sent_conv_fusion
    )
    
    # Jika mode test, buat file CSV dari CTM untuk keperluan submission atau analisis lebih lanjut
    if mode.startswith('test'):
        #  Buat file CSV dengan format id dan gloss dari file CTM hasil BiLSTM fusion
        csv_file = os.path.join(results_dir, f'{mode}.csv')
        # Baca file CTM, ekstrak id dan kata, lalu simpan dalam format CSV dengan kolom id dan gloss
        ctm_file = os.path.join(results_dir, "output-hypothesis-fusion-{}.ctm".format(mode))
        # Buka file CTM, baca setiap baris, ekstrak id dan kata, lalu simpan dalam dictionary berdasarkan id. Setelah itu, tulis ke file CSV dengan kolom id dan gloss (gabungan kata-kata).
        with open(ctm_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        # Initialisasi dictionary untuk menyimpan id dan kata-kata yang terkait
        data = {}
        # Iterasi setiap baris dalam file CTM, ekstrak id (kolom pertama) dan kata (kolom kelima), lalu simpan dalam dictionary berdasarkan id. Jika id sudah ada, tambahkan kata ke list yang terkait dengan id tersebut.
        for line_idx, line in enumerate(lines):
            # Setiap baris diharapkan memiliki format: id 1 start_time end_time word. Kita ekstrak id dan word, lalu simpan dalam dictionary. Jika id sudah ada, kita tambahkan kata ke list yang terkait dengan id tersebut.
            parts = line.strip().split()
            #  Jika format baris valid (minimal 5 bagian), ekstrak id dan kata, lalu simpan dalam dictionary. Jika id sudah ada, tambahkan kata ke list yang terkait dengan id tersebut.
            if len(parts) >= 5:
                # Ekstrak id (kolom pertama) dan kata (kolom kelima), lalu simpan dalam dictionary. Jika id sudah ada, tambahkan kata ke list yang terkait dengan id tersebut.
                id = parts[0]
                word = parts[4]
                # Jika id belum ada dalam dictionary, buat entry baru dengan list kosong. Kemudian tambahkan kata ke list yang terkait dengan id tersebut.
                if id not in data:
                    data[id] = []
                # Tambahkan kata ke list yang terkait dengan id tersebut. Jika id sudah ada, kita tambahkan kata ke list yang terkait dengan id tersebut.
                data[id].append(word)
        # Setelah membaca semua baris, kita memiliki dictionary yang berisi id dan list kata-kata terkait. Selanjutnya, kita tulis ke file CSV dengan kolom id dan gloss (gabungan kata-kata). Kita urutkan dictionary berdasarkan id agar hasil CSV terurut.
        data = dict(sorted(data.items(), key=lambda item: item[0]))
        # Tulis ke file CSV dengan kolom id dan gloss (gabungan kata-kata). Kita gabungkan list kata-kata menjadi satu string untuk kolom gloss. Setiap baris CSV akan berisi id dan gloss yang terkait.
        with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "gloss"])
            for id, words in data.items():
                gloss = " ".join(words)
                writer.writerow([id, gloss])

    try:
        # Evaluasi hasil BiLSTM
        lstm_ret_fusion = evaluate(
            prefix=results_dir + "/",
            mode=mode,
            output_file="output-hypothesis-fusion-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=None,
            python_evaluate=python_eval,
            triplet=True,
        )
        # Evaluasi hasil Conv1D
        conv_ret_fusion = evaluate(
            prefix=results_dir + "/",
            mode=mode,
            output_file="output-hypothesis-conv-fusion-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=None,
            python_evaluate=python_eval,
        )
    except Exception as e:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret_fusion = 100.0
        conv_ret_fusion = 100.0
        
    recoder.print_log(
        f"[{mode.upper()}] Conv1D WER: {conv_ret_fusion: 2.2f}%, BiLSTM WER: {lstm_ret_fusion: 2.2f}%", os.path.join(results_dir, f"{mode}_wer.txt")
    )
    return min([conv_ret_fusion, lstm_ret_fusion])


def write2file(path, info, output):
    """Menulis hasil prediksi ke file CTM.

    Input:
    1. path: path file output CTM.
    2. info: daftar id sample atau nama file.
    3. output: daftar hasil prediksi per sample.

    Proses:
    1. Membuka file output untuk ditulis.
    2. Menulis setiap kata prediksi sebagai satu baris CTM.
    3. Menggunakan waktu dummy karena format CTM membutuhkan start/end time.

    Output:
    1. File CTM berisi hasil prediksi yang siap dipakai evaluator.
    """
    filereader = open(path, "w")  # Buka file untuk ditulis
    # Iterasi setiap sample (per video/sequence)
    for sample_idx, sample in enumerate(output):
        # Iterasi setiap kata hasil prediksi
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(
                    info[sample_idx],  # ID sample
                    word_idx * 1.0 / 100,  # Start time (dummy)
                    (word_idx + 1) * 1.0 / 100,  # End time (dummy)
                    word[0],  # Kata hasil prediksi
                )
            )
