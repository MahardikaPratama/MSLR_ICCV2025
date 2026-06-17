
"""
Training and evaluation sequence functions for CSLR pipeline.

This file contains the main helpers to run a single training epoch,
perform evaluation on the validation/test split, and write prediction
results to a CTM file used by an external evaluator.
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
    """
    Run training for one full epoch.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader for training data.
    model : torch.nn.Module
        Model to be trained.
    optimizer : utils.Optimizer
        Optimizer wrapper including the scheduler.
    device : utils.GpuDataParallel
        Utility to move data to the active device.
    epoch_idx : int
        Current epoch index.
    recoder : utils.Recorder
        Logger object to record training progress.

    Returns
    -------
    list
        List of loss values for all valid batches.
    """
    model.train()  # Set model ke mode training
    loss_value = []  # List untuk menyimpan nilai loss tiap batch
    total_samples = 0  # Jumlah sample yang benar-benar diproses model pada epoch ini
    total_batches = 0  # Jumlah batch valid yang diproses
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]  # Ambil learning rate saat ini

    # Iterasi setiap batch data
    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)  # Pindahkan data ke device (CPU/GPU)
        total_batches += 1
        total_samples += len(data['origin_info'])
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
    recoder.print_log(
        f'\tEpoch {epoch_idx} processed {total_samples} samples in {total_batches} batches.'
    )
    return loss_value  # Kembalikan list loss


def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"
):
    """
    Run model evaluation on a specific split.

    Parameters
    ----------
    cfg : argparse.Namespace
        Main configuration object containing dataset info.
    loader : torch.utils.data.DataLoader
        DataLoader for the evaluated split.
    model : torch.nn.Module
        Model to be evaluated.
    device : utils.GpuDataParallel
        Utility to move data to the active device.
    mode : str
        Split name, e.g., train, dev, or test.
    epoch : int
        Epoch indicator for result logging.
    work_dir : str
        Working directory to save evaluation output.
    recoder : utils.Recorder
        Logger object.
    task : str
        Task name or dataset suffix.
    evaluate_tool : str
        Evaluator name, either python or external.

    Returns
    -------
    float
        The best WER value from the two evaluated prediction paths.
    """
    model.eval()  # Set model ke mode evaluasi
    total_info = []  # List untuk menyimpan info file
    total_sent_fusion = []  # Hasil decoding dari jalur BiLSTM/kontekstual
    total_sent_conv_fusion = []  # Hasil decoding dari jalur Conv1D temporal
    
    total_inference_time_wo_decoding = 0.0
    total_inference_time_w_decoding = 0.0
    total_sequences = 0
    
    # Iterasi setiap batch data
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")  # Catat waktu pemindahan ke device
        data = device.dict_data_to_device(data)  # Pindahkan data ke device
        
        # Hitung jumlah sequence dalam batch untuk kecepatan
        batch_sequences = len(data['origin_info'])
        
        data['skip_decoding'] = True # Do not decode during forward pass
        
        with torch.no_grad():
            # W/O Decoding Forward Pass
            start_time_wo = time.time()
            ret_dict = model(data)  # Forward pass tanpa gradien dan tanpa decoding
            end_time_wo = time.time()
            
            # W/ Decoding = W/O Decoding time + Decoding Time
            real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            # Explicit Decoding
            start_time_decoding = time.time()
            conv_sents_fusion = real_model.decoder.decode(
                ret_dict['conv_logits_fusion'] * real_model.norm_scale, ret_dict['feat_len'], batch_first=False, probs=False
            )
            recognized_sents_fusion = real_model.decoder.decode(
                ret_dict['seq_logits_fusion'] * real_model.norm_scale, ret_dict['feat_len'], batch_first=False, probs=False
            )
            end_time_decoding = time.time()

        # Update total waktu inferensi dan jumlah sequence 
        total_inference_time_wo_decoding += (end_time_wo - start_time_wo)
        total_inference_time_w_decoding += (end_time_wo - start_time_wo) + (end_time_decoding - start_time_decoding)
        total_sequences += batch_sequences

        # Simpan info file dan hasil prediksi dari kedua jalur evaluasi
        total_info += [file_name.split("|")[0] for file_name in data['origin_info']]
        total_sent_fusion += recognized_sents_fusion
        total_sent_conv_fusion += conv_sents_fusion

    # Hitung kecepatan inferensi
    sps_wo = total_sequences / total_inference_time_wo_decoding if total_inference_time_wo_decoding > 0 else 0
    sps_w = total_sequences / total_inference_time_w_decoding if total_inference_time_w_decoding > 0 else 0
    
    # Log waktu inferensi dan kecepatan
    recoder.print_log(f"Inference Speed w/o Decoding : {sps_wo:.2f} seq/s")
    recoder.print_log(f"Inference Speed w/ Decoding  : {sps_w:.2f} seq/s")

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

    # Tulis hasil prediksi ke file CTM untuk kedua jalur evaluasi
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
        # Evaluasi hasil jalur BiLSTM/kontekstual
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
        # Evaluasi hasil jalur Conv1D temporal
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
        f"[{mode.upper()}] Conv1D temporal WER: {conv_ret_fusion: 2.2f}%, BiLSTM contextual WER: {lstm_ret_fusion: 2.2f}%",
        os.path.join(results_dir, f"{mode}_wer.txt")
    )
    return min([conv_ret_fusion, lstm_ret_fusion])


def write2file(path, info, output):
    """
    Write prediction results to a CTM file.

    Parameters
    ----------
    path : str
        Output path for the CTM file.
    info : list
        List of sample IDs or file names.
    output : list
        List of prediction results per sample.
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
