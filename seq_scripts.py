
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


# Fungsi untuk melatih model satu epoch
def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
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


# Fungsi untuk evaluasi model pada data validasi/test
def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"
):
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
        batch_sequences = len(data['origin_info'])
        
        with torch.no_grad():
            start_time = time.time()
            ret_dict = model(data)  # Forward pass tanpa gradien
            end_time = time.time()
            
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
    
    recoder.print_log(f"[{mode.upper()} EVAL] Total Inference Time: {total_inference_time:.2f}s")
    recoder.print_log(f"[{mode.upper()} EVAL] Inference Speed: {fps:.2f} Frames/s, {sps:.2f} Sequences/s")

    # Pilih mode evaluasi (python atau eksternal)
    python_eval = True if evaluate_tool == "python" else False

    # Buat subfolder khusus untuk hasil test jika mode test
    test_results_dir = work_dir
    if mode == 'test':
        test_results_dir = os.path.join(work_dir, 'test_results' + os.sep)
        if not os.path.exists(test_results_dir):
            os.makedirs(test_results_dir)
    
    # Tulis hasil prediksi ke file CTM
    write2file(
        test_results_dir + "output-hypothesis-fusion-{}.ctm".format(mode), total_info, total_sent_fusion
    )
    write2file(
        test_results_dir + "output-hypothesis-conv-fusion-{}.ctm".format(mode), total_info, total_sent_conv_fusion
    )

    # Jika mode test, hasil akhir ditulis ke CSV di subfolder
    if mode == 'test':
        csv_file = f'{test_results_dir}test.csv'
        if task == 'us':
            ctm_file = f'{test_results_dir}output-hypothesis-conv-fusion-test.ctm'
        else:
            ctm_file = f'{test_results_dir}output-hypothesis-fusion-test.ctm'
        # Baca file CTM hasil prediksi
        with open(ctm_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data = {}
        # Proses setiap baris CTM menjadi dictionary id -> list kata
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:  
                id = parts[0]  
                word = parts[4]  
                if id not in data:
                    data[id] = []
                data[id].append(word)

        data = dict(sorted(data.items(), key=lambda item: item[0]))  # Urutkan berdasarkan id (string)

        # Tulis hasil ke file CSV
        with open(csv_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "gloss"])
            for id, words in data.items():
                gloss = " ".join(words)
                writer.writerow([id, gloss])
        return csv_file
    else:
        try:
            # Evaluasi hasil BiLSTM
            lstm_ret_fusion = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-fusion-{}.ctm".format(mode),
                evaluate_dir=cfg.dataset_info['evaluation_dir'],
                evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                triplet=True,
            )
            # Evaluasi hasil Conv1D
            conv_ret_fusion = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-conv-fusion-{}.ctm".format(mode),
                evaluate_dir=cfg.dataset_info['evaluation_dir'],
                evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                # triplet=True,
            )
        except:
            print("Unexpected error:", sys.exc_info()[0])
            lstm_ret = 100.0
        finally:
            pass
        # Log hasil evaluasi WER
        recoder.print_log(
            f"Epoch {epoch}, {mode} Conv1D WER: {conv_ret_fusion: 2.2f}%, BiLSTM WER: {lstm_ret_fusion: 2.2f}%", f"{work_dir}/{mode}.txt"
        )
        return min([conv_ret_fusion, lstm_ret_fusion])       


# Fungsi untuk menulis hasil prediksi ke file CTM
def write2file(path, info, output):
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
