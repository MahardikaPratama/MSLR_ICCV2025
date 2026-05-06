import pickle
import os

def downsample_data(input_file, output_file, ratio=0.5):
    if not os.path.exists(input_file):
        print(f"File {input_file} tidak ditemukan.")
        return

    print(f"Membaca {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Melakukan downsampling (ratio={ratio}) pada {len(data)} sequence...")
    downsampled_data = {}
    
    # Menghitung stride berdasarkan ratio
    # Ratio 0.5 berarti mengambil 1 dari setiap 2 frame (stride = 2)
    stride = int(1 / ratio) if ratio > 0 else 1
    
    for vid_id, vid_info in data.items():
        new_vid_info = {}
        for key, value in vid_info.items():
            # Biasanya data temporal (frame) ada di key 'keypoints' atau 'keypoint_score'
            # Kita potong berdasarkan dimensi pertama (waktu/frame)
            if isinstance(value, list):
                new_vid_info[key] = value[::stride]
            elif hasattr(value, 'shape'): # Untuk numpy array
                new_vid_info[key] = value[::stride]
            else:
                new_vid_info[key] = value # Biarkan metadata lain apa adanya
                
        downsampled_data[vid_id] = new_vid_info
        
    print(f"Menyimpan hasil ke {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(downsampled_data, f)
    print("Selesai!\n")

if __name__ == "__main__":
    # Sesuaikan path direktori datasets Anda jika diperlukan
    base_dir = "datasets" # atau "c:/TA/Source-Code/MSLR_ICCV2025/datasets"
    
    file_train_dev = os.path.join(base_dir, "pose_bisindo_train_dev.pkl")
    file_train_dev_out = os.path.join(base_dir, "pose_bisindo_train_dev_v2.pkl")
    
    file_test = os.path.join(base_dir, "pose_bisindo_test.pkl")
    file_test_out = os.path.join(base_dir, "pose_bisindo_test_v2.pkl")
    
    # Jika path base_dir tidak ada tapi file ada di current directory
    if not os.path.exists(file_train_dev) and os.path.exists("pose_bisindo_train_dev.pkl"):
        file_train_dev = "pose_bisindo_train_dev.pkl"
        file_train_dev_out = "pose_bisindo_train_dev_v2.pkl"
        file_test = "pose_bisindo_test.pkl"
        file_test_out = "pose_bisindo_test_v2.pkl"
        
    downsample_data(file_train_dev, file_train_dev_out, ratio=0.5)
    downsample_data(file_test, file_test_out, ratio=0.5)
