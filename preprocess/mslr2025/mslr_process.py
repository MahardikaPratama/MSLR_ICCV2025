import os
import json
from tqdm import tqdm


# =========================================================
# PROJECT PATH
# =========================================================

def get_project_root():
    """
    Deskripsi:
        Mengembalikan path absolut dari root direktori proyek,
        yaitu dua level di atas lokasi file ini.

    Input:
        Tidak ada input parameter.

    Proses:
        1. Mengambil path absolut file saat ini menggunakan __file__
        2. Mengambil direktori parent pertama (satu level ke atas) dengan os.path.dirname()
        3. Mengambil direktori parent kedua (dua level ke atas) dengan os.path.join(..., "../../")
        4. Menormalkan path ke bentuk absolut dengan os.path.abspath()

    Output:
        str: Path absolut root direktori proyek.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )


PROJECT_ROOT = get_project_root()

PREPROCESS_ROOT = os.path.dirname(__file__)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "mslr2025"
)

os.makedirs(DATASET_ROOT, exist_ok=True)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

DATASET_SPLITS = {
    "train": {
        "folder": "SD",
        "file": "train_list.txt",
    },

    "dev": {
        "folder": "SD",
        "file": "dev_list.txt",
    },

    "test_sd": {
        "folder": "SD",
        "file": "test_list.txt",
    },

    "test_si_major": {
        "folder": "SI-MAJ",
        "file": "test_list.txt",
    },

    "test_si_minor": {
        "folder": "SI-MIN",
        "file": "test_list.txt",
    },
}


# =========================================================
# CORE FUNCTIONS
# =========================================================

def sign_dict_update(total_dict, info):
    """
    Deskripsi:
        Memperbarui dictionary akumulasi gloss dengan menghitung frekuensi
        kemunculan setiap gloss dari daftar info yang diberikan.

    Input:
        total_dict (dict): Dictionary akumulasi gloss yang sudah ada,
                           dengan key berupa string gloss dan value berupa int frekuensi.
        info (list[dict]): Daftar dictionary info sample, masing-masing memiliki
                           key 'gloss_sequence' berupa string urutan gloss.

    Proses:
        1. Iterasi setiap item dalam list info
        2. Memecah nilai 'gloss_sequence' pada item menjadi list token gloss dengan .split()
        3. Iterasi setiap gloss dalam list token hasil split
        4. Menambahkan 1 ke nilai gloss pada total_dict (default 0 jika belum ada) dengan dict.get()

    Output:
        dict: total_dict yang telah diperbarui dengan frekuensi gloss terbaru.
    """

    for item in info:

        split_label = item['gloss_sequence'].split()

        for gloss in split_label:

            total_dict[gloss] = (
                total_dict.get(gloss, 0) + 1
            )

    return total_dict


def generate_gt_stm(info, save_path):
    """
    Deskripsi:
        Menghasilkan file ground truth berformat STM (Segment Time Mark)
        dari daftar info sample, untuk keperluan evaluasi WER.

    Input:
        info (list[dict]): Daftar dictionary info sample, masing-masing memiliki
                           key 'video_id', 'signer', dan 'gloss_sequence'.
        save_path (str): Path lengkap file STM yang akan disimpan.

    Proses:
        1. Membuka file pada save_path dengan mode tulis ('w') dan encoding UTF-8
        2. Iterasi setiap item dalam list info
        3. Menulis satu baris per item ke file dengan format:
           "<video_id> 1 <signer> 0.0 1.79769e+308 <gloss_sequence>"
           - Kolom 1: video_id (identifier unik video)
           - Kolom 2: channel number (tetap 1)
           - Kolom 3: signer (id penanda)
           - Kolom 4: waktu mulai segment (tetap 0.0)
           - Kolom 5: waktu akhir segment (tetap 1.79769e+308, mewakili tak hingga)
           - Kolom 6: gloss_sequence (label urutan gloss)

    Output:
        None. File STM tersimpan pada save_path.
    """

    with open(save_path, "w", encoding="utf-8") as f:

        for item in info:

            f.write(
                f"{item['video_id']} "
                f"1 "
                f"{item['signer']} "
                f"0.0 "
                f"1.79769e+308 "
                f"{item['gloss_sequence']}\n"
            )


def info2dict(anno_path):
    """
    Deskripsi:
        Membaca file anotasi teks berformat pipe-separated (|) dan
        mengubahnya menjadi list of dictionary berisi metadata tiap sample.

    Input:
        anno_path (str): Path lengkap ke file anotasi (.txt) yang akan dibaca.

    Proses:
        1. Memeriksa apakah file pada anno_path ada; jika tidak, raise FileNotFoundError
        2. Membuka file dengan mode baca ('r') dan encoding UTF-8
        3. Membaca seluruh baris file ke dalam list lines dengan readlines()
        4. Memeriksa apakah baris pertama adalah header (mengandung kata 'video' atau 'gloss'); jika ya, melewatinya dengan lines = lines[1:]
        5. Menginisialisasi list kosong info_list untuk menampung hasil parsing
        6. Iterasi setiap baris dalam lines
        7. Memecah baris pada karakter '|' menjadi list parts dengan .strip().split('|')
        8. Melewati baris jika jumlah parts kurang dari 2 (data tidak valid) dengan continue
        9. Mengambil video_id dari parts[0]
        10. Mengambil gloss_seq dari parts[1]
        11. Memecah video_id pada karakter '_' menjadi list split_vid
        12. Mengambil signer dari elemen pertama split_vid (split_vid[0])
        13. Mengambil sentence_id dari elemen kedua split_vid (split_vid[1])
        14. Membuat dictionary item berisi signer, video_id, gloss_sequence, sentence_id, dan original_info
        15. Menambahkan dictionary item ke info_list

    Output:
        list[dict]: Daftar dictionary metadata sample, masing-masing berisi:
                    - 'signer' (str): ID penanda/signer
                    - 'video_id' (str): ID unik video
                    - 'gloss_sequence' (str): Urutan label gloss
                    - 'sentence_id' (str): ID kalimat (bagian video_id setelah signer)
                    - 'original_info' (str): Baris asli dari file anotasi
    """

    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found:\n{anno_path}"
        )

    with open(anno_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header if exists
    if (
        len(lines) > 0 and
        (
            "video" in lines[0].lower() or
            "gloss" in lines[0].lower()
        )
    ):
        lines = lines[1:]

    info_list = []

    for line in tqdm(lines):

        parts = line.strip().split('|')

        if len(parts) < 2:
            continue

        video_id = parts[0]
        gloss_seq = parts[1]

        split_vid = video_id.split('_')

        signer = split_vid[0]

        sentence_id = split_vid[1]

        info_list.append({

            "signer": signer,

            "video_id": video_id,

            "gloss_sequence": gloss_seq.strip(),

            "sentence_id": sentence_id,

            "original_info": line,

        })

    return info_list


def save_json(obj, save_path):
    """
    Deskripsi:
        Menyimpan objek Python (dict atau list) ke file JSON
        dengan format indentasi yang mudah dibaca.

    Input:
        obj (dict | list): Objek Python yang akan disimpan sebagai JSON.
        save_path (str): Path lengkap file JSON tujuan penyimpanan.

    Proses:
        1. Membuka file pada save_path dengan mode tulis ('w') dan encoding UTF-8
        2. Menulis obj ke file menggunakan json.dump() dengan:
           - indent=4 agar output terformat rapi dengan indentasi 4 spasi
           - ensure_ascii=False agar karakter non-ASCII (misal: aksara lokal) tidak di-escape

    Output:
        None. File JSON tersimpan pada save_path.
    """

    with open(save_path, "w", encoding="utf-8") as f:

        json.dump(
            obj,
            f,
            indent=4,
            ensure_ascii=False
        )


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    """
    Deskripsi:
        Entry point utama preprocessing MSLR2025. Memproses semua split dataset,
        menghasilkan file metadata JSON, file ground truth STM, dan
        kamus gloss global dari split train dan dev.

    Input:
        Tidak ada input langsung. Konfigurasi dibaca dari DATASET_SPLITS
        dan path file anotasi di direktori PREPROCESS_ROOT.

    Proses:
        1.  Menginisialisasi global_sign_dict sebagai dict kosong untuk akumulasi vocabulary
        2.  Iterasi setiap pasangan (split_name, cfg) dalam DATASET_SPLITS
        3.  Membangun anno_path dari PREPROCESS_ROOT, folder, dan file_name menggunakan os.path.join()
        4.  Memanggil info2dict(anno_path) untuk memuat dan mem-parsing data split ke split_info
        5.  Membangun json_save_path untuk file metadata JSON di DATASET_ROOT
        6.  Memanggil save_json(split_info, json_save_path) untuk menyimpan metadata split
        7.  Membangun stm_save_path untuk file STM di DATASET_ROOT
        8.  Memanggil generate_gt_stm(split_info, stm_save_path) untuk menyimpan file ground truth STM
        9.  Memeriksa apakah split_name termasuk ["train", "dev"]; jika ya, memanggil sign_dict_update() untuk menambah vocabulary global
        10. Mengurutkan global_sign_dict secara alfabetis berdasarkan key gloss menggunakan sorted()
        11. Menginisialisasi save_dict dengan dua sub-dict kosong: "id2gloss" dan "gloss2id"
        12. Iterasi setiap pasangan (gloss, freq) dari global_sign_dict dengan enumerate() untuk membangun mapping dua arah gloss2id dan id2gloss berbasis indeks 1
        13. Memanggil save_json(save_dict, gloss_dict_path) untuk menyimpan kamus gloss global

    Output:
        Tidak ada return value. Menghasilkan file-file berikut di DATASET_ROOT:
        - <split_name>_info.json              : Metadata tiap split (train, dev, test_*)
        - mslr-groundtruth-<split_name>.stm   : File ground truth STM tiap split
        - global_gloss_dict.json              : Kamus gloss global dari split train dan dev,
                                                berisi mapping dua arah gloss2id dan id2gloss
    """

    print("\n===================================")
    print("MSLR2025 PREPROCESSING START")
    print("===================================\n")

    # =====================================================
    # GLOBAL GLOSS DICTIONARY
    # =====================================================

    # 1. Menginisialisasi global_sign_dict sebagai dict kosong untuk akumulasi vocabulary
    global_sign_dict = dict()

    # =====================================================
    # PROCESS EACH SPLIT
    # =====================================================

    # 2.  Iterasi setiap pasangan (split_name, cfg) dalam DATASET_SPLITS
    for split_name, cfg in DATASET_SPLITS.items():

        folder = cfg["folder"]

        file_name = cfg["file"]

        # 3.  Membangun anno_path dari PREPROCESS_ROOT, folder, dan file_name menggunakan os.path.join()
        anno_path = os.path.join(
            PREPROCESS_ROOT,
            folder,
            file_name
        )

        print(f"\nProcessing: {split_name}")

        # =================================================
        # LOAD SPLIT
        # =================================================
        # 4.  Memanggil info2dict(anno_path) untuk memuat dan mem-parsing data split ke split_info
        split_info = info2dict(anno_path)

        # =================================================
        # SAVE METADATA JSON
        # =================================================

        # 5.  Membangun json_save_path untuk file metadata JSON di DATASET_ROOT
        json_save_path = os.path.join(
            DATASET_ROOT,
            f"{split_name}_info.json"
        )

        # 6.  Memanggil save_json(split_info, json_save_path) untuk menyimpan metadata split
        save_json(split_info, json_save_path)

        # =================================================
        # SAVE STM GROUND TRUTH
        # =================================================

        # 7.  Membangun stm_save_path untuk file STM di DATASET_ROOT
        stm_save_path = os.path.join(
            DATASET_ROOT,
            f"mslr-groundtruth-{split_name}.stm"
        )

        # 8.  Memanggil generate_gt_stm() untuk menghasilkan file STM ground truth
        generate_gt_stm(
            split_info,
            stm_save_path
        )

        # =================================================
        # BUILD GLOBAL VOCABULARY
        # ONLY FROM TRAIN + DEV
        # =================================================
        # 9.  Memeriksa apakah split_name termasuk ["train", "dev"]; jika ya, memanggil sign_dict_update() untuk menambah vocabulary global
        if split_name in ["train", "dev"]:

            sign_dict_update(
                global_sign_dict,
                split_info
            )

        print(
            f"{split_name} completed "
            f"({len(split_info)} samples)"
        )

    # =====================================================
    # SORT GLOSS DICTIONARY
    # =====================================================

    # 10. Mengurutkan global_sign_dict secara alfabetis berdasarkan key gloss menggunakan sorted()
    global_sign_dict = sorted(
        global_sign_dict.items(),
        key=lambda d: d[0]
    )

    # =====================================================
    # BUILD GLOSS MAPPING
    # =====================================================
    # 11. Menginisialisasi save_dict dengan dua sub-dict kosong: "id2gloss" dan "gloss2id"
    save_dict = {

        "id2gloss": {},

        "gloss2id": {},

    }

    # 12. Iterasi setiap pasangan (gloss, freq) dari global_sign_dict dengan enumerate() untuk membangun mapping dua arah gloss2id dan id2gloss berbasis indeks 1
    for idx, (gloss, freq) in enumerate(global_sign_dict):

        gloss_index = idx + 1

        # gloss -> id
        save_dict["gloss2id"][gloss] = {

            "index": gloss_index,

            "frequency": freq,

        }

        # id -> gloss
        save_dict["id2gloss"][gloss_index] = {

            "gloss": gloss,

            "frequency": freq,

        }

    # =====================================================
    # SAVE GLOBAL GLOSS DICTIONARY
    # =====================================================

    gloss_dict_path = os.path.join(
        DATASET_ROOT,
        "global_gloss_dict.json"
    )

    # 13. Memanggil save_json(save_dict, gloss_dict_path) untuk menyimpan kamus gloss global
    save_json(save_dict, gloss_dict_path)

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n===================================")
    print("PREPROCESSING FINISHED")
    print("===================================\n")

    print(
        f"Total gloss vocabulary : "
        f"{len(global_sign_dict)}"
    )

    print(
        f"Gloss dictionary saved at:\n"
        f"{gloss_dict_path}"
    )

    print("\nAll preprocessing completed.\n")