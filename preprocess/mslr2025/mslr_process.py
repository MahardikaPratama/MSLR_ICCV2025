# import pdb
import os
import json
import numpy as np
from tqdm import tqdm


# ===============================
# ROBUST PATH HANDLING
# ===============================

def get_project_root():
    """
    Returns the project root directory.
    Works for both local execution and Google Colab.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


PROJECT_ROOT = get_project_root()
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "mslr2025")
PREPROCESS_ROOT = os.path.dirname(__file__)

# Ensure dataset directory exists
os.makedirs(DATASET_ROOT, exist_ok=True)


# ===============================
# CORE FUNCTIONS
# ===============================

def sign_dict_update(total_dict, info):
    """
    Updates gloss frequency dictionary.
    """
    for item in info:
        split_label = item['gloss_sequence'].split()
        for gloss in split_label:
            total_dict[gloss] = total_dict.get(gloss, 0) + 1
    return total_dict


def generate_gt_stm(info, save_path):
    """
    Generates ground truth STM file for evaluation.
    """
    with open(save_path, "w") as f:
        for item in info:
            f.write(
                f"{item['video_id']} 1 {item['signer']} 0.0 1.79769e+308 {item['gloss_sequence']}\n"
            )


def info2dict(anno_filename):
    """
    Converts annotation txt file into structured dictionary format.
    """
    anno_path = os.path.join(PREPROCESS_ROOT, anno_filename)

    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")

    with open(anno_path, 'r') as f:
        inputs_list = f.readlines()[1:]

    info_list = []

    for line in tqdm(inputs_list):
        parts = line.strip().split('|')

        if len(parts) > 2:
            video_id, gloss_seq, _ = parts
        else:
            video_id, gloss_seq = parts

        signer, sentence_id = video_id.split('_')

        info_list.append({
            'signer': signer,
            'video_id': video_id,
            'gloss_sequence': gloss_seq.strip(),
            'sentence_id': sentence_id,
            'original_info': line,
        })

    return info_list


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == '__main__':

    for setting in ['si', 'us']:

        sign_dict = dict()

        for md in ['train', 'dev']:

            split_info = info2dict(f'{setting}_{md}_list.txt')

            # Save JSON metadata
            info_save_path = os.path.join(
                DATASET_ROOT, f"{setting}_{md}_info.json"
            )
            with open(info_save_path, "w") as f:
                json.dump(split_info, f, indent=4)

            # Update gloss frequency
            sign_dict_update(sign_dict, split_info)

            # Save STM ground truth file
            stm_save_path = os.path.join(
                DATASET_ROOT, f"mslr-{setting}-groundtruth-{md}.stm"
            )
            generate_gt_stm(split_info, stm_save_path)

            print(f"{setting}-{md} preprocessing completed.")

        # ===============================
        # SAVE GLOSS DICTIONARY
        # ===============================

        sign_dict_sorted = sorted(sign_dict.items(), key=lambda d: d[0])

        save_dict = {'id2gloss': {}, 'gloss2id': {}}

        for idx, (key, value) in enumerate(sign_dict_sorted):
            save_dict['gloss2id'][key] = {
                'index': idx + 1,
                'frequency': value,
            }
            save_dict['id2gloss'][idx + 1] = {
                'gloss': key,
                'frequency': value,
            }

        gloss_dict_path = os.path.join(
            DATASET_ROOT, f"{setting}_gloss_dict.json"
        )

        with open(gloss_dict_path, "w") as f:
            json.dump(save_dict, f, indent=4)

        print(f"{setting} gloss dictionary successfully saved.\n")

    print("Preprocessing completed successfully without errors.")