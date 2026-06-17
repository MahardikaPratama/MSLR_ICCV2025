import os
import json
from tqdm import tqdm


# =========================================================
# PROJECT PATH
# =========================================================

def get_project_root():
    """
    Return the absolute path of the project root directory.

    Returns
    -------
    str
        Absolute path of the project root directory.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

# Project root directory
PROJECT_ROOT = get_project_root()

# Preprocess root directory
PREPROCESS_ROOT = os.path.dirname(__file__)

# Dataset root directory
DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "mslr2025"
)

# Create dataset root directory
os.makedirs(DATASET_ROOT, exist_ok=True)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

# Dataset splits
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
    Update a dictionary of sign frequencies.

    Parameters
    ----------
    total_dict : dict
        Dictionary containing sign frequencies.
    info : list of dict
        List of dictionaries containing sign sequences.

    Returns
    -------
    dict
        Updated dictionary of sign frequencies.
    """

    # Iterate through each item in the info list
    for item in info:

        # Split the gloss sequence into individual glosses
        split_label = item['gloss_sequence'].split()

        # Iterate through each gloss in the split label
        for gloss in split_label:

            # Update the count of each gloss in the total_dict
            total_dict[gloss] = (
                total_dict.get(gloss, 0) + 1
            )

    # Return the updated dictionary of sign frequencies
    return total_dict


def generate_gt_stm(info, save_path):
    """
    Generate a ground truth STM (Segment Time Mark) file for WER evaluation.

    Parameters
    ----------
    info : list of dict
        List containing 'video_id', 'signer', and 'gloss_sequence'.
    save_path : str
        Path to save the STM file.
    """

    # Open the file at save_path in write mode ('w') and UTF-8 encoding
    with open(save_path, "w", encoding="utf-8") as f:

        # Iterate through each item in the info list
        for item in info:

            # Write one line per item to the file with the format:
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
    Read a pipe-separated text annotation file and convert it into a list of dictionaries.

    Parameters
    ----------
    anno_path : str
        Path to the text annotation file.

    Returns
    -------
    list of dict
        List of dictionaries containing sample metadata.
    """

    # Check if the annotation file exists at anno_path; if not, raise FileNotFoundError
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found:\n{anno_path}"
        )

    # Open the annotation file in read mode ('r') and UTF-8 encoding
    with open(anno_path, "r", encoding="utf-8") as f:

        # Read all lines from the file into the lines list
        lines = f.readlines()

    # Check if the first line is a header (contains 'video' or 'gloss' in lowercase)
    # If it is, skip the header line by slicing the list
    if (
        len(lines) > 0 and
        (
            "video" in lines[0].lower() or
            "gloss" in lines[0].lower()
        )
    ):
        # Skip the header line
        lines = lines[1:]

    # Initialize an empty list to store the parsed sample information
    info_list = []

    # Iterate through each line in the lines list using tqdm for progress tracking
    for line in tqdm(lines):

        # Split the line by the '|' character
        parts = line.strip().split('|')

        # Skip the line if it does not contain at least 2 parts
        if len(parts) < 2:
            continue

        # Extract video_id and gloss_sequence from the parts list
        video_id = parts[0]
        gloss_seq = parts[1]

        # Split the video_id by the '_' character to extract signer and sentence_id
        split_vid = video_id.split('_')
        signer = split_vid[0]
        sentence_id = split_vid[1]

        # Append the extracted information to the info_list
        info_list.append({

            "signer": signer,

            "video_id": video_id,

            "gloss_sequence": gloss_seq.strip(),

            "sentence_id": sentence_id,

            "original_info": line,

        })

    # Return the list of dictionaries
    return info_list


def save_json(obj, save_path):
    """
    Save a Python object to a formatted JSON file.

    Parameters
    ----------
    obj : dict or list
        Python object to save.
    save_path : str
        Full path to the destination JSON file.
    """

    # Open the file at save_path in write mode ('w') and UTF-8 encoding
    with open(save_path, "w", encoding="utf-8") as f:

        # Dump the object to the file as JSON
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
    Process MSLR2025 dataset splits to generate metadata JSONs, ground truth STM files, and global gloss dictionary.
    """

    # =====================================================
    # PRINT START MESSAGE
    # =====================================================
    
    print("\n===================================")
    print("MSLR2025 PREPROCESSING START")
    print("===================================\n")

    # =====================================================
    # GLOBAL GLOSS DICTIONARY
    # =====================================================

    # Initialize an empty dictionary to accumulate the global sign vocabulary
    global_sign_dict = dict()

    # =====================================================
    # PROCESS EACH SPLIT
    # =====================================================

    # Iterate through each split defined in DATASET_SPLITS
    for split_name, cfg in DATASET_SPLITS.items():

        # Get the folder and file name for the current split
        folder = cfg["folder"]
        file_name = cfg["file"]

        # Construct the full path to the annotation file
        anno_path = os.path.join(
            PREPROCESS_ROOT,
            folder,
            file_name
        )

        # Print start message for the current split
        print(f"\nProcessing: {split_name}")

        # =================================================
        # LOAD SPLIT
        # =================================================
        
        # Call info2dict() to load and parse the split data into split_info
        split_info = info2dict(anno_path)

        # =================================================
        # SAVE METADATA JSON
        # =================================================
        
        # Construct the JSON file path in DATASET_ROOT
        json_save_path = os.path.join(
            DATASET_ROOT,
            f"{split_name}_info.json"
        )

        # Call save_json() to save the split metadata
        save_json(split_info, json_save_path)

        # =================================================
        # SAVE STM GROUND TRUTH
        # =================================================

        # Construct the STM file path in DATASET_ROOT
        stm_save_path = os.path.join(
            DATASET_ROOT,
            f"mslr-groundtruth-{split_name}.stm"
        )

        # Call generate_gt_stm() to produce the STM ground truth file
        generate_gt_stm(
            split_info,
            stm_save_path
        )

        # =================================================
        # BUILD GLOBAL VOCABULARY
        # ONLY FROM TRAIN + DEV
        # =================================================

        # Check if the split name is "train" or "dev"; if so, call sign_dict_update() to add to the global vocabulary
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

    # Sort the global_sign_dict alphabetically by gloss key using sorted()
    global_sign_dict = sorted(
        global_sign_dict.items(),
        key=lambda d: d[0]
    )

    # =====================================================
    # BUILD GLOSS MAPPING
    # =====================================================
    
    # Initialize save_dict with two empty sub-dictionaries: "id2gloss" and "gloss2id"
    save_dict = {

        "id2gloss": {},

        "gloss2id": {},

    }

    # Iterate through each pair of (gloss, freq) from global_sign_dict using enumerate() to build a two-way mapping of gloss2id and id2gloss based on index 1
    for idx, (gloss, freq) in enumerate(global_sign_dict):
        
        # Assign gloss_index starting from 1 (not 0)
        gloss_index = idx + 1

        # Save gloss2id mapping: gloss -> index + frequency
        save_dict["gloss2id"][gloss] = {

            "index": gloss_index,

            "frequency": freq,

        }

        # Save id2gloss mapping: index -> gloss + frequency
        save_dict["id2gloss"][gloss_index] = {

            "gloss": gloss,

            "frequency": freq,

        }

    # =====================================================
    # SAVE GLOBAL GLOSS DICTIONARY
    # =====================================================

    # Construct the file path for the global gloss dictionary
    gloss_dict_path = os.path.join(
        DATASET_ROOT,
        "global_gloss_dict.json"
    )

    # Call save_json() to save the global gloss dictionary
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