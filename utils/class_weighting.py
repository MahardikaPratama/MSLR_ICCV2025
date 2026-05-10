"""
Class Weighting untuk mengatasi class imbalance dalam Sign Language Recognition.
Support multiple strategies: inverse_frequency, balanced, smooth
"""
import json
import torch
import numpy as np
from collections import Counter


def calculate_class_weights(gloss_dict, train_info_path, method='inverse_frequency', gamma=0.8):
    """
    Hitung bobot untuk setiap kelas berdasarkan frekuensi di training set.
    
    Args:
        gloss_dict: Dictionary dengan keys 'id2gloss' dan 'gloss2id'
                   gloss2id[gloss] = {'index': idx, 'frequency': freq}
                   Index range: 1-93 untuk 93 glosses
        train_info_path: Path ke file *_train_info.json
        method: 'inverse_frequency', 'balanced', atau 'smooth'
        gamma: Parameter untuk smooth method (0.0 - 1.0)
    
    Returns:
        torch.Tensor: Weights dengan shape (num_classes,)
                     Index 0: blank class
                     Index 1-93: gloss classes
    """
    
    # Baca training info dan hitung frekuensi setiap gloss
    with open(train_info_path, 'r') as f:
        data = json.load(f)
    
    all_glosses = []
    for item in data:
        glosses = item['gloss_sequence'].split()
        all_glosses.extend(glosses)
    
    counts = Counter(all_glosses)
    total_samples = len(all_glosses)
    num_classes = len(gloss_dict['id2gloss']) + 1  # +1 untuk blank class (index 0)
    
    # Initialize weights (size 94: indices 0-93)
    weights = torch.ones(num_classes)
    
    # Method 1: Inverse Frequency (paling umum)
    if method == 'inverse_frequency':
        for gloss, gloss_info in gloss_dict['gloss2id'].items():
            idx = gloss_info['index']  # Index 1-93
            if gloss in counts:
                # weight = total / (num_classes * count)
                # Semakin banyak muncul, semakin rendah weight
                weights[idx] = total_samples / (num_classes * counts[gloss])
        weights[0] = 1.0  # Blank class weight = 1.0 (normal)
        weights = weights / weights.mean()  # Normalize agar mean = 1
    
    # Method 2: Balanced (1 - frequency)
    elif method == 'balanced':
        for gloss, gloss_info in gloss_dict['gloss2id'].items():
            idx = gloss_info['index']
            if gloss in counts:
                # weight = 1 - (count / total)
                freq = counts[gloss] / total_samples
                weights[idx] = 1.0 - freq
        weights[0] = 1.0
        weights = weights / weights.mean()
    
    # Method 3: Smooth (less extreme)
    elif method == 'smooth':
        # weight = ((total - count) / total) ^ gamma
        for gloss, gloss_info in gloss_dict['gloss2id'].items():
            idx = gloss_info['index']
            if gloss in counts:
                freq = counts[gloss] / total_samples
                weights[idx] = (1.0 - freq) ** gamma
        weights[0] = 1.0
        weights = weights / weights.mean()
    
    return weights


def log_class_weights(gloss_dict, train_info_path, method='inverse_frequency', top_k=10):
    """
    Print class weights untuk debugging dan visualization.
    
    Args:
        gloss_dict: Dictionary dengan keys 'id2gloss' dan 'gloss2id'
        train_info_path: Path ke file *_train_info.json
        method: Method yang digunakan
        top_k: Jumlah top kelas yang ditampilkan
    """
    
    with open(train_info_path, 'r') as f:
        data = json.load(f)
    
    all_glosses = []
    for item in data:
        glosses = item['gloss_sequence'].split()
        all_glosses.extend(glosses)
    
    counts = Counter(all_glosses)
    weights = calculate_class_weights(gloss_dict, train_info_path, method)
    
    print(f"\n{'='*60}")
    print(f"Class Weighting Report (Method: {method})")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_glosses)}")
    print(f"Total classes: {len(gloss_dict['id2gloss'])}")
    print(f"\nTop {top_k} most frequent glosses and their weights:")
    print(f"{'Rank':<6} {'Gloss':<15} {'Count':<8} {'Weight':<10} {'Normalized':<10}")
    print("-" * 60)
    
    # Sort by frequency
    sorted_glosses = counts.most_common(top_k)
    for rank, (gloss, count) in enumerate(sorted_glosses, 1):
        gloss_info = gloss_dict['gloss2id'][gloss]
        idx = gloss_info['index']  # Index 1-93
        weight = weights[idx].item()
        normalized_weight = weight / weights.mean().item()
        print(f"{rank:<6} {gloss:<15} {count:<8} {weight:<10.4f} {normalized_weight:<10.4f}")
    
    print(f"\nBlank class weight: {weights[0].item():.4f}")
    print(f"Mean weight: {weights.mean().item():.4f}")
    print(f"Min weight: {weights.min().item():.4f}")
    print(f"Max weight: {weights.max().item():.4f}")
    print(f"{'='*60}\n")
