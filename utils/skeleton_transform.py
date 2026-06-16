import random
import numpy as np
import torch
from scipy.interpolate import interp1d

def downsample(video, ratio=0.5):
    """
    Temporal downsampling by frame skipping.

    Parameters
    ----------
    video : ndarray or Tensor
        Skeleton sequence with shape (T, K, C).
    ratio : float
        Currently designed for ratio=0.5.

    Returns
    -------
    ndarray or Tensor
        Downsampled sequence.
    """
    if ratio != 0.5:
        raise ValueError(
            "This implementation only supports ratio=0.5"
        )

    T = video.shape[0]

    if T <= 1:
        return video

    start_idx = (
        0 if random.uniform(0, 1) > 0.5
        else 1
    )

    idx = np.arange(start_idx, T, 2)

    return video[idx]

import torch


def spatial_normalize_anchor_paper(
    origin_input_data,
    norm_div,
    split=None,
    used_part=None,
):
    """
    Adaptasi Anchor-Based Normalization (Roh et al., 2024)

    Hand:
        Hand_i = Hand_i - Palm

    Upper limb:
        Upper_i =
        (Upper_i - ShoulderCenter)
        / ShoulderWidth

    Input:
        (T, K, C)

    Output:
        (T, K, C)
    """

    if not isinstance(origin_input_data, torch.Tensor):
        origin_input_data = torch.as_tensor(origin_input_data)

    out = origin_input_data.clone().float()

    # ----------------------------------------
    # Coordinate normalization
    # ----------------------------------------
    out[:, :, 0:2] = out[:, :, 0:2] / norm_div - 1.0

    if split is None or used_part is None:
        return out

    eps = 1e-8
    split_points = [0] + list(split)

    for idx, part in enumerate(used_part):

        start = split_points[idx]
        end = split_points[idx + 1]

        part_data = out[:, start:end, 0:2]

        # =====================================
        # LEFT HAND
        # =====================================
        if part == "left_hand":

            palm = part_data[:, 0, :]

            out[:, start:end, 0:2] = (
                part_data
                - palm[:, None, :]
            )

        # =====================================
        # RIGHT HAND
        # =====================================
        elif part == "right_hand":

            palm = part_data[:, 0, :]

            out[:, start:end, 0:2] = (
                part_data
                - palm[:, None, :]
            )

        # =====================================
        # UPPER LIMB
        # =====================================
        elif part == "upper_limb":

            # sesuaikan indeks ini
            LEFT_SHOULDER = 0
            RIGHT_SHOULDER = 1

            left_shoulder = part_data[:, LEFT_SHOULDER, :]
            right_shoulder = part_data[:, RIGHT_SHOULDER, :]

            center = (
                left_shoulder
                + right_shoulder
            ) / 2.0

            shoulder_width = torch.linalg.norm(
                right_shoulder - left_shoulder,
                dim=1,
                keepdim=True,
            ).clamp_min(eps)

            out[:, start:end, 0:2] = (
                part_data
                - center[:, None, :]
            ) / shoulder_width[:, None, :]

    return out

def missing_keypoint_reconstruction(origin_input_data):
    """
    Missing keypoint reconstruction using temporal interpolation.

    Parameters
    ----------
    origin_input_data : Tensor
        Skeleton sequence with shape (T, K, C).

    Returns
    -------
    Tensor
        Reconstructed skeleton sequence.
    """
    # Buat salinan untuk hasil rekonstruksi
    result = origin_input_data.clone()

    # Ekstrak koordinat x dan y
    kp_xy = result[:, :, 0:2].cpu().numpy().astype(np.float32)
    # T = jumlah frame, K = jumlah keypoint
    T, K, _ = kp_xy.shape

    for k in range(K):

        coords = kp_xy[:, k, :]  # example: (T, 2)

        # Buat array boolean untuk menandai frame yang valid (bukan missing)
        valid_mask = ~(
            (coords[:, 0] == 0) &
            (coords[:, 1] == 0)
        )
        # example: [True, True, False, True, False, ...]

        # Dapatkan indeks frame yang valid
        valid_idx = np.where(valid_mask)[0] # example: [0, 1, 3, ...]

        # Cek apakah ada frame yang valid untuk keypoint ini
        if len(valid_idx) == 0:
            continue

        # Iterasi melalui semua frame untuk keypoint ini
        for t in range(T):

            # Cek apakah frame ini valid
            if valid_mask[t]:
                continue
            
            # prev_arr = indeks frame valid yang lebih kecil dari t
            prev_arr = valid_idx[valid_idx < t] # example: [0, 1] untuk t=2
            # next_arr = indeks frame valid yang lebih besar dari t
            next_arr = valid_idx[valid_idx > t] # example: [3] untuk t=2

            # Cek apakah ada frame valid sebelum dan sesudah t
            if len(prev_arr) and len(next_arr):

                # Lakukan interpolasi linier antara frame terakhir sebelum t dan frame pertama setelah t
                p = prev_arr[-1]
                n = next_arr[0]

                alpha = t - p      # jarak ke frame valid sebelumnya
                beta = n - t       # jarak ke frame valid sesudahnya

                coords[t] = (
                    beta * coords[p] +
                    alpha * coords[n]
                ) / (alpha + beta)

            # Jika hanya ada frame valid sebelum t, gunakan koordinat dari frame tersebut
            elif len(prev_arr):
                coords[t] = coords[prev_arr[-1]]

            # Jika hanya ada frame valid sesudah t, gunakan koordinat dari frame tersebut
            elif len(next_arr):
                coords[t] = coords[next_arr[0]]

        # Masukkan kembali koordinat yang sudah direkonstruksi ke array hasil
        kp_xy[:, k, :] = coords

    # Masukkan kembali hasil rekonstruksi
    result[:, :, 0:2] = torch.from_numpy(kp_xy).to(
        device=result.device,
        dtype=result.dtype
    )

    return result

def temporal_normalize(origin_input_data, target_length):
    """
    Temporal normalization by resampling to a target sequence length.

    Parameters
    ----------
    origin_input_data : Tensor
        Skeleton sequence with shape (T, K, C).
    target_length : int
        Target number of frames.

    Returns
    -------
    Tensor
        Temporally normalized sequence.
    """
    T, K, C = origin_input_data.shape

    if T == 0:
        raise ValueError("Empty sequence.")
    
    if T == 1:
        return origin_input_data.repeat(
            target_length,
            1,
            1
        )
    # Jika panjang sudah sesuai
    if T == target_length:
        return origin_input_data.clone()

    data = origin_input_data.cpu().numpy()

    orig_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_length)

    result = np.zeros(
        (target_length, K, C),
        dtype=data.dtype
    )

    # Interpolasi setiap keypoint dan channel
    for k in range(K):
        for c in range(C):

            fn = interp1d(
                orig_idx,
                data[:, k, c],
                kind='linear'
            )

            result[:, k, c] = fn(new_idx)

    return torch.from_numpy(result).to(
        device=origin_input_data.device,
        dtype=origin_input_data.dtype
    )
