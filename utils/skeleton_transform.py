import numpy as np
import torch
from scipy.interpolate import interp1d

def downsample(video, ratio=0.5):
    """
    Uniform temporal downsampling.

    Parameters
    ----------
    video : ndarray or Tensor
        Skeleton sequence with shape (T, K, C).
    ratio : float
        Downsampling ratio (0 < ratio < 1).

    Returns
    -------
    ndarray or Tensor
        Downsampled sequence.
    """
    if not (0 < ratio < 1):
        return video

    T = video.shape[0]
    new_len = max(1, round(T * ratio))

    idx = np.round(
        np.linspace(0, T - 1, new_len)
    ).astype(np.int64)

    return video[idx]

def spatial_normalize(
    origin_input_data,
    norm_div,
    norm_point=None,
    split=None,
    used_part=None,
):
    """
    Spatial normalization.

    For hand skeletons:
        1. Coordinate normalization.
        2. Translation normalization using wrist.
        3. Scale normalization using wrist-to-middle-MCP distance.

    Input:
        (T, K, C)

    Output:
        Tensor with shape (T, K, C)
    """
    if not isinstance(origin_input_data, torch.Tensor):
        origin_input_data = torch.as_tensor(origin_input_data)

    out = origin_input_data.clone().float()

    # --------------------------------------------------
    # Coordinate normalization
    # --------------------------------------------------
    out[:, :, 0:2] = out[:, :, 0:2] / norm_div - 1.0

    if split is None or used_part is None:
        return out

    eps = 1e-8
    split_points = [0] + list(split)

    for idx, part in enumerate(used_part):

        start = split_points[idx]
        end = split_points[idx + 1]

        part_data = out[:, start:end, 0:2]

        # ==================================================
        # LEFT HAND
        # ==================================================
        if part == "left_hand":

            wrist = part_data[:, 0, :]
            middle_mcp = part_data[:, 9, :]

            scale = torch.linalg.norm(
                middle_mcp - wrist,
                dim=1,
                keepdim=True,
            ).clamp_min(eps)

            out[:, start:end, 0:2] = (
                part_data - wrist[:, None, :]
            ) / scale[:, None, :]

        # ==================================================
        # RIGHT HAND
        # ==================================================
        elif part == "right_hand":

            wrist = part_data[:, 0, :]
            middle_mcp = part_data[:, 9, :]

            scale = torch.linalg.norm(
                middle_mcp - wrist,
                dim=1,
                keepdim=True,
            ).clamp_min(eps)

            out[:, start:end, 0:2] = (
                part_data - wrist[:, None, :]
            ) / scale[:, None, :]

        # ==================================================
        # MOUTH
        # ==================================================
        elif part == "mouth_8":

            # TODO:
            # Implement mouth normalization when mouth
            # becomes part of the experimental setup.
            pass

        # ==================================================
        # BODY
        # ==================================================
        elif part == "body":

            # TODO:
            # Implement body normalization when body
            # becomes part of the experimental setup.
            pass

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
    result = origin_input_data.clone()

    # Ambil koordinat xy
    kp_xy = result[:, :, 0:2].cpu().numpy().astype(np.float32)
    T, K, _ = kp_xy.shape

    for k in range(K):

        coords = kp_xy[:, k, :]  # (T, 2)

        # Keypoint dianggap missing jika x == 0 dan y == 0
        valid_mask = ~(
            (coords[:, 0] == 0) &
            (coords[:, 1] == 0)
        )

        valid_idx = np.where(valid_mask)[0]

        # Semua frame missing
        if len(valid_idx) == 0:
            continue

        for t in range(T):

            # Skip jika valid
            if valid_mask[t]:
                continue

            prev_arr = valid_idx[valid_idx < t]
            next_arr = valid_idx[valid_idx > t]

            # Interpolasi linier
            if len(prev_arr) and len(next_arr):

                p = prev_arr[-1]
                n = next_arr[0]

                alpha = (t - p) / (n - p)

                coords[t] = (
                    (1 - alpha) * coords[p] +
                    alpha * coords[n]
                )

            # Gunakan frame sebelumnya
            elif len(prev_arr):
                coords[t] = coords[prev_arr[-1]]

            # Gunakan frame berikutnya
            elif len(next_arr):
                coords[t] = coords[next_arr[0]]

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
