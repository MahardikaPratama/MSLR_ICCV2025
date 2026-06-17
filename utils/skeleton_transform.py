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
    # Check if ratio is 0.5 and raise error if not
    if ratio != 0.5:
        raise ValueError(
            "This implementation only supports ratio=0.5"
        )

    # Get the number of frames
    T = video.shape[0]

    # If the number of frames is less than or equal to 1, return the video
    if T <= 1:
        return video

    # Randomly select the starting index
    # If random value > 0.5, start_idx = 0, else start_idx = 1
    start_idx = (
        0 if random.uniform(0, 1) > 0.5
        else 1
    )

    # Create an array of indices with a step of 2
    idx = np.arange(start_idx, T, 2)

    # Return the downsampled sequence
    return video[idx]

def spatial_normalize(
    origin_input_data,
    norm_div,
    norm_point=None,
    split=None,
    used_part=None,
):
    """
    Adopted from Anchor-Based Normalization (Roh et al., 2024)

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

    # Convert input data to tensor if not already a tensor
    if not isinstance(origin_input_data, torch.Tensor):
        origin_input_data = torch.as_tensor(origin_input_data)

    # Clone tensor to avoid modifying the original tensor
    out = origin_input_data.clone().float()

    # ----------------------------------------
    # Coordinate normalization
    # ----------------------------------------

    # If split or used_part is None, return the result
    if split is None or used_part is None:
        return out

    # Add epsilon value to prevent division by zero
    eps = 1e-8
    # Add split_points[idx] + 1 to split_points to calculate the end of each part
    split_points = [0] + list(split)

    # Iterate through each part
    for idx, part in enumerate(used_part):

        # Get the start and end indices of each part
        start = split_points[idx]
        end = split_points[idx + 1]

        # Get the data for each part
        part_data = out[:, start:end, 0:2]

        # =====================================
        # LEFT HAND
        # =====================================
        if part == "left_hand":

            # Get the palm keypoint (index 0)
            palm = part_data[:, 0, :]

            # Subtract each keypoint with the palm keypoint
            out[:, start:end, 0:2] = (
                part_data
                - palm[:, None, :]
            )

        # =====================================
        # RIGHT HAND
        # =====================================
        elif part == "right_hand":

            # Get the palm keypoint (index 0)
            palm = part_data[:, 0, :]

            # Subtract each keypoint with the palm keypoint
            out[:, start:end, 0:2] = (
                part_data
                - palm[:, None, :]
            )

        # =====================================
        # UPPER LIMB
        # =====================================
        elif part == "upper_limb":

            # Define the left and right shoulder indices
            LEFT_SHOULDER = 0
            RIGHT_SHOULDER = 1

            # Get the left and right shoulder keypoints
            left_shoulder = part_data[:, LEFT_SHOULDER, :]
            right_shoulder = part_data[:, RIGHT_SHOULDER, :]

            # Get the center of the left and right shoulder keypoints
            center = (
                left_shoulder
                + right_shoulder
            ) / 2.0

            # Calculate the distance between the left and right shoulder keypoints
            shoulder_width = torch.linalg.norm(
                right_shoulder - left_shoulder,
                dim=1,
                keepdim=True,
            ).clamp_min(eps)

            # Subtract each keypoint with the center keypoint
            # and divide by the shoulder width
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
    # Create a copy of the input data for reconstruction
    result = origin_input_data.clone()

    # Extract x and y coordinates
    kp_xy = result[:, :, 0:2].cpu().numpy().astype(np.float32)
    # T = number of frames, K = number of keypoints
    T, K, _ = kp_xy.shape

    # Iterate through each keypoint
    for k in range(K):

        # Get the coordinates of the current keypoint
        coords = kp_xy[:, k, :]  # example: (T, 2)

        # Create a boolean array to mark valid frames (not missing)
        valid_mask = ~(
            (coords[:, 0] == 0) &
            (coords[:, 1] == 0)
        )
        # example: [True, True, False, True, False, ...]

        # Get the indices of valid frames
        valid_idx = np.where(valid_mask)[0] # example: [0, 1, 3, ...]

        # Check if there are valid frames for this keypoint
        if len(valid_idx) == 0:
            continue

        # Iterate through all frames for this keypoint
        for t in range(T):

            # Check if the current frame is valid
            if valid_mask[t]:
                continue
            
            # prev_arr = indices of valid frames smaller than t
            prev_arr = valid_idx[valid_idx < t] # example: [0, 1] untuk t=2
            # next_arr = indices of valid frames larger than t
            next_arr = valid_idx[valid_idx > t] # example: [3] untuk t=2

            # Check if there are valid frames before and after t
            if len(prev_arr) and len(next_arr):

                # Perform linear interpolation between the last valid frame before t and the first valid frame after t
                p = prev_arr[-1]
                n = next_arr[0]

                alpha = t - p      # distance to the previous valid frame
                beta = n - t       # distance to the next valid frame

                # Perform linear interpolation between the last valid frame before t and the first valid frame after t
                coords[t] = (
                    beta * coords[p] +
                    alpha * coords[n]
                ) / (alpha + beta)

            # If there are only valid frames before t, use the coordinates from the previous valid frame
            elif len(prev_arr):
                coords[t] = coords[prev_arr[-1]]

            # If there are only valid frames after t, use the coordinates from the next valid frame
            elif len(next_arr):
                coords[t] = coords[next_arr[0]]

        # Insert the reconstructed coordinates back into the result array
        kp_xy[:, k, :] = coords

    # Insert the reconstructed result
    result[:, :, 0:2] = torch.from_numpy(kp_xy).to(
        device=result.device,
        dtype=result.dtype
    )

    # Return the reconstructed result
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
    # Get the number of frames, keypoints, and channels
    T, K, C = origin_input_data.shape

    # If the number of frames is 0, raise an error
    if T == 0:
        raise ValueError("Empty sequence.")
    
    # If the number of frames is 1, repeat the sequence to the target length
    if T == 1:
        return origin_input_data.repeat(
            target_length,
            1,
            1
        )
    # If the number of frames is already equal to the target length
    if T == target_length:
        return origin_input_data.clone()

    # Convert the input data to numpy array
    data = origin_input_data.cpu().numpy()

    # Create an array of original indices
    orig_idx = np.linspace(0, T - 1, T)
    # Create an array of new indices
    new_idx = np.linspace(0, T - 1, target_length)

    # Create an array to store the result
    result = np.zeros(
        (target_length, K, C),
        dtype=data.dtype
    )

    # Interpolate each keypoint and channel
    for k in range(K):
        for c in range(C):

            # Create an interpolation function
            fn = interp1d(
                orig_idx,
                data[:, k, c],
                kind='linear'
            )

            # Interpolate the data
            result[:, k, c] = fn(new_idx)

    # Convert the result back to tensor
    return torch.from_numpy(result).to(
        device=origin_input_data.device,
        dtype=origin_input_data.dtype
    )
