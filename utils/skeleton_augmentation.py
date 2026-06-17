try:
    import torch
except ImportError:
    pass
import random
import numpy as np

class Compose(object):
    """
    Compose multiple skeleton transformations.
    
    Args:
        transforms (list): List of skeleton transformations to apply.
    """
    # Initialize the transform
    def __init__(self, transforms):
        self.transforms = transforms

    # Apply the transforms
    def __call__(self, skeleton):
        # Iterate through each transform
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    """
    Convert skeleton to PyTorch tensor.
    
    Args:
        skeleton (np.ndarray): Skeleton sequence.
    """
    # Convert the skeleton data to tensor
    def __call__(self, skeleton):
        # Check if the skeleton data is a numpy array
        if isinstance(skeleton, np.ndarray):
            # Convert the skeleton data to numpy array
            skeleton = np.asarray(skeleton)
            # Convert the skeleton data to tensor
            skeleton = torch.from_numpy(skeleton).float()
        # Return the transformed skeleton
        return skeleton


class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.
    
    Args:
        std_dev (float): Standard deviation of the Gaussian noise.
    """
    # Initialize the transform
    def __init__(self, std_dev=0.01) -> None:
        self.std_dev = std_dev
    # Apply the transform
    def __call__(self, skeleton):
        # Create Gaussian noise
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        # Add the Gaussian noise to the skeleton
        return skeleton + noise


class TemporalDropout(object):
    """
    Apply temporal dropout by randomly removing a contiguous segment of frames.
    
    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """
    # Initialize the transform
    def __init__(self, max_dp=0.2):
        self.max_dp = max_dp
    # Apply the transform
    def __call__(self, clip):
        # Get the length of the clip
        vid_len = len(clip)
        # Calculate the dropout length
        dp_len = int(vid_len * self.max_dp * np.random.random())
        # Calculate the start and end indices of the dropout
        start = np.random.randint(0, vid_len - dp_len + 1)
        end = start + dp_len
        # Create a list of indices for the remaining frames
        index = list(range(0, start)) + list(range(end, vid_len))
        # Return the clip with the dropout applied
        return clip[index]

class Scale(object):
    """
    Scale skeleton sequences by applying random scaling factors.
    
    Args:
        scale_range (tuple): Range of scaling factors (min, max).
    """
    # Initialize the transform
    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range
    # Apply the transform
    def __call__(self, skeleton):
        # Get the length of the skeleton
        T = skeleton.shape[0]
        # Generate random scale factor
        scales = np.random.uniform(*self.scale_range, size=T)
        # Scale the skeleton
        scaled_skeleton = skeleton * scales[:, np.newaxis, np.newaxis]
        # Return the scaled skeleton
        return scaled_skeleton


class TemporalRescale(object):
    """
    Temporally rescale video by resampling frames.
    
    Args:
        temp_scaling (float): Temporal scaling factor. Video length is scaled 
            between [1 - temp_scaling, 1 + temp_scaling].
    """
    # Initialize the transform
    def __init__(self, temp_scaling=0.2) -> None:
        # Set the minimum and maximum lengths
        self.min_len = 32
        self.max_len = 230
        # Calculate the lower and upper bounds for temporal scaling
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling
    # Apply the transform
    def __call__(self, clip):
        # Get the length of the clip
        vid_len = len(clip)
        # Calculate the new length
        new_len = int(vid_len * np.random.uniform(self.L, self.U))
        # Check if the new length is less than the minimum length
        if new_len < self.min_len:
            # Set the new length to the minimum length
            new_len = self.min_len
        # Check if the new length is greater than the maximum length
        if new_len > self.max_len:
            # Set the new length to the maximum length
            new_len = self.max_len
        # Check if the new length minus 4 is not divisible by 4
        if (new_len - 4) % 4 != 0:
            # Add 4 to the new length
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            # Get the indices for the new length
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            # Get the indices for the new length
            index = sorted(random.choices(range(vid_len), k=new_len))
        # Return the clip with the new length
        return clip[index]