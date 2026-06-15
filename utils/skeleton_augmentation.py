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
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    """
    Convert skeleton to PyTorch tensor.
    
    Args:
        skeleton (np.ndarray): Skeleton sequence.
    """
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton


class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.
    
    Args:
        std_dev (float): Standard deviation of the Gaussian noise.
    """
    
    def __init__(self, std_dev=0.01) -> None:
        self.std_dev = std_dev

    def __call__(self, skeleton):
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        return skeleton + noise


class TemporalDropout(object):
    """
    Apply temporal dropout by randomly removing a contiguous segment of frames.
    
    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """

    def __init__(self, max_dp=0.2):
        self.max_dp = max_dp

    def __call__(self, clip):
        vid_len = len(clip)
        dp_len = int(vid_len * self.max_dp * np.random.random())
        start = np.random.randint(0, vid_len - dp_len + 1)
        end = start + dp_len
        index = list(range(0, start)) + list(range(end, vid_len))
        return clip[index]

class Scale(object):
    """
    Apply spatial scaling to skeleton features.

    Args:
        scale_range (tuple): Range of scaling factors.
    """

    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, skeleton):

        scale = np.random.uniform(*self.scale_range)

        output = skeleton.copy()

        output[..., :-1] *= scale

        return output


class TemporalRescale(object):
    """
    Temporally rescale video by resampling frames.
    
    Args:
        temp_scaling (float): Temporal scaling factor. Video length is scaled 
            between [1 - temp_scaling, 1 + temp_scaling].
    """
    
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32 # jadi parameter
        self.max_len = 230 # jadi parameter
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = int(vid_len * np.random.uniform(self.L, self.U))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]