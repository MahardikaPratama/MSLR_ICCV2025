try:
    import torch
except ImportError:
    pass
import random
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton


class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.
    """
    def __init__(self, std_dev=0.01) -> None:
        self.std_dev = std_dev

    def __call__(self, skeleton):
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        return skeleton + noise


class TemporalDropout(object):
    """
    Remove a contiguous segment of frames.
    """
    def __init__(self, max_dp=0.2):
        self.max_dp = max_dp

    def __call__(self, clip):
        T = len(clip)
        dp_len = int(T * self.max_dp * np.random.random())

        if dp_len == 0:
            return clip

        start = np.random.randint(0, T - dp_len + 1)
        end = start + dp_len

        index = list(range(0, start)) + list(range(end, T))
        return clip[index]


class TemporalCrop(object):
    """
    Crop frames from beginning and end (NO hidden downsampling).
    """
    def __init__(self, max_dp=0.2) -> None:
        self.max_dp = max_dp

    def __call__(self, clip):
        T = len(clip)
        dp_len = int(T * self.max_dp * np.random.random())

        if dp_len == 0:
            return clip

        drop_head = random.randint(0, dp_len)
        drop_tail = dp_len - drop_head

        start = drop_head
        end = T - drop_tail

        return clip[start:end]


class Dropout_kp(object):
    """
    Randomly drop keypoints.
    """
    def __init__(self, drop_prob=0.1) -> None:
        self.drop_prob = drop_prob    

    def __call__(self, skeleton):
        T, K, _ = skeleton.shape
        mask = np.random.rand(T, K) > self.drop_prob
        return skeleton * mask[..., np.newaxis]


class Spatial_flip(object):
    """
    Horizontal flip (mirror).
    """
    def __init__(self, prob=0.5) -> None:
        self.prob = prob

    def __call__(self, skeleton):
        if random.random() < self.prob:
            flipped = skeleton.copy()

            # flip x-axis
            flipped[..., 0] = -flipped[..., 0]

            # swap left-right hands (assumed index layout)
            flipped[:, 0:21], flipped[:, 21:42] = \
                flipped[:, 21:42].copy(), flipped[:, 0:21].copy()

            return flipped
        return skeleton


class Scale(object):
    """
    Apply GLOBAL spatial scaling (physically plausible).
    """
    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, skeleton):
        scale = np.random.uniform(*self.scale_range)
        return skeleton * scale


class TemporalRescale(object):
    """
    Temporal rescaling using linear interpolation (smooth).
    """
    def __init__(self, temp_scaling=0.2):
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        T = len(clip)

        scale = self.L + (self.U - self.L) * np.random.random()
        new_T = max(2, int(round(T * scale)))

        # original time indices
        old_idx = np.arange(T)
        new_idx = np.linspace(0, T - 1, new_T)

        # interpolate
        resampled = []
        for i in range(clip.shape[1]):  # keypoints
            kp_traj = clip[:, i, :]  # (T, C)

            interp_kp = np.zeros((new_T, kp_traj.shape[1]))
            for c in range(kp_traj.shape[1]):
                interp_kp[:, c] = np.interp(new_idx, old_idx, kp_traj[:, c])

            resampled.append(interp_kp)

        resampled = np.stack(resampled, axis=1)  # (new_T, K, C)
        return resampled


class TemporalRescale_test(object):
    def __call__(self, clip):
        vid_len = len(clip)
        new_len = vid_len

        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4

        index = [i for i in range(new_len)]
        for i in range(vid_len, new_len):
            index[i] = vid_len - 1

        return clip[index]