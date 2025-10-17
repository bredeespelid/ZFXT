from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import math


class PatchDataset(Dataset):
    def __init__(self, data: np.ndarray, context_len: int = 512, patch_len: int = 32, stride: int = 8,
                 mask_ratio_min: float = 0.2, mask_ratio_max: float = 0.3):
        """
        data: (N, C) normalized array
        returns sequences of length T=context_len with masked patches.
        """
        assert data.ndim == 2
        self.data = data.astype(np.float32)
        self.C = data.shape[1]
        self.T = context_len
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

        self.num_patches = math.floor((self.T - patch_len) / stride) + 1
        self.total_steps = len(self.data) - self.T
        self.total_steps = max(self.total_steps, 1)

    def __len__(self):
        return self.total_steps

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.T
        if end > len(self.data):
            # pad by repeating last row
            pad_len = end - len(self.data)
            segment = np.concatenate([self.data[start:], np.repeat(self.data[-1:], pad_len, axis=0)], axis=0)
        else:
            segment = self.data[start:end]
        x = torch.from_numpy(segment)  # (T, C)

        # create patch mask
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        mask_ratio = np.random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        num_mask = max(1, int(self.num_patches * mask_ratio))
        idxs = np.random.choice(self.num_patches, size=num_mask, replace=False)
        mask[idxs] = True
        # expand mask to time steps
        time_mask = torch.zeros(self.T, dtype=torch.bool)
        for p in range(self.num_patches):
            s = p * self.stride
            e = s + self.patch_len
            if e > self.T:
                e = self.T
            if mask[p]:
                time_mask[s:e] = True
        # targets only where masked
        target = x.clone()
        target[~time_mask] = 0.0
        # input can be corrupted on masked positions (set to 0)
        x_masked = x.clone()
        x_masked[time_mask] = 0.0

        return {
            'input': x_masked,     # (T, C)
            'target': target,      # (T, C) but zeros elsewhere
            'time_mask': time_mask # (T,)
        }


def create_dataloader(arr: np.ndarray, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0,
                      context_len: int = 512, patch_len: int = 32, stride: int = 8,
                      mask_ratio_min: float = 0.2, mask_ratio_max: float = 0.3):
    ds = PatchDataset(arr, context_len=context_len, patch_len=patch_len, stride=stride,
                      mask_ratio_min=mask_ratio_min, mask_ratio_max=mask_ratio_max)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
