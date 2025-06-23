from torch.utils.data import Dataset, get_worker_info
import h5py
import torch
import numpy as np

class VLSIOpenMaskDataset(Dataset):
    def __init__(self, h5_paths):
        self.h5_paths = h5_paths
        self.index_map = []
        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, 'r') as hf:
                i = 0
                while f'img_{i}' in hf:
                    self.index_map.append((file_idx, i))
                    i += 1

        self._h5_handles = None  # will be initialized per-worker

    def _init_handles(self):
        # Called once per worker
        # for path in self.h5_paths:
        #     print(f"Opening {path}")     
        self._h5_handles = [h5py.File(p, 'r') for p in self.h5_paths]

    def __len__(self):
        return len(self.index_map)

    def __del__(self):
        if self._h5_handles:
            for f in self._h5_handles:
                f.close()

    def __getitem__(self, idx):
        # Per-worker handle initialization
        if self._h5_handles is None:
            worker_info = get_worker_info()
            if worker_info is None:
                # Running in main process (single-process DataLoader or debugging)
                self._h5_handles = [h5py.File(p, 'r') for p in self.h5_paths]
            else:
                # We're inside a worker process
                self._init_handles()

        file_idx, inner_idx = self.index_map[idx]
        hf = self._h5_handles[file_idx]

        img_np = hf[f'img_{inner_idx}'][:]
        center_np = hf[f'center_{inner_idx}'][:]

        img_np = img_np.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        label = torch.tensor([center_np[0] * 48 + center_np[1]], dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {
            "labels": label,
            "image_id": image_id
        }

        return img_tensor, target