from torch.utils.data import Dataset, get_worker_info
import h5py
import torch
import numpy as np
import random
import cv2

class VLSIOpenMaskDataset(Dataset):
    def __init__(self, h5_paths, noise_gaussian=False, noise_salt_and_pepper=False, noise_blur=False, noise_prob=0.0, gaussian_std=0.1, salt_and_pepper_amount=0.01, kernel_size=(3,3)):
        self.h5_paths = h5_paths
        self.index_map = []
        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, 'r') as hf:
                i = 0
                Limit = np.inf if 'rectrees' not in path else 10000
                while f'img_{i}' in hf and i < Limit:
                    self.index_map.append((file_idx, i))
                    i += 1

        self._h5_handles = None  # will be initialized per-worker
        self.noise_gaussian = noise_gaussian
        self.noise_salt_and_pepper = noise_salt_and_pepper
        self.noise_blur = noise_blur
        self.noise_prob = noise_prob
        self.gaussian_std = gaussian_std
        self.salt_and_pepper_amount = salt_and_pepper_amount
        self.kernel_size = kernel_size

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

    def _noise_img(self, img):
        if self.noise_gaussian:
            noise = np.random.normal(0, self.gaussian_std, img.shape)
            img = img + noise
            img = np.clip(img, 0.0, 1.0)

        if self.noise_salt_and_pepper:
            ratio = random.random()
            num_salt = np.ceil(self.salt_and_pepper_amount * img.size * ratio)
            num_pepper = np.ceil(self.salt_and_pepper_amount * img.size * (1.0 - ratio))

            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            img[tuple(coords)] = 1

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            img[tuple(coords)] = 0

        if self.noise_blur:
            img = (img * 255).astype(np.uint8)
            img = cv2.GaussianBlur(img, self.kernel_size, 0)
            img = img.astype(np.float32) / 255.0

        return img

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
        try: center_np = hf[f'center_{inner_idx}'][:]
        except: center_np = (48,0) #dummy value for circuit with no defects
        
        img_np = img_np.astype(np.float32) / 255.0
        
        if random.random() < self.noise_prob:
            img_np = self._noise_img(img_np)
        
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        label = torch.tensor([center_np[0] * 48 + center_np[1]], dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {
            "labels": label,
            "image_id": image_id
        }

        return img_tensor, target