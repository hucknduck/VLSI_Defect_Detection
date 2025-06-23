import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class VLSIOpenMaskDataset(Dataset):
    """
    PyTorch Dataset for loading VLSI “open defect” samples from multiple H5 files.
    Each sample returns:
      - image: FloatTensor [1, 48, 48], grayscale in [0,1] (original resolution)
      - target: dict with keys:
            'boxes'     : FloatTensor [1, 4] (x_min, y_min, x_max, y_max) at 48x48 scale
            'labels'    : Int64Tensor [1] (class index = 1 for defect)
            'masks'     : UInt8Tensor [1, 48, 48] binary mask for the defect rectangle
            'image_id'  : Int64Tensor [1] unique ID
            'area'      : FloatTensor [1] area of box
            'iscrowd'   : Int64Tensor [1] = 0
    """
    def __init__(self, h5_paths):
        super().__init__()
        self.h5_paths = h5_paths
        #self.transforms = transforms
        self.index_map = []
        # Build index map: (file_index, sample_index) for each of the 500K entries
        for file_idx, path in enumerate(self.h5_paths):
            with h5py.File(path, 'r') as hf:
                current_file_length = 0
                while f'img_{current_file_length}' in hf:
                    self.index_map.append((file_idx, current_file_length))
                    current_file_length += 1

        # print(f"Total samples indexed across all H5 files: {len(self.index_map)}")


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, inner_idx = self.index_map[idx]
        h5_path = self.h5_paths[file_idx]
        with h5py.File(h5_path, 'r') as hf:
            img_np = hf[f'img_{inner_idx}'][:]      # shape (48, 48), uint8 or float
            # box_np = hf[f'open_{inner_idx}'][:]     # [x_min, y_min, x_max, y_max] at 48x48 scale
            center_np = hf[f'center_{inner_idx}'][:] # [center_x, center_y] at 48x48 scale

        img_np = img_np.astype(np.float32) / 255.0  # (48,48) float32
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # [1, 48, 48]

        # 3) Prepare target (boxes and area are already at 48x48 scale)
        # boxes = torch.tensor([box_np.tolist()], dtype=torch.float32)  # [1,4]
        labels = torch.tensor([center_np[0]*48 + center_np[1]], dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)
        # area = torch.tensor([(x_max - x_min) * (y_max - y_min)], dtype=torch.float32)
        # iscrowd = torch.tensor([0], dtype=torch.int64)

        target = {
            # "boxes": boxes,
            "labels": labels, #label in (48x + y)
            # "masks": mask_tensor,
            "image_id": image_id,
            # "area": area,
            # "iscrowd": iscrowd,
            # "center": center_np # label in (x,y)
        }

        # if self.transforms:
        #     img_tensor = self.transforms(img_tensor)

        return img_tensor, target