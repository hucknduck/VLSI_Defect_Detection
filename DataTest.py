import os
import h5py
import torch
from NewDataSet import VLSIOpenMaskDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def main():
    DataDir = "./Data"

    h5_paths = [
        os.path.join(DataDir, "opens1.h5"),
        os.path.join(DataDir, "opens2.h5"),
        os.path.join(DataDir, "opens3.h5"),
        os.path.join(DataDir, "opens4.h5"),
        os.path.join(DataDir, "opens5.h5")
    ]

    # count = [0 for _ in range(48*48)]

    # for file_idx, path in enumerate(h5_paths):
    #             with h5py.File(path, 'r') as hf:
    #                 print("opening new file")
    #                 current_file_length = 0
    #                 while f'img_{current_file_length}' in hf:
    #                     center = hf[f'center_{current_file_length}']
    #                     count[center[0]*48 + center[1]] += 1
    #                     current_file_length += 1


    # for idx, i in enumerate(count):
    #        print(f'label {idx} appears {i} times')

    Data = VLSIOpenMaskDataset(h5_paths)

    train_loader = DataLoader(
            Data,
            batch_size=512,
            num_workers=1,
            pin_memory=True
        )

    ###### PLOT AS HISTOGRAM ######
    
    # all_labels = []
    # for _, targets in train_loader:
    #     labels = targets['labels'].squeeze(1)
    #     all_labels.append(labels.cpu())
    # all_labels = torch.cat(all_labels).numpy()
    # plt.hist(all_labels, bins=2304)
    # plt.title("Class distribution")
    # plt.show()

    ###### PLOT AS HEATMAP ######
    
    heatmap = np.zeros((48, 48), dtype=np.int32)

    for _, targets in train_loader:
        labels = targets['labels'].squeeze(1)  # Shape: [B]
        coords = labels.cpu().numpy()

        for label in coords:
            x = label // 48
            y = label % 48
            heatmap[x, y] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("Label Frequency Heatmap (48x48)")
    plt.xlabel("y-coordinate")
    plt.ylabel("x-coordinate")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # fix 2
    main()