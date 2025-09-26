import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from NewModel import CNNLarge
from NewDataSet import VLSIOpenMaskDataset
import random
from torch.utils.data import Subset
import argparse
import os
from collections import OrderedDict

def visualize_predictions_grid(model, dataset, device, ModelPath, save_name="predictions_grid.png"):
    model.eval()

    # Sample 4 random indices
    idxs = random.sample(range(len(dataset)), 3)
    # idxs.append(488595)

    for idx, sample in enumerate(dataset):
        if idx in idxs: continue
        if sample[1]["labels"].item() == 48*48:
            idxs.append(idx)
            break


    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        img_tensor, target = dataset[idx]
        image_id = target["image_id"].item()
        label = target["labels"].item()
        X = label // 48
        Y = label % 48

        # move to device with batch dim
        img = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)

        # remove batch dim
        out = out.squeeze(0)

        # ensure image is on CPU for numpy
        img_np = img_tensor.detach().cpu().squeeze(0).numpy()  # (256, 256)
        img_disp = img_np

        # ensure logits are handled on CPU for numpy
        logits = out.detach().cpu()

        # softmax over classes (last dimension works for 1D or 2D)
        scores = torch.softmax(logits, dim=-1).numpy().squeeze()

        best_idx = int(np.argmax(scores))  # use scores, already on CPU
        X_pred = best_idx // 48
        Y_pred = best_idx % 48

        NoDefect = (best_idx == 48*48)
        pred_name = str((X_pred, Y_pred)) if not NoDefect else "No Defect"

        NoDefectGT = (label == 48*48)
        GTName = str((X, Y)) if not NoDefectGT else "No Defect"

        if not NoDefectGT:
            # Plot predicted center
            ax.scatter(Y, X, c='red', marker='o', s=100, label='Prediction')

        if not NoDefect:
            # Plot ground truth center
            ax.scatter(Y_pred, X_pred, c='lime', marker='x', s=100, label='Ground Truth')
        
        ax.imshow(img_disp, cmap="gray")

        ax.set_title(f"Image {image_id} - {GTName} (score={scores[best_idx]:.2f})")
        ax.axis("off")

    plt.tight_layout()
    FinalSavePath = os.path.join(ModelPath, save_name)
    plt.savefig(FinalSavePath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved predictions grid to {FinalSavePath}")

def visualize_failure_cases(model, dataset, device, ModelPath, save_name="failed_predictions_grid.png"):
    model.eval()

    #idxs = []

    #for idx, sample in enumerate(dataset):
    #    X, Y = sample[1]["labels"].item() // 48, sample[1]["labels"].item() % 48
    #    MoveX, MoveY = X <= 7, Y <= 7
    #    img_proc = sample[0]
    #    if MoveX or MoveY:
    #        nonzero_coords = img_proc.nonzero(as_tuple=False)
    #        if nonzero_coords.numel() > 0:
    #            min_y, min_x = nonzero_coords[:, -2].min().item(), nonzero_coords[:, -1].min().item()
    #        if (MoveX and X - min_y < 3) or (MoveY and Y - min_x < 3):
    #            idxs.append((idx, X<=7, Y<=7))
    #    if len(idxs) >= 4:
    #        break


    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    curr_ax = 0

    for img_tensor, target in dataset:
        if curr_ax == 4:
          break
        
        #img_tensor, target = dataset[idx]
        image_id = target["image_id"].item()
        label = target["labels"].item()
        X = label // 48
        Y = label % 48

        img_proc = img_tensor.clone()  # (C, H, W)

        # find bounding box of nonzero region
        nonzero_coords = img_proc.nonzero(as_tuple=False)
        if nonzero_coords.numel() > 0:
            min_y, min_x = nonzero_coords[:, -2].min().item(), nonzero_coords[:, -1].min().item()

            # shift up if needed
            if min_y > 0:
                img_proc = img_proc[..., min_y:, :]                # remove top rows
                img_proc = F.pad(img_proc, (0, 0, 0, min_y))       # pad at bottom
                label -= min_y * 48

            # shift left if needed
            elif min_x > 0:
                img_proc = img_proc[..., :, min_x:]                # remove left cols
                img_proc = F.pad(img_proc, (0, min_x, 0, 0))       # pad at right
                label -= min_x

        # add batch dim + move to device
        img = img_proc.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
        
        # remove batch dim
        out = out.squeeze(0)

        # ensure image is on CPU for numpy
        img_np = img_tensor.detach().cpu().squeeze(0).numpy()  # (256, 256)
        img_disp = img_np

        # ensure logits are handled on CPU for numpy
        logits = out.detach().cpu()

        # softmax over classes (last dimension works for 1D or 2D)
        scores = torch.softmax(logits, dim=-1).numpy().squeeze()

        best_idx = int(np.argmax(scores))  # use scores, already on CPU

        
        X_pred = best_idx // 48
        Y_pred = best_idx % 48

        if X-1 <= X_pred <= X+1 and Y-1 <= Y_pred <= Y+1:
          continue

        ax = axes[curr_ax]
        curr_ax += 1

        NoDefect = (best_idx == 48*48)
        pred_name = str((X_pred, Y_pred)) if not NoDefect else "No Defect"

        NoDefectGT = (label == 48*48)
        GTName = str((X, Y)) if not NoDefectGT else "No Defect"

        if not NoDefectGT:
            # Plot predicted center
            ax.scatter(Y, X, c='red', marker='o', s=100, label='Prediction')

        if not NoDefect:
            # Plot ground truth center
            ax.scatter(Y_pred, X_pred, c='lime', marker='x', s=100, label='Ground Truth')
        
        ax.imshow(img_disp, cmap="gray")

        ax.set_title(f"Image {image_id} - {GTName} (score={scores[best_idx]:.2f})")
        ax.axis("off")

    plt.tight_layout()
    FinalSavePath = os.path.join(ModelPath, save_name)
    plt.savefig(FinalSavePath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved predictions grid to {FinalSavePath}")

def TestRun(ModelPath, DataPath, Fails, noise_gaussian=False, noise_salt_and_pepper=False, noise_blur=False, noise_prob=0.0, gaussian_std=0.1, salt_and_pepper_amount=0.01, kernel_size=(3,3)):
    #load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLarge().to(device)
    LoadModelPath = os.path.join(ModelPath, "CNNLarge")
    LoadModelPath = os.path.join(LoadModelPath, "Best.pth")
    # Load the checkpoint
    state_dict = torch.load(LoadModelPath)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  
        new_state_dict[name] = v

    # Load into model
    model.load_state_dict(new_state_dict)
    model.eval()

    h5_paths = [
        os.path.join(DataPath, "opens1.h5"),
        os.path.join(DataPath, "opens2.h5"),
        os.path.join(DataPath, "opens3.h5"),
        os.path.join(DataPath, "opens4.h5"),
        os.path.join(DataPath, "opens5.h5"),
        os.path.join(DataPath, "rectrees1.h5")
    ]

    # train_path = os.path.join(ModelPath, "train_indices.npy")
    val_path   = os.path.join(ModelPath, "val_indices.npy")

    # train_indices = np.load(train_path) #load from previous training run
    val_indices   = np.load(val_path)

    #load data
    full_dataset = VLSIOpenMaskDataset(h5_paths, noise_gaussian=noise_gaussian, noise_salt_and_pepper=noise_salt_and_pepper, noise_blur=noise_blur, noise_prob=noise_prob, gaussian_std=gaussian_std, salt_and_pepper_amount=salt_and_pepper_amount, kernel_size=kernel_size)
    # val_indices   = list(range(len(full_dataset)*0.9, len(full_dataset)))
    val_dataset = Subset(full_dataset, val_indices)
    
    # Visualize random validation examples
    if Fails:
        visualize_failure_cases(model, val_dataset, "cuda" if torch.cuda.is_available() else "cpu", ModelPath)
    else:
        visualize_predictions_grid(model, val_dataset, "cuda" if torch.cuda.is_available() else "cpu", ModelPath)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and sample images from the dataset")

    parser.add_argument("ModelPath", type=str, help="Path to model statedict")
    parser.add_argument("DataDir", type=str, help="path to h5 directory")
    parser.add_argument("--Failure_Cases", action='store_true', help="Optional - Sample from the model failure cases", default=False)
    # parser.add_argument("--NumSamples", type=int, help="Optional - Number of images to sample", default=3)
    parser.add_argument("--Noise_Gaussian", action='store_true', help="Optional - Add Gaussian noise to images during training", default=False)
    parser.add_argument("--Noise_SaltAndPepper", action='store_true', help="Optional - Add Salt and Pepper noise to images during training", default=False)
    parser.add_argument("--Noise_Blur", action='store_true', help="Optional - Add Gaussian Blur to images during training", default=False)
    parser.add_argument("--Noise_Prob", type=float, help="Optional - Probability of adding noise to each image during training", default=0.0)
    parser.add_argument("--Gaussian_Std", type=float, help="Optional - Standard deviation of Gaussian noise", default=0.1)
    parser.add_argument("--SaltAndPepper_Amount", type=float, help="Optional - Amount of Salt and Pepper noise", default=0.01)
    parser.add_argument("--Blur_KernelSize", type=int, help="Optional - Kernel size for Gaussian Blur (must be odd)", default=3)


    args = parser.parse_args()
    TestRun(args.ModelPath, args.DataDir, args.Failure_Cases, args.Noise_Gaussian, args.Noise_SaltAndPepper, args.Noise_Blur, args.Noise_Prob, args.Gaussian_Std, args.SaltAndPepper_Amount, (args.Blur_KernelSize, args.Blur_KernelSize))
