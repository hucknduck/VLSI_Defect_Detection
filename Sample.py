import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
from Model import DiscreteCenterPredictorCNN
from DataSet import VLSIOpenMaskDataset
import random
from torch.utils.data import Subset
import argparse

def visualize_prediction(model, dataset, idx, device):
    model.eval()
    img_tensor, target = dataset[idx]
    image_id = target["image_id"].item()
    img = img_tensor.to(device)
    with torch.no_grad():
        out = model([img])[0]

    img_np = img_tensor.cpu().squeeze(0).numpy()  # (256,256)
    img_disp = np.stack([img_np]*3, axis=2)  # convert to 3-channel for display

    gt_box = target["boxes"][0].cpu().numpy()
    gt_mask = target["masks"][0].cpu().numpy()

    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    boxes = out["boxes"].cpu().numpy()
    masks = out["masks"].cpu().numpy()

    # Pick highest-score defect
    keep_idxs = np.where(labels == 1)[0]
    if keep_idxs.size == 0:
        print(f"No defect detected for image {image_id}")
        return
    best_idx = keep_idxs[np.argmax(scores[keep_idxs])]
    pred_box = boxes[best_idx]
    pred_mask = (masks[best_idx,0] >= 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1,2,figsize=(12,6))
    # Ground truth
    axes[0].imshow(img_disp, cmap='gray')
    x0,y0,x1,y1 = gt_box
    rect_gt = plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='g', linewidth=2)
    axes[0].add_patch(rect_gt)
    axes[0].imshow(gt_mask, cmap='Greens', alpha=0.4)
    axes[0].set_title(f"Image {image_id} — Ground Truth")
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(img_disp, cmap='gray')
    x0p,y0p,x1p,y1p = pred_box
    rect_pred = plt.Rectangle((x0p,y0p), x1p-x0p, y1p-y0p, fill=False, edgecolor='r', linewidth=2)
    axes[1].add_patch(rect_pred)
    mask_up = TF.resize(torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float(),
                        size=[256,256], interpolation=TF.InterpolationMode.NEAREST).squeeze().numpy()
    axes[1].imshow(mask_up, cmap='Reds', alpha=0.4)
    axes[1].set_title(f"Image {image_id} — Prediction (score={scores[best_idx]:.2f})")
    axes[1].axis('off')

    plt.show()


def TestRun(ModelPath, DataPath, NumSamples):
    #load model
    model = DiscreteCenterPredictorCNN()
    model.load_state_dict(torch.load(ModelPath))
    model.eval()

    #load data
    full_dataset = VLSIOpenMaskDataset(DataPath)
    val_indices   = list(range(len(full_dataset)*0.9, len(full_dataset)))
    val_dataset = Subset(full_dataset, val_indices)
    
    # Visualize random validation examples
    for _ in range(NumSamples):
        idx = random.randrange(len(val_dataset))
        visualize_prediction(model, val_dataset, idx, "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and sample images from the dataset")

    parser.add_argument("ModelPath", type=str, help="Path to model statedict")
    parser.add_argument("DataDir", type=str, help="path to h5 directory")
    parser.add_argument("--NumSamples", type=int, help="Optional - Number of images to sample", default=3)

    args = parser.parse_args()
    TestRun(args.ModelPath, args.DataDir, args.NumSamples)
