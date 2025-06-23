from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import h5py
import numpy as np
import torchvision.transforms.functional as TF
import torch
import json
import argparse
from Model import DiscreteCenterPredictorCNN
from DataSet import VLSIOpenMaskDataset
from torch.utils.data import Subset, DataLoader

def create_coco_gt_json(dataset_subset, out_json_path="/content/vlsi_val_gt.json"):
    images = []
    annotations = []
    ann_id = 1
    for idx_in_subset, global_idx in enumerate(dataset_subset.indices):
        file_idx, inner_idx = dataset_subset.dataset.index_map[global_idx]
        h5_path = dataset_subset.dataset.h5_paths[file_idx]
        with h5py.File(h5_path, 'r') as hf:
            box_np = hf[f'open_{inner_idx}'][:]
            mask48 = np.zeros((48,48), dtype=np.uint8)
            x1,y1,x2,y2 = box_np.astype(int)
            x1c, y1c = max(0,x1), max(0,y1)
            x2c, y2c = min(47,x2), min(47,y2)
            mask48[y1c:y2c, x1c:x2c] = 1

        img_info = {
            "id": global_idx,
            "width": 256,
            "height": 256,
            "file_name": f"{global_idx}.png"
        }
        images.append(img_info)

        # Upsample mask to 256
        mask256 = TF.resize(torch.from_numpy(mask48).unsqueeze(0).unsqueeze(0).float(),
                            size=[256,256], interpolation=TF.InterpolationMode.NEAREST).squeeze().numpy().astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask256))
        rle['counts'] = rle['counts'].decode('ascii')

        x1_s, y1_s, x2_s, y2_s = box_np * (256.0/48.0)
        w_box = x2_s - x1_s
        h_box = y2_s - y1_s

        ann = {
            "id": ann_id,
            "image_id": global_idx,
            "category_id": 1,
            "bbox": [float(x1_s), float(y1_s), float(w_box), float(h_box)],
            "area": float(w_box * h_box),
            "iscrowd": 0,
            "segmentation": rle
        }
        annotations.append(ann)
        ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id":1, "name":"defect", "supercategory":"none"}]
    }
    with open(out_json_path, 'w') as f:
        json.dump(coco_dict, f)
    return out_json_path

def get_coco_predictions(model, data_loader, device, pred_json_path="/content/vlsi_val_preds.json"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for tgt, out in zip(targets, outputs):
                image_id = tgt["image_id"].item()
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                masks  = out["masks"].cpu().numpy()  # (N,1,28,28)
                for i in range(boxes.shape[0]):
                    if labels[i] != 1:
                        continue
                    x1,y1,x2,y2 = boxes[i]
                    w_box = x2 - x1
                    h_box = y2 - y1
                    mask_pred = (masks[i,0] >= 0.5).astype(np.uint8)  # (28,28)
                    # Upsample mask_pred to 256×256
                    mask_up = TF.resize(torch.from_numpy(mask_pred).unsqueeze(0).unsqueeze(0).float(),
                                        size=[256,256], interpolation=TF.InterpolationMode.NEAREST).squeeze().numpy().astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask_up))
                    rle['counts'] = rle['counts'].decode('ascii')
                    pred = {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                        "score": float(scores[i]),
                        "segmentation": rle
                    }
                    all_preds.append(pred)
    with open(pred_json_path, 'w') as f:
        json.dump(all_preds, f)
    return pred_json_path


def Eval(ModelPath, DataPath):
    #load model
    model = DiscreteCenterPredictorCNN()
    model.load_state_dict(torch.load(ModelPath))
    model.eval()

    #load data
    full_dataset = VLSIOpenMaskDataset(DataPath)
    val_indices   = list(range(len(full_dataset)*0.9, len(full_dataset)))
    val_dataset = Subset(full_dataset, val_indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=2, # Reduced num_workers
        # collate_fn=collate_fn
    )

    gt_json = create_coco_gt_json(val_dataset, out_json_path="/content/vlsi_val_gt.json")
    pred_json = get_coco_predictions(model, val_loader, "cuda" if torch.cuda.is_available() else "cpu", pred_json_path="/content/vlsi_val_preds.json")

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval_segm = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval_segm.params.imgIds = list(coco_gt.getImgIds())
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()

    coco_eval_bbox = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval_bbox.params.imgIds = list(coco_gt.getImgIds())
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference to evaluate model accuracy")

    parser.add_argument("ModelPath", type=str, help="Path to model statedict")
    parser.add_argument("DataDir", type=str, help="path to h5 directory")

    args = parser.parse_args()
    Eval(args.ModelPath, args.DataDir)