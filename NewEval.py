import h5py
import numpy as np
import torchvision.transforms.functional as TF
import torch
import torch.distributed as dist
import argparse
from Model import DiscreteCenterPredictorCNN
from NewDataSet import VLSIOpenMaskDataset
from torch.utils.data import Subset, DataLoader, DistributedSampler
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.nn.functional as F

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def Eval(ModelPath, OutputDir, DataPath, local_rank):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    print(f"[Rank {rank}] Using device: {device}")
    
    #load model
    model = DiscreteCenterPredictorCNN()
    model.to(torch.device(f"cuda:{local_rank}"))
    model = DDP(model, device_ids=[local_rank])
    model.load_state_dict(torch.load(ModelPath))
    model.eval()


    h5_paths = [
        os.path.join(DataPath, "opens1.h5"),
        os.path.join(DataPath, "opens2.h5"),
        os.path.join(DataPath, "opens3.h5"),
        os.path.join(DataPath, "opens4.h5"),
        os.path.join(DataPath, "opens5.h5"),
        os.path.join(DataPath, "rectrees1.h5")
    ]

    full_dataset = VLSIOpenMaskDataset(h5_paths)

    #load data
    split = int(len(full_dataset)*0.9)

    train_path = os.path.join(OutputDir, "train_indices.npy")
    val_path   = os.path.join(OutputDir, "val_indices.npy")

    train_indices = np.load(train_path) #load from previous training run
    val_indices   = np.load(val_path)

    # train_indices   = list(range(split))
    # val_indices   = list(range(split, len(full_dataset)))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    


    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler   = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    running_loss = 0
    train_total = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            labels = targets['labels'].squeeze(1).to(device, non_blocking=True)

            logits = model(images).float()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        # Validation loop (compute losses only)
        val_loss = 0.0
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)
            labels = targets['labels'].squeeze(1).to(device, non_blocking=True)
            
            logits = model(images).float()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_loss += (preds == labels).sum().item()
            train_total += labels.size(0)
            val_total += labels.size(0)



    running_loss += val_loss
    val_loss /= val_total
    running_loss /= train_total
    print(f"Model achieves {val_loss} test accuracy")
    print(f"Model achieves {running_loss} total accuracy")
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference to evaluate model accuracy")

    local_rank = setup_ddp()

    parser.add_argument("ModelPath", type=str, help="Path to model statedict")
    parser.add_argument("DataDir", type=str, help="path to h5 directory")

    args = parser.parse_args()
    Eval(args.ModelPath, args.DataDir, local_rank)