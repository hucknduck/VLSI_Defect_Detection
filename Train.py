import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm import tqdm
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from NewModel import DiscreteCenterPredictorCNN
from NewDataSet import VLSIOpenMaskDataset

matplotlib.use('Agg')

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def Train(DataDir, OutputDir, NumEpochs, TrainTestSplit, CheckPointPath, local_rank):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    print(f"[Rank {rank}] Using device: {device}")
    
    
    # Set device
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(f"Using device: {device}")

    # Point to the exact Drive paths under DataDir:
    h5_paths = [
        os.path.join(DataDir, "opens1.h5"),
        os.path.join(DataDir, "opens2.h5"),
        os.path.join(DataDir, "opens3.h5"),
        os.path.join(DataDir, "opens4.h5"),
        os.path.join(DataDir, "opens5.h5"),
        os.path.join(DataDir, "rectrees1.h5")
    ]

    full_dataset = VLSIOpenMaskDataset(h5_paths)
    print("Number of samples in full_dataset:", len(full_dataset))

    # Split into train/validation
    num_total = len(full_dataset)
    indices = list(range(num_total))
    random.seed(42)
    random.shuffle(indices)
    split = int(TrainTestSplit * num_total)

    train_indices = indices[:split]
    val_indices   = indices[split:]

    #Save indices for reproducibility and evaluation
    if rank == 0:
        if not os.path.exists(OutputDir):
            os.makedirs(OutputDir)
        TrainPath = os.path.join(OutputDir, "train_indices.npy")
        ValPath   = os.path.join(OutputDir, "val_indices.npy") 
        np.save(TrainPath, train_indices)
        np.save(ValPath, val_indices)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

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
        batch_size=512,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader   = DataLoader(
        val_dataset,
        batch_size=512,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    print("Number of samples in train_loader:", len(train_loader))
    print("Number of samples in val_loader:", len(val_loader))

    model = DiscreteCenterPredictorCNN()
    model.to(torch.device(f"cuda:{local_rank}"))
    if CheckPointPath != None:
       map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
       model.load_state_dict(torch.load(CheckPointPath, map_location=map_location))
    
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    
    best_val_loss = float('inf')
    BatchLoss = []
    EpochLoss = []
    ValLoss = []

    for epoch in range(NumEpochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1}/{NumEpochs}")):
            images = images.to(device, non_blocking=True)
            labels = targets['labels'].squeeze(1).to(device, non_blocking=True)

            logits = model(images).float()
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0:
                BatchLoss.append(loss.item())

        lr_scheduler.step()
        avg_loss = running_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{NumEpochs}]  Training Loss: {avg_loss:.4f}")
            EpochLoss.append(avg_loss)

        # Validation loop (compute losses only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{NumEpochs}")):
                images = images.to(device, non_blocking=True)
                labels = targets['labels'].squeeze(1).to(device, non_blocking=True)
                
                logits = model(images).float()
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        if rank == 0:
            ValLoss.append(val_loss)
            print(f"→ Validation Loss: {val_loss:.4f}")

            # Save checkpoint if this is the best validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(OutputDir, f"Best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"*** Saved Best Model (Epoch {epoch+1}) with val_loss={val_loss:.4f} ***")
    if rank == 0:
        # === Plot 1: Batch Loss ===
        plt.figure(figsize=(10, 5))
        plt.plot(BatchLoss, label='Batch Loss', color='blue')
        plt.title('Training Loss per Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        # plt.legend()
        plt.tight_layout()
        plot1_path = os.path.join(OutputDir, f"BatchLoss.png")
        plt.savefig(plot1_path)
        # plt.show()

        # === Plot 2: Epoch vs Validation Loss ===
        plt.figure(figsize=(10, 5))
        plt.plot(EpochLoss, label='Epoch Loss', color='blue')
        plt.plot(ValLoss, label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot2_path = os.path.join(OutputDir, f"EpochLoss.png")
        plt.savefig(plot2_path)
        # plt.show()

        # At the end, also save the final epoch model
        final_path = os.path.join(OutputDir, f"Final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Saved Final Model at '{final_path}'")

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model")

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
            print(f"Device {i} capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No CUDA devices found by PyTorch.")

    local_rank = setup_ddp()

    parser.add_argument("DataDir", type=str, help="Path to h5 dir")
    parser.add_argument("OutputDir", type=str, help="Path to dir for outputs")
    parser.add_argument("--NumEpochs", type=int, help="Optional - Num epochs to train", default=20)
    parser.add_argument("--CheckpointPath", type=str, help="Optional - Path to checkpoint state dict", default=None)
    parser.add_argument("--TrainTestSplit", type=float, help="Optional - Part of the data that will be set in train dataset", default=0.9)
    parser.add_argument("--local_rank", type=int, help="Optional - Local rank passed from torchrun", default=0)
    

    args = parser.parse_args()
    Train(args.DataDir, args.OutputDir, args.NumEpochs, args.TrainTestSplit, args.CheckpointPath, local_rank)