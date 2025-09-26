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

from NewModel import DiscreteCenterPredictorCNN, CNNTiny, CNNSmall, CNNMedium, CNNLarge
from NewDataSet import VLSIOpenMaskDataset

matplotlib.use('Agg')

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def Train(DataDir, OutputDir, NumEpochs, TrainTestSplit, CheckPointPath, local_rank, argv):
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

    full_dataset = VLSIOpenMaskDataset(h5_paths, argv.Noise_Gaussian, argv.Noise_SaltAndPepper, argv.Noise_Blur, argv.Noise_Prob, argv.Gaussian_Std, argv.SaltAndPepper_Amount, (argv.Blur_KernelSize, argv.Blur_KernelSize))
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

    modeltypes = [CNNTiny, CNNSmall, CNNMedium, CNNLarge]
    BatchLossPerModel = {}
    EpochLossPerModel = {}
    ValLossPerModel = {}
    BestValLossPerModel = {}

    for modeltype in modeltypes:
        model = modeltype()
        model.to(torch.device(f"cuda:{local_rank}"))
        if CheckPointPath != None:
            CheckPointPathModel = os.path.join(CheckPointPath, modeltype.__name__, "Best.pth")
            print(f"Loading checkpoint from {CheckPointPathModel}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            model.load_state_dict(torch.load(CheckPointPath, map_location=map_location))
    
        model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        OutputDirModel = os.path.join(OutputDir, modeltype.__name__)
        if rank == 0:
            if not os.path.exists(OutputDirModel):
                os.makedirs(OutputDirModel)

        best_val_loss = float('inf')
        BatchLoss = []
        EpochLoss = []
        ValLoss = []

        for epoch in range(NumEpochs):
            model.train()
            train_sampler.set_epoch(epoch)
            running_loss = 0.0

            for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"[Rank {rank}] {modeltype.__name__}, Epoch {epoch+1}/{NumEpochs}")):
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
                    best_path = os.path.join(OutputDirModel, f"Best.pth")
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
            plot1_path = os.path.join(OutputDirModel, f"BatchLoss.png")
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
            plot2_path = os.path.join(OutputDirModel, f"EpochLoss.png")
            plt.savefig(plot2_path)
            # plt.show()

            # At the end, also save the final epoch model
            final_path = os.path.join(OutputDirModel, f"Final.pth")
            torch.save(model.state_dict(), final_path)
            print(f"Saved Final Model at '{final_path}'")

            BatchLossPerModel[modeltype.__name__] = BatchLoss
            EpochLossPerModel[modeltype.__name__] = EpochLoss
            ValLossPerModel[modeltype.__name__] = ValLoss
            BestValLossPerModel[modeltype.__name__] = best_val_loss
    
    if rank == 0:
        # Plot comparison of models
        plt.figure(figsize=(10, 5))
        # model_names = ["CNNTiny", "CNNSmall", "CNNMedium", "CNNLarge"]
        for name, EpochLoss in EpochLossPerModel.items():
            plt.plot(EpochLoss, label=name)
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        comparison_path = os.path.join(OutputDir, f"ModelComparison_EpochLoss.png")
        plt.savefig(comparison_path)

        # Print best validation losses
        # print("Best Validation Losses per Model:")
        plt.figure(figsize=(10, 5))
        for name, val_loss in ValLossPerModel.items():
            plt.plot(val_loss, label=name)
        plt.title('Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        comparison_path = os.path.join(OutputDir, f"ModelComparison_ValidationLoss.png")
        plt.savefig(comparison_path)

        # Print batch losses
        # print("Best Validation Losses per Model:")
        plt.figure(figsize=(10, 5))
        for name, batch_loss in BatchLossPerModel.items():
            plt.plot(batch_loss, label=name)
        plt.title('Training Loss per Batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        comparison_path = os.path.join(OutputDir, f"ModelComparison_BatchLoss.png")
        plt.savefig(comparison_path)

        # Plot Training and validation loss comparison
        plt.figure(figsize=(10, 5))
        keys = list(EpochLossPerModel.keys())
        colors = ['blue', 'orange', 'green', 'red']
        for name, color in zip(keys, colors):
            plt.plot(ValLossPerModel[name], label=f"{name} Validation", linestyle='--', color=color)
            plt.plot(EpochLossPerModel[name], label=f"{name} Training", color=color)
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        comparison_path = os.path.join(OutputDir, f"ModelComparison.png")
        plt.savefig(comparison_path)

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
    parser.add_argument("--Noise_Gaussian", action='store_true', help="Optional - Add Gaussian noise to images during training", default=False)
    parser.add_argument("--Noise_SaltAndPepper", action='store_true', help="Optional - Add Salt and Pepper noise to images during training", default=False)
    parser.add_argument("--Noise_Blur", action='store_true', help="Optional - Add Gaussian Blur to images during training", default=False)
    parser.add_argument("--Noise_Prob", type=float, help="Optional - Probability of adding noise to each image during training", default=0.0)
    parser.add_argument("--Gaussian_Std", type=float, help="Optional - Standard deviation of Gaussian noise", default=0.1)
    parser.add_argument("--SaltAndPepper_Amount", type=float, help="Optional - Amount of Salt and Pepper noise", default=0.01)
    parser.add_argument("--Blur_KernelSize", type=int, help="Optional - Kernel size for Gaussian Blur (must be odd)", default=3)

    args = parser.parse_args()
    Train(args.DataDir, args.OutputDir, args.NumEpochs, args.TrainTestSplit, args.CheckpointPath, local_rank, args)