import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Import your custom modules
from core.model import build_model, warm_start_new_pipeline
from core.preprocess import ASVDataset, WaveFakeDataset, InTheWildDataset, MultiDataset

# --- Configuration ---
WORLD_SIZE = 4
BATCH_SIZE_PER_GPU = 8  # 8 * 4 = 32 effective batch size
EPOCHS = 10
CHECKPOINT_PATH = 'deepfake_detector.pth'
# Update WORLD_SIZE to handle actual GPU count
WORLD_SIZE = min(4, torch.cuda.device_count())

# Dataset Directories
ASV_DIR = "../data/ASVspoof2019/LA/flac/"
WAVEFAKE_DIR = "../data/wavefake-test/"
ITW_DIR = "../data/release_in_the_wild/"
CHECKPOINT_PATH = "deepfake_detector.pth"

# Training Hyper-parameters
BATCH_SIZE_PER_GPU = 16
NUM_EPOCHS = 10
LR_BACKBONE = 1e-6 # Layers 3-4
LR_HEAD = 1e-4     # New FC Head
SAVE_DIR = 'checkpoints'

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    # Stability Fixes for Jupyter/Shared environments
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def get_dataloaders(rank, world_size):
    print(f"[*] Rank {rank}: Initializing Datasets...")
    asv_data = ASVDataset(ASV_DIR)
    wavefake_data = WaveFakeDataset(WAVEFAKE_DIR)
    itw_data = InTheWildDataset(ITW_DIR)
    
    train_dataset = MultiDataset(asv_data, wavefake_data, itw_data)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE_PER_GPU, 
        sampler=sampler, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    return loader, sampler

def train(rank, world_size):
    setup_ddp(rank, world_size)
    torch.manual_seed(42)
    device = rank
    
    model = EchoTraceResNet()
    if os.path.exists(CHECKPOINT_PATH):
        if rank == 0:
            print(f"[*] Warm starting from {CHECKPOINT_PATH}")
        model = warm_start_new_pipeline(model, CHECKPOINT_PATH, device)
    
    # --- Strategy 2: Differential Unfreezing ---
    # Layers 1 & 2 remain frozen (default from ImageNet init)
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model = DDP(model, device_ids=[rank])
    
    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.module.resnet.layer3.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.resnet.layer4.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.fc.parameters(), 'lr': LR_HEAD}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    loader, sampler = get_dataloaders(rank, world_size)
    
    print(f"[*] Rank {rank}: Starting DDP Training Loop...")
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        
        # Warmup Check for first epoch
        if epoch == 0 and rank == 0:
            print(f"[Epoch 0] Warmup active: Layer 3 LR set to {LR_BACKBONE}")
            
        for batch_idx, (images, scalars, labels) in enumerate(loader):
            images = images.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss = criterion(outputs, labels)

            # Scaled Backward Pass
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step & Scaler update
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Logging on Master Node (Rank 0)
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{EPOCHS-1}] | Batch [{batch_idx}/{len(loader)}] | Loss: {loss.item():.4f}")

        # --- Epoch Conclusion & Checkpointing ---
        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {(time.time()-start_time)/60:.2f} min ===")
            
            # Save Checkpoint
            save_path = os.path.join(SAVE_DIR, f"ensemble_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), save_path)
            
            # Keep latest as ensemble_model.pth
            torch.save(model.module.state_dict(), "ensemble_model.pth")
            print(f"[*] Saved checkpoint to {save_path}")

    cleanup_ddp()

if __name__ == "__main__":
    try:
        mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    except Exception as e:
        print(f"[!] DDP Training crashed: {e}")