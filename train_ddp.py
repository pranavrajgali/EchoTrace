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
SAVE_DIR = 'checkpoints'

# Dataset Paths (relative to ~/work/EchoTrace/)
ASV_PROTOCOL = '../data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
ASV_DIR = '../data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/'
WAVEFAKE_DIR = '../data/wavefake/wavefake-test/'
ITW_DIR = '../data/in_the_wild/release_in_the_wild/'

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    # Stability Fixes for Jupyter/Shared environments
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

def prepare_dataloader(rank, world_size):
    """Builds the combined dataset and wraps it in a DistributedSampler."""
    if rank == 0:
        print("[*] Initializing Datasets...")
    
    # 1. Initialize individual datasets (Augmentation ON for training)
    asv_data = ASVDataset(ASV_PROTOCOL, ASV_DIR, augment=True)
    wavefake_data = WaveFakeDataset(WAVEFAKE_DIR)
    itw_data = InTheWildDataset(ITW_DIR, subset='train')
    
    # 2. Combine using the round-robin MultiDataset
    train_dataset = MultiDataset(asv_data, wavefake_data, itw_data)
    
    # 3. Distributed Sampler ensures each GPU gets a unique subset of data
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # 4. DataLoader
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
    device = torch.device(f"cuda:{rank}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    loader, sampler = prepare_dataloader(rank, world_size)
    
    # --- Model Initialization & Warm Start ---
    model = build_model(device)
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
    # FC is unfrozen by default upon instantiation

    # --- DDP Wrapping ---
    model = DDP(model, device_ids=[rank], output_device=rank)

    # --- Optimizer Setup ---
    # Note: Access attributes via `model.module` after DDP wrapping
    optimizer = torch.optim.Adam([
        {'params': model.module.resnet.layer3.parameters(), 'lr': 1e-6},
        {'params': model.module.resnet.layer4.parameters(), 'lr': 1e-5},
        {'params': model.module.fc.parameters(), 'lr': 1e-4}
    ])
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler("cuda") # For Mixed Precision (FP16)

    # --- Training Loop ---
    if rank == 0:
        print("[*] Starting DDP Training Loop...")
        start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        sampler.set_epoch(epoch) # Crucial for shuffling in DDP
        
        # Layer 3 Warmup Logic (First epoch only)
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = 1e-7
            if rank == 0:
                print(f"[Epoch 0] Warmup active: Layer 3 LR set to 1e-7")
        elif epoch == 1:
            optimizer.param_groups[0]['lr'] = 1e-6
            if rank == 0:
                print(f"[Epoch 1] Warmup complete: Layer 3 LR restored to 1e-6")

        epoch_loss = 0.0
        
        for batch_idx, (images, scalars, labels) in enumerate(loader):
            # Move data to specific GPU
            images = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with autocast("cuda"):
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