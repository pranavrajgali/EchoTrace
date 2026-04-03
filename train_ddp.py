import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Import your custom modules
from core.model import EchoTraceResNet, build_model, get_loss, get_optimizer
from core.preprocess import ASVDataset, WaveFakeDataset, InTheWildDataset, MultiDataset

# --- Configuration ---
WORLD_SIZE = min(4, torch.cuda.device_count())
assert WORLD_SIZE > 0, "No CUDA GPUs found"

# Dataset Directories
ASV_PROTOCOL = "../data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASV_DIR = "../data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/"
WAVEFAKE_DIR = "../data/wavefake-test/"
ITW_DIR = "../data/release_in_the_wild/"

# Training Hyper-parameters
BATCH_SIZE_PER_GPU = 16
NUM_EPOCHS = 10
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
    asv_data = ASVDataset(ASV_PROTOCOL, ASV_DIR, subset_size=None, augment=True, augment_prob=0.3)
    wavefake_data = WaveFakeDataset(WAVEFAKE_DIR, subset_size=None, augment=True, augment_prob=0.3)
    itw_data = InTheWildDataset(ITW_DIR, subset_size=None, augment=True, augment_prob=0.3)
    
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
    device = torch.device(f"cuda:{rank}")
    
    # Build model with ImageNet weights and explicit layer freezing
    model = build_model(device)
    
    # Belt-and-suspenders: explicitly freeze layer1 and layer2 again
    for param in model.resnet.layer1.parameters():
        param.requires_grad = False
    for param in model.resnet.layer2.parameters():
        param.requires_grad = False
    
    # First, freeze ALL parameters
    for param in model.resnet.parameters():
        param.requires_grad = False
    
    # Then selectively unfreeze layer3, layer4, and fc head
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Get loss and optimizer from model functions
    criterion = get_loss()
    optimizer = get_optimizer(model.module)
    
    scaler = torch.amp.GradScaler("cuda")
    loader, sampler = get_dataloaders(rank, world_size)
    
    # Create checkpoint directory if not exists
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"[*] Rank {rank}: Starting DDP Training Loop...")
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        if epoch == 0 and rank == 0:
            print(f"[Epoch 0] Training from scratch with ImageNet backbone + new FC head")
            
        for batch_idx, (images, scalars, labels) in enumerate(loader):
            images = images.to(device)
            scalars = scalars.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS-1}] | Batch [{batch_idx}/{len(loader)}] | Loss: {loss.item():.4f}")

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            epoch_elapsed = (time.time() - epoch_start_time) / 60
            print(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_elapsed:.2f} min ===")
            
            save_path = os.path.join(SAVE_DIR, f"ensemble_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), save_path)
            torch.save(model.module.state_dict(), "ensemble_model.pth")
            print(f"[*] Saved checkpoint to {save_path}")

    cleanup_ddp()


if __name__ == "__main__":
    try:
        mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    except Exception as e:
        print(f"[!] DDP Training crashed: {e}")