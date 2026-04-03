"""
EchoTrace DDP Training Script
4× RTX 2080 Ti | float16 mixed precision | ImageNet init (no warm-start)
Absolute paths — data at /home/jovyan/work/data/
"""
import os
import time
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from core.model import EchoTraceResNet, get_loss, get_optimizer
from core.preprocess import ASVDataset, WaveFakeDataset, InTheWildDataset, MultiDataset

# ── Config ────────────────────────────────────────────────────
WORLD_SIZE     = min(4, torch.cuda.device_count())

# Absolute paths
ASV_PROTOCOL   = "/home/jovyan/work/data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASV_DIR        = "/home/jovyan/work/data/asvspoof2019/LA/ASVspoof2019_LA_train/flac"
WAVEFAKE_DIR   = "/home/jovyan/work/data/wavefake/wavefake-test"
ITW_DIR        = "/home/jovyan/work/data/in_the_wild/release_in_the_wild"

CHECKPOINT_DIR = "/home/jovyan/work/EchoTrace/checkpoints"
FINAL_PATH     = "/home/jovyan/work/EchoTrace/ensemble_model.pth"
LOG_PATH       = "/home/jovyan/work/EchoTrace/ddp_train.log"

BATCH_PER_GPU  = 8      # 8 × 4 GPUs = 32 effective batch size
NUM_EPOCHS     = 10
SUBSET_SIZE    = None   # None = full dataset
AUGMENT_PROB   = 0.3


# ── Logging ───────────────────────────────────────────────────
import logging, sys

def get_logger(rank):
    logger = logging.getLogger(f"EchoTrace.rank{rank}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"[%(asctime)s][rank{rank}] %(message)s", "%H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0:
        fh = logging.FileHandler(LOG_PATH, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── DDP setup / teardown ──────────────────────────────────────
def setup(rank, world_size):
    os.environ["MASTER_ADDR"]      = "localhost"
    os.environ["MASTER_PORT"]      = "12359"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"]  = "1"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=60),
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


# ── DataLoader ────────────────────────────────────────────────
def get_loader(rank, world_size, logger):
    logger.info("Loading ASVspoof dataset ...")
    asv = ASVDataset(
        protocol_file=ASV_PROTOCOL,
        data_dir=ASV_DIR,
        subset_size=SUBSET_SIZE,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    logger.info("Loading WaveFake dataset ...")
    wf = WaveFakeDataset(
        data_dir=WAVEFAKE_DIR,
        subset_size=SUBSET_SIZE,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    logger.info("Loading InTheWild dataset ...")
    itw = InTheWildDataset(
        data_dir=ITW_DIR,
        subset="train",
        subset_size=SUBSET_SIZE,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    dataset = MultiDataset(asv, wf, itw)
    logger.info(f"Total training samples: {len(dataset)}")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_PER_GPU,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return loader, sampler


# ── Training process ──────────────────────────────────────────
def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    logger = get_logger(rank)

    # Model: ImageNet init, freeze applied inside __init__
    model = EchoTraceResNet(num_scalars=8).to(device)

    if rank == 0:
        l1 = any(p.requires_grad for p in model.resnet.layer1.parameters())
        l2 = any(p.requires_grad for p in model.resnet.layer2.parameters())
        l3 = any(p.requires_grad for p in model.resnet.layer3.parameters())
        l4 = any(p.requires_grad for p in model.resnet.layer4.parameters())
        fc = any(p.requires_grad for p in model.fc.parameters())
        logger.info(f"Freeze check  -- L1:{l1} L2:{l2} L3:{l3} L4:{l4} FC:{fc}")
        logger.info("Expected      -- L1:False L2:False L3:False L4:True FC:True")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,}")

    model     = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = get_loss()
    optimizer = get_optimizer(model.module)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    loader, sampler = get_loader(rank, world_size, logger)

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger.info("=" * 60)
        logger.info("  EchoTrace DDP Training")
        logger.info(f"  GPUs           : {world_size}")
        logger.info(f"  Batch/GPU      : {BATCH_PER_GPU}")
        logger.info(f"  Effective batch: {BATCH_PER_GPU * world_size}")
        logger.info(f"  Epochs         : {NUM_EPOCHS}")
        logger.info("=" * 60)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for batch_idx, (images, scalars, labels) in enumerate(loader):
            images  = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels  = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if rank == 0 and batch_idx % 50 == 0:
                pct = batch_idx / len(loader) * 100
                logger.info(
                    f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx:4d}/{len(loader)} ({pct:4.1f}%) | "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()

        if rank == 0:
            avg     = epoch_loss / len(loader)
            elapsed = (time.time() - t0) / 60
            saved   = " <- best" if avg < best_loss else ""
            if avg < best_loss:
                best_loss = avg

            logger.info(
                f"[epoch {epoch+1:02d}] avg_loss={avg:.4f} | "
                f"time={elapsed:.1f}m | best={best_loss:.4f}{saved}"
            )

            ckpt = os.path.join(CHECKPOINT_DIR, f"echo_epoch_{epoch+1:02d}.pth")
            torch.save(model.module.state_dict(), ckpt)
            torch.save(model.module.state_dict(), FINAL_PATH)
            logger.info(f"Saved -> {ckpt}")

    cleanup()
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training complete.")
        logger.info(f"Final model: {FINAL_PATH}")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    assert torch.cuda.is_available(), "No CUDA GPUs found."
    assert WORLD_SIZE > 0,            "WORLD_SIZE must be > 0."
    print(f"[launch] {WORLD_SIZE} GPU(s) detected. Spawning processes ...")
    print(f"[launch] Log -> {LOG_PATH}")
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)