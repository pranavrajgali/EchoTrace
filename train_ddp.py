"""
EchoTrace DDP Training Script
4× RTX 2080 Ti | float16 mixed precision | ImageNet init (no warm-start)
Absolute paths — data at /home/jovyan/work/data/
"""
import os
import glob
import time
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# ── Librosa thread control (critical for 16 worker processes) ──
# Prevent worker thread explosion: each worker uses 1 thread max
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from core.model import EchoTraceResNet, get_loss, get_optimizer
from core.preprocess import ASVDataset, WaveFakeDataset, InTheWildDataset, LibriSpeechDataset, build_combined_dataset

# ── Evaluation imports ──
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, average_precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ── Config ────────────────────────────────────────────────────
WORLD_SIZE     = min(4, torch.cuda.device_count())

# Absolute paths
ASV_PROTOCOL    = "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASV_DIR         = "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_train/flac"
WAVEFAKE_DIR    = "/home/jovyan/work/data/wavefake-test/wavefake-test"
ITW_DIR         = "/home/jovyan/work/data/release_in_the_wild/release_in_the_wild"
LIBRISPEECH_DIR = "/home/jovyan/work/data/LibriSpeech"

CHECKPOINT_DIR = "/home/jovyan/work/EchoTrace/checkpoints"
FINAL_PATH     = "/home/jovyan/work/EchoTrace/ensemble_model.pth"
LOG_PATH       = "/home/jovyan/work/EchoTrace/ddp_train.log"

BATCH_PER_GPU  = 32

# ── TRAINING CONFIG (edit these values for different runs) ──
NUM_EPOCHS         = 3
AUGMENT_PROB       = 0.2

# Dataset subset sizes
ASV_SUBSET         = 10000
WAVEFAKE_SUBSET    = 50000
ITW_SUBSET         = 12000
LIBRISPEECH_SUBSET = 28000

# Validation set size (from InTheWild val split)
VAL_SIZE           = 3000


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
    os.environ["MASTER_PORT"]      = "12365"
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
        subset_size=ASV_SUBSET,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    logger.info("Loading WaveFake dataset ...")
    wf = WaveFakeDataset(
        data_dir=WAVEFAKE_DIR,
        subset_size=WAVEFAKE_SUBSET,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    logger.info("Loading InTheWild dataset ...")
    itw = InTheWildDataset(
        data_dir=ITW_DIR,
        subset="train",
        subset_size=ITW_SUBSET,
        augment=True,
        augment_prob=AUGMENT_PROB,
    )

    logger.info("Loading LibriSpeech dataset ...")
    librispeech = LibriSpeechDataset(
        data_dir=LIBRISPEECH_DIR,
        subset_size=LIBRISPEECH_SUBSET,
        augment=True,
        augment_prob=0.5,
    )

    dataset = build_combined_dataset(asv, wf, itw, librispeech)
    logger.info(f"Total training samples: {len(dataset)}")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_PER_GPU,
        sampler=sampler,
        num_workers=6,           # Increased to 6 per GPU for 100k run
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, # don't restart workers between epochs
        prefetch_factor=2,       # each worker prefetches 2 batches ahead
    )
    return loader, sampler


# ── Validation DataLoader ─────────────────────────────────────
def get_val_loader(rank, world_size, logger):
    """
    Create validation loader using InTheWild 'val' split.
    No augmentation on validation data.
    """
    logger.info("Loading InTheWild validation dataset ...")
    val_dataset = InTheWildDataset(
        data_dir=ITW_DIR,
        subset="val",
        subset_size=VAL_SIZE,
        augment=False,  # no augmentation during validation
        augment_prob=0.0,
    )
    
    # Use SequentialSampler for validation (no shuffling, reproducible)
    from torch.utils.data import SequentialSampler
    sampler = SequentialSampler(val_dataset)
    
    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_PER_GPU,
        sampler=sampler,
        num_workers=2,      # Less intensive than training
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    logger.info(f"Validation samples: {len(val_dataset)}")
    return loader


# ── Evaluation Function ───────────────────────────────────────
def evaluate(model, val_loader, device, criterion):
    """
    Run validation pass and compute metrics.
    Returns: val_loss, balanced_accuracy, eer
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    val_loss = 0.0

    with torch.no_grad():
        for images, scalars, labels in val_loader:
            images = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss = criterion(outputs, labels.unsqueeze(1).float())

            val_loss += loss.item()

            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            labels_np = labels.cpu().numpy()

            # Handle single sample
            if probabilities.ndim == 0:
                probabilities = np.array([probabilities])
                predictions = np.array([predictions])

            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
            all_probabilities.extend(probabilities)

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Metrics
    val_loss /= len(val_loader)
    accuracy = np.mean(all_predictions == all_labels) * 100
    cm = confusion_matrix(all_labels, all_predictions)

    # Per-class recall
    real_recall = (cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fake_recall = (cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    balanced_acc = (real_recall + fake_recall) / 2

    # EER with boundary protection
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        fpr_clipped = np.clip(fpr, 1e-6, 1 - 1e-6)
        fnr_clipped = np.clip(fnr, 1e-6, 1 - 1e-6)
        sort_idx = np.argsort(fpr_clipped)
        fpr_sorted = fpr_clipped[sort_idx]
        fnr_sorted = fnr_clipped[sort_idx]
        _, unique_idx = np.unique(fpr_sorted, return_index=True)
        fpr_unique = fpr_sorted[unique_idx]
        fnr_unique = fnr_sorted[unique_idx]
        eer_fraction = brentq(
            lambda x: x - interp1d(fpr_unique, fnr_unique,
                                    bounds_error=False,
                                    fill_value=(fnr_unique[0], fnr_unique[-1]))(x),
            fpr_unique[0], fpr_unique[-1]
        )
        eer = eer_fraction * 100
    except Exception as e:
        roc_auc = 0.0
        eer = None

    return val_loss, balanced_acc, real_recall, fake_recall, eer, roc_auc


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

    torch.backends.cudnn.benchmark = True # Free CNN acceleration
    model     = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # Removed pos_weight downweighting of fakes to improve recall
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model.module)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    loader, sampler = get_loader(rank, world_size, logger)
    val_loader = get_val_loader(rank, world_size, logger) if rank == 0 else None

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger.info("=" * 60)
        logger.info("  EchoTrace DDP Training")
        logger.info(f"  GPUs           : {world_size}")
        logger.info(f"  Batch/GPU      : {BATCH_PER_GPU}")
        logger.info(f"  Effective batch: {BATCH_PER_GPU * world_size}")
        logger.info(f"  Epochs         : {NUM_EPOCHS}")
        logger.info(f"  Aug Prob       : {AUGMENT_PROB}")
        logger.info(f"  Scheduler      : CosineAnnealingLR (T_max={NUM_EPOCHS})")
        logger.info("=" * 60)

    best_loss = float("inf")
    start_epoch = 0

    # Checkpoint resumption: load latest checkpoint on EVERY rank
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.module.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            if rank == 0:
                logger.info(f"Resumed from checkpoint: {latest_ckpt}")
                logger.info(f"Starting from epoch {start_epoch + 1}")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to load checkpoint: {e}")
    
    # Broadcast resumption state to all ranks
    if world_size > 1:
        start_epoch_tensor = torch.tensor([start_epoch], device=device, dtype=torch.long)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()

    for epoch in range(start_epoch, NUM_EPOCHS):
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
            
            logger.info(
                f"[epoch {epoch+1:02d}] train_loss={avg:.4f} | "
                f"time={elapsed:.1f}m"
            )

            # ── Validation ──
            if val_loader is not None:
                model.eval()
                val_loss, val_bal_acc, val_real_recall, val_fake_recall, val_eer, val_roc_auc = evaluate(
                    model.module, val_loader, device, criterion
                )
                model.train()
                
                eer_str = f"{val_eer:.4f}%" if val_eer is not None else "N/A"
                logger.info(
                    f"[val    {epoch+1:02d}] val_loss={val_loss:.4f} | "
                    f"bal_acc={val_bal_acc:.2f}% | "
                    f"real_recall={val_real_recall:.2f}% | "
                    f"fake_recall={val_fake_recall:.2f}% | "
                    f"eer={eer_str} | "
                    f"roc_auc={val_roc_auc:.4f}"
                )

            # Save full checkpoint (model + optimizer + scheduler + epoch + best_loss)
            checkpoint = {
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:02d}.pth")
            torch.save(checkpoint, ckpt_path)
            
            # Also save final model weights for inference
            torch.save(model.module.state_dict(), FINAL_PATH)
            logger.info(f"Saved checkpoint -> {ckpt_path}")

        # Ensure all ranks wait for Rank 0 to finish evaluation and saving
        if world_size > 1:
            dist.barrier()

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