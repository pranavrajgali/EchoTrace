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
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys, pathlib
_project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.model import EchoTraceResNet, FocalLoss, get_optimizer
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

# ── TRAINING CONFIG ──────────────────────────────────────────
NUM_EPOCHS         = 10
AUGMENT_PROB       = 0.5

# Dataset subset sizes (Balanced 50/50 Split — verified against on-disk counts)
# Real:  ASV 2,580 + WaveFake 18,100 + ITW 12,500 + LibriSpeech 38,000 = 71,180
# Fake:  ASV 22,800 + WaveFake 40,000 + ITW 8,271                      = 71,071
# Total: ~142,251 samples | Ratio: 50.04% real / 49.96% fake
ASV_SUBSET         = None     # ~25,380 samples (2,580 real + 22,800 fake)
WAVEFAKE_SUBSET    = 80000    # 18,100 real (disk limit) + 40,000 fake
ITW_SUBSET         = 25000    # 12,500 real + 8,271 fake (disk limit)
LIBRISPEECH_SUBSET = 38000    # 38,000 real (compensates for WaveFake/ITW shortfall)

# Validation set size (None = full split)
VAL_SIZE           = None


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
    os.environ["MASTER_PORT"]      = "12366"
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
    logger.info("Loading datasets (Full Budget) ...")
    asv = ASVDataset(
        protocol_file=ASV_PROTOCOL, data_dir=ASV_DIR,
        subset_size=ASV_SUBSET, augment=True, augment_prob=AUGMENT_PROB
    )
    wf = WaveFakeDataset(
        data_dir=WAVEFAKE_DIR, subset_size=WAVEFAKE_SUBSET,
        augment=True, augment_prob=AUGMENT_PROB
    )
    itw = InTheWildDataset(
        data_dir=ITW_DIR, subset="train", subset_size=ITW_SUBSET,
        augment=True, augment_prob=AUGMENT_PROB
    )
    librispeech = LibriSpeechDataset(
        data_dir=LIBRISPEECH_DIR, subset_size=LIBRISPEECH_SUBSET,
        augment=True, augment_prob=0.5
    )

    dataset = build_combined_dataset(asv, wf, itw, librispeech)
    logger.info(f"Total training samples: {len(dataset)} (Natural Ratio)")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_PER_GPU,
        sampler=sampler,
        num_workers=6,           # 6 workers per GPU = 24 total CPU workers across 4 GPUs
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, # Keeps workers alive between epochs (avoids respawn overhead)
        prefetch_factor=3,       # Pre-load 3 batches per worker for GPU saturation
    )
    return loader, sampler


# ── Validation DataLoader ─────────────────────────────────────
def get_val_loader(rank, world_size, logger):
    logger.info("Loading InTheWild validation dataset (Full) ...")
    val_dataset = InTheWildDataset(
        data_dir=ITW_DIR,
        subset="val",
        subset_size=VAL_SIZE, # Now loads full split
        augment=False,
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
    # ── CRITICAL: Seed ALL RNGs identically across ranks ──
    # Dataset constructors use Python's random.shuffle(), so we must seed
    # Python's random module too. Otherwise each rank builds a different
    # file list order, and DistributedSampler partitions different data.
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
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
    model     = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model     = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # FocalLoss with alpha=0.5 for balanced 50/50 dataset — focuses on hard examples
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
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
        logger.info("  DATASET SUMMARY")
        logger.info(f"  ASVspoof 2019  : subset={ASV_SUBSET or 'FULL'}")
        logger.info(f"  WaveFake       : subset={WAVEFAKE_SUBSET or 'FULL'}")
        logger.info(f"  InTheWild      : subset={ITW_SUBSET or 'FULL'}")
        logger.info(f"  LibriSpeech    : subset={LIBRISPEECH_SUBSET}")
        logger.info(f"  Val split      : {VAL_SIZE or 'FULL'}")
        logger.info(f"  Total training : {len(loader.dataset)} samples")
        logger.info(f"  Batches/epoch  : {len(loader)}")
        logger.info(f"  Est. time/epoch: ~{len(loader) * 4 / 3600:.1f} hrs")
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
            if 'scaler_state' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float("inf"))
            if rank == 0:
                logger.info(f"Resumed from checkpoint: {latest_ckpt}")
                logger.info(f"Starting from epoch {start_epoch + 1} | best_loss={best_loss:.4f}")
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

            optimizer.zero_grad(set_to_none=True)  # Faster than zeroing — sets grads to None

            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # NaN guard: check BEFORE stepping to avoid corrupting optimizer state
            if torch.isnan(loss):
                if rank == 0: logger.error(f"NaN Loss detected at batch {batch_idx}, skipping update!")
                scaler.update()  # Must still call update to keep scaler in sync
                continue

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

        # ── Global loss reduction: average across all ranks ──
        loss_tensor = torch.tensor([epoch_loss], device=device)
        if world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / (len(loader) * world_size)

        if rank == 0:
            elapsed = (time.time() - t0) / 60
            
            logger.info(
                f"[epoch {epoch+1:02d}] train_loss={global_avg_loss:.4f} | "
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

            # Save full checkpoint (model + optimizer + scheduler + scaler + epoch + best_loss)
            checkpoint = {
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),   # AMP scaler state for safe resumption
                'best_loss': best_loss,
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:02d}.pth")
            torch.save(checkpoint, ckpt_path)
            
            # If this is the best model so far, save it as the final model
            if val_loader is not None and val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(), FINAL_PATH)
                logger.info(f"*** New best model (loss: {best_loss:.4f}) saved to {FINAL_PATH} ***")
            
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