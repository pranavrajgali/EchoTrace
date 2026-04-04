"""
EchoTrace GPU-accelerated DDP trainer.
torchaudio loads audio on CPU workers → GPU feature extraction in training loop.
No librosa. No cache files. Full dataset.
"""
import os, time, datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from core.model import EchoTraceResNet, get_loss, get_optimizer
from core.preprocess_gpu import (
    FastASVDataset, FastWaveFakeDataset, FastITWDataset,
    GPUFeatureExtractor, _extract_scalars_gpu
)
import logging, sys

WORLD_SIZE     = min(4, torch.cuda.device_count())
CHECKPOINT_DIR = "/home/jovyan/work/EchoTrace/checkpoints"
FINAL_PATH     = "/home/jovyan/work/EchoTrace/ensemble_model.pth"
LOG_PATH       = "/home/jovyan/work/EchoTrace/ddp_gpu.log"
BATCH_PER_GPU  = 64
NUM_EPOCHS     = 10
MASTER_PORT    = "12375"


def get_logger(rank):
    logger = logging.getLogger(f"rank{rank}")
    if logger.handlers: return logger
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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"]      = "localhost"
    os.environ["MASTER_PORT"]      = MASTER_PORT
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"]  = "1"
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)
    device  = torch.device(f"cuda:{rank}")
    logger  = get_logger(rank)

    # GPU feature extractor — one per GPU process
    extractor = GPUFeatureExtractor(device)

    # Datasets — torchaudio loading only, no librosa
    asv  = FastASVDataset()
    wf   = FastWaveFakeDataset()
    itw  = FastITWDataset()
    dataset = ConcatDataset([asv, wf, itw])

    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_PER_GPU,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model     = EchoTraceResNet(num_scalars=8).to(device)
    model     = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = get_loss()
    optimizer = get_optimizer(model.module)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info("=" * 60)
        logger.info("  EchoTrace GPU-Accelerated DDP Training")
        logger.info(f"  GPUs            : {world_size}")
        logger.info(f"  Total samples   : {len(dataset)}")
        logger.info(f"  Batch/GPU       : {BATCH_PER_GPU}")
        logger.info(f"  Effective batch : {BATCH_PER_GPU * world_size}")
        logger.info(f"  Epochs          : {NUM_EPOCHS}")
        logger.info(f"  Trainable params: {trainable:,} / {total:,}")
        logger.info("=" * 60)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (waveforms, labels) in enumerate(loader):
            # waveforms: (B, T) float32  — move to GPU
            waveforms = waveforms.to(device, non_blocking=True)
            labels    = labels.float().unsqueeze(1).to(device, non_blocking=True)

            # GPU feature extraction — no CPU bottleneck
            with torch.no_grad():
                images  = extractor(waveforms)          # (B, 3, 224, 224)
                scalars = _extract_scalars_gpu(waveforms)  # (B, 8)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images, scalars)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if rank == 0 and batch_idx % 20 == 0:
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
            tag     = " <- best" if avg < best_loss else ""
            if avg < best_loss:
                best_loss = avg
            logger.info(
                f"[epoch {epoch+1:02d}] avg_loss={avg:.4f} | "
                f"time={elapsed:.1f}m | best={best_loss:.4f}{tag}"
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


if __name__ == "__main__":
    assert torch.cuda.is_available(), "No CUDA GPUs found."
    print(f"[launch] {WORLD_SIZE} GPU(s) detected.")
    print(f"[launch] Log -> {LOG_PATH}")
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
