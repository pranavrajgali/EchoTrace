import torch
import os
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from .model import build_model, get_loss, get_optimizer
from .preprocess import ASVDataset, InTheWildDataset, MultiDataset


def setup_logger(log_path):
    """Setup logger that writes to both terminal and log file simultaneously."""
    logger = logging.getLogger("EchoTrace_Trainer")
    logger.setLevel(logging.DEBUG)

    # Format: [2026-03-30 21:07:00] MESSAGE
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Terminal handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler (training_output.log)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(BASE_DIR, "training_output.log")

    logger = setup_logger(log_path)
    logger.info(f"{'='*60}")
    logger.info(f"  EchoTrace Model Training  —  {datetime.now().strftime('%A, %d %B %Y')}")
    logger.info(f"{'='*60}")

    # 1. Hardware Setup (CUDA for Windows/Linux, MPS for Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ Device: NVIDIA {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("⚠️  Device: CPU (no GPU detected)")

    # 2. Paths (Linked to Documents folder)
    protocol = r"C:\Users\Admin\Documents\ASVPOOF 2019 LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    asv_data_dir = r"C:\Users\Admin\Documents\ASVPOOF 2019 LA\LA\ASVspoof2019_LA_train\flac"
    wild_data_dir = r"C:\Users\Admin\Documents\InTheWild\release_in_the_wild"

    # 3. Balanced Dataset Loading (10,000 per dataset = 20,000 total target)
    logger.info(f"{'─'*60}")
    logger.info("📂 Loading Datasets...")
    logger.info(f"   ASVspoof : {asv_data_dir}")
    logger.info(f"   InTheWild: {wild_data_dir}")

    asv_dataset = ASVDataset(protocol, asv_data_dir, subset_size=10000, augment=True, augment_prob=0.3)
    wild_dataset = InTheWildDataset(wild_data_dir, subset='train', subset_size=10000, augment=True, augment_prob=0.3)
    dataset = MultiDataset(asv_dataset, wild_dataset)

    asv_real = sum(1 for label in asv_dataset.labels if label == 0)
    asv_fake = len(asv_dataset.labels) - asv_real
    wild_real = sum(1 for label in wild_dataset.labels if label == 0)
    wild_fake = len(wild_dataset.labels) - wild_real

    logger.info(f"✅ Loaded {len(asv_dataset)} ASVspoof samples + {len(wild_dataset)} InTheWild samples")
    logger.info(f"✅ Total training samples (len(dataset)): {len(dataset)}")
    logger.info(
        f"   ASV class balance       -> real: {asv_real}, fake: {asv_fake}"
    )
    logger.info(
        f"   InTheWild class balance -> real: {wild_real}, fake: {wild_fake}"
    )

    # 4. Optimized DataLoader for NVIDIA RTX GPUs
    batch_size = 64
    num_workers = 4

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    batches_per_epoch = len(train_loader)
    full_batches_samples = batches_per_epoch * batch_size
    last_batch_size = len(dataset) % batch_size if len(dataset) % batch_size != 0 else batch_size

    logger.info(f"   DataLoader batches/epoch: {batches_per_epoch}")
    logger.info(f"   Approx samples/epoch (batches*batch_size): {full_batches_samples}")
    logger.info(f"   Last batch size: {last_batch_size}")

    # 5. Model & Differential Optimizer
    model = build_model(device)
    optimizer = get_optimizer(model)
    criterion = get_loss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 6. Training Loop
    num_epochs = 10
    best_loss = float('inf')
    epoch_summary = []

    logger.info(f"{'─'*60}")
    logger.info(f"🚀 Starting Training: {len(dataset)} samples | Batch Size: {batch_size} | Epochs: {num_epochs}")
    logger.info(f"{'─'*60}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = datetime.now()

        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs, labels = specs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                pct = (batch_idx + 1) / len(train_loader) * 100
                bar = "█" * int(pct / 5) + "▒" * (20 - int(pct / 5))
                logger.info(
                    f"Epoch {epoch+1:02d}/{num_epochs} [{bar}] {pct:5.1f}% "
                    f"Batch {batch_idx+1:3d}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        elapsed = (datetime.now() - epoch_start).seconds
        epoch_summary.append((epoch + 1, avg_loss))

        saved = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "deepfake_detector.pth"))
            saved = " ✅ CHECKPOINT SAVED"

        logger.info(f"{'─'*60}")
        logger.info(f"📊 Epoch {epoch+1:02d}/{num_epochs} Complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed}s{saved}")
        logger.info(f"{'─'*60}")

        scheduler.step()

    # Final Summary
    logger.info(f"{'='*60}")
    logger.info("🏆 TRAINING COMPLETE — EPOCH SUMMARY")
    logger.info(f"{'='*60}")
    for ep, loss in epoch_summary:
        bar = "█" * int((1 - loss) * 20)
        logger.info(f"  Epoch {ep:02d}: Loss {loss:.4f}  {bar}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Best Loss   : {best_loss:.4f}")
    logger.info(f"  Model saved : deepfake_detector.pth")
    logger.info(f"  Log saved   : {log_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()