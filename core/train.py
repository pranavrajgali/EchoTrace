import torch
import os
from torch.utils.data import DataLoader
from model import build_model, get_loss, get_optimizer
from preprocess import ASVDataset, InTheWildDataset, MultiDataset

def main():
    # 1. Hardware Setup (MPS for Mac, CUDA for Windows/Linux)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. Paths
    protocol = os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    asv_data_dir = os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_train/flac")
    wild_data_dir = os.path.join(BASE_DIR, "data/release_in_the_wild")

    # 3. Balanced Dataset Loading
    print("Loading datasets...")
    # Lower augment_prob to 0.3 to prevent signal destruction during learning
    asv_dataset = ASVDataset(protocol, asv_data_dir, subset_size=2000, augment=True, augment_prob=0.3)
    wild_dataset = InTheWildDataset(wild_data_dir, subset='train', subset_size=2000, augment=True, augment_prob=0.3)
    
    dataset = MultiDataset(asv_dataset, wild_dataset)

    # 4. Optimized DataLoader Settings
    batch_size = 32 # Standardized for 16GB RAM stability
    num_workers = 0 # Avoids multiprocessing overhead on macOS
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False # Keep False for Apple Silicon MPS
    )

    # 5. Model & Differential Optimizer
    model = build_model(device)
    # This now uses the custom Adam with differential learning rates
    optimizer = get_optimizer(model)
    criterion = get_loss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 6. Training Loop
    num_epochs = 10
    best_loss = float('inf')

    print(f"Starting training: {len(dataset)} samples | Batch Size: {batch_size}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (specs, labels) in enumerate(train_loader):
            specs, labels = specs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Avg Loss: {avg_loss:.4f} ---")

        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "deepfake_detector.pth"))
        
        scheduler.step()

    print("Training Complete. Model saved as 'deepfake_detector.pth'")

if __name__ == '__main__':
    main()