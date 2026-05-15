# EchoTrace Training Guide — v4.0 (Technical Edition)
**Status:** Production Optimized (Full DDP Pipeline Audit)

This guide provides a comprehensive technical breakdown of the EchoTrace training pipeline as implemented in `scripts/train_ddp.py` and `core/model.py`.

---

## 🏗️ 1. Architecture & Strategy

EchoTrace uses a **Dual-Stream Late-Fusion** strategy, combining visual spectral features with physical biometric markers.

### The Model: `EchoTraceResNet`
*   **Backbone**: ResNet-50 (ImageNet-1K pretrained).
*   **Freeze Strategy**: To preserve robust feature detectors, **Layers 1-3 are frozen**. Only **Layer 4** and the **FC Head** are trainable.
*   **Late Fusion**: The 2048-dimensional visual embedding is concatenated with an 8-dimensional scalar vector (vocal tract physics) at the neck, resulting in a **2056-dimensional** input to the final classifier.

### The Optimizer: Differential Learning Rates
The system uses **Adam** with split learning rates to ensure stability during fine-tuning:
*   **Backbone (Layer 4)**: `1e-5` (Gentle fine-tuning).
*   **Forensic Head**: `1e-4` (Aggressive training for classification).
*   **Weight Decay**: `1e-4` (Regularization to prevent overfitting).

---

## ⚖️ 2. Loss & Optimization

### Focal Loss
Instead of standard Cross-Entropy, we use **Focal Loss** (`gamma=2.0`, `alpha=0.5`). 
*   **Hard Example Mining**: It mathematically penalizes the model more for missing "hard" examples (where deepfakes are nearly perfect) and down-weights easy, confident predictions.
*   **Balance**: Optimized for a 50/50 real-to-fake distribution.

### Scheduler: Cosine Annealing
We use `CosineAnnealingLR` with an `eta_min` of `1e-7`.
*   This provides a smooth, non-linear decay of the learning rate across all epochs, which is more effective for the short (5-10 epoch) training cycles used in EchoTrace.

---

## 🚀 3. DDP Infrastructure (Multi-GPU)

The script is built for **Distributed Data Parallel (DDP)** using the `nccl` backend.

### Performance Controls:
*   **Automatic Mixed Precision (AMP)**: Heavy math is performed in `float16` to double throughput, while critical weights stay in `float32`.
*   **SyncBatchNorm**: Automatically converts the model to use synchronized batch normalization, preventing the "rank drift" that occurs when GPUs see slightly different data distributions.
*   **NaN Guard**: If a corrupted audio file causes a `NaN` loss, the script **skips the optimizer update** for that batch, preventing the model weights from being corrupted.

---

## ⚙️ 4. Training Configuration

All parameters are managed in the `CONFIG` block of `train_ddp.py`.

| Parameter | Current Default | Notes |
| :--- | :--- | :--- |
| `NUM_EPOCHS` | 10 | The standard production budget. |
| `AUGMENT_PROB` | 0.1 | Reduced to 10% to improve generalization to clean laptop mics. |
| `BATCH_PER_GPU` | 32 | Results in an effective batch size of 128 on a 4-GPU system. |
| `ASV_SUBSET` | `None` | Loads the full ASVspoof 2019 dataset (~25k samples). |
| `WAVEFAKE_SUBSET`| 80,000 | Balanced real/fake split from WaveFake. |
| `ITW_SUBSET` | 25,000 | Combined train/val split from In-The-Wild. |
| `LIBRISPEECH_SUBSET`| 38,000 | Used as a "real-world real" anchor dataset. |

---

## 🏃 5. Execution Workflow

### Starting/Resuming Training:
The script is **fully stateful**. It automatically looks for the latest checkpoint and resumes the epoch, optimizer state, scheduler state, and AMP scaler state.

```bash
cd scripts
nohup python train_ddp.py > ddp_train.log 2>&1 &
tail -f ddp_train.log
```

### Monitoring Metrics (Rank 0):
The model is evaluated after every epoch on the **In-The-Wild Validation set**.
*   **Best Model**: The checkpoint with the **lowest validation loss** is automatically copied to `ensemble_model.pth`.
*   **Success Indicator**: Look for an **EER below 1.0%** and **Balanced Accuracy above 98%**.

---
*Last updated: May 15, 2026 (Full Pipeline Audit v4.0)*
