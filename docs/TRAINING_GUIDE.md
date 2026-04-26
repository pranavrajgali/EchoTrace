# EchoTrace Training Guide — v2.0

**Updated:** April 6, 2026 | **Deadline:** April 10 demo

---

## What Changed

✅ **Per-epoch validation**: Model now evaluates on InTheWild 'val' split after each epoch
✅ **CosineAnnealingLR scheduler**: Smooth decay replaces fixed step schedule (much better for 5-7 epoch runs)
✅ **Reduced augmentation**: 0.3 → 0.2 (less I/O pressure on librosa workers)
✅ **Three training configs**: Smoke test, medium, full — just change one line to switch

---

## Quick Start

### 1. Select Your Config

Edit `train_ddp.py` line 42:
```python
CONFIG = "smoke"      # for quick pipeline test
# CONFIG = "medium"   # for backup checkpoint
# CONFIG = "full"     # for final demo model
```

### 2. Run Training

```bash
nohup python train_ddp.py > ddp_train.log 2>&1 &
tail -f ddp_train.log
```

### 3. Monitor Output

Each epoch prints:
```
[14:32:10][rank0] [epoch 01] train_loss=0.4521 | time=45.2m
[14:32:10][rank0] [val    01] val_loss=0.3821 | bal_acc=68.45% | real_recall=72.15% | fake_recall=64.75% | eer=8.35% | roc_auc=0.9823
[14:32:10][rank0] Saved checkpoint -> checkpoints/checkpoint_epoch_01.pth
```

---

## The Three Configs

### SMOKE TEST — Day 1 (20–30 min training)

**Purpose:** Verify pipeline, no crashes, workers keep up, validation loop runs.

```python
CONFIG = "smoke"
NUM_EPOCHS         = 2
AUGMENT_PROB       = 0.2
ASV_SUBSET         = 1500
WAVEFAKE_SUBSET    = 8000
ITW_SUBSET         = 2500
LIBRISPEECH_SUBSET = 8000
VAL_SIZE           = 1000
```

**Target real/fake split:** ~11,650 (58%) / ~8,350 (42%)

**Time per epoch:** 8–15 min → Total ~20–30 min

**What to check:**
- No crashes or worker deadlocks?
- Is validation loss printing after each epoch?
- Is balanced accuracy > 50%? (random = 50%)

---

### MEDIUM RUN — Day 2 (1–1.5 hr training)

**Purpose:** Get a usable backup checkpoint in case something goes wrong during full 12-hour run.

```python
CONFIG = "medium"
NUM_EPOCHS         = 3
AUGMENT_PROB       = 0.2
ASV_SUBSET         = 4000
WAVEFAKE_SUBSET    = 20000
ITW_SUBSET         = 6000
LIBRISPEECH_SUBSET = 20000
VAL_SIZE           = 2000
```

**Target real/fake split:** ~29,000 (58%) / ~21,000 (42%)

**Time per epoch:** 20–30 min → Total ~1–1.5 hrs

**What to check:**
- Is balanced accuracy improving epoch-to-epoch?
- Is EER trending down?
- Is the model learning or stuck?

If metrics look reasonable, proceed to FULL config. If they're stalled, debug before long run.

---

### FULL TRAINING — Day 3 (7–9 hrs training)

**Purpose:** Final demo model with maximum data budget and epochs.

```python
CONFIG = "full"
NUM_EPOCHS         = 7
AUGMENT_PROB       = 0.2
ASV_SUBSET         = 20000
WAVEFAKE_SUBSET    = 80000
ITW_SUBSET         = 11000
LIBRISPEECH_SUBSET = 85000
VAL_SIZE           = 5000
```

**Target real/fake split:** ~113,200 (58%) / ~82,800 (42%)

**Time per epoch:** 60–75 min → Total ~7–9 hrs (fits comfortably in 12-hour window)

**What to pick for the demo:**
- Choose the checkpoint with the **highest balanced accuracy** on validation
- OR manually evaluate the last checkpoint: `python evaluate_pc.py`
- Copy the best checkpoint to `ensemble_model.pth` before April 9

---

## What the Validation Metrics Mean

| Metric | Meaning | Good | Bad |
|---|---|---|---|
| **val_loss** | Validation cross-entropy loss | Decreases each epoch (< 0.3) | Stays high or increases |
| **bal_acc** | Balanced accuracy (avg of real & fake recall) | > 70% for demo | < 60% = model not learning |
| **real_recall** | % of real audio correctly classified as real | > 75% | < 50% = model thinks everything is real |
| **fake_recall** | % of fake audio correctly classified as fake | > 65% | < 50% = model thinks everything is fake |
| **EER** | Equal Error Rate (lower is better) | < 10% | > 20% = poor |
| **roc_auc** | ROC-AUC score | > 0.95 | < 0.85 = marginal |

**Rule of thumb for demo:**
- Balanced accuracy ≥ 70% → Ready
- Balanced accuracy 60–70% → Borderline, test on live audio
- Balanced accuracy < 60% → Debug before demo

---

## Key Improvements in v2.0

### Per-Epoch Validation ✅
Previously: no held-out validation during training, model state unknown
Now: After each epoch, model tested on held-out InTheWild val split → clear metrics

### CosineAnnealingLR Scheduler ✅
**Old:** `StepLR(step_size=3, gamma=0.5)` — LR halved every 3 epochs
- With 5 epochs: fires at epoch 4 → only 1 decay, useless
- With 7 epochs: fires at epochs 4, 7 → 2 decays, late for first

**New:** `CosineAnnealingLR(T_max=NUM_EPOCHS, eta_min=1e-7)` — smooth cosine decay
- LR smoothly decays from start → finish
- Works the same regardless of epoch count
- Better for short runs (convergence is smooth, not jerky)

Learning rate of layer4 backbone: 1e-5 → cosine decay to ~1e-7
Learning rate of FC head: 1e-4 → cosine decay to ~1e-6

### Reduced Augmentation ✅
**Old:** 30% augmentation
- 16 librosa workers total
- ~5 workers loading/mixing MUSAN noise each batch
- I/O bottleneck, batch arrival delayed

**New:** 20% augmentation
- ~3 workers doing augmentation
- Easier on I/O, prefetch works better
- Same learning signal (noise still helps), but faster throughput

---

## After Training Complete

### Evaluate Best Checkpoint

```bash
python evaluate_pc.py
```

This runs full assessment on ASVspoof dev, ASVspoof eval, and InTheWild test.
Output goes to `pc_eval_logs/eval_YYYYMMDD_HHMMSS/`

### Copy Best Model to Demo Location

```bash
cp checkpoints/checkpoint_epoch_XX.pth ensemble_model.pth
```

Replace XX with the epoch that had the highest validation balanced accuracy.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Smoke test takes > 30 min per epoch | Workers bottleneck | Reduce num_workers (currently 4) or AUGMENT_PROB (currently 0.2) |
| Validation metrics stuck/not improving | Learning rate too low | Already optimized for layer freeze — likely data issue |
| Model crashes during validation | OOM (out of memory) | Reduce VAL_SIZE (currently 1000–5000) or batch size |
| Validation never prints | Bug in evaluate() | Check logs, likely numpy/sklearn import issue |

---

## Checkpoints Storage

```
checkpoints/
├── checkpoint_epoch_01.pth  ← resume here if interrupted
├── checkpoint_epoch_02.pth
├── checkpoint_epoch_03.pth
├── checkpoint_epoch_04.pth
├── checkpoint_epoch_05.pth
├── checkpoint_epoch_06.pth
└── checkpoint_epoch_07.pth
```

Training auto-resumes from the latest checkpoint.

---

## Timeline

| Day | Time | Config | Task | Output |
|---|---|---|---|---|
| April 6 | 2h | SMOKE | Test pipeline | checkpoint_epoch_01.pth, checkpoint_epoch_02.pth |
| April 7 | 1.5h | MEDIUM | Backup ckpt | checkpoint_epoch_01.pth, checkpoint_epoch_02.pth, checkpoint_epoch_03.pth |
| April 8 | 12h | FULL | Final model | checkpoint_epoch_01–07.pth, best selected |
| April 9 | 2h | — | Evaluate best, copy to ensemble_model.pth | evaluate_pc.py output |
| April 10 | Demo day | — | Live demo | 🎙️ |

---

## Quick Config Switch Cheat Sheet

```python
# In train_ddp.py, line 42:

# For rapid testing:
CONFIG = "smoke"

# For backup checkpoint:
CONFIG = "medium"

# For final model:
CONFIG = "full"
```

That's it. Everything else updates automatically (NUM_EPOCHS, subsets, VAL_SIZE, etc).

---

**Questions?** Check `ddp_train.log` for each run's full output.
