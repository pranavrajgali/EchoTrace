# EchoTrace Overfitting Analysis

This note summarizes the most likely reasons the current EchoTrace training setup can appear overfit, or can genuinely overfit, based on the code in this repo.

## Most Likely Repo-Specific Causes

### 1. DDP dataset ordering can differ across ranks
This is the highest-impact issue found in the current training setup.

In `core/preprocess.py`, several dataset constructors call `random.shuffle(...)` while building file lists. In `scripts/train_ddp.py`, each rank sets a different seed with:

- `torch.manual_seed(42 + rank)`

but Python's `random` module is also used by the dataset constructors. Because each rank can build the dataset in a different order before `DistributedSampler` runs, the sampler may no longer be partitioning one shared global ordering. That can create:

- duplicated training samples across ranks
- missing samples on some epochs
- skewed class/domain exposure per rank
- unstable or misleading train/validation behavior

This can look like overfitting even when the root problem is broken distributed sampling.

## 2. Reported training loss may be misleading in DDP
In `scripts/train_ddp.py`, each rank accumulates its own `epoch_loss`, but only rank 0 prints the result. That means the displayed training loss is rank 0's local shard average, not the true global average across all GPUs.

Possible effects:

- train loss may look much better or worse than the real multi-GPU average
- train vs validation gaps can be exaggerated
- debugging overfitting becomes harder because the comparison is not apples-to-apples

This does not itself cause overfitting, but it can make you think overfitting is worse than it really is.

## 3. Domain mismatch between training mixture and validation split
Training uses a combined dataset:

- ASVspoof
- WaveFake
- InTheWild train
- LibriSpeech

Validation uses only `InTheWildDataset(subset="val")`.

That means the model trains on a multi-domain mixture but is judged on a single-domain validation set. If the model becomes very good at artifacts specific to ASVspoof/WaveFake/LibriSpeech composition while validation reflects a narrower real-world split, the train/val gap can widen even without a classic memorization failure.

## 4. Capacity is still fairly high for the effective signal
The model uses:

- ImageNet-pretrained `ResNet50`
- trainable `layer4`
- trainable fully connected head
- 2048-d visual backbone features plus scalar features

Even with partial freezing, this is still a strong model. If the most informative spoof cues are narrow, repetitive, or dataset-specific, the classifier head and upper backbone can lock onto shortcuts faster than they generalize.

Signs this is the real issue:

- tiny-set overfit works perfectly
- DDP diagnostics are clean
- stronger regularization improves validation
- smaller models reduce the train/val gap

## 5. BatchNorm behavior can be noisy across GPUs
The DDP script does not convert the model to `SyncBatchNorm`. Standard BatchNorm runs independently on each GPU.

If per-GPU batches become effectively small or data distributions differ across ranks, BatchNorm statistics can drift and make validation behavior unstable. That instability can resemble overfitting or amplify it.

## 6. Validation may be honest while training data is augmented
Training datasets use augmentation, while validation does not. That is expected and usually correct, but it can complicate interpretation:

- training examples are noisier and more variable
- validation examples may be cleaner or from a different distribution
- the model may fit dataset-specific structure despite augmentation

So augmentation alone does not rule out overfitting.

## Things That Look Less Likely

### Softmax / loss misuse
This does **not** currently look like the main bug.

The repo's main binary classifier is consistent with:

- raw logits from the model
- `BCEWithLogitsLoss`
- `torch.sigmoid(...)` only for metrics/inference

That is the correct pattern for binary or multi-label logits.

### Wrong target shape for the current binary setup
The training loop uses:

- logits shaped like `(B, 1)`
- labels converted to `float().unsqueeze(1)`

That matches `BCEWithLogitsLoss`.

## Recommended Diagnostic Order

1. Run `scripts/ddp_diagnostic.py --mode ddp` with multiple GPUs.
2. Run `scripts/ddp_diagnostic.py --mode tiny-overfit` on one GPU.
3. Fix dataset ordering so every rank builds the exact same sample order.
4. Reduce confusion by logging globally reduced training loss.
5. Re-test before changing major hyperparameters.

## Practical Conclusion

The strongest current hypothesis is:

- the model may not be "just overfitting"
- DDP data-order inconsistency may be distorting or breaking training

After that, the next most plausible contributors are:

- domain mismatch between the mixed training pool and InTheWild-only validation
- high model capacity relative to the forensic signal
- non-synchronized BatchNorm behavior

So the first step should be to validate the DDP pipeline, not just add more dropout or weight decay.
