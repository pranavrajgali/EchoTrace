"""
EchoTrace evaluate_quick.py — Fast Evaluation (ASVspoof Dev + InTheWild Test only)
Skips the large ASVspoof Eval set for quick iteration.
Usage: python scripts/evaluate_quick.py --checkpoint 7
"""

# ── Thread control (set before ANY imports that use librosa) ──
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import warnings
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob
from tqdm import tqdm
import io
import base64

warnings.filterwarnings('ignore', message='Trying to estimate tuning from empty frequency set')

from sklearn.metrics import (
    roc_curve, confusion_matrix, balanced_accuracy_score,
    precision_score, f1_score, roc_auc_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import Image

from core.model import EchoTraceResNet
from core.preprocess import load_audio, build_feature_image, extract_scalar_features

# ── Hyperparameters ──
SR = 16000
DURATION = 4.0
BATCH_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ── EER ──
def compute_eer(y_true, y_score):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
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
        return eer_fraction * 100
    except Exception as e:
        print(f"EER computation failed: {e}")
        return None


# ── Dataset ──
class SimpleAudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, dataset_name=""):
        self.file_list = file_list
        self.label_list = label_list
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            audio = load_audio(self.file_list[idx], target_sr=SR, duration=DURATION, random_crop=False)
            image = build_feature_image(audio, sr=SR)
            scalars = extract_scalar_features(audio, sr=SR)
            image_pil = Image.fromarray(image)
            image_tensor = _EVAL_TRANSFORM(image_pil)
            scalars_tensor = torch.tensor(scalars, dtype=torch.float32)
            label_tensor = torch.tensor(self.label_list[idx], dtype=torch.long)
            return image_tensor, scalars_tensor, label_tensor
        except Exception as e:
            print(f"Error loading {self.file_list[idx]}: {e}")
            normalized_zeros = torch.zeros(3, 224, 224, dtype=torch.float32)
            for c in range(3):
                normalized_zeros[c] -= IMAGENET_MEAN[c] / IMAGENET_STD[c]
            return (
                normalized_zeros,
                torch.zeros(8, dtype=torch.float32),
                torch.tensor(self.label_list[idx], dtype=torch.long)
            )


# ── Protocol parsing ──
def parse_asv_protocol(protocol_path, audio_root):
    file_list, label_list = [], []
    if not Path(protocol_path).exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = 0 if parts[4] == "bonafide" else 1
            audio_path = Path(audio_root) / f"{file_id}.flac"
            if audio_path.exists():
                file_list.append(str(audio_path))
                label_list.append(label)
    print(f"  Loaded {len(file_list)} samples from protocol")
    return file_list, label_list


def load_inthe_wild_test(test_root):
    test_root = Path(test_root)
    real_files = glob(str(test_root / "real" / "**" / "*.wav"), recursive=True) + \
                 glob(str(test_root / "real" / "**" / "*.flac"), recursive=True)
    fake_files = glob(str(test_root / "fake" / "**" / "*.wav"), recursive=True) + \
                 glob(str(test_root / "fake" / "**" / "*.flac"), recursive=True)
    file_list = real_files + fake_files
    label_list = [0] * len(real_files) + [1] * len(fake_files)
    print(f"  Loaded {len(real_files)} real + {len(fake_files)} fake samples")
    return file_list, label_list


# ── Model loading ──
def load_model(checkpoint_path, device):
    model = EchoTraceResNet(num_scalars=8).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    if isinstance(state_dict, dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Using all {torch.cuda.device_count()} GPUs for parallel evaluation!")
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


# ── Inference ──
def evaluate_dataset(model, dataloader, device, dataset_name):
    all_labels, all_scores = [], []
    with torch.no_grad():
        for images, scalars, labels in tqdm(dataloader, desc=f"Eval {dataset_name}", leave=True):
            images = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            outputs = model(images, scalars)
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
            if scores.ndim == 0:
                scores = np.array([scores])
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_scores)


# ── Metrics ──
def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    eer = compute_eer(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    real_total = (y_true == 0).sum()
    real_recall = tn / real_total * 100 if real_total > 0 else 0
    fake_total = (y_true == 1).sum()
    fake_recall = tp / fake_total * 100 if fake_total > 0 else 0

    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'bal_acc': bal_acc * 100, 'eer': eer, 'roc_auc': roc_auc,
        'real_recall': real_recall, 'fake_recall': fake_recall,
        'f1': f1, 'cm': cm, 'y_pred': y_pred
    }


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description='EchoTrace quick evaluation (ASVspoof Dev + InTheWild Test)')
    parser.add_argument('--checkpoint', required=True, help='Path or epoch number (e.g. 7)')
    parser.add_argument('--tag', default='quick_eval', help='Output folder tag')
    parser.add_argument('--asv_root', default='/home/jovyan/work/data/LA/LA')
    parser.add_argument('--itw_test_root', default='/home/jovyan/work/data/release_in_the_wild/release_in_the_wild/test')
    parser.add_argument('--output_dir', default='/home/jovyan/work/EchoTrace/eval_results')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path.isdigit():
        checkpoint_path = f"/home/jovyan/work/EchoTrace/checkpoints/checkpoint_epoch_{int(checkpoint_path):02d}.pth"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"{'='*60}")
    print(f"  EchoTrace Quick Evaluation")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Threshold  : {args.threshold}")
    print(f"  Device     : {device}")
    print(f"  Datasets   : ASVspoof Dev + InTheWild Test")
    print(f"{'='*60}\n")

    model = load_model(checkpoint_path, device)

    # ── ASVspoof Dev ──
    print("Loading ASVspoof Dev...")
    asv_files, asv_labels = parse_asv_protocol(
        Path(args.asv_root) / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        Path(args.asv_root) / "ASVspoof2019_LA_dev/flac"
    )
    asv_dataset = SimpleAudioDataset(asv_files, asv_labels, "ASVspoof Dev")
    asv_loader = torch.utils.data.DataLoader(asv_dataset, batch_size=BATCH_SIZE, num_workers=8)

    # ── InTheWild Test ──
    print("\nLoading InTheWild Test...")
    itw_files, itw_labels = load_inthe_wild_test(args.itw_test_root)
    itw_dataset = SimpleAudioDataset(itw_files, itw_labels, "InTheWild Test")
    itw_loader = torch.utils.data.DataLoader(itw_dataset, batch_size=BATCH_SIZE, num_workers=8)

    # ── Evaluate ──
    print("\nEvaluating...\n")
    results = {}

    for name, loader in [("ASVspoof Dev", asv_loader), ("InTheWild Test", itw_loader)]:
        y_true, y_score = evaluate_dataset(model, loader, device, name)
        metrics = compute_metrics(y_true, y_score, threshold=args.threshold)

        results[name] = {
            'n_samples': len(y_true),
            'bal_acc': metrics['bal_acc'],
            'eer': metrics['eer'],
            'roc_auc': metrics['roc_auc'],
            'real_recall': metrics['real_recall'],
            'fake_recall': metrics['fake_recall'],
            'f1': metrics['f1'],
            'cm': metrics['cm'].tolist(),
        }

        eer_str = f"{metrics['eer']:.2f}%" if metrics['eer'] is not None else "N/A"
        print(f"\n  {name} ({len(y_true)} samples):")
        print(f"    Bal. Accuracy : {metrics['bal_acc']:.2f}%")
        print(f"    EER           : {eer_str}")
        print(f"    ROC-AUC       : {metrics['roc_auc']:.4f}")
        print(f"    Real Recall   : {metrics['real_recall']:.2f}%")
        print(f"    Fake Recall   : {metrics['fake_recall']:.2f}%")
        print(f"    F1            : {metrics['f1']:.4f}")

    # ── Save JSON ──
    json_path = output_dir / "quick_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Metrics saved to: {json_path}")
    print("✅ Quick evaluation complete!")


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(0)
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
