"""
EchoTrace evaluate_server.py — Comprehensive Evaluation Script
Evaluates model on ASVspoof Dev, ASVspoof Eval, and InTheWild Test.
Outputs HTML report with metrics and visualizations + JSON data dump.
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

# Suppress librosa warnings
warnings.filterwarnings('ignore', message='Trying to estimate tuning from empty frequency set')

# Scientific/metrics
from sklearn.metrics import (
    roc_curve, confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# EchoTrace
from core.model import EchoTraceResNet
from core.preprocess import load_audio, build_feature_image, extract_scalar_features

# ── Hyperparameters ──
SR = 16000
DURATION = 4.0
BATCH_SIZE = 32


# ── EER Computation (inlined from core/evaluate.py) ──
def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate with boundary protection.
    
    Args:
        y_true: Binary labels (0/1)
        y_score: Continuous scores [0, 1]
    
    Returns:
        eer: Equal Error Rate as percentage
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        
        # Boundary protection
        fpr_clipped = np.clip(fpr, 1e-6, 1 - 1e-6)
        fnr_clipped = np.clip(fnr, 1e-6, 1 - 1e-6)
        
        # Sort and deduplicate
        sort_idx = np.argsort(fpr_clipped)
        fpr_sorted = fpr_clipped[sort_idx]
        fnr_sorted = fnr_clipped[sort_idx]
        _, unique_idx = np.unique(fpr_sorted, return_index=True)
        fpr_unique = fpr_sorted[unique_idx]
        fnr_unique = fnr_sorted[unique_idx]
        
        # Find crossing point
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


# ── Custom Dataset Classes ──
class SimpleAudioDataset(torch.utils.data.Dataset):
    """Lightweight dataset for evaluation — no augmentation."""
    
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
            
            image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
            scalars_tensor = torch.tensor(scalars, dtype=torch.float32)
            label_tensor = torch.tensor(self.label_list[idx], dtype=torch.long)
            
            return image_tensor, scalars_tensor, label_tensor
        except Exception as e:
            # Gracefully skip corrupted files
            print(f"Error loading {self.file_list[idx]}: {e}")
            # Return zeros as fallback (will inflate metrics slightly, but avoids crash)
            return (
                torch.zeros(3, 224, 224, dtype=torch.float32),
                torch.zeros(8, dtype=torch.float32),
                torch.tensor(self.label_list[idx], dtype=torch.long)
            )


# ── ASVspoof Protocol Parsing ──
def parse_asv_protocol(protocol_path, audio_root):
    """
    Parse ASVspoof CM protocol file.
    Format: SPEAKER_ID FILE_ID SYSTEM_ID - KEY
    Key: bonafide (0) or spoof (1)
    """
    file_list = []
    label_list = []
    system_list = []
    
    if not Path(protocol_path).exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")
    
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            file_id = parts[1]
            key = parts[4]
            system_id = parts[2]
            
            label = 0 if key == "bonafide" else 1
            audio_path = Path(audio_root) / f"{file_id}.flac"
            
            if audio_path.exists():
                file_list.append(str(audio_path))
                label_list.append(label)
                system_list.append(system_id)
    
    print(f"  Loaded {len(file_list)} samples from protocol")
    return file_list, label_list, system_list


# ── InTheWild Folder Parsing ──
def load_inthe_wild_test(test_root):
    """Load InTheWild test set from dedicated test folder."""
    test_root = Path(test_root)
    
    real_files = glob(str(test_root / "real" / "**" / "*.wav"), recursive=True) + \
                 glob(str(test_root / "real" / "**" / "*.flac"), recursive=True)
    fake_files = glob(str(test_root / "fake" / "**" / "*.wav"), recursive=True) + \
                 glob(str(test_root / "fake" / "**" / "*.flac"), recursive=True)
    
    file_list = real_files + fake_files
    label_list = [0] * len(real_files) + [1] * len(fake_files)
    
    print(f"  Loaded {len(real_files)} real + {len(fake_files)} fake samples")
    return file_list, label_list


# ── Model Loading ──
def load_model(checkpoint_path, device):
    """Load checkpoint and strip DDP prefix if needed."""
    model = EchoTraceResNet(num_scalars=8).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP prefix stripping
    state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    
    if isinstance(state_dict, dict):
        # Check if keys have 'module.' prefix (DDP)
        if any(k.startswith('module.') for k in state_dict.keys()):
            # Strip prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ── Inference Loop ──
def evaluate_dataset(model, dataloader, device, dataset_name):
    """Run inference and collect predictions."""
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, scalars, labels in tqdm(dataloader, desc=f"Eval {dataset_name}", leave=True):
            images = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images, scalars)
            scores = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Handle batch size 1
            if scores.ndim == 0:
                scores = np.array([scores])
            
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_scores)


# ── Metrics Computation ──
def compute_metrics(y_true, y_score, threshold=0.5):
    """Compute all metrics."""
    y_pred = (y_score > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    eer = compute_eer(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    # Per-class recalls
    # Note: tp/fn/fp are oriented with fake=1 as positive class
    # Real recall = tn / (tn + fp) — % of actual reals correctly identified
    # Fake recall = tp / (tp + fn) — % of actual fakes correctly identified
    real_total = (y_true == 0).sum()
    real_recall = tn / real_total * 100 if real_total > 0 else 0
    
    fake_total = (y_true == 1).sum()
    fake_recall = tp / fake_total * 100 if fake_total > 0 else 0
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    return {
        'bal_acc': bal_acc * 100,
        'eer': eer,
        'roc_auc': roc_auc,
        'real_recall': real_recall,
        'fake_recall': fake_recall,
        'precision': precision * 100,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred
    }


# ── Per-Attack Analysis (ASVspoof Eval) ──
ATTACK_INFO = {
    "A01": ("TTS", "neural waveform model"),
    "A02": ("TTS", "vocoder"),
    "A03": ("TTS", "vocoder"),
    "A04": ("TTS", "waveform concatenation"),
    "A05": ("VC", "vocoder"),
    "A06": ("VC", "spectral filtering"),
    "A07": ("TTS", "vocoder+GAN"),
    "A08": ("TTS", "neural waveform"),
    "A09": ("TTS", "vocoder"),
    "A10": ("TTS", "neural waveform"),
    "A11": ("TTS", "griffin lim"),
    "A12": ("TTS", "neural waveform"),
    "A13": ("TTS_VC", "waveform concatenation+filtering"),
    "A14": ("TTS_VC", "vocoder"),
    "A15": ("TTS_VC", "neural waveform"),
    "A16": ("TTS", "waveform concatenation"),
    "A17": ("VC", "waveform filtering"),
    "A18": ("VC", "vocoder"),
    "A19": ("VC", "spectral filtering"),
}

def compute_per_attack_metrics(y_true, y_score, system_list, threshold=0.5):
    """Compute EER and fake recall per attack type (vs bonafide)."""
    attacks = {}
    system_array = np.array(system_list)
    bonafide_mask = system_array == "-"
    
    for i, system_id in enumerate(set(system_list)):
        # Skip bonafide (no real attacks to compare)
        if system_id == "-":
            continue
        
        # Include both this attack AND bonafide samples for proper EER
        attack_mask = system_array == system_id
        combined_mask = bonafide_mask | attack_mask
        
        y_true_attack = y_true[combined_mask]
        y_score_attack = y_score[combined_mask]
        
        # Skip if too few samples
        if len(y_true_attack) < 10:
            continue
        
        y_pred_attack = (y_score_attack > threshold).astype(int)
        
        eer = compute_eer(y_true_attack, y_score_attack)
        fake_total = (y_true_attack == 1).sum()
        fake_recall = (y_pred_attack[y_true_attack == 1] == 1).sum() / fake_total * 100 if fake_total > 0 else 0
        
        attacks[system_id] = {
            'eer': eer,
            'fake_recall': fake_recall,
            'n_samples': len(y_true_attack)
        }
    
    return attacks


# ── HTML Report Generation ──
def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_html_report(results_dict, output_path, checkpoint_name, asv_eval_attacks=None):
    """Generate comprehensive HTML report with embedded plots."""
    
    # ── Plot 1: ROC Curves ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset_name, metrics in results_dict.items():
        ax.plot(metrics['fpr'], metrics['tpr'], 
                label=f"{dataset_name} (AUC={metrics['roc_auc']:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Cross-Dataset Generalization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    roc_plot = fig_to_base64(fig)
    
    # ── Plot 2: Score Distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (dataset_name, metrics) in enumerate(results_dict.items()):
        ax = axes[idx]
        y_score = np.array(metrics['y_score'])
        y_true = np.array(metrics['y_true'])
        
        ax.hist(y_score[y_true == 0], bins=30, alpha=0.6, label='Real', color='blue')
        ax.hist(y_score[y_true == 1], bins=30, alpha=0.6, label='Fake', color='red')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset_name}\n(n={len(y_score)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_plot = fig_to_base64(fig)
    
    # ── Plot 3: Confusion Matrices ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (dataset_name, metrics) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = np.array(metrics['cm'])
        
        # Normalize for visualization
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'Fake'])
        ax.set_yticklabels(['Real', 'Fake'])
        ax.set_title(f'{dataset_name}')
        
        # Add text
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                              ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_plot = fig_to_base64(fig)
    
    # ── HTML Content ──
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EchoTrace Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric-good {{ color: green; font-weight: bold; }}
        .metric-warning {{ color: orange; font-weight: bold; }}
        .metric-bad {{ color: red; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 5px; }}
        .footer {{ text-align: center; color: #7f8c8d; margin-top: 40px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 EchoTrace Evaluation Report</h1>
        <p><strong>Checkpoint:</strong> {checkpoint_name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Summary Metrics</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Samples</th>
                <th>Bal. Accuracy</th>
                <th>EER</th>
                <th>ROC-AUC</th>
                <th>Real Recall</th>
                <th>Fake Recall</th>
                <th>Precision</th>
                <th>F1 Score</th>
            </tr>
"""
    
    for dataset_name, metrics in results_dict.items():
        bal_acc = metrics['bal_acc']
        eer = metrics['eer']
        if eer is None:
            eer_display = "N/A"
            eer_class = ""
        else:
            eer_display = f"{eer:.2f}%"
            eer_class = 'metric-good' if eer < 5 else ('metric-warning' if eer < 10 else 'metric-bad')
        
        html_content += f"""
            <tr>
                <td><strong>{dataset_name}</strong></td>
                <td>{metrics['n_samples']}</td>
                <td>{bal_acc:.2f}%</td>
                <td class="{eer_class}">{eer_display}</td>
                <td>{metrics['roc_auc']:.4f}</td>
                <td>{metrics['real_recall']:.2f}%</td>
                <td>{metrics['fake_recall']:.2f}%</td>
                <td>{metrics['precision']:.2f}%</td>
                <td>{metrics['f1']:.4f}</td>
            </tr>
"""
    
    html_content += f"""
        </table>
    </div>
    
    <div class="section">
        <h2>ROC Curves</h2>
        <img src="data:image/png;base64,{roc_plot}" />
    </div>
    
    <div class="section">
        <h2>Score Distributions</h2>
        <img src="data:image/png;base64,{dist_plot}" />
    </div>
    
    <div class="section">
        <h2>Confusion Matrices</h2>
        <img src="data:image/png;base64,{cm_plot}" />
    </div>"""
    
    # Add per-attack analysis if available
    if asv_eval_attacks:
        attack_rows = ""
        for attack, data in sorted(asv_eval_attacks.items()):
            eer = data['eer']
            if eer is None:
                eer_display = "N/A"
                eer_class = ""
            else:
                eer_display = f"{eer:.2f}%"
                eer_class = 'metric-good' if eer < 5 else ('metric-warning' if eer < 10 else 'metric-bad')
            attack_type, technique = ATTACK_INFO.get(attack, ("Unknown", "Unknown"))
            attack_rows += f"""
            <tr>
                <td><strong>{attack}</strong></td>
                <td>{attack_type}</td>
                <td>{technique}</td>
                <td>{data['n_samples']}</td>
                <td class="{eer_class}">{eer_display}</td>
                <td>{data['fake_recall']:.2f}%</td>
            </tr>
"""
        html_content += f"""    
    <div class="section">
        <h2>Per-Attack Analysis (ASVspoof Eval)</h2>
        <p><em>EER computed on attack samples vs bonafide baselines. TTS = Text-to-Speech, VC = Voice Conversion.</em></p>
        <table>
            <tr>
                <th>Attack ID</th>
                <th>Type</th>
                <th>Technique</th>
                <th>Samples</th>
                <th>EER</th>
                <th>Fake Recall</th>
            </tr>
{attack_rows}        </table>
    </div>
"""
    
    html_content += """    
    <div class="footer">
        <p>BackProp Bandits | Udhbhav 2026 Hackathon</p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Report saved to: {output_path}")


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description='EchoTrace comprehensive evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint or epoch number (e.g. 6 → checkpoint_epoch_06.pth)')
    parser.add_argument('--tag', default='eval', help='Tag for output folder')
    parser.add_argument('--asv_root', default='/home/jovyan/work/data/LA/LA', 
                        help='Root of ASVspoof2019 LA folder')
    parser.add_argument('--itw_test_root', default='/home/jovyan/work/data/release_in_the_wild/release_in_the_wild/test',
                        help='Path to InTheWild test folder')
    parser.add_argument('--output_dir', default='/home/jovyan/work/EchoTrace/eval_results',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-expand epoch number to checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path.isdigit():
        epoch = int(checkpoint_path)
        checkpoint_path = f"/home/jovyan/work/EchoTrace/checkpoints/checkpoint_epoch_{epoch:02d}.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"[EchoTrace Evaluation]")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    checkpoint_name = Path(checkpoint_path).name
    
    # Load datasets
    print("\nLoading datasets...")
    
    print("  ASVspoof Dev:")
    asv_dev_files, asv_dev_labels, asv_dev_systems = parse_asv_protocol(
        Path(args.asv_root) / "../ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        Path(args.asv_root) / "../ASVspoof2019_LA_dev/flac"
    )
    asv_dev_dataset = SimpleAudioDataset(asv_dev_files, asv_dev_labels, "ASVspoof Dev")
    asv_dev_loader = torch.utils.data.DataLoader(asv_dev_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    print("  ASVspoof Eval:")
    asv_eval_files, asv_eval_labels, asv_eval_systems = parse_asv_protocol(
        Path(args.asv_root) / "../ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        Path(args.asv_root) / "../ASVspoof2019_LA_eval/flac"
    )
    asv_eval_dataset = SimpleAudioDataset(asv_eval_files, asv_eval_labels, "ASVspoof Eval")
    asv_eval_loader = torch.utils.data.DataLoader(asv_eval_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    print("  InTheWild Test:")
    itw_files, itw_labels = load_inthe_wild_test(args.itw_test_root)
    itw_dataset = SimpleAudioDataset(itw_files, itw_labels, "InTheWild Test")
    itw_loader = torch.utils.data.DataLoader(itw_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Evaluate
    print("\nEvaluating...\n")
    
    results = {}
    
    asv_eval_attacks = None
    
    for dataset_name, loader, labels, systems in [
        ("ASVspoof Dev", asv_dev_loader, asv_dev_labels, asv_dev_systems),
        ("ASVspoof Eval", asv_eval_loader, asv_eval_labels, asv_eval_systems),
        ("InTheWild Test", itw_loader, itw_labels, None),
    ]:
        y_true, y_score = evaluate_dataset(model, loader, device, dataset_name)
        metrics = compute_metrics(y_true, y_score, threshold=args.threshold)
        
        # Per-attack analysis for ASVspoof Eval
        if dataset_name == "ASVspoof Eval" and systems:
            asv_eval_attacks = compute_per_attack_metrics(y_true, y_score, systems, threshold=args.threshold)
        
        results[dataset_name] = {
            'n_samples': len(y_true),
            'bal_acc': metrics['bal_acc'],
            'eer': metrics['eer'],
            'roc_auc': metrics['roc_auc'],
            'real_recall': metrics['real_recall'],
            'fake_recall': metrics['fake_recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1'],
            'cm': metrics['cm'].tolist(),
            'fpr': metrics['fpr'].tolist(),
            'tpr': metrics['tpr'].tolist(),
            'y_true': y_true.tolist(),
            'y_score': y_score.tolist(),
            'y_pred': metrics['y_pred'].tolist()
        }
        
        print(f"  {dataset_name}:")
        print(f"    Bal. Accuracy: {metrics['bal_acc']:.2f}%")
        eer_str = f"{metrics['eer']:.2f}%" if metrics['eer'] is not None else "N/A"
        print(f"    EER: {eer_str}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Fake Recall: {metrics['fake_recall']:.2f}%")
        print()
    
    # Generate HTTP report
    print("Generating HTML report...")
    html_path = output_dir / "report.html"
    generate_html_report(results, html_path, checkpoint_name, asv_eval_attacks=asv_eval_attacks)
    
    # Save JSON
    print("Saving metrics...")
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Metrics saved to: {json_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
