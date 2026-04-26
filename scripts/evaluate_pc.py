import os
import sys
import pathlib

# Try to add torch lib to DLL search path for Windows
try:
    lib_path = pathlib.Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if lib_path.exists():
        os.add_dll_directory(str(lib_path))
except Exception:
    pass

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
import pandas as pd

import sys
import pathlib

# Add project root to sys.path for absolute imports
root_path = pathlib.Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Core imports
from core.model import build_model
from core.preprocess import ASVDataset, InTheWildDataset

def setup_pc_device():
    """Setup the local PC GPU with optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU Mode Active: Using {gpu_count}x {gpu_name}")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not found: Using CPU (This will be SLOW)")
    return device

def evaluate_dataset(model, dataloader, device, dataset_name):
    """Evaluate model on a specific dataset and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print(f"\n🔍 Evaluating on {dataset_name}...")
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Progress: {dataset_name}", leave=True)

    with torch.no_grad():
        for specs, scalars, labels in pbar:
            specs = specs.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            outputs = model(specs, scalars)

            # Get probabilities and predictions
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            labels = labels.cpu().numpy()

            # Handle single sample batches
            if probabilities.ndim == 0:
                probabilities = np.array([probabilities])
                predictions = np.array([predictions])

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Metrics
    accuracy = np.mean(all_predictions == all_labels) * 100
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    # ROC AUC & Correct EER
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Correct EER: Point where FPR == FNR (FNR = 1 - TPR)
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
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
        print(f"⚠️ EER calculation failed: {e}")
        fpr, tpr, roc_auc, eer = None, None, None, None

    # Numerically stable PR AUC (Average Precision)
    from sklearn.metrics import average_precision_score, precision_recall_curve
    pr_auc = average_precision_score(all_labels, all_probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)

    # Balanced Accuracy
    real_recall = (cm[0,0] / (cm[0,0] + cm[0,1]) * 100) if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_recall = (cm[1,1] / (cm[1,0] + cm[1,1]) * 100) if (cm[1,0] + cm[1,1]) > 0 else 0
    balanced_acc = (real_recall + fake_recall) / 2

    return {
        'dataset': dataset_name, 'accuracy': accuracy, 'f1_score': f1,
        'precision': precision, 'recall': recall,
        'real_recall': real_recall, 'fake_recall': fake_recall, 'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'eer': eer,
        'fpr': fpr, 'tpr': tpr, 'confusion_matrix': cm, 'total_samples': len(all_labels),
        'predictions': all_predictions, 'labels': all_labels, 'probabilities': all_probabilities,
        'precision_curve': precision_curve, 'recall_curve': recall_curve
    }

def print_metrics(results):
    print(f"\n{'='*60}")
    print(f"📊 {results['dataset'].upper()} RESULTS")
    print(f"{'='*60}")
    print(f"⚖️  Balanced Accuracy: {results['balanced_accuracy']:.2f}%")
    print(f"🎯 F1 Score:          {results['f1_score']:.4f}")
    print(f"📏 Precision:         {results['precision']:.4f}")
    print(f"🔄 Recall (Fake):     {results['fake_recall']:.2f}%")
    print(f"🔄 Recall (Real):     {results['real_recall']:.2f}%")
    print(f"📊 ROC AUC Score:     {results['roc_auc']:.4f}" if results['roc_auc'] else "N/A")
    if results['eer'] is not None:
        print(f"🔐 EER:               {results['eer']:.4f}%")
    print(f"📋 Total Samples:      {results['total_samples']}")

def create_plots(results_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('dark_background')
    
    for results in results_list:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Forensic Analysis: {results["dataset"]}', fontsize=20, fontweight='bold', color='#E8443A')
        
        # 1. Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0], 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                    annot_kws={"size": 16, "weight": "bold"})
        axes[0].set_title('Confusion Matrix', fontsize=14, pad=10)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # 2. ROC Curve
        if results['fpr'] is not None:
            axes[1].plot(results['fpr'], results['tpr'], color='#E8443A', lw=3, label=f'ROC (AUC={results["roc_auc"]:.4f})')
            axes[1].plot([0, 1], [0, 1], color='#5A5A5E', lw=1, linestyle='--')
            
            if results['eer'] is not None:
                val = results['eer'] / 100.0
                axes[1].plot(val, 1-val, 'wo', markersize=8, label=f'EER={results["eer"]:.2f}%')
            
            axes[1].set_title('ROC Curve (Detection Sensitivity)', fontsize=14, pad=10)
            axes[1].set_xlabel('False Positive Rate', fontsize=12)
            axes[1].set_ylabel('True Positive Rate', fontsize=12)
            axes[1].legend(loc='lower right', fontsize=10)
            axes[1].grid(alpha=0.2)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"eval_{results['dataset'].lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
        print(f"📈 Saved plot: {filename}")

def run_pc_evaluation():
    device = setup_pc_device()
    
    # Correct Pathing: BASE_DIR is 'scripts', so model is in parent
    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
    model_path = os.path.join(PROJECT_ROOT, "ensemble_model.pth")
    
    # Dataset Paths (PC Windows Structure)
    DATA_ROOT = r"C:\Users\Admin\Documents\data"
    LA_ROOT = os.path.join(DATA_ROOT, "LA")
    CM_PROTOCOLS = os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols")
    
    configs = [
        {
            'name': 'ASVspoof 2019 Dev',
            'protocol': os.path.join(CM_PROTOCOLS, "ASVspoof2019.LA.cm.dev.trl.txt"),
            'data_dir': os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac"),
            'class': ASVDataset, 'subset_size': None
        },
        {
            'name': 'InTheWild Test',
            'data_dir': os.path.join(DATA_ROOT, "release_in_the_wild"),
            'class': InTheWildDataset, 'subset': 'test', 'subset_size': None
        }
    ]

    # Load Model
    model = build_model(device)
    if os.path.exists(model_path):
        print(f"📂 Loading weights from: {model_path}")
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state: 
            state = state["model_state"]
        
        # Clean DDP prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        
        msg = model.load_state_dict(state, strict=False)
        print(f"✅ Load status: {msg}")
        
        if torch.cuda.device_count() > 1: 
            model = torch.nn.DataParallel(model)
    else:
        print(f"❌ Error: {model_path} missing!")
        return

    results_list = []
    for cfg in configs:
        print(f"\n📂 Initializing {cfg['name']}...")
        try:
            if cfg['class'] == ASVDataset:
                ds = cfg['class'](cfg['protocol'], cfg['data_dir'], subset_size=cfg['subset_size'], augment=False)
            else:
                ds = cfg['class'](cfg['data_dir'], subset=cfg['subset'], subset_size=cfg['subset_size'], augment=False)

            loader = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
            res = evaluate_dataset(model, loader, device, cfg['name'])
            results_list.append(res)
            print_metrics(res)
        except Exception as e:
            print(f"❌ Failed to evaluate {cfg['name']}: {e}")

    # Save outputs to reports folder for easy access
    reports_dir = os.path.join(PROJECT_ROOT, "reports", f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    create_plots(results_list, reports_dir)
    print(f"\n✨ Evaluation Complete. Presentation graphs saved to: {reports_dir}")

if __name__ == "__main__":
    run_pc_evaluation()
