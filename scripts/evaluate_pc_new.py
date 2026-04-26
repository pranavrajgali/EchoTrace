import os
import sys
import pathlib
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc, roc_curve, f1_score, 
    precision_score, recall_score, average_precision_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# --- DLL FIX FOR WINDOWS ---
# This attempts to resolve "DLL load failed" by explicitly adding the torch lib path.
# However, if an "Application Control policy" is blocking the file, this code cannot override it.
try:
    project_root = pathlib.Path(__file__).parent.parent.absolute()
    venv_lib = project_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if venv_lib.exists():
        os.add_dll_directory(str(venv_lib))
except Exception as e:
    print(f"DEBUG: DLL path injection skipped: {e}")

# --- TORCH IMPORT ---
try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:
    print("\n" + "!"*80)
    print("CRITICAL IMPORT ERROR: PyTorch could not be loaded.")
    print(f"Error: {e}")
    print("\nEXPLANATION:")
    print("Your Windows system is blocking 'torch/_C.pyd' due to an 'Application Control' policy.")
    print("This is likely 'Smart App Control' or a corporate security policy.")
    print("\nSUGGESTED FIXES:")
    print("1. Move the project folder to C:\\Projects (outside of C:\\Users).")
    print("2. If on a personal PC, check 'Smart App Control' settings in Windows Security.")
    print("3. Run the terminal as Administrator.")
    print("!"*80 + "\n")
    sys.exit(1)

# --- LOCAL IMPORTS ---
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from core.model import build_model
    from core.preprocess import ASVDataset, InTheWildDataset
except ImportError as e:
    print(f"❌ Failed to load project modules: {e}")
    sys.exit(1)

def setup_device(force_cpu=False):
    """Configures the computation device (GPU/CPU)."""
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 [GPU] Using: {torch.cuda.get_device_name(0)}")
        # Enable benchmark for speed
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        mode = "FORCED" if force_cpu else "FALLBACK"
        print(f"🐌 [{mode} CPU] Using CPU for evaluation. This will be significantly slower.")
    return device

def calculate_eer(labels, scores):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Interp function to find where fpr == fnr
    interp_fnr = interp1d(fpr, fnr)
    eer = brentq(lambda x: x - interp_fnr(x), 0, 1)
    return eer * 100, fpr, tpr

def evaluate(model, loader, device, name):
    """Runs evaluation on a dataset and returns metrics."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    print(f"\n🔍 [EVAL] Starting inference on: {name}")
    with torch.no_grad():
        for specs, scalars, labels in tqdm(loader, desc=f"   {name[:15]}"):
            specs, scalars = specs.to(device), scalars.to(device)
            outputs = model(specs, scalars)
            
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            # Ensure probs is an array even for batch size 1
            if probs.ndim == 0: probs = np.array([probs])
            
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(labels.numpy())
            y_prob.extend(probs)
            y_pred.extend(preds)
            
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    # Metrics calculation
    acc = np.mean(y_true == y_pred) * 100
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # EER and ROC
    try:
        eer, fpr, tpr = calculate_eer(y_true, y_prob)
        auc_score = auc(fpr, tpr)
    except Exception:
        eer, fpr, tpr, auc_score = 0, None, None, 0

    return {
        'name': name, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
        'eer': eer, 'auc': auc_score, 'cm': cm, 'fpr': fpr, 'tpr': tpr,
        'total': len(y_true)
    }

def plot_results(results_list, save_path):
    """Generates and saves performance visualizations."""
    os.makedirs(save_path, exist_ok=True)
    plt.style.use('ggplot') # High-fidelity look
    
    for res in results_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"EchoTrace Performance: {res['name']}", fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax1.set_title("Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        
        # 2. ROC Curve
        if res['fpr'] is not None:
            ax2.plot(res['fpr'], res['tpr'], color='darkorange', lw=2, label=f"ROC (AUC = {res['auc']:.4f})")
            ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax2.set_title("ROC Curve")
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.legend(loc="lower right")
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"eval_{res['name'].lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=150)
        plt.close()

def run_main(cpu_only=False):
    """Main execution flow."""
    device = setup_device(force_cpu=cpu_only)
    
    # Paths
    model_path = project_root / "ensemble_model.pth"
    data_root = pathlib.Path(r"C:\Users\Admin\Documents\data")
    
    if not model_path.exists():
        print(f"❌ FATAL: Model weights not found at {model_path}")
        return

    # Initialize Model
    print(f"📂 Loading model weights...")
    model = build_model(device)
    state = torch.load(model_path, map_location=device)
    if "model_state" in state: state = state["model_state"]
    # Clean DDP strings
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    
    # Dataset Configurations
    datasets = [
        {
            'name': 'ASVspoof2019_Dev',
            'class': ASVDataset,
            'args': [
                str(data_root / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"),
                str(data_root / "LA" / "ASVspoof2019_LA_dev" / "flac")
            ]
        },
        {
            'name': 'InTheWild_Test',
            'class': InTheWildDataset,
            'args': [str(data_root / "release_in_the_wild")],
            'kwargs': {'subset': 'test'}
        }
    ]

    results = []
    for d_cfg in datasets:
        try:
            ds = d_cfg['class'](*d_cfg['args'], **d_cfg.get('kwargs', {}), augment=False)
            loader = DataLoader(ds, batch_size=32, num_workers=0 if cpu_only else 4, pin_memory=not cpu_only)
            
            res = evaluate(model, loader, device, d_cfg['name'])
            results.append(res)
            
            print(f"   ✅ {d_cfg['name']}: ACC={res['acc']:.2f}% | EER={res['eer']:.2f}% | F1={res['f1']:.3f}")
        except Exception as e:
            print(f"   ❌ Failed {d_cfg['name']}: {e}")

    # Output
    report_path = project_root / "reports" / f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plot_results(results, report_path)
    print(f"\n✨ EVALUATION COMPLETE.")
    print(f"📊 Visualization reports saved to: {report_path}")

if __name__ == "__main__":
    # Change to True to force CPU if GPU DLLs are the issue
    FORCE_CPU = False 
    run_main(cpu_only=FORCE_CPU)
