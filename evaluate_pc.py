import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime
import pandas as pd

# Core imports (assuming running from root)
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
    f1 = f1_score(all_labels, all_predictions, average='binary')
    cm = confusion_matrix(all_labels, all_predictions)

    # ROC AUC & Correct EER
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Correct EER: Point where FPR == FNR (FNR = 1 - TPR)
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        fnr = 1 - tpr
        eer = brentq(lambda x: x - interp1d(fpr, fnr)(x), 0., 1.)
        eer = eer * 100 # Percentage
    except Exception as e:
        fpr, tpr, roc_auc, eer = None, None, None, None

    # Numerically stable PR AUC (Average Precision)
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(all_labels, all_probabilities)

    # Balanced Accuracy
    real_recall = (cm[0,0] / (cm[0,0] + cm[0,1]) * 100) if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_recall = (cm[1,1] / (cm[1,0] + cm[1,1]) * 100) if (cm[1,0] + cm[1,1]) > 0 else 0
    balanced_acc = (real_recall + fake_recall) / 2

    return {
        'dataset': dataset_name, 'accuracy': accuracy, 'f1_score': f1,
        'real_recall': real_recall, 'fake_recall': fake_recall, 'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'eer': eer,
        'fpr': fpr, 'tpr': tpr, 'confusion_matrix': cm, 'total_samples': len(all_labels),
        'predictions': all_predictions, 'labels': all_labels, 'probabilities': all_probabilities
    }

def print_metrics(results):
    print(f"\n{'='*60}")
    print(f"📊 {results['dataset'].upper()} RESULTS")
    print(f"{'='*60}")
    print(f"⚖️  Balanced Accuracy: {results['balanced_accuracy']:.2f}%")
    print(f"🎯 F1 Score:          {results['f1_score']:.4f}")
    print(f"📊 ROC AUC Score:     {results['roc_auc']:.4f}" if results['roc_auc'] else "N/A")
    print(f"📊 PR AUC Score:      {results['pr_auc']:.4f}")
    print(f"📋 Total Samples:      {results['total_samples']}")

def create_plots(results_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    for results in results_list:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Forensic Analysis: {results["dataset"]}', fontsize=16, fontweight='bold')
        
        # CM
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title('Confusion Matrix')
        
        # ROC
        if results['fpr'] is not None:
            axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2, label=f'ROC (AUC={results["roc_auc"]:.4f})')
            axes[1].plot([0, 1], [0, 1], 'navy', linestyle='--')
            
            axes[1].set_title('ROC Curve')
            axes[1].legend()
            
        # PR
        axes[2].plot(results['recall_curve'], results['precision_curve'], color='green', lw=2, label=f'AUC={results["pr_auc"]:.4f}')
        axes[2].set_title('PR Curve')
        axes[2].legend()
        
        plt.savefig(os.path.join(save_dir, f"plot_{results['dataset'].lower().replace(' ', '_')}.png"))
        plt.close()

def run_pc_evaluation():
    device = setup_pc_device()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "ensemble_model.pth")
    
    # Dataset Paths (PC Windows Structure)
    DATA_ROOT = r"C:\Users\Admin\Documents\data"
    LA_ROOT = os.path.join(DATA_ROOT, "LA")
    CM_PROTOCOLS = os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols")
    
    configs = [
        {
            'name': 'ASVspoof Dev',
            'protocol': os.path.join(CM_PROTOCOLS, "ASVspoof2019.LA.cm.dev.trl.txt"),
            'data_dir': os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac"),
            'class': ASVDataset, 'subset_size': None
        },
        {
            'name': 'ASVspoof Eval (Full)',
            'protocol': os.path.join(CM_PROTOCOLS, "ASVspoof2019.LA.cm.eval.trl.txt"),
            'data_dir': os.path.join(LA_ROOT, "ASVspoof2019_LA_eval", "flac"),
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
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state: state = state["model_state"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
        print(f"✅ Weights loaded from {model_path}")
    else:
        print(f"❌ Error: {model_path} missing in EchoTrace folder!")
        return

    results_list = []
    for cfg in configs:
        print(f"\n📂 Initializing {cfg['name']}...")
        if cfg['class'] == ASVDataset:
            ds = cfg['class'](cfg['protocol'], cfg['data_dir'], subset_size=cfg['subset_size'], augment=False)
        else:
            ds = cfg['class'](cfg['data_dir'], subset=cfg['subset'], subset_size=cfg['subset_size'], augment=False)

        loader = DataLoader(ds, batch_size=32 * max(1, torch.cuda.device_count()), num_workers=4, pin_memory=True)
        res = evaluate_dataset(model, loader, device, cfg['name'])
        results_list.append(res)
        print_metrics(res)

    logs_dir = os.path.join(BASE_DIR, "pc_eval_logs", f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    create_plots(results_list, logs_dir)
    print(f"\n✨ Evaluation Complete. Full forensic logs saved to: {logs_dir}")

if __name__ == "__main__":
    run_pc_evaluation()
