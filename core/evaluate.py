import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from .model import build_model
from .preprocess import ASVDataset, InTheWildDataset
from tqdm import tqdm
import datetime
import pandas as pd

def setup_device():
    """Setup the best available device with optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

def evaluate_dataset(model, dataloader, device, dataset_name):
    """Evaluate model on a specific dataset and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print(f"\nEvaluating on {dataset_name}...")
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Eval {dataset_name}", leave=False)

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

            # Handle single sample batches (squeeze makes it 0-dim)
            if probabilities.ndim == 0:
                probabilities = np.array([probabilities])
                predictions = np.array([predictions])

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)
            
            # Update description with current accuracy (optional, adds overhead)
            # pbar.set_postfix({'acc': f"{np.mean(np.array(all_predictions) == np.array(all_labels))*100:.1f}%"})

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels) * 100

    # Calculate F1 Score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # PER-CLASS METRICS: Real (label=0) vs Fake (label=1)
    real_correct = cm[0, 0]
    real_total = cm[0, 0] + cm[0, 1]
    fake_correct = cm[1, 1]
    fake_total = cm[1, 0] + cm[1, 1]
    
    real_recall_val = real_correct / real_total * 100 if real_total > 0 else 0
    fake_recall_val = fake_correct / fake_total * 100 if fake_total > 0 else 0
    balanced_accuracy = (real_recall_val + fake_recall_val) / 2

    # ROC AUC & Correct EER
    try:
        from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Proper EER Calculation: Point where FPR == FNR (FNR = 1 - TPR)
        fnr = 1 - tpr
        eer = brentq(lambda x: x - interp1d(fpr, fnr)(x), 0., 1.)
        eer = eer * 100 # Percentage

        # PR Curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
    except Exception as e:
        print(f"⚠️ Metric Error: {e}")
        fpr, tpr, roc_auc, eer, precision, recall = None, None, None, None, None, None

    # Numerically stable Average Precision (PR AUC)
    pr_auc = average_precision_score(all_labels, all_probabilities)

    results = {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'real_recall': real_recall_val,
        'fake_recall': fake_recall_val,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'eer': eer,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision,
        'recall_curve': recall,
        'confusion_matrix': cm,
        'total_samples': len(all_labels),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

    return results

def print_evaluation_results(results):
    """Print comprehensive evaluation results with balanced metrics"""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS - {results['dataset'].upper()}")
    print(f"{'='*60}")

    print(f"✅ Real Recall (Real audio detected as real): {results['real_recall']:.2f}%")
    print(f"❌ Fake Recall (Fake audio detected as fake): {results['fake_recall']:.2f}%")
    print(f"⚖️  Balanced Accuracy: {results['balanced_accuracy']:.2f}%")
    print(f"🎯 F1 Score: {results['f1_score']:.4f}")
    print(f"📈 Overall Accuracy: {results['accuracy']:.2f}%")

    if results['roc_auc'] is not None:
        print(f"📊 ROC AUC Score: {results['roc_auc']:.4f}")

    print(f"📊 Precision-Recall AUC (PR AUC): {results['pr_auc']:.4f}")

    print(f"📋 Total Samples: {results['total_samples']}")

    # Confusion Matrix
    cm = results['confusion_matrix']
    print(f"\n🔢 Confusion Matrix:")
    print(f"           Predicted")
    print(f"           Real    Fake")
    print(f"Actual Real {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"      Fake {cm[1,0]:4d}   {cm[1,1]:4d}")

    # Classification Report
    print(f"\n📋 Detailed Classification Report:")
    report = classification_report(results['labels'], results['predictions'],
                                 target_names=['Real', 'Fake'], digits=4)
    print(report)

def create_evaluation_plots(results_list, save_dir):
    """Create and save comprehensive evaluation plots (CM, ROC, PR)"""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    for results in results_list:
        dataset_name = results['dataset']
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Forensic Analysis: {dataset_name}', fontsize=18, fontweight='bold', y=1.05)

        # 1. Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   ax=axes[0], cbar=False)
        axes[0].set_title('Confusion Matrix', fontsize=14)
        axes[0].set_ylabel('Actual Label')
        axes[0].set_xlabel('Predicted Label')

        # 2. ROC Curve + EER
        if results['fpr'] is not None and results['tpr'] is not None:
            axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
                         label=f'ROC (AUC = {results["roc_auc"]:.4f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
            axes[1].legend(loc="lower right")

        # 3. Precision-Recall Curve
        axes[2].plot(results['recall_curve'], results['precision_curve'], color='green', lw=2,
                     label=f'PR Curve (AUC = {results["pr_auc"]:.4f})')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_title('Precision-Recall Curve', fontsize=14)
        axes[2].legend(loc="lower left")

        plt.tight_layout()
        filename = f"visual_report_{dataset_name.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # Save detailed metrics to text file
    with open(os.path.join(save_dir, 'evaluation_summary.txt'), 'w') as f:
        for results in results_list:
            f.write(f"\n{'='*60}\n")
            f.write(f"EVALUATION RESULTS - {results['dataset'].upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Real Recall (Real detected as real): {results['real_recall']:.2f}%\n")
            f.write(f"Fake Recall (Fake detected as fake): {results['fake_recall']:.2f}%\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.2f}%\n")
            f.write(f"F1 Score: {results['f1_score']:.4f}\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
            if results['roc_auc'] is not None:
                f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {results['pr_auc']:.4f}\n")
            f.write(f"Total Samples: {results['total_samples']}\n")

            # Confusion matrix
            cm = results['confusion_matrix']
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"           Predicted\n")
            f.write(f"           Real    Fake\n")
            f.write(f"Actual Real {cm[0,0]:4d}   {cm[0,1]:4d}\n")
            f.write(f"      Fake {cm[1,0]:4d}   {cm[1,1]:4d}\n")
            
            # Save raw confusion matrix as CSV for each dataset
            cm_df = pd.DataFrame(cm, index=['Actual_Real', 'Actual_Fake'], columns=['Pred_Real', 'Pred_Fake'])
            cm_df.to_csv(os.path.join(save_dir, f"cm_{results['dataset'].replace(' ', '_').lower()}.csv"))

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on both datasets"""
    device = setup_device()

    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "ensemble_model.pth")

    # Load model
    model = build_model(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        # Handle state dict if it's a checkpoint dict
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        
        # Strip DDP prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"✅ Successfully loaded trained weights from {model_path}")
        
        # 4-GPU Maximization: Use DataParallel if multiple GPUs found
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"🚀 Using 4-GPU Max Mode: Distributing evaluation across {num_gpus} GPUs!")
            model = torch.nn.DataParallel(model)
        else:
            print(f"💡 One GPU detected: Using single-GPU inference.")
    else:
        print(f"❌ Error: {model_path} not found. Ensure training is complete!")
        return

    # Setup evaluation datasets
    # Using FULL datasets (subset_size=None) for maximum precision
    datasets_config = [
        {
            'name': 'ASVspoof Dev',
            'protocol': "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            'data_dir': "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_dev/flac",
            'dataset_class': ASVDataset,
            'subset_size': None
        },
        {
            'name': 'ASVspoof Eval (Test)',
            'protocol': "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            'data_dir': "/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_eval/flac",
            'dataset_class': ASVDataset,
            'subset_size': None
        },
        {
            'name': 'InTheWild Test',
            'data_dir': "/home/jovyan/work/data/release_in_the_wild/release_in_the_wild",
            'dataset_class': InTheWildDataset,
            'subset': 'test',
            'subset_size': None
        }
    ]

    results_list = []

    for config in datasets_config:
        print(f"\n🔍 Setting up {config['name']} dataset...")

        # Create dataset
        if config['dataset_class'] == ASVDataset:
            dataset = config['dataset_class'](
                config['protocol'],
                config['data_dir'],
                subset_size=config['subset_size'],
                augment=False
            )
        else:  # InTheWildDataset
            dataset = config['dataset_class'](
                config['data_dir'],
                subset=config['subset'],
                subset_size=config['subset_size'],
                augment=False
            )

        # Create dataloader with 4-GPU optimized settings
        num_gpus = torch.cuda.device_count()
        base_batch_size = 32  # Reduced from 64 to avoid OOM on 11GB GPUs
        batch_size = base_batch_size * num_gpus if torch.cuda.is_available() else 32
        
        # Scale workers by GPU count (4 per GPU for max saturation)
        num_workers = 4 * num_gpus if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()  # Pin memory only works with CUDA

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None
        )

        # Evaluate
        results = evaluate_dataset(model, dataloader, device, config['name'])
        results_list.append(results)
        print_evaluation_results(results)

    # Create plots and save results in a timestamped logs folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(BASE_DIR, "evaluation_logs", f"eval_{timestamp}")
    create_evaluation_plots(results_list, logs_dir)
    print(f"\n📊 Detailed results, confusion matrices, and plots saved to: {logs_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("📈 EVALUATION SUMMARY")
    print(f"{'='*60}")

    for results in results_list:
        print(f"{results['dataset']:<15}: {results['accuracy']:6.2f}% accuracy, "
              f"ROC AUC: {results['roc_auc']:.4f}")

if __name__ == "__main__":
    run_comprehensive_evaluation()