import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from .model import build_model
from .preprocess import ASVDataset, InTheWildDataset

def setup_device():
    """Setup the best available device with optimizations"""
    # Check for MPS (Metal Performance Shaders) - Mac M1/M2
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    return device

def evaluate_dataset(model, dataloader, device, dataset_name):
    """Evaluate model on a specific dataset and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print(f"\nEvaluating on {dataset_name}...")

    with torch.no_grad():
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)

            # Get probabilities and predictions
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels) * 100

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    except:
        roc_auc = None

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
    pr_auc = auc(recall, precision)

    # Per-class metrics
    bonafide_correct = cm[0, 0]
    bonafide_total = cm[0, 0] + cm[0, 1]
    spoof_correct = cm[1, 1]
    spoof_total = cm[1, 0] + cm[1, 1]

    bonafide_acc = bonafide_correct / bonafide_total * 100 if bonafide_total > 0 else 0
    spoof_acc = spoof_correct / spoof_total * 100 if spoof_total > 0 else 0

    # EER (Equal Error Rate) calculation
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    fpr = 1 - recall  # False Positive Rate = 1 - Specificity
    fnr = 1 - precision  # False Negative Rate = 1 - Sensitivity

    # Find EER
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, fnr)(x), 0., 1.)
    except:
        eer = None

    results = {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'bonafide_accuracy': bonafide_acc,
        'spoof_accuracy': spoof_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'eer': eer,
        'confusion_matrix': cm,
        'total_samples': len(all_labels),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

    return results

def print_evaluation_results(results):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS - {results['dataset'].upper()}")
    print(f"{'='*60}")

    print(f"📈 Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"✅ Bonafide (Real) Accuracy: {results['bonafide_accuracy']:.2f}%")
    print(f"❌ Spoof (Fake) Accuracy: {results['spoof_accuracy']:.2f}%")

    if results['roc_auc'] is not None:
        print(f"🎯 ROC AUC Score: {results['roc_auc']:.4f}")

    print(f"📊 Precision-Recall AUC: {results['pr_auc']:.4f}")

    if results['eer'] is not None:
        print(f"⚖️  Equal Error Rate (EER): {results['eer']:.4f}")

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
    """Create and save evaluation plots"""
    os.makedirs(save_dir, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deepfake Audio Detection - Evaluation Results', fontsize=16, fontweight='bold')

    for i, results in enumerate(results_list):
        dataset_name = results['dataset']

        # Confusion Matrix Heatmap
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   ax=axes[i, 0])
        axes[i, 0].set_title(f'{dataset_name} - Confusion Matrix')
        axes[i, 0].set_ylabel('Actual')
        axes[i, 0].set_xlabel('Predicted')

        # ROC Curve placeholder (would need FPR/TPR data)
        axes[i, 1].text(0.5, 0.5, f'{dataset_name}\nROC AUC: {results["roc_auc"]:.4f}',
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[i, 1].set_title(f'{dataset_name} - ROC AUC')
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed metrics to text file
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        for results in results_list:
            f.write(f"\n{'='*60}\n")
            f.write(f"EVALUATION RESULTS - {results['dataset'].upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Bonafide Accuracy: {results['bonafide_accuracy']:.2f}%\n")
            f.write(f"Spoof Accuracy: {results['spoof_accuracy']:.2f}%\n")
            if results['roc_auc'] is not None:
                f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {results['pr_auc']:.4f}\n")
            if results['eer'] is not None:
                f.write(f"EER: {results['eer']:.4f}\n")
            f.write(f"Total Samples: {results['total_samples']}\n")

            # Confusion matrix
            cm = results['confusion_matrix']
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"           Predicted\n")
            f.write(f"           Real    Fake\n")
            f.write(f"Actual Real {cm[0,0]:4d}   {cm[0,1]:4d}\n")
            f.write(f"      Fake {cm[1,0]:4d}   {cm[1,1]:4d}\n")

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on both datasets"""
    device = setup_device()

    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "deepfake_detector.pth")

    # Load model
    model = build_model(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Successfully loaded trained weights from {model_path}")
    except FileNotFoundError:
        print(f"❌ Error: {model_path} not found. Run train.py first!")
        return

    # Setup evaluation datasets
    datasets_config = [
        {
            'name': 'ASVspoof Dev',
            'protocol': os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
            'data_dir': os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_dev/flac"),
            'dataset_class': ASVDataset,
            'subset_size': 2000
        },
        {
            'name': 'InTheWild Val',
            'data_dir': os.path.join(BASE_DIR, "data/release_in_the_wild"),
            'dataset_class': InTheWildDataset,
            'subset': 'val',
            'subset_size': 1000
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

        # Create dataloader with device-optimized settings
        batch_size = 64 if torch.cuda.is_available() else 32  # Larger batch for CUDA, standard for MPS/CPU
        num_workers = 0  # Disable multiprocessing on macOS to avoid issues
        pin_memory = torch.cuda.is_available()  # Pin memory only works with CUDA, not MPS

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=None if num_workers == 0 else 2
        )

        # Evaluate
        results = evaluate_dataset(model, dataloader, device, config['name'])
        results_list.append(results)
        print_evaluation_results(results)

    # Create plots and save results
    plots_dir = os.path.join(BASE_DIR, "evaluation_results")
    create_evaluation_plots(results_list, plots_dir)
    print(f"\n📊 Detailed results and plots saved to: {plots_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("📈 EVALUATION SUMMARY")
    print(f"{'='*60}")

    for results in results_list:
        print(f"{results['dataset']:<15}: {results['accuracy']:6.2f}% accuracy, "
              f"ROC AUC: {results['roc_auc']:.4f}")

if __name__ == "__main__":
    run_comprehensive_evaluation()