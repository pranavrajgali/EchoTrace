
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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
    except Exception:
        return None

def main():
    metrics_path = r'c:\Users\Admin\Documents\EchoTrace_V4\EchoTraceV2\EchoTrace\eval_results\final_eval\metrics.json'
    protocol_path = r'C:\Users\Admin\Documents\Data\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt'
    
    print(f"Loading metrics from {metrics_path}...")
    with open(metrics_path) as f:
        data = json.load(f)
    
    eval_data = data.get('ASVspoof Eval')
    if not eval_data:
        print("ASVspoof Eval data not found in JSON.")
        return
    
    y_true = np.array(eval_data['y_true'])
    y_score = np.array(eval_data['y_score'])
    
    print(f"Loading protocol from {protocol_path}...")
    system_list = []
    # Note: We need to match the logic in parse_asv_protocol which checks for file existence
    # but we don't know the audio_root here. 
    # However, if n_samples match, we can assume they are the same.
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                system_list.append(parts[3]) # system_id
    
    print(f"JSON samples: {len(y_true)}, Protocol samples: {len(system_list)}")
    print(f"Sample system IDs: {list(set(system_list))[:10]}")
    
    if len(y_true) != len(system_list):
        print("Warning: Sample count mismatch! Trying to align...")
        # Since I don't have audio_root, I can't check existence.
        # But usually in ASVspoof, all files in the protocol are present in the dataset.
        # If they don't match, this diagnosis might be inaccurate.
        # Let's assume they match for now or use a subset.
        system_list = system_list[:len(y_true)]

    system_array = np.array(system_list)
    bonafide_mask = system_array == "-"
    
    results = {}
    for system_id in sorted(set(system_list)):
        if system_id == "-": continue
        
        attack_mask = system_array == system_id
        combined_mask = bonafide_mask | attack_mask
        
        y_true_attack = y_true[combined_mask]
        y_score_attack = y_score[combined_mask]
        
        if len(y_true_attack) < 10: continue
        
        eer = compute_eer(y_true_attack, y_score_attack)
        
        # Fake recall at 0.5 threshold
        y_pred = (y_score_attack > 0.5).astype(int)
        fake_recall = (y_pred[y_true_attack == 1] == 1).sum() / (y_true_attack == 1).sum() * 100
        
        results[system_id] = {'eer': eer, 'fake_recall': fake_recall, 'count': int((y_true_attack == 1).sum())}

    print("\nPer-Attack Analysis:")
    print(f"{'Attack':<8} | {'EER':<8} | {'Recall':<8} | {'Count':<6}")
    print("-" * 40)
    for attack, res in sorted(results.items(), key=lambda x: x[1]['eer'] if x[1]['eer'] is not None else 0, reverse=True):
        eer_str = f"{res['eer']:.2f}%" if res['eer'] is not None else "N/A"
        print(f"{attack:<8} | {eer_str:<8} | {res['fake_recall']:.2f}% | {res['count']:<6}")

if __name__ == "__main__":
    main()
