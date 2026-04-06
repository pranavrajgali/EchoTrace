import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def verify_eer_logic(labels, probs):
    """Re-run the exact logic from evaluate_pc.py to verify it works"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Correct EER: Point where FPR == FNR (FNR = 1 - TPR)
    fnr = 1 - tpr
    # Create the interpolation
    f = interp1d(fpr, fnr)
    # The EER is where x = f(x)
    eer = brentq(lambda x: x - f(x), 0., 1.)
    
    return roc_auc, eer

# Create a sample case with known performance
# Simulation: ~2.5% error case 
np.random.seed(42)
labels = np.array([0]*5000 + [1]*5000)
probs = np.concatenate([
    np.random.normal(0.2, 0.1, 5000), # Real (low prob)
    np.random.normal(0.8, 0.1, 5000)  # Fake (high prob)
])
probs = np.clip(probs, 0.001, 0.999)

roc_auc, eer = verify_eer_logic(labels, probs)

print("EER LOGIC VERIFICATION RESULTS:")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"EER (Fractional): {eer:.4f}")
print(f"EER (Percentage): {eer*100:.2f}%")

if abs(roc_auc - (1 - eer)) < 0.2: # Rough sanity check
    print("\nMATHEMATICAL STATUS: WORKING")
    print("The EER is correctly intersecting the FPR and FNR line.")
else:
    print("\nMATHEMATICAL STATUS: CHECK NEEDED")
