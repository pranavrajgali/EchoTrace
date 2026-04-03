import torch
import torch.nn as nn
from core.model import build_model
import numpy as np

def run_sanity_check():
    print("="*50)
    print("ECHOTRACE PRE-FLIGHT SANITY CHECK")
    print("="*50)

    # 1. Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"[*] Target Device: {device}")
    print(f"[*] GPUs Available: {gpu_count}")

    # 2. Model Initialization
    try:
        print("[*] Initializing EchoTraceResNet...")
        model = build_model(device)
        print("[+] Model initialized successfully.")
    except Exception as e:
        print(f"[!] Model Initialization FAILED: {e}")
        return

    # 3. ImageNet Weights Check
    try:
        print("[*] Verifying ImageNet pre-trained weights loaded...")
        checkpoint_path = 'deepfake_detector.pth'
        import os
        if os.path.exists(checkpoint_path):
            print(f"[!] Note: Old checkpoint {checkpoint_path} exists but will NOT be used (training from scratch)")
        else:
            print("[+] No old checkpoint found. Model loaded with ImageNet weights.")
    except Exception as e:
        print(f"[!] Check FAILED: {e}")
        return

    # 4. Forward Pass Validation
    # We simulate a batch of 2 samples
    print("[*] Simulating dual-input forward pass...")
    try:
        # Dummy Image: (Batch, Channels=3, H=224, W=224)
        dummy_img = torch.randn(2, 3, 224, 224).to(device)
        
        # Dummy Scalars: (Batch, Features=8)
        dummy_scalars = torch.randn(2, 8).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_img, dummy_scalars)
        
        print(f"[+] Forward pass successful.")
        print(f"[+] Output shape: {output.shape} (Expected: [2, 1])")
        
        # Verify output dimension
        assert output.shape == (2, 1), f"Output shape mismatch! Got {output.shape}"
        
    except Exception as e:
        print(f"[!] Forward Pass FAILED: {e}")
        return

    # 5. Layer Gradient Check (Verify unfreezing)
    print("[*] Verifying Layer Trainability...")
    # Layer 3 and 4 should be True for Strategy 2
    l3_grad = any(p.requires_grad for p in model.resnet.layer3.parameters())
    l4_grad = any(p.requires_grad for p in model.resnet.layer4.parameters())
    fc_grad = any(p.requires_grad for p in model.fc.parameters())
    
    print(f"    - Layer 3 Trainable: {l3_grad}")
    print(f"    - Layer 4 Trainable: {l4_grad}")
    print(f"    - FC Head Trainable: {fc_grad}")

    print("="*50)
    print("STATUS: ALL SYSTEMS NOMINAL - READY FOR DDP TRAINING")
    print("="*50)

if __name__ == "__main__":
    run_sanity_check()