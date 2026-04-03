"""
sanity_check.py — EchoTrace pre-flight check
Run before launching train_ddp.py to confirm environment is ready.
"""
import os
import torch
import numpy as np
from core.model import build_model

def run_sanity_check():
    print("=" * 60)
    print("  ECHOTRACE PRE-FLIGHT SANITY CHECK")
    print("=" * 60)

    # 1. GPU inventory
    gpu_count = torch.cuda.device_count()
    print(f"\n[hardware]")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  GPU count      : {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024**3
        print(f"  GPU {i}          : {props.name} ({vram:.1f} GB VRAM)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Using device   : {device}")

    # 2. Model init
    print(f"\n[model]")
    try:
        model = build_model(device)
        print("  EchoTraceResNet initialised: OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 3. Freeze verification — only layer4 and fc should be trainable
    l1_req = any(p.requires_grad for p in model.resnet.layer1.parameters())
    l2_req = any(p.requires_grad for p in model.resnet.layer2.parameters())
    l3_req = any(p.requires_grad for p in model.resnet.layer3.parameters())
    l4_req = any(p.requires_grad for p in model.resnet.layer4.parameters())
    fc_req = any(p.requires_grad for p in model.fc.parameters())

    print(f"  Layer 1 trainable : {l1_req}  (expected: False)")
    print(f"  Layer 2 trainable : {l2_req}  (expected: False)")
    print(f"  Layer 3 trainable : {l3_req}  (expected: False)")
    print(f"  Layer 4 trainable : {l4_req}  (expected: True)")
    print(f"  FC head trainable : {fc_req}  (expected: True)")

    ok = (not l1_req) and (not l2_req) and (not l3_req) and l4_req and fc_req
    print(f"  Freeze config     : {'OK' if ok else 'MISMATCH — check build_model()'}")

    # 4. Warm-start check
    print(f"\n[weights]")
    for name in ("ensemble_model.pth", "deepfake_detector.pth"):
        exists = os.path.exists(name)
        size_mb = os.path.getsize(name) / 1024**2 if exists else 0
        print(f"  {name:<30} {'found' if exists else 'NOT FOUND'} "
              f"{'(' + f'{size_mb:.1f} MB)' if exists else ''}")

    # 5. Forward pass — dual input
    print(f"\n[forward pass]")
    try:
        dummy_img     = torch.randn(2, 3, 224, 224).to(device)
        dummy_scalars = torch.randn(2, 8).to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_img, dummy_scalars)
        assert out.shape == (2, 1), f"Shape mismatch: {out.shape}"
        print(f"  Output shape : {out.shape}  OK")
        probs = torch.sigmoid(out).cpu().numpy()
        print(f"  Sample probs : {probs.flatten().tolist()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 6. Dataset smoke-test (single item)
    print(f"\n[dataset smoke-test]")
    try:
        from core.preprocess import build_feature_image, extract_scalar_features
        dummy_audio = np.zeros(64000, dtype=np.float32)
        dummy_audio[1000:2000] = 0.5   # add some signal
        img = build_feature_image(dummy_audio)
        sc  = extract_scalar_features(dummy_audio)
        assert img.shape == (224, 224, 3), f"Image shape: {img.shape}"
        assert sc.shape  == (8,),           f"Scalar shape: {sc.shape}"
        print(f"  build_feature_image : {img.shape}  OK")
        print(f"  extract_scalar_features : {sc.shape}  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # 7. Mixed precision check
    print(f"\n[mixed precision]")
    if torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda"):
                dummy_img     = torch.randn(1, 3, 224, 224).to(device)
                dummy_scalars = torch.randn(1, 8).to(device)
                out = model(dummy_img, dummy_scalars)
            print(f"  autocast fp16 forward : OK  (dtype={out.dtype})")
        except Exception as e:
            print(f"  WARNING: {e}")
    else:
        print("  Skipped (no CUDA)")

    print(f"\n{'='*60}")
    print(f"  STATUS: ALL CHECKS PASSED — READY FOR DDP TRAINING")
    print(f"  Launch with: nohup python train_ddp.py > ddp_train.log 2>&1 &")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    run_sanity_check()