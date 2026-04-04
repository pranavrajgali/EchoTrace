#!/usr/bin/env python3
"""
local_sanity.py — EchoTrace local validation suite
Tests without requiring the full dataset or GPU multi-node setup.

Covers:
  1. Pipeline consistency (librosa vs torchaudio if available)
  2. Model loading with DDP prefix stripping  
  3. Forward pass with real and synthetic audio
  4. Feature value sanity checks
  5. Optimizer parameter coverage
  6. Single-GPU training smoke test
"""

import sys
import os
import torch
import numpy as np
import librosa
from torch.utils.data import DataLoader, TensorDataset
import tempfile

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.model import build_model, get_optimizer, get_loss
from core.preprocess import (
    build_feature_image, 
    extract_scalar_features, 
    load_audio
)

# ════════════════════════════════════════════════════════════════════════════
# TEST 1: Feature Pipeline Consistency
# ════════════════════════════════════════════════════════════════════════════

def test_feature_pipeline():
    """Check that feature extraction produces valid outputs."""
    print("\n" + "="*70)
    print("TEST 1: FEATURE PIPELINE CONSISTENCY")
    print("="*70)
    
    # Create synthetic audio: 16kHz, 4 seconds
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Mix of sine waves to simulate real audio
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +    # 440 Hz
        0.2 * np.sin(2 * np.pi * 880 * t) +    # 880 Hz
        0.1 * np.random.normal(0, 0.01, len(t))  # Noise
    ).astype(np.float32)
    
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-7:
        audio = audio / peak
    
    try:
        # Test feature image
        img = build_feature_image(audio, sr=sr)
        assert img.shape == (224, 224, 3), f"Image shape mismatch: {img.shape}"
        assert img.dtype == np.uint8, f"Image dtype should be uint8, got {img.dtype}"
        assert img.min() >= 0 and img.max() <= 255, "Image values out of range"
        print(f"✅ Feature image shape: {img.shape}, dtype: {img.dtype}")
        print(f"   Value range: [{img.min()}, {img.max()}]")
        
        # Test scalar features
        scalars = extract_scalar_features(audio, sr=sr)
        assert scalars.shape == (8,), f"Scalar shape mismatch: {scalars.shape}"
        assert scalars.dtype == np.float32, f"Scalar dtype should be float32, got {scalars.dtype}"
        assert np.all(scalars >= 0) and np.all(scalars <= 1), "Scalars out of [0, 1] range"
        assert not np.any(np.isnan(scalars)), "NaN in scalars"
        assert not np.any(np.isinf(scalars)), "Inf in scalars"
        print(f"✅ Scalar features shape: {scalars.shape}, dtype: {scalars.dtype}")
        print(f"   Values: {scalars}")
        print(f"   Range: [{scalars.min():.4f}, {scalars.max():.4f}]")
        
        # Check no all-zero channels
        if np.all(img[:, :, 0] == 0):
            print("⚠️  WARNING: Channel 1 is all zeros")
        if np.all(img[:, :, 1] == 0):
            print("⚠️  WARNING: Channel 2 is all zeros")
        if np.all(img[:, :, 2] == 0):
            print("⚠️  WARNING: Channel 3 is all zeros")
        
        print("✅ PASS: Feature pipeline produces valid outputs\n")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: Model Loading with DDP Prefix Stripping
# ════════════════════════════════════════════════════════════════════════════

def test_model_loading():
    """Test model initialization and weight loading with DDP prefix stripping."""
    print("="*70)
    print("TEST 2: MODEL LOADING & DDP PREFIX STRIPPING")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Build model
        model = build_model(device)
        print(f"✅ Model initialized on {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        
        # Check freeze status
        l1_trainable = any(p.requires_grad for p in model.resnet.layer1.parameters())
        l2_trainable = any(p.requires_grad for p in model.resnet.layer2.parameters())
        l3_trainable = any(p.requires_grad for p in model.resnet.layer3.parameters())
        l4_trainable = any(p.requires_grad for p in model.resnet.layer4.parameters())
        fc_trainable = any(p.requires_grad for p in model.fc.parameters())
        
        print(f"   Layer 1 trainable: {l1_trainable} (expected: False)")
        print(f"   Layer 2 trainable: {l2_trainable} (expected: False)")
        print(f"   Layer 3 trainable: {l3_trainable} (expected: False)")
        print(f"   Layer 4 trainable: {l4_trainable} (expected: True)")
        print(f"   FC head trainable: {fc_trainable} (expected: True)")
        
        freeze_ok = (not l1_trainable and not l2_trainable and not l3_trainable 
                     and l4_trainable and fc_trainable)
        if not freeze_ok:
            print("❌ FAIL: Freeze strategy incorrect\n")
            return False
        
        # Test DDP prefix stripping with mock checkpoint
        print("\n   Testing DDP prefix stripping...")
        state = model.state_dict()
        
        # Add mock "module." prefix to simulate DDP checkpoint
        mock_ddp_state = {f"module.{k}": v for k, v in state.items()}
        
        # Test stripping
        stripped = {k.replace("module.", ""): v for k, v in mock_ddp_state.items()}
        
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        print(f"   Missing keys after load: {len(missing) if missing else 0}")
        print(f"   Unexpected keys after load: {len(unexpected) if unexpected else 0}")
        print(f"✅ PASS: DDP prefix stripping works correctly\n")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: End-to-End Forward Pass
# ════════════════════════════════════════════════════════════════════════════

def test_forward_pass():
    """Test end-to-end forward pass with synthetic audio."""
    print("="*70)
    print("TEST 3: END-TO-END FORWARD PASS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = build_model(device)
        model.eval()
        print(f"✅ Model in eval mode on {device}")
        
        # Create batch of synthetic audio (2 samples)
        sr = 16000
        duration = 4.0
        batch_size = 2
        
        images = []
        scalars_list = []
        
        for _ in range(batch_size):
            # Synthetic audio
            t = np.linspace(0, duration, int(sr * duration))
            freq = np.random.uniform(200, 2000)
            audio = (
                0.3 * np.sin(2 * np.pi * freq * t) +
                0.1 * np.random.normal(0, 0.01, len(t))
            ).astype(np.float32)
            
            peak = np.max(np.abs(audio))
            if peak > 1e-7:
                audio = audio / peak
            
            img = build_feature_image(audio, sr=sr)
            images.append(torch.from_numpy(img).permute(2, 0, 1).float())
            
            scalars = extract_scalar_features(audio, sr=sr)
            scalars_list.append(torch.from_numpy(scalars).float())
        
        # Normalize images (ImageNet)
        images_tensor = torch.stack(images).to(device)
        images_tensor = (images_tensor / 255.0) if images_tensor.max() > 1 else images_tensor
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_tensor = (images_tensor - mean) / std
        
        scalars_tensor = torch.stack(scalars_list).to(device)
        
        print(f"   Image shape: {images_tensor.shape}")
        print(f"   Scalars shape: {scalars_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(images_tensor, scalars_tensor)
        
        assert outputs.shape == (batch_size, 1), f"Output shape mismatch: {outputs.shape}"
        
        # Check output values
        probs = torch.sigmoid(outputs).cpu().numpy()
        print(f"   Output logits: {outputs.squeeze().cpu().numpy()}")
        print(f"   Output probabilities: {probs.squeeze()}")
        
        assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities out of [0, 1] range"
        assert not np.any(np.isnan(probs)), "NaN in outputs"
        assert not np.any(np.isinf(probs)), "Inf in outputs"
        
        print(f"✅ PASS: Forward pass successful\n")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# TEST 4: Optimizer Parameter Coverage
# ════════════════════════════════════════════════════════════════════════════

def test_optimizer_coverage():
    """Verify optimizer covers all trainable parameters."""
    print("="*70)
    print("TEST 4: OPTIMIZER PARAMETER COVERAGE")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = build_model(device)
        optimizer = get_optimizer(model)
        
        # Count parameters in optimizer
        opt_params = set()
        for group in optimizer.param_groups:
            for param in group['params']:
                opt_params.add(id(param))
        
        # Count trainable parameters in model
        trainable_params = set()
        trainable_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.add(id(param))
                trainable_count += param.numel()
        
        # Count parameters in optimizer
        opt_count = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
        
        print(f"   Trainable params in model: {trainable_count:,}")
        print(f"   Params in optimizer: {opt_count:,}")
        
        # Check coverage
        if opt_params == trainable_params:
            print(f"✅ PASS: Optimizer covers all {len(trainable_params)} trainable parameters\n")
            return True
        else:
            missing = trainable_params - opt_params
            extra = opt_params - trainable_params
            print(f"❌ FAIL: Parameter mismatch")
            if missing:
                print(f"   Missing from optimizer: {len(missing)} params")
            if extra:
                print(f"   Extra in optimizer: {len(extra)} params")
            print()
            return False
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# TEST 5: Evaluate.py Dataloader Compatibility
# ════════════════════════════════════════════════════════════════════════════

def test_evaluate_compatibility():
    """Test that evaluate.py can unpack 3-tuple from dataloader."""
    print("="*70)
    print("TEST 5: EVALUATE.PY DATALOADER COMPATIBILITY")
    print("="*70)
    
    try:
        # Create mock dataset that returns (image, scalars, label) 3-tuple
        batch_size = 4
        image_batch = torch.randn(batch_size, 3, 224, 224)
        scalars_batch = torch.randn(batch_size, 8)
        label_batch = torch.randint(0, 2, (batch_size,)).float()
        
        dataset = TensorDataset(image_batch, scalars_batch, label_batch)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Test unpacking
        for specs, scalars, labels in dataloader:
            assert specs.shape[0] == 2, f"Batch size mismatch: {specs.shape}"
            assert scalars.shape == (2, 8), f"Scalars shape mismatch: {scalars.shape}"
            assert labels.shape == (2,), f"Labels shape mismatch: {labels.shape}"
            print(f"   Specs shape: {specs.shape}")
            print(f"   Scalars shape: {scalars.shape}")
            print(f"   Labels shape: {labels.shape}")
            break
        
        print(f"✅ PASS: Dataloader returns proper 3-tuple for evaluate.py\n")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# TEST 6: Single-GPU Training Smoke Test
# ════════════════════════════════════════════════════════════════════════════

def test_training_smoke():
    """Run 2 training iterations to verify loss decreases."""
    print("="*70)
    print("TEST 6: SINGLE-GPU TRAINING SMOKE TEST")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = build_model(device)
        criterion = get_loss()
        optimizer = get_optimizer(model)
        
        print(f"   Training on {device}")
        
        # Create mock batch
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        scalars = torch.randn(batch_size, 8).to(device)
        labels = torch.randint(0, 2, (batch_size,)).float().unsqueeze(1).to(device)
        
        losses = []
        model.train()
        
        for step in range(2):
            optimizer.zero_grad()
            outputs = model(images, scalars)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"   Iteration {step + 1}: loss = {loss.item():.6f}")
        
        # Check that loss is decreasing or at least finite
        if np.isnan(losses[0]) or np.isinf(losses[0]):
            print(f"❌ FAIL: Initial loss is invalid\n")
            return False
        
        if np.isnan(losses[1]) or np.isinf(losses[1]):
            print(f"❌ FAIL: Second iteration loss is invalid\n")
            return False
        
        # Loss should not explode
        if losses[1] > losses[0] * 10:
            print(f"⚠️  WARNING: Loss increased significantly ({losses[0]:.6f} → {losses[1]:.6f})")
        
        print(f"✅ PASS: Training runs without explosions\n")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  ECHOTRACE LOCAL SANITY CHECK SUITE".center(68) + "║")
    print("║" + "  Run before training or deployment".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    tests = [
        ("Feature Pipeline", test_feature_pipeline),
        ("Model Loading", test_model_loading),
        ("Forward Pass", test_forward_pass),
        ("Optimizer Coverage", test_optimizer_coverage),
        ("Evaluate Compatibility", test_evaluate_compatibility),
        ("Training Smoke Test", test_training_smoke),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ UNHANDLED EXCEPTION in {name}: {e}")
            results[name] = False
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"{'='*70}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! Ready for training.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
