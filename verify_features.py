import numpy as np
from core.preprocess import load_audio, build_feature_image, extract_scalar_features

# Load a sample audio
audio = load_audio("/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac")
print(f"✓ Audio shape: {audio.shape}")  # Should be (64000,) for 4 sec @ 16kHz
print(f"✓ Audio dtype: {audio.dtype}")
print(f"✓ Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

# Extract features
img = build_feature_image(audio)
print(f"\n✓ Feature image shape: {img.shape}")  # Should be (224, 224, 3)
print(f"✓ Feature image dtype: {img.dtype}")
print(f"✓ Feature image range: [{img.min()}, {img.max()}]")

# Check each channel
print(f"\n✓ Channel 1 (Mel): min={img[:,:,0].min()}, max={img[:,:,0].max()}, mean={img[:,:,0].mean():.2f}")
print(f"✓ Channel 2 (MFCC): min={img[:,:,1].min()}, max={img[:,:,1].max()}, mean={img[:,:,1].mean():.2f}")
print(f"✓ Channel 3 (Contrast): min={img[:,:,2].min()}, max={img[:,:,2].max()}, mean={img[:,:,2].mean():.2f}")

# Extract scalars
scalars = extract_scalar_features(audio)
print(f"\n✓ Scalars shape: {scalars.shape}")  # Should be (8,)
print(f"✓ Scalars dtype: {scalars.dtype}")
print(f"✓ Scalars values: {scalars}")
print(f"✓ Scalars range: [{scalars.min():.4f}, {scalars.max():.4f}]")