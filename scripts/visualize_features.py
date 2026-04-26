import matplotlib.pyplot as plt
from core.preprocess import load_audio, build_feature_image
import numpy as np

# Load sample audio
audio = load_audio("/home/jovyan/work/data/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac")

# Build feature image
img = build_feature_image(audio)

# Display the 3 channels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Channel 1: Mel Spectrogram
im1 = axes[0].imshow(img[:, :, 0], cmap='viridis', aspect='auto')
axes[0].set_title('Channel 1: Mel Spectrogram', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time Frames')
axes[0].set_ylabel('Mel Bins')
plt.colorbar(im1, ax=axes[0])

# Channel 2: MFCC + Delta + Delta²
im2 = axes[1].imshow(img[:, :, 1], cmap='viridis', aspect='auto')
axes[1].set_title('Channel 2: MFCC + Delta + Δ²', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time Frames')
axes[1].set_ylabel('Cepstral Coefficient')
plt.colorbar(im2, ax=axes[1])

# Channel 3: Spectral Contrast + Chroma
im3 = axes[2].imshow(img[:, :, 2], cmap='viridis', aspect='auto')
axes[2].set_title('Channel 3: Spectral Contrast + Chroma', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Time Frames')
axes[2].set_ylabel('Feature Index')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('/home/jovyan/work/feature_map_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved to /home/jovyan/work/feature_map_visualization.png")
plt.close()

print(f"Image shape: {img.shape}")
print(f"All features extracted successfully!")