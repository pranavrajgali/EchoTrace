import sys
import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import base64
import asyncio
from io import BytesIO

# --- DYNAMIC PATH INJECTION ---
# 1. Add the project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# 2. Add the core folder specifically to sys.path
# This allows 'from model import ...' to work even if called from inference.py
core_path = os.path.join(root_path, 'core')
if core_path not in sys.path:
    sys.path.append(core_path)
# ------------------------------

# Now these imports will find each other correctly
from core.model import build_model
from core.inference import run_inference
from core.preprocess import load_audio, build_feature_image, extract_scalar_features

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

# ── Scalar feature names ──
SCALAR_NAMES = [
    'Spectral Flatness',
    'Zero Crossing Rate',
    'F1 Formant',
    'F2 Formant',
    'F3 Formant',
    'Voiced Ratio',
    'Harmonic-to-Noise Ratio',
    'Cepstral Peak Prominence',
]

def generate_forensic_report(audio_path):
    """
    Comprehensive forensic report: analyzes audio, extracts features, runs inference,
    and generates a detailed visualization with 3-channel features + 8 scalar values.
    """
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        error_msg = f"Unsupported File Format ({file_ext}). Please use: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        print(f"\n[ERROR] {error_msg}")
        return {"result": "ERROR", "confidence": "0%", "error": error_msg}, None

    if not os.path.exists(audio_path):
        print(f"[X] Error: File not found at {audio_path}")
        return

    print(f"\n[*] Analyzing: {os.path.basename(audio_path)}")
    
    # Load audio with librosa (for visualization)
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Load audio with core.preprocess (for features)
    audio_core = load_audio(audio_path, target_sr=16000, duration=4.0, random_crop=False)
    
    # Extract features
    print("   • Extracting 3-channel feature image...")
    feature_image = build_feature_image(audio_core, sr=sr)  # (224, 224, 3) uint8
    
    print("   • Extracting 8-dimensional scalar features...")
    scalars = extract_scalar_features(audio_core, sr=sr)  # (8,) normalized to [0, 1]
    
    # Run inference
    print("   • Running model inference...")
    with open(audio_path, 'rb') as f:
        file_bytes = f.read()
    result = asyncio.run(run_inference(file_bytes))
    
    # --- VISUALIZATION (3x3 grid) ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#080d1a')
    plt.suptitle(f"EchoTrace Forensic Analysis Report\n{os.path.basename(audio_path)}", 
                 color='white', fontsize=24, fontweight='bold', y=0.98)
    
    # ── ROW 1: Waveform, Spectrogram, Grad-CAM ──
    
    # (1,1) Waveform
    ax1 = plt.subplot(3, 3, 1)
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8', alpha=0.8)
    ax1.set_title("① Time-Domain Waveform", color='#8aa4cc', fontweight='bold', fontsize=11)
    ax1.set_facecolor('#0f172a')
    ax1.set_ylabel('Amplitude')
    
    # (1,2) Spectrogram
    ax2 = plt.subplot(3, 3, 2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224, n_fft=1024)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title("② Spectral Fingerprint (Mel)", color='#8aa4cc', fontweight='bold', fontsize=11)
    plt.colorbar(img, ax=ax2, label='dB')
    
    # (1,3) Grad-CAM Heatmap
    ax3 = plt.subplot(3, 3, 3)
    try:
        heatmap_img = Image.open(BytesIO(base64.b64decode(result['heatmap'])))
        ax3.imshow(heatmap_img)
        ax3.set_title("③ AI Layer-4 Attention", color='#8aa4cc', fontweight='bold', fontsize=11)
    except:
        ax3.text(0.5, 0.5, 'Heatmap unavailable', ha='center', va='center', color='#f87171', fontsize=12)
    ax3.axis('off')
    
    # ── ROW 2: 3-Channel Feature Image ──
    
    # (2,1) Channel 1: Mel Spectrogram
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(feature_image[:, :, 0], cmap='viridis', aspect='auto')
    ax4.set_title("④ Ch1: Mel Spectrogram", color='#8aa4cc', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Time Frames')
    ax4.set_ylabel('Mel Bins')
    plt.colorbar(im4, ax=ax4)
    
    # (2,2) Channel 2: MFCC + Delta + ΔΔ
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(feature_image[:, :, 1], cmap='plasma', aspect='auto')
    ax5.set_title("⑤ Ch2: MFCC+Δ+Δ²", color='#8aa4cc', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Time Frames')
    ax5.set_ylabel('Cepstral Coeff')
    plt.colorbar(im5, ax=ax5)
    
    # (2,3) Channel 3: Spectral Contrast + Chroma
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(feature_image[:, :, 2], cmap='coolwarm', aspect='auto')
    ax6.set_title("⑥ Ch3: Contrast+Chroma", color='#8aa4cc', fontweight='bold', fontsize=11)
    ax6.set_xlabel('Time Frames')
    ax6.set_ylabel('Feature Index')
    plt.colorbar(im6, ax=ax6)
    
    # ── ROW 3: Scalars + Verdict ──
    
    # (3,1-3,2) Scalar Features Bar Chart
    ax7 = plt.subplot(3, 3, 7)
    colors = ['#34d399' if 0.3 < s < 0.7 else '#f87171' for s in scalars]
    bars = ax7.barh(range(len(SCALAR_NAMES)), scalars, color=colors, alpha=0.8, edgecolor='#8aa4cc', linewidth=1.5)
    ax7.set_yticks(range(len(SCALAR_NAMES)))
    ax7.set_yticklabels([name[:25] for name in SCALAR_NAMES], fontsize=9)
    ax7.set_xlabel('Normalized Value [0, 1]')
    ax7.set_title("⑦ Forensic Scalar Features", color='#8aa4cc', fontweight='bold', fontsize=11)
    ax7.set_facecolor('#0f172a')
    ax7.set_xlim(0, 1)
    ax7.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, scalars)):
        ax7.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8, color='#8aa4cc')
    
    # (3,3) Verdict Label + Stats
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    v_color = '#34d399' if result['result'] == "BONAFIDE" else '#f87171'
    is_bonafide = result['result'] == "BONAFIDE"
    
    # Large verdict text
    verdict_text = f"{'✓ BONAFIDE' if is_bonafide else '✗ SPOOF'}\n{'AUTHENTIC' if is_bonafide else 'FAKE'}"
    ax8.text(0.5, 0.75, verdict_text, 
             ha='center', va='center', fontsize=28, fontweight='bold',
             color=v_color, transform=ax8.transAxes)
    
    # Confidence bar
    raw_prob = result['raw_prob']
    conf_display = raw_prob if not is_bonafide else (1.0 - raw_prob)
    ax8.text(0.5, 0.52, f"Confidence: {conf_display:.1%}", 
             ha='center', va='center', fontsize=14, color='white', transform=ax8.transAxes)
    
    # Metadata
    metadata = (f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model: ResNet50 + Dual-Input\n"
                f"Threshold: 0.50\n"
                f"Raw Score: {raw_prob:.4f}")
    ax8.text(0.5, 0.18, metadata, 
             ha='center', va='center', fontsize=10, color='#8aa4cc',
             transform=ax8.transAxes, family='monospace',
             bbox=dict(facecolor=v_color, alpha=0.1, edgecolor=v_color, boxstyle='round,pad=0.8'))
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    
    # Save the report
    output_dir = os.path.join(root_path, "reports")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    report_name = f"Report_{os.path.basename(audio_path)}.png"
    save_path = os.path.join(output_dir, report_name)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches='tight')
    print(f"\n[OK] Report saved to: {save_path}\n")
    plt.close(fig)
    
    return result, save_path

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  [*] EchoTrace Forensic Report Generator")
    print("="*60)
    print(f"  Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("="*60 + "\n")
    
    audio_to_test = input("[>] Enter the path to the audio file for analysis: ").strip()
    
    if not audio_to_test:
        print("[X] No file path provided. Exiting.")
        sys.exit(1)
    
    result, report_path = generate_forensic_report(audio_to_test)
    
    if result and 'error' not in result:
        print("━" * 60)
        print(f"[*] FINAL VERDICT: {result['result']}")
        print(f"[+] Confidence: {result['confidence']}")
        print(f"[+] Report: {report_path}")
        print("━" * 60 + "\n")
    else:
        print(f"[X] Analysis failed: {result.get('error', 'Unknown error')}\n")