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

def generate_forensic_report(audio_path, model_path="../deepfake_detector.pth"):
    # Resolve the model path relative to this script
    model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(device)
    
    if os.path.exists(model_abs_path):
        model.load_state_dict(torch.load(model_abs_path, map_location=device))
        model.eval()
    else:
        print(f"❌ Error: Model weights not found at {model_abs_path}")
        return

    # Run Analysis
    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found at {audio_path}")
        return

    with open(audio_path, 'rb') as f:
        file_bytes = f.read()
    
    # Execute the backend inference
    result = asyncio.run(run_inference(file_bytes))

    # --- VISUALIZATION ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor='#080d1a')
    plt.suptitle(f"EchoTrace Forensic Analysis\n{os.path.basename(audio_path)}", 
                 color='white', fontsize=22, fontweight='bold', y=0.96)

    # Waveform
    ax1 = plt.subplot(2, 2, 1)
    y, sr = librosa.load(audio_path, sr=16000)
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#38bdf8', alpha=0.7)
    ax1.set_title("Time-Domain Waveform", color='#8aa4cc', fontweight='bold')
    ax1.set_facecolor('#0f172a')

    # Spectrogram
    ax2 = plt.subplot(2, 2, 2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224, n_fft=1024)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title("Spectral Fingerprint (224x224)", color='#8aa4cc', fontweight='bold')

    # Grad-CAM Heatmap
    ax3 = plt.subplot(2, 2, 3)
    heatmap_img = Image.open(BytesIO(base64.b64decode(result['heatmap'])))
    ax3.imshow(heatmap_img)
    ax3.set_title("AI Artifact Detection (Layer 4)", color='#8aa4cc', fontweight='bold')
    ax3.axis('off')

    # Stats Panel
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    v_color = '#34d399' if result['result'] == "BONAFIDE" else '#f87171'
    summary = (f"VERDICT: {result['result']}\nCONFIDENCE: {result['confidence']}\n\n"
               f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"Sensor: ResNet-50 Fine-Tuned")
    ax4.text(0.1, 0.5, summary, color='white', fontsize=14, fontweight='bold',
             bbox=dict(facecolor=v_color, alpha=0.15, edgecolor=v_color, boxstyle='round,pad=1.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    report_name = f"Report_{os.path.basename(audio_path)}.png"
    plt.savefig(report_name, facecolor=fig.get_facecolor(), dpi=300)
    print(f"✅ Report saved: {report_name}")
    plt.show()

if __name__ == "__main__":
    # Update this with your audio path
    audio_to_test = input("Enter the path to the audio file for analysis: ")
    generate_forensic_report(audio_to_test)