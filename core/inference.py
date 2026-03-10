import torch
import cv2
import librosa
import numpy as np
import os
from model import build_model
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# 1. Device and Model Initialization
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detector.pth")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = build_model(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 2. Grad-CAM Hooks
activations, gradients = None, None
def hook_fn(module, input, output): 
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out): 
    global gradients
    gradients = grad_out[0]

target_layer = model.layer4[-1]
target_layer.register_forward_hook(hook_fn)
target_layer.register_full_backward_hook(backward_hook)

def process_gradcam(cam, original_img_np):
    cam = np.maximum(cam.cpu().detach().numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')

async def run_inference(file_bytes):
    # 1. Load Audio with High-Quality Resampling
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=16000, duration=4.0, res_type='soxr_hq')
    
    # 2. MANDATORY PEAK NORMALIZATION (Match Training)
    peak = np.max(np.abs(audio))
    if peak > 1e-7:
        audio = audio / peak

    # Standardize length
    if len(audio) < 64000:
        audio = np.pad(audio, (0, 64000 - len(audio)))
    else:
        audio = audio[:64000]

    # 3. Generate 224x224 Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=224, n_fft=1024)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Process for ResNet
    img_np = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img_resized).unsqueeze(0).to(device)

    # 4. Predict & Explain
    model.zero_grad()
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    output.backward()
    
    # Grad-CAM math
    weights = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * activations[0, i, :, :]
    
    heatmap_b64 = process_gradcam(cam, img_resized)

    return {
        "result": "SPOOF" if prob > 0.5 else "BONAFIDE",
        "confidence": f"{prob if prob > 0.5 else 1-prob:.2%}",
        "heatmap": heatmap_b64
    }