import torch
import cv2
import librosa
import numpy as np
import os
from .model import build_model
from .preprocess import build_feature_image, extract_scalar_features
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# ── Device & Model ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer ensemble_model.pth if it exists (post-DDP), else fall back to original weights
def _find_weights():
    for name in ("ensemble_model.pth", "deepfake_detector.pth"):
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    return None

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = build_model(device)

_weights_path = _find_weights()
if _weights_path:
    state = torch.load(_weights_path, map_location=device)
    # Strip DDP "module." prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # Load with strict=False so old single-channel weights don't crash the new head
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[inference] Missing keys (expected for new head): {missing[:5]}")
model.eval()

# ── Grad-CAM hooks on layer4 (spectrogram branch) ───────────
_activations, _gradients = {}, {}

def _fwd_hook(module, input, output):
    _activations["value"] = output

def _bwd_hook(module, grad_in, grad_out):
    _gradients["value"] = grad_out[0]

_target_layer = model.resnet.layer4[-1]
_target_layer.register_forward_hook(_fwd_hook)
_target_layer.register_full_backward_hook(_bwd_hook)

# ── Transforms ───────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Grad-CAM ─────────────────────────────────────────────────
def _compute_gradcam(original_img_np: np.ndarray) -> str:
    """Returns base64-encoded PNG of Grad-CAM overlay on the mel channel."""
    act = _activations.get("value")
    grad = _gradients.get("value")
    if act is None or grad is None:
        return ""

    weights = torch.mean(grad, dim=[0, 2, 3])
    cam = torch.zeros(act.shape[2:], dtype=torch.float32, device=act.device)
    for i, w in enumerate(weights):
        cam += w * act[0, i, :, :]

    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)
    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode("utf-8")


# ── Main inference entry-point ────────────────────────────────
async def run_inference(file_bytes: bytes) -> dict:
    """
    Accepts raw audio bytes, returns verdict/confidence/heatmap dict.
    Compatible with the existing Streamlit + report generator call sites.
    """
    SR = 16000
    DURATION = 4.0
    TARGET_LEN = int(SR * DURATION)

    # Load & normalize
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=SR, duration=DURATION, res_type="soxr_hq")
    peak = np.max(np.abs(audio))
    if peak > 1e-7:
        audio = audio / peak

    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)), mode='reflect')
    else:
        audio = audio[:TARGET_LEN]

    # 3-channel feature image
    feature_img = build_feature_image(audio, sr=SR)          # (224, 224, 3) uint8
    pil_img = Image.fromarray(feature_img)
    input_tensor = _transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Scalar features
    scalars_np = extract_scalar_features(audio, sr=SR)        # (8,) float32
    scalars_tensor = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward + backward for Grad-CAM
    model.zero_grad()
    output = model(input_tensor, scalars_tensor)
    prob = torch.sigmoid(output).item()
    output.backward()

    # Grad-CAM overlay uses the mel channel (ch0) of the feature image as background
    mel_ch = feature_img[:, :, 0]                             # 224×224 grayscale
    mel_rgb = cv2.cvtColor(mel_ch, cv2.COLOR_GRAY2RGB)
    heatmap_b64 = _compute_gradcam(mel_rgb)

    is_spoof = prob > 0.5
    confidence = prob if is_spoof else 1.0 - prob

    return {
        "result":     "SPOOF" if is_spoof else "BONAFIDE",
        "confidence": f"{confidence:.2%}",
        "heatmap":    heatmap_b64,
        "raw_prob":   prob,
    }