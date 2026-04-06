import streamlit as st
import os
import sys
import tempfile
import pathlib
import time
from PIL import Image
import torch

# ── Download model weights from HF Hub if not present ──
try:
    from download_model import ensure_model_exists
    ensure_model_exists()
except Exception as _e:
    pass  # Will surface a clearer error later when model load fails

# Add the directory containing single_example_report_generator.py to sys.path
project_root = pathlib.Path(__file__).parent.absolute()
tests_path = project_root / "tests"
if str(tests_path) not in sys.path:
    sys.path.append(str(tests_path))

try:
    from single_example_report_generator import generate_forensic_report
except ImportError as e:
    st.error(f"Failed to import forensic report generator: {e}")
    st.stop()

# ── Import EchoTrace Core ──
from core.model import build_model
from core.preprocess import build_feature_image, extract_scalar_features
from core.inference import device, _transform
from utils.audio import validate_and_load, AudioValidationError
from utils.llm_report import generate_llm_report

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

import io
import numpy as np
import plotly.graph_objects as go

def clean_wav_bytes(raw_bytes: bytes) -> bytes:
    """Re-encode raw mic bytes into a clean, browser-compatible WAV via pydub."""
    if not PYDUB_AVAILABLE:
        return raw_bytes
    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(raw_bytes), format="wav")
        out = io.BytesIO()
        audio_seg.export(out, format="wav")
        return out.getvalue()
    except Exception:
        return raw_bytes  # fall back to raw if re-encode fails

# --- Page Configuration ---
st.set_page_config(
    page_title="EchoTrace | Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        background-color: #0A0A0B;
        color: #C8C5BF;
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* ─── Noise Overlay ─── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
        pointer-events: none;
        z-index: 0;
        opacity: 0.35;
    }

    /* ─── Scanlines ─── */
    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
        pointer-events: none;
        z-index: 1;
    }

    /* ─── Hero ─── */
    .hero-wrap {
        position: relative;
        background: linear-gradient(180deg, #0D0D0E 0%, #0A0A0B 100%);
        border-bottom: 1px solid #1E1E20;
        padding: 4rem 2rem 3rem;
        text-align: center;
        overflow: hidden;
    }

    .hero-wrap::before {
        content: 'ECHOTRACE';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(6rem, 18vw, 14rem);
        letter-spacing: 0.1em;
        color: transparent;
        -webkit-text-stroke: 1px rgba(255,255,255,0.03);
        white-space: nowrap;
        pointer-events: none;
        user-select: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(200, 40, 30, 0.12);
        border: 1px solid rgba(200, 40, 30, 0.3);
        color: #E8443A;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        padding: 4px 12px;
        border-radius: 2px;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }

    .hero-badge::before {
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #E8443A;
        animation: pulse-dot 2s infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.7); }
    }

    .wordmark {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(3.5rem, 8vw, 6.5rem);
        letter-spacing: 0.12em;
        line-height: 1;
        margin-bottom: 0.75rem;
    }

    .echo-part { color: #F0EDE8; }
    .trace-part { color: #E8443A; }

    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        color: #5A5A5E;
        text-transform: uppercase;
        margin: 0;
    }

    .hero-rule { width: 60px; height: 2px; background: #E8443A; margin: 1.5rem auto 0; }

    /* ─── Section Label ─── */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        color: #5A5A5E;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after { content: ''; flex: 1; height: 1px; background: #1E1E20; }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1E1E20 !important;
        gap: 0 !important;
        margin-bottom: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #3A3A3E !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 10px 20px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.2s !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #C8C5BF !important;
        background: rgba(255,255,255,0.02) !important;
    }

    .stTabs [aria-selected="true"] {
        color: #E8443A !important;
        border-bottom: 2px solid #E8443A !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

    /* ─── Upload Zone ─── */
    .stFileUploader > div { background: transparent !important; }

    .stFileUploader section {
        background: #111113 !important;
        border: 1px solid #1E1E20 !important;
        border-radius: 4px !important;
        padding: 2.5rem !important;
        transition: border-color 0.25s !important;
        position: relative;
        overflow: hidden;
    }

    .stFileUploader section::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(232, 68, 58, 0.04) 0%, transparent 70%);
        pointer-events: none;
    }

    .stFileUploader section:hover { border-color: #E8443A !important; }

    .stFileUploader label p { color: #F0EDE8 !important; font-weight: 600 !important; font-size: 0.95rem !important; }
    .stFileUploader section div div { color: #5A5A5E !important; font-size: 0.82rem !important; }
    [data-testid="stFileUploaderFileName"] {
        color: #E8443A !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ─── Mic Zone ─── */
    .mic-zone {
        background: #111113;
        border: 1px solid #1E1E20;
        border-radius: 4px;
        padding: 2rem 2rem 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .mic-zone::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 100%, rgba(232, 68, 58, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .mic-icon { font-size: 2.2rem; margin-bottom: 0.6rem; display: block; }

    .mic-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #F0EDE8;
        margin-bottom: 0.3rem;
    }

    .mic-hint {
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        color: #3A3A3E;
        letter-spacing: 0.12em;
        margin-bottom: 1.25rem;
    }

    /* ─── Audio preview ─── */
    .stAudio { margin-top: 0.5rem !important; }
    .stAudio > div {
        background: #111113 !important;
        border: 1px solid #1E1E20 !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.75rem !important;
    }
    audio {
        width: 100% !important;
        border-radius: 2px !important;
    }

    /* ─── Buttons ─── */
    div.stButton > button {
        background: #E8443A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 14px 28px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        margin-top: 1rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    div.stButton > button::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 50%);
        pointer-events: none;
    }

    div.stButton > button:hover {
        background: #D03530 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(232, 68, 58, 0.35) !important;
    }

    div.stButton > button:active { transform: translateY(0px) !important; }

    /* ─── Progress Bar ─── */
    .stProgress { margin-top: 1.5rem !important; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #E8443A, #FF6B5E) !important; border-radius: 1px !important; }
    .stProgress > div > div > div { background: #1E1E20 !important; border-radius: 1px !important; height: 3px !important; }

    .stMarkdown p { color: #C8C5BF !important; }
    hr { border-color: #1E1E20 !important; margin: 2.5rem 0 !important; }

    /* ─── Verdict ─── */
    .verdict-wrap {
        background: #111113;
        border: 1px solid #1E1E20;
        border-top: 3px solid #E8443A;
        padding: 2rem;
        border-radius: 0 0 4px 4px;
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
    }

    .verdict-wrap::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 0% 0%, rgba(232, 68, 58, 0.06) 0%, transparent 60%);
        pointer-events: none;
    }

    .verdict-eyebrow { font-family: 'Space Mono', monospace; font-size: 0.62rem; letter-spacing: 0.22em; color: #5A5A5E; text-transform: uppercase; margin-bottom: 0.4rem; }

    .verdict-source-tag {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.14em;
        color: #3A3A3E;
        border: 1px solid #1E1E20;
        padding: 2px 8px;
        border-radius: 1px;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }

    .verdict-result-fake { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; letter-spacing: 0.06em; color: #E8443A !important; line-height: 1; margin-bottom: 0.75rem; }
    .verdict-result-real { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; letter-spacing: 0.06em; color: #52D98A !important; line-height: 1; margin-bottom: 0.75rem; }

    .verdict-confidence { font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #5A5A5E; border-top: 1px solid #1E1E20; padding-top: 0.75rem; margin-top: 0.75rem; }
    .verdict-confidence span { color: #C8C5BF !important; font-weight: 700; }

    /* ─── Report Image ─── */
    .stImage { margin-top: 1.5rem !important; }
    .stImage img { border-radius: 4px !important; border: 1px solid #1E1E20 !important; }

    /* ─── Download Button ─── */
    .stDownloadButton > button {
        background: transparent !important;
        color: #E8443A !important;
        border: 1px solid #E8443A !important;
        border-radius: 2px !important;
        padding: 10px 20px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        margin-top: 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(232, 68, 58, 0.1) !important;
        box-shadow: 0 4px 16px rgba(232, 68, 58, 0.2) !important;
    }

    .stAlert { border-radius: 2px !important; border-left-width: 3px !important; }

    /* ─── Footer ─── */
    .footer-wrap { border-top: 1px solid #1E1E20; padding: 1.5rem; text-align: center; }
    .footer-text { font-family: 'Space Mono', monospace; font-size: 0.62rem; color: #2E2E32; letter-spacing: 0.14em; text-transform: uppercase; }

    /* ─── Stat Pills ─── */
    .stat-row { display: flex; gap: 12px; margin-bottom: 2rem; flex-wrap: wrap; }
    .stat-pill { flex: 1; min-width: 120px; background: #111113; border: 1px solid #1E1E20; border-radius: 2px; padding: 0.85rem 1rem; text-align: center; }
    .stat-val { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; color: #F0EDE8 !important; line-height: 1; }
    .stat-key { font-family: 'Space Mono', monospace; font-size: 0.58rem; color: #3A3A3E; text-transform: uppercase; letter-spacing: 0.18em; margin-top: 3px; }

    </style>
""", unsafe_allow_html=True)




# ─── Confidence Timeline ─────────────────────────────────────
def compute_confidence_timeline(audio_np: np.ndarray, window_sec=2.0, hop_sec=0.5):
    """
    Slice audio into overlapping windows, run inference on each,
    return list of (timestamp, confidence, verdict) tuples.
    """
    SR = 16000
    WINDOW = int(window_sec * SR)
    HOP    = int(hop_sec * SR)

    global model
    try:
        model.eval()
    except NameError:
        # If model isn't globally loaded, initialize it via core.inference logic
        from core.inference import model as inference_model
        model = inference_model

    results = []
    total_samples = len(audio_np)
    starts = list(range(0, max(1, total_samples - WINDOW + 1), HOP))

    for start in starts:
        chunk = audio_np[start:start + WINDOW]
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))

        # 3-channel feature image
        feature_img = build_feature_image(chunk, sr=SR)
        pil_img = Image.fromarray(feature_img)
        input_tensor = _transform(pil_img).unsqueeze(0).to(device)

        # Scalar features
        scalars_np = extract_scalar_features(chunk, sr=SR)
        scalars_tensor = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor, scalars_tensor)
            prob = torch.sigmoid(output).item()

        timestamp = start / SR
        results.append((round(timestamp, 2), round(prob, 4), "SPOOF" if prob > 0.5 else "BONAFIDE"))

    return results


def render_confidence_timeline(audio_np: np.ndarray, model_path: str, ctx):
    ctx.markdown('''<div class="section-label" style="margin-top:2rem">Confidence Timeline</div>''',
                 unsafe_allow_html=True)
    ctx.markdown('''
        <p style="font-family:Space Mono,monospace;font-size:0.68rem;color:#5A5A5E;
        letter-spacing:0.1em;margin-bottom:1.25rem;">
        Sliding 2-second window across the audio — spikes above 50% indicate
        regions the model flagged as synthetic.
        </p>
    ''', unsafe_allow_html=True)

    _spin = ctx.empty()
    _spin.markdown('''<p style="font-family:Space Mono,monospace;font-size:0.68rem;
        color:#5A5A5E;letter-spacing:0.1em;text-align:center;">
        Running sliding-window inference…</p>''', unsafe_allow_html=True)
    try:
        points = compute_confidence_timeline(audio_np)
    except Exception as e:
        ctx.error(f"Timeline error: {e}")
        _spin.empty()
        return
    _spin.empty()

    times  = [p[0] for p in points]
    confs  = [p[1] * 100 for p in points]   # as percentage
    labels = [p[2] for p in points]

    # Color each point
    colors = ["#E8443A" if l == "SPOOF" else "#52D98A" for l in labels]

    fig = go.Figure()

    # Danger zone fill above 50%
    fig.add_hrect(
        y0=50, y1=100,
        fillcolor="rgba(232,68,58,0.06)",
        line_width=0,
    )

    # Threshold line
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(232,68,58,0.4)",
        line_width=1.5,
        annotation_text="Detection Threshold",
        annotation_font_color="rgba(232,68,58,0.5)",
        annotation_font_size=10,
        annotation_position="top right",
    )

    # Shaded area under the curve
    fig.add_trace(go.Scatter(
        x=times, y=confs,
        fill="tozeroy",
        fillcolor="rgba(232,68,58,0.08)",
        line=dict(color="rgba(232,68,58,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=times, y=confs,
        mode="lines+markers",
        line=dict(color="#E8443A", width=2.5, shape="spline", smoothing=0.8),
        marker=dict(
            size=7,
            color=colors,
            line=dict(color="#0A0A0B", width=1.5),
        ),
        hovertemplate=(
            "<b>%{x:.2f}s</b><br>"
            "Confidence: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
        name="Spoof Confidence",
    ))

    fig.update_layout(
        paper_bgcolor="#0A0A0B",
        plot_bgcolor="#111113",
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(
            title="Time (seconds)",
            title_font=dict(color="#5A5A5E", size=10, family="Space Mono"),
            tickfont=dict(color="#5A5A5E", size=9, family="Space Mono"),
            gridcolor="#1E1E20",
            zerolinecolor="#1E1E20",
            showgrid=True,
        ),
        yaxis=dict(
            title="Spoof Confidence %",
            title_font=dict(color="#5A5A5E", size=10, family="Space Mono"),
            tickfont=dict(color="#5A5A5E", size=9, family="Space Mono"),
            gridcolor="#1E1E20",
            zerolinecolor="#1E1E20",
            range=[-2, 108],
            showgrid=True,
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1E1E20",
            font_color="#F0EDE8",
            font_family="Space Mono",
            font_size=11,
            bordercolor="#E8443A",
        ),
    )

    ctx.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Summary stats below chart
    spoof_pct = sum(1 for l in labels if l == "SPOOF") / len(labels) * 100 if labels else 0
    peak_conf  = max(confs) if confs else 0
    peak_time  = times[confs.index(peak_conf)] if confs else 0

    ctx.markdown(f'''
        <div style="display:flex;gap:12px;margin-top:0.75rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;background:#111113;border:1px solid #1E1E20;
            border-radius:2px;padding:0.75rem 1rem;text-align:center;">
                <div style="font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:#E8443A;">{spoof_pct:.0f}%</div>
                <div style="font-family:Space Mono,monospace;font-size:0.55rem;color:#3A3A3E;
                letter-spacing:0.16em;text-transform:uppercase;margin-top:2px;">Flagged Windows</div>
            </div>
            <div style="flex:1;min-width:120px;background:#111113;border:1px solid #1E1E20;
            border-radius:2px;padding:0.75rem 1rem;text-align:center;">
                <div style="font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:#F0EDE8;">{peak_conf:.1f}%</div>
                <div style="font-family:Space Mono,monospace;font-size:0.55rem;color:#3A3A3E;
                letter-spacing:0.16em;text-transform:uppercase;margin-top:2px;">Peak Confidence</div>
            </div>
            <div style="flex:1;min-width:120px;background:#111113;border:1px solid #1E1E20;
            border-radius:2px;padding:0.75rem 1rem;text-align:center;">
                <div style="font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:#F0EDE8;">{peak_time:.2f}s</div>
                <div style="font-family:Space Mono,monospace;font-size:0.55rem;color:#3A3A3E;
                letter-spacing:0.16em;text-transform:uppercase;margin-top:2px;">Peak Timestamp</div>
            </div>
            <div style="flex:1;min-width:120px;background:#111113;border:1px solid #1E1E20;
            border-radius:2px;padding:0.75rem 1rem;text-align:center;">
                <div style="font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:#F0EDE8;">{len(points)}</div>
                <div style="font-family:Space Mono,monospace;font-size:0.55rem;color:#3A3A3E;
                letter-spacing:0.16em;text-transform:uppercase;margin-top:2px;">Windows Analyzed</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

# ─── Shared analysis function ────────────────────────────────
def run_analysis(audio_bytes: bytes, suffix: str, source_label: str, container=None):
    ctx = container if container is not None else st
    
    # ── VAD and Audio Validation ──
    try:
        audio_np, voiced_ratio = validate_and_load(audio_bytes)
    except AudioValidationError as e:
        ctx.markdown(f"""
            <div style="background: rgba(232, 68, 58, 0.1); border-left: 3px solid #E8443A; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                <div style="color: #E8443A; font-family: 'Space Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;">Validation Error</div>
                <div style="color: #F0EDE8; font-family: 'DM Sans', sans-serif; font-size: 0.88rem; margin-top: 4px;">{str(e)}</div>
            </div>
        """, unsafe_allow_html=True)
        return

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_bytes)
        os.close(fd)

        progress_bar = ctx.progress(0)
        status = ctx.empty()
        status.markdown(
            "<p style='font-family:Space Mono,monospace;font-size:0.72rem;"
            "color:#5A5A5E;letter-spacing:0.12em;text-align:center;margin-top:0.75rem;'>"
            "Deep learning core online. Identifying synthetic artifacts…</p>",
            unsafe_allow_html=True,
        )

        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)

        # Main Inference
        analysis_result, report_path = generate_forensic_report(tmp_path)

        status.empty()
        progress_bar.empty()

        if report_path and os.path.exists(report_path):
            verdict = analysis_result.get("result", "UNKNOWN")
            is_fake = verdict != "BONAFIDE"
            verdict_label = "AI-GENERATED (SPOOF)" if is_fake else "AUTHENTIC (BONAFIDE)"
            verdict_cls = "verdict-result-fake" if is_fake else "verdict-result-real"
            confidence_str = analysis_result.get("confidence", "N/A")
            raw_prob = analysis_result.get("raw_prob", 0.5)

            ctx.divider()

            # Result Dashboard
            ctx.markdown(f"""
                <div class="verdict-wrap">
                    <div class="verdict-eyebrow">EchoTrace Verdict</div>
                    <div class="verdict-source-tag">Processing: ResNet-50 v4 · DDP-Ensemble</div>
                    <div class="{verdict_cls}">{verdict_label}</div>
                    <div class="verdict-confidence">
                        Confidence Score &nbsp;·&nbsp; <span>{confidence_str}</span>
                        <div style="font-size: 0.65rem; color: #3A3A3E; margin-top: 2px;">(Probability: {raw_prob:.4f})</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # LLM Forensic Report integration
            ctx.markdown('<div class="section-label" style="margin-top:2rem">Forensic Analysis Report (AI-Generated)</div>', unsafe_allow_html=True)
            
            # Extract scalars for the LLM from the actual audio logic
            # (In a real app we'd pass these from analysis_result, but we can compute them here too)
            sc = extract_scalar_features(audio_np[:64000], sr=16000)
            
            llm_text = generate_llm_report(
                verdict=verdict,
                confidence=float(confidence_str.strip('%'))/100,
                f0_jitter=sc[6],
                spectral_contrast=sc[2], # using centroid as proxy or sc[2]
                voiced_ratio=voiced_ratio
            )
            
            ctx.markdown(f"""
                <div style="background: #111113; border: 1px solid #1E1E20; padding: 1.5rem; border-radius: 4px; border-left: 2px solid #5A5A5E;">
                    <p style="color: #C8C5BF; font-family: 'DM Sans', sans-serif; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                        {llm_text}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Confidence Timeline
            with ctx:
                render_confidence_timeline(audio_np, None, ctx)

            ctx.markdown('<div class="section-label" style="margin-top:2rem">Grad-CAM Spatial Pulse</div>', unsafe_allow_html=True)
            ctx.image(str(report_path), use_container_width=True)

            with open(report_path, "rb") as f:
                ctx.download_button(
                    label="↓ Download Forensic Image",
                    data=f,
                    file_name=os.path.basename(report_path),
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            ctx.error("Analysis engine failed — please check system logs.")

    except Exception as e:
        ctx.error(f"Analysis error: {str(e)}")
        if "deepfake_detector.pth" in str(e):
            ctx.warning("Model weights not found. Ensure `deepfake_detector.pth` is in the project root.")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─── HERO ────────────────────────────────────────────────────
st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">Forensic Analysis Suite · v3.0</div>
        <div class="wordmark">
            <span class="echo-part">ECHO</span><span class="trace-part">TRACE</span>
        </div>
        <p class="hero-sub">AI-Powered Deepfake Audio Detection</p>
        <div class="hero-rule"></div>
    </div>
""", unsafe_allow_html=True)

# ─── MAIN CONTENT ────────────────────────────────────────────
_, col, _ = st.columns([1, 2.4, 1])

with col:
    st.markdown("""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-val">3-Channel</div>
                <div class="stat-key">Spectral Input</div>
            </div>
            <div class="stat-pill">
                <div class="stat-val">8</div>
                <div class="stat-key">Physics Scalars</div>
            </div>
            <div class="stat-pill">
                <div class="stat-val">300k+</div>
                <div class="stat-key">Training Clips</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_record = st.tabs(["⬡  Upload File", "⏺  Record Live"])

    # ── Tab 1: File Upload ────────────────────────────────────
    with tab_upload:
        st.markdown('<div class="section-label">Upload Audio Sample</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")

        if uploaded_file is not None:
            # Preview player for uploaded file
            st.markdown('<div class="section-label" style="margin-top:1rem">Preview</div>', unsafe_allow_html=True)
            import base64
            file_bytes = uploaded_file.getvalue()
            suffix = pathlib.Path(uploaded_file.name).suffix.lower()
            try:
                if PYDUB_AVAILABLE:
                    seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=suffix.strip("."))
                    mp3_buf = io.BytesIO()
                    seg.export(mp3_buf, format="mp3", bitrate="128k")
                    b64 = base64.b64encode(mp3_buf.getvalue()).decode("utf-8")
                    mime = "audio/mpeg"
                else:
                    b64 = base64.b64encode(file_bytes).decode("utf-8")
                    mime = "audio/mpeg" if suffix == ".mp3" else "audio/wav"
            except Exception:
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                mime = "audio/mpeg" if suffix == ".mp3" else "audio/wav"

            st.markdown(f'''
                <audio controls style="width:100%;margin-bottom:0.75rem;border-radius:4px;background:#111113;">
                    <source src="data:{mime};base64,{b64}" type="{mime}">
                </audio>
            ''', unsafe_allow_html=True)

            # Static waveform for uploaded file
            st.markdown('<div class="section-label" style="margin-top:1rem">Waveform</div>', unsafe_allow_html=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                seg_wf = AudioSegment.from_file(io.BytesIO(file_bytes), format=suffix.strip("."))
                samples_wf = np.array(seg_wf.get_array_of_samples(), dtype=np.float32)
                if seg_wf.channels == 2:
                    samples_wf = samples_wf[::2]
                samples_wf /= float(2 ** (seg_wf.sample_width * 8 - 1))

                target_pts = 1200
                step_wf = max(1, len(samples_wf) // target_pts)
                display_wf = samples_wf[::step_wf]
                duration_wf = len(seg_wf) / 1000.0
                times_wf = np.linspace(0, duration_wf, len(display_wf))

                fig_wf, ax_wf = plt.subplots(figsize=(8, 1.6))
                fig_wf.patch.set_facecolor('#111113')
                ax_wf.set_facecolor('#111113')
                ax_wf.fill_between(times_wf, display_wf, alpha=0.25, color='#E8443A')
                ax_wf.plot(times_wf, display_wf, color='#E8443A', linewidth=0.8, alpha=0.9)
                ax_wf.axhline(0, color='#2A2A2E', linewidth=0.5)
                ax_wf.set_xlim(0, duration_wf)
                ax_wf.set_ylim(-1.05, 1.05)
                ax_wf.tick_params(colors='#3A3A3E', labelsize=7)
                for spine in ax_wf.spines.values():
                    spine.set_edgecolor('#1E1E20')
                ax_wf.set_xlabel('seconds', color='#3A3A3E', fontsize=7)
                fig_wf.tight_layout(pad=0.4)

                wf_buf = io.BytesIO()
                fig_wf.savefig(wf_buf, format='png', dpi=120, bbox_inches='tight',
                               facecolor='#111113', edgecolor='none')
                plt.close(fig_wf)
                wf_buf.seek(0)
                st.image(wf_buf, use_container_width=True)
            except Exception as e:
                st.caption(f"Waveform unavailable: {e}")

            if st.button("⬡  Run Forensic Analysis", key="btn_upload"):
                try:
                    run_analysis(file_bytes, suffix, "File Upload", container=col)
                except Exception as e:
                    st.error(f"File handling error: {e}")

    # ── Tab 2: Live Recording ─────────────────────────────────
    with tab_record:
        if not MIC_AVAILABLE:
            st.warning(
                "Mic recording requires **streamlit-mic-recorder**. "
                "Install it with: `pip install streamlit-mic-recorder`"
            )
        elif not PYDUB_AVAILABLE:
            st.warning(
                "Audio playback requires **pydub** for re-encoding. "
                "Install it with: `pip install pydub`"
            )
        else:
            st.markdown('<div class="section-label">Record Audio Sample</div>', unsafe_allow_html=True)

            st.markdown("""
                <div class="mic-zone">
                    <span class="mic-icon">🎙️</span>
                    <div class="mic-title">Live Microphone Input</div>
                    <div class="mic-hint">Click start &nbsp;·&nbsp; speak &nbsp;·&nbsp; click stop &nbsp;·&nbsp; then analyze</div>
                </div>
            """, unsafe_allow_html=True)

            # Only show the recorder widget if we don't already have audio
            if "last_mic_audio" not in st.session_state:

                # Live waveform visualizer using Web Audio API
                st.components.v1.html("""
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { background: transparent; }
                    #viz-wrap {
                        background: #111113;
                        border: 1px solid #1E1E20;
                        border-radius: 4px;
                        padding: 1rem;
                        margin-top: 0.5rem;
                        position: relative;
                        overflow: hidden;
                    }
                    #viz-label {
                        font-family: 'Space Mono', monospace;
                        font-size: 0.6rem;
                        letter-spacing: 0.2em;
                        color: #3A3A3E;
                        text-transform: uppercase;
                        margin-bottom: 0.5rem;
                    }
                    canvas {
                        width: 100%;
                        height: 80px;
                        display: block;
                        border-radius: 2px;
                    }
                    #status-dot {
                        display: inline-block;
                        width: 6px; height: 6px;
                        border-radius: 50%;
                        background: #3A3A3E;
                        margin-right: 6px;
                        vertical-align: middle;
                        transition: background 0.3s;
                    }
                    #status-dot.active { background: #E8443A; animation: blink 1s infinite; }
                    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
                </style>
                <div id="viz-wrap">
                    <div id="viz-label">
                        <span id="status-dot"></span>
                        <span id="status-text">Waiting for input…</span>
                    </div>
                    <canvas id="waveform"></canvas>
                </div>
                <script>
                    const canvas = document.getElementById('waveform');
                    const ctx = canvas.getContext('2d');
                    const dot = document.getElementById('status-dot');
                    const statusText = document.getElementById('status-text');
                    let animId = null;
                    let analyser = null;
                    let dataArray = null;
                    let isLive = false;

                    function drawIdle() {
                        canvas.width = canvas.offsetWidth;
                        canvas.height = 80;
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.strokeStyle = '#2A2A2E';
                        ctx.lineWidth = 1.5;
                        ctx.beginPath();
                        ctx.moveTo(0, canvas.height / 2);
                        ctx.lineTo(canvas.width, canvas.height / 2);
                        ctx.stroke();
                    }

                    function drawLive() {
                        if (!analyser) return;
                        animId = requestAnimationFrame(drawLive);
                        canvas.width = canvas.offsetWidth;
                        canvas.height = 80;
                        analyser.getByteTimeDomainData(dataArray);

                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        // Glow effect
                        ctx.shadowBlur = 8;
                        ctx.shadowColor = '#E8443A';
                        ctx.strokeStyle = '#E8443A';
                        ctx.lineWidth = 2;
                        ctx.beginPath();

                        const sliceW = canvas.width / dataArray.length;
                        let x = 0;
                        for (let i = 0; i < dataArray.length; i++) {
                            const v = dataArray[i] / 128.0;
                            const y = (v * canvas.height) / 2;
                            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                            x += sliceW;
                        }
                        ctx.lineTo(canvas.width, canvas.height / 2);
                        ctx.stroke();
                        ctx.shadowBlur = 0;
                    }

                    async function startViz() {
                        try {
                            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                            const source = audioCtx.createMediaStreamSource(stream);
                            analyser = audioCtx.createAnalyser();
                            analyser.fftSize = 1024;
                            dataArray = new Uint8Array(analyser.frequencyBinCount);
                            source.connect(analyser);
                            isLive = true;
                            dot.classList.add('active');
                            statusText.textContent = 'Mic active — press record above';
                            drawLive();
                        } catch(e) {
                            statusText.textContent = 'Mic access denied';
                            drawIdle();
                        }
                    }

                    drawIdle();
                    startViz();
                </script>
                """, height=130)

                audio = mic_recorder(
                    start_prompt="⏺  Start Recording",
                    stop_prompt="⏹  Stop Recording",
                    just_once=False,
                    use_container_width=True,
                    format="wav",
                    key="mic_input",
                )
                if audio and audio.get("bytes"):
                    st.session_state["last_mic_audio"] = clean_wav_bytes(audio["bytes"])
                    st.rerun()
            else:
                raw = st.session_state["last_mic_audio"]

                # Static waveform from actual audio samples
                st.markdown('<div class="section-label" style="margin-top:1.5rem">Waveform</div>', unsafe_allow_html=True)
                try:
                    import wave, struct, matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    seg = AudioSegment.from_file(io.BytesIO(raw), format="wav")
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    if seg.channels == 2:
                        samples = samples[::2]  # take left channel
                    samples /= float(2 ** (seg.sample_width * 8 - 1))  # normalize to -1..1

                    # Downsample for display
                    target_pts = 1200
                    step = max(1, len(samples) // target_pts)
                    display = samples[::step]

                    duration = len(seg) / 1000.0
                    times = np.linspace(0, duration, len(display))

                    fig, ax = plt.subplots(figsize=(8, 1.6))
                    fig.patch.set_facecolor('#111113')
                    ax.set_facecolor('#111113')

                    # Fill waveform
                    ax.fill_between(times, display, alpha=0.25, color='#E8443A')
                    ax.plot(times, display, color='#E8443A', linewidth=0.8, alpha=0.9)
                    ax.axhline(0, color='#2A2A2E', linewidth=0.5)

                    ax.set_xlim(0, duration)
                    ax.set_ylim(-1.05, 1.05)
                    ax.tick_params(colors='#3A3A3E', labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1E1E20')
                    ax.set_xlabel('seconds', color='#3A3A3E', fontsize=7)
                    fig.tight_layout(pad=0.4)

                    waveform_buf = io.BytesIO()
                    fig.savefig(waveform_buf, format='png', dpi=120, bbox_inches='tight',
                                facecolor='#111113', edgecolor='none')
                    plt.close(fig)
                    waveform_buf.seek(0)
                    st.image(waveform_buf, use_container_width=True)
                except Exception as e:
                    st.caption(f"Waveform unavailable: {e}")

                # Audio player
                st.markdown('<div class="section-label" style="margin-top:1rem">Preview Recording</div>', unsafe_allow_html=True)
                import base64
                try:
                    if PYDUB_AVAILABLE:
                        seg = AudioSegment.from_file(io.BytesIO(raw), format="wav")
                        mp3_buf = io.BytesIO()
                        seg.export(mp3_buf, format="mp3", bitrate="128k")
                        mp3_bytes = mp3_buf.getvalue()
                        b64 = base64.b64encode(mp3_bytes).decode("utf-8")
                        mime = "audio/mpeg"
                    else:
                        b64 = base64.b64encode(raw).decode("utf-8")
                        mime = "audio/wav"
                except Exception:
                    b64 = base64.b64encode(raw).decode("utf-8")
                    mime = "audio/wav"

                st.markdown(f'''
                    <audio controls style="width:100%;margin-top:0.5rem;border-radius:4px;background:#111113;">
                        <source src="data:{mime};base64,{b64}" type="{mime}">
                    </audio>
                ''', unsafe_allow_html=True)

                col_rerecord, col_analyze = st.columns([1, 2])
                with col_rerecord:
                    if st.button("↺  Re-record", key="btn_rerecord"):
                        del st.session_state["last_mic_audio"]
                        st.rerun()
                with col_analyze:
                    if st.button("⬡  Analyze Recording", key="btn_record"):
                        run_analysis(st.session_state["last_mic_audio"], ".wav", "Microphone", container=col)

# ─── FOOTER ──────────────────────────────────────────────────
st.markdown("""
    <div class="footer-wrap">
        <div class="footer-text">EchoTrace Deepfake Detection · v3.0 · Built by the BackProp Bandits</div>
    </div>
""", unsafe_allow_html=True)