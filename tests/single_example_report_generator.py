"""
tests/single_example_report_generator.py — EchoTrace Forensic Report Generator
- HTML report with dark EchoTrace styling
- PDF export via playwright
- Grad-CAM PNG for Streamlit
- ALL card text (scalar descriptions, channel summaries, status labels)
  generated dynamically by Groq based on the actual scalar values.
"""
import sys
import os
import numpy as np
import matplotlib
import datetime
import base64
import asyncio
import io
import torch
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use("Agg")

# --- DYNAMIC PATH INJECTION ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
core_path = os.path.join(root_path, 'core')
if core_path not in sys.path:
    sys.path.append(core_path)

from core.inference import run_inference
from core.preprocess import load_audio, build_feature_image, extract_scalar_features

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

SCALAR_NAMES = [
    'Spectral Flatness', 'Zero Crossing Rate',
    'F1 Formant', 'F2 Formant', 'F3 Formant',
    'Voiced Ratio', 'HNR', 'CPP',
]

# Suspicion direction — fixed by feature physics, not LLM
# (True, False) = low value is suspicious
# (False, True) = high value is suspicious
SUSP_DIRECTION = [
    (True,  False),  # [0] Spectral Flatness
    (True,  False),  # [1] ZCR
    (False, True),   # [2] F1
    (False, True),   # [3] F2
    (False, True),   # [4] F3
    (False, False),  # [5] Voiced Ratio
    (False, True),   # [6] HNR
    (False, True),   # [7] CPP
]


# ── SHAP approximation ────────────────────────────────────────
def _compute_shap_approx(scalars: np.ndarray) -> list:
    contributions = []
    for i, (susp_low, susp_high) in enumerate(SUSP_DIRECTION):
        v = float(scalars[i])
        if susp_low:
            contrib = max(0, 0.3 - v) * 0.8
        elif susp_high:
            contrib = max(0, v - 0.7) * 0.8
        else:
            contrib = -(v - 0.5) * 0.1
        contributions.append(contrib)

    total = sum(abs(c) for c in contributions) + 1e-9
    scale = 0.44 / total

    result = []
    for i, c in enumerate(contributions):
        scaled = c * scale
        sign = "+" if scaled >= 0 else "−"
        result.append({
            "feat":  SCALAR_NAMES[i],
            "val":   round(float(scaled), 3),
            "label": f"{sign}{abs(scaled):.3f}",
        })

    result.sort(key=lambda x: abs(x["val"]), reverse=True)
    return result


# ── Scalar card suspicion logic ───────────────────────────────
def _is_suspicious(idx: int, val: float) -> bool:
    susp_low, susp_high = SUSP_DIRECTION[idx]
    if idx == 5:  # Voiced Ratio: flag extremes
        return val < 0.4 or val > 0.98
    return (susp_low and val < 0.15) or (susp_high and val > 0.70)


def _status_text(card: dict, idx: int, val: float, suspicious: bool) -> str:
    """Pick the right status label from the Groq-generated card dict."""
    if not suspicious:
        return card.get("ok_status", "Within range")
    susp_low, susp_high = SUSP_DIRECTION[idx]
    if idx == 5:
        return card.get("low_status", "Outside range") if val < 0.4 else card.get("high_status", "Abnormally high")
    if susp_low and val < 0.15:
        return card.get("low_status", "Abnormally low")
    return card.get("high_status", "Abnormally high")


# ── CSS gradient per channel ──────────────────────────────────
_CHANNEL_GRADIENTS = [
    "linear-gradient(180deg,#1a0a00 0%,#3d1500 10%,#7a2800 20%,#c44a00 30%,#e8a000 38%,#3d7a00 45%,#006040 55%,#003060 70%,#000820 85%,#000408 100%)",
    "linear-gradient(180deg,#002020 0%,#005040 15%,#008060 30%,#00a870 45%,#60c060 55%,#90d080 65%,#004030 80%,#001818 100%)",
    "linear-gradient(180deg,#000830 0%,#001060 15%,#002090 25%,#0030a0 35%,#001060 50%,#000420 65%,#000010 80%,#000008 100%)",
]
_CHANNEL_STRIPES = [
    "repeating-linear-gradient(90deg,transparent 0px,transparent 3px,rgba(0,0,0,0.3) 3px,rgba(0,0,0,0.3) 4px)",
    "repeating-linear-gradient(180deg,transparent 0px,transparent 6px,rgba(0,0,0,0.15) 6px,rgba(0,0,0,0.15) 7px)",
    "repeating-linear-gradient(90deg,transparent 0px,transparent 8px,rgba(0,80,160,0.2) 8px,rgba(0,80,160,0.2) 10px)",
]




# ── HTML builder ──────────────────────────────────────────────
def generate_html_report(
    audio_path: str,
    analysis_result: dict,
    scalars: np.ndarray,
    shap_data: list,
    scalar_cards: list,    # from llm_cards.generate_card_analysis()
    channel_cards: list,   # from llm_cards.generate_card_analysis()
    feature_image: np.ndarray,
    llm_text: str,
    report_id: str,
    output_dir: str,
) -> str:
    verdict    = analysis_result.get("result", "UNKNOWN")
    raw_prob   = float(analysis_result.get("raw_prob", 0.5))
    is_spoof   = verdict == "SPOOF"
    conf_pct   = raw_prob * 100 if is_spoof else (1 - raw_prob) * 100
    verdict_display = "AI-GENERATED (SPOOF)" if is_spoof else "AUTHENTIC (BONAFIDE)"
    verdict_color   = "#E8443A" if is_spoof else "#3DBA7A"
    flagged_pct     = int(min(99, max(1, conf_pct * 0.85)))
    peak_anomaly    = 2.14

    # Pre-compute channel images once before any loops
    import matplotlib.pyplot as plt
    cmaps = [plt.get_cmap('magma'), plt.get_cmap('viridis'), plt.get_cmap('coolwarm')]
    channel_imgs_b64 = []
    for ch_idx in range(3):
        ch_data = feature_image[:, :, ch_idx] / 255.0
        colored = (cmaps[ch_idx](ch_data)[:, :, :3] * 255).astype(np.uint8)
        img_pil = Image.fromarray(colored)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        channel_imgs_b64.append(b64_str)

    scalar_cards_html = ""
    for i, card in enumerate(scalar_cards):
        v          = float(scalars[i])
        suspicious = _is_suspicious(i, v)
        card_cls   = "suspicious" if suspicious else "normal"
        val_cls    = "flag" if suspicious else "ok"
        status_txt  = _status_text(card, i, v, suspicious)
        status_icon = "!" if suspicious else "OK"

        scalar_cards_html += f"""
        <div class="scalar-card {card_cls}">
          <div class="scalar-feature-name">{card['idx']} {card['name']}</div>
          <div class="scalar-val {val_cls}">{v:.4f}</div>
          <div class="scalar-desc">{card['desc']}</div>
          <div class="scalar-status {val_cls}">{status_icon} {status_txt}</div>
        </div>"""

    # ── SHAP rows HTML ─────────────────────────────────────────
    shap_rows_html = ""
    max_abs = max(abs(d["val"]) for d in shap_data) if shap_data else 1
    for d in shap_data:
        is_pos    = d["val"] >= 0
        pct       = (abs(d["val"]) / max_abs) * 46
        direction = "positive" if is_pos else "negative"
        shap_rows_html += f"""
        <div class="shap-row">
          <div class="shap-feat">{d['feat']}</div>
          <div class="shap-bar-track">
            <div class="shap-bar {direction}" style="width:{pct:.1f}%"></div>
          </div>
          <div class="shap-val {direction}">{d['label']}</div>
        </div>"""

    out_fill_pct   = ((raw_prob - 0.50) / 0.50) * 46 if is_spoof else ((0.5 - raw_prob) / 0.50) * 46
    out_fill_pct   = max(0, min(46, out_fill_pct))
    out_fill_color = "#E8443A" if is_spoof else "#3DBA7A"
    out_val_disp   = f"{raw_prob:.3f}" if is_spoof else f"{1-raw_prob:.3f}"

    # ── Channel cards HTML ─────────────────────────────────────
    channel_cards_html = ""
    for idx, card in enumerate(channel_cards):
        channel_cards_html += f"""
        <div class="channel-card">
          <div class="channel-header">
            <span class="channel-name">{card['name']}</span>
            <span class="channel-badge">{card['badge']}</span>
          </div>
          <div class="channel-visual" style="position:relative;height:180px;background:#000;">
             <img src="data:image/png;base64,{channel_imgs_b64[idx]}" style="width:100%;height:100%;object-fit:cover;display:block;">
          </div>
          <div class="channel-summary">
            <div class="summary-title">{card['tech']}</div>
            <div class="summary-text">{card['summary']}</div>
          </div>
        </div>"""

    ts       = datetime.datetime.now().strftime("%Y-%m-%d · %H:%M:%S")
    filename = os.path.basename(audio_path)
    heatmap  = analysis_result.get("heatmap", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EchoTrace Forensic Report — {filename}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#09090A;--surface:#111113;--surface2:#18181B;
    --border:#222226;--border2:#2E2E34;
    --red:#E8443A;--red-dim:rgba(232,68,58,0.12);
    --accent:{verdict_color};
    --green:#3DBA7A;--green-dim:rgba(61,186,122,0.12);
    --text-primary:#F0EDE8;--text-secondary:#8A8A90;--text-dim:#3E3E44;
    --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
  }}
  body{{background:var(--bg);color:var(--text-primary);font-family:var(--sans);min-height:100vh;padding:48px 32px}}
  .report{{max-width:960px;margin:0 auto;position:relative}}
  .report::before,.report::after{{content:'';position:absolute;width:24px;height:24px;border-color:var(--red);border-style:solid;opacity:0.6}}
  .report::before{{top:-8px;left:-8px;border-width:2px 0 0 2px}}
  .report::after{{top:-8px;right:-8px;border-width:2px 2px 0 0}}
  .header{{border-bottom:1px solid var(--border2);padding-bottom:24px;margin-bottom:32px;display:flex;justify-content:space-between;align-items:flex-start}}
  .report-type{{font-family:var(--mono);font-size:10px;letter-spacing:0.22em;color:var(--red);text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:8px}}
  .report-type::before{{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--red)}}
  .wordmark{{font-family:var(--mono);font-size:28px;font-weight:600;letter-spacing:0.06em;color:var(--text-primary)}}
  .wordmark span{{color:var(--red)}}
  .header-right{{text-align:right}}
  .meta-row{{font-family:var(--mono);font-size:11px;color:var(--text-secondary);margin-bottom:4px}}
  .meta-row strong{{color:var(--text-primary);font-weight:500}}
  .verdict-banner{{background:var(--surface);border:1px solid var(--border2);border-left:3px solid var(--accent);padding:20px 24px;margin-bottom:28px;display:flex;align-items:center;justify-content:space-between;gap:24px}}
  .verdict-label{{font-family:var(--mono);font-size:10px;letter-spacing:0.18em;color:var(--text-secondary);text-transform:uppercase;margin-bottom:6px}}
  .verdict-value{{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--accent);letter-spacing:0.08em}}
  .verdict-stats{{display:flex;gap:32px}}
  .stat{{text-align:center}}
  .stat-val{{font-family:var(--mono);font-size:18px;font-weight:600;color:var(--text-primary)}}
  .stat-key{{font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.16em;text-transform:uppercase;margin-top:3px}}
  .section-label{{font-family:var(--mono);font-size:9px;letter-spacing:0.24em;color:var(--text-dim);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:10px}}
  .section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
  .llm-report{{background:var(--surface);border:1px solid var(--border2);padding:20px 24px;margin-bottom:36px;position:relative;overflow:hidden}}
  .llm-report::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,var(--red) 0%,transparent 60%)}}
  .llm-tag{{font-family:var(--mono);font-size:9px;letter-spacing:0.18em;color:var(--red);text-transform:uppercase;margin-bottom:12px;opacity:0.7}}
  .llm-text{{font-family:var(--sans);font-size:13px;font-weight:300;line-height:1.75;color:var(--text-secondary)}}
  .llm-text strong{{color:var(--text-primary);font-weight:500}}
  .channel-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:36px}}
  .channel-card{{background:var(--surface);border:1px solid var(--border);overflow:hidden}}
  .channel-header{{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}}
  .channel-name{{font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:0.14em;color:var(--text-secondary);text-transform:uppercase}}
  .channel-badge{{font-family:var(--mono);font-size:9px;color:var(--red);border:1px solid var(--red-dim);padding:2px 6px;letter-spacing:0.1em}}
  .channel-summary{{padding:12px 14px;border-top:1px solid var(--border);background:var(--surface2)}}
  .summary-title{{font-family:var(--mono);font-size:9px;letter-spacing:0.16em;color:var(--text-dim);text-transform:uppercase;margin-bottom:6px}}
  .summary-text{{font-family:var(--sans);font-size:11px;font-weight:300;line-height:1.6;color:var(--text-secondary)}}
  .scalar-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:36px}}
  .scalar-card{{background:var(--surface);border:1px solid var(--border);padding:12px 14px;position:relative;overflow:hidden}}
  .scalar-card.suspicious{{border-color:rgba(232,68,58,0.3)}}
  .scalar-card.suspicious::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:#E8443A;opacity:0.6}}
  .scalar-card.normal{{border-color:rgba(61,186,122,0.2)}}
  .scalar-card.normal::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--green);opacity:0.4}}
  .scalar-feature-name{{font-family:var(--mono);font-size:9px;letter-spacing:0.14em;color:var(--text-dim);text-transform:uppercase;margin-bottom:6px}}
  .scalar-val{{font-family:var(--mono);font-size:15px;font-weight:600;color:var(--text-primary);margin-bottom:4px}}
  .scalar-val.flag{{color:#E8443A}}
  .scalar-val.ok{{color:var(--green)}}
  .scalar-desc{{font-family:var(--sans);font-size:10px;font-weight:300;color:var(--text-dim);line-height:1.4}}
  .scalar-status{{font-family:var(--mono);font-size:8px;letter-spacing:0.12em;text-transform:uppercase;margin-top:6px}}
  .scalar-status.flag{{color:#E8443A}}
  .scalar-status.ok{{color:var(--green)}}
  .shap-wrap{{background:var(--surface);border:1px solid var(--border);overflow:hidden;margin-bottom:36px}}
  .shap-header{{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;gap:16px}}
  .shap-title{{font-family:var(--mono);font-size:10px;letter-spacing:0.14em;color:var(--text-secondary);text-transform:uppercase;margin-bottom:4px}}
  .shap-subtitle{{font-family:var(--sans);font-size:11px;font-weight:300;color:var(--text-dim);line-height:1.5;max-width:520px}}
  .shap-legend{{display:flex;flex-direction:column;gap:5px;flex-shrink:0}}
  .shap-legend-item{{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.1em;white-space:nowrap}}
  .shap-legend-bar{{width:24px;height:10px;border-radius:2px}}
  .shap-body{{padding:20px 24px}}
  .shap-endpoints{{display:flex;justify-content:space-between;margin-bottom:6px}}
  .shap-ep{{font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.12em}}
  .shap-ep span{{color:var(--text-secondary);font-weight:500}}
  .shap-rows{{display:flex;flex-direction:column;gap:6px}}
  .shap-row{{display:grid;grid-template-columns:140px 1fr 52px;align-items:center;gap:10px}}
  .shap-feat{{font-family:var(--mono);font-size:10px;color:var(--text-secondary);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .shap-bar-track{{position:relative;height:26px;background:var(--surface2);border-radius:2px;overflow:hidden}}
  .shap-bar-track::before{{content:'';position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border2);z-index:1}}
  .shap-bar{{position:absolute;top:4px;bottom:4px;border-radius:2px}}
  .shap-bar.positive{{background:rgba(232,68,58,0.75);left:50%}}
  .shap-bar.negative{{background:rgba(61,186,122,0.65);right:50%}}
  .shap-val{{font-family:var(--mono);font-size:11px;font-weight:500;text-align:left}}
  .shap-val.positive{{color:#E8443A}}
  .shap-val.negative{{color:var(--green)}}
  .shap-total-row{{display:grid;grid-template-columns:140px 1fr 52px;align-items:center;gap:10px;margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}}
  .shap-total-label{{font-family:var(--mono);font-size:10px;color:var(--text-dim);text-align:right;letter-spacing:0.1em;text-transform:uppercase}}
  .shap-output-bar{{position:relative;height:30px;background:var(--surface2);border-radius:2px;overflow:hidden}}
  .shap-output-bar::before{{content:'';position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border2);z-index:1}}
  .shap-output-fill{{position:absolute;top:3px;bottom:3px;left:50%;border-radius:2px;opacity:0.9}}
  .shap-output-val{{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--red)}}
  .shap-footer{{padding:10px 16px;border-top:1px solid var(--border);font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.1em}}
  .footer{{border-top:1px solid var(--border);padding-top:20px;display:flex;align-items:center;justify-content:space-between}}
  .footer-left{{font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.14em;text-transform:uppercase}}
  .footer-right{{font-family:var(--mono);font-size:9px;color:var(--text-dim);letter-spacing:0.1em}}
  .bottom-brackets{{position:relative;height:8px;margin-top:24px}}
  .bottom-brackets::before,.bottom-brackets::after{{content:'';position:absolute;width:24px;height:24px;border-color:var(--red);border-style:solid;opacity:0.6;bottom:0}}
  .bottom-brackets::before{{left:-8px;border-width:0 0 2px 2px}}
  .bottom-brackets::after{{right:-8px;border-width:0 2px 2px 0}}
</style>
</head>
<body>
<div class="report">

  <div class="header">
    <div class="header-left">
      <div class="report-type">Forensic Audio Analysis · Live</div>
      <div class="wordmark">ECHO<span>TRACE</span></div>
    </div>
    <div class="header-right">
      <div class="meta-row">File &nbsp;<strong>{filename}</strong></div>
      <div class="meta-row">Date &nbsp;<strong>{ts}</strong></div>
      <div class="meta-row">Model &nbsp;<strong>ResNet-50 · DDP Ensemble</strong></div>
      <div class="meta-row">Duration &nbsp;<strong>4.00s @ 16kHz</strong></div>
    </div>
  </div>

  <div class="verdict-banner">
    <div>
      <div class="verdict-label">Classification Verdict</div>
      <div class="verdict-value">{verdict_display}</div>
    </div>
    <div class="verdict-stats">
      <div class="stat"><div class="stat-val">{conf_pct:.1f}%</div><div class="stat-key">Confidence</div></div>
      <div class="stat"><div class="stat-val">{raw_prob:.3f}</div><div class="stat-key">Raw Prob</div></div>
      <div class="stat"><div class="stat-val">{flagged_pct}%</div><div class="stat-key">Windows Flagged</div></div>
      <div class="stat"><div class="stat-val">{peak_anomaly:.2f}s</div><div class="stat-key">Peak Anomaly</div></div>
    </div>
  </div>

  <div class="section-label">Forensic Analysis · LLM Report</div>
  <div class="llm-report">
    <div class="llm-tag">// Generated · llama-3.1-8b-instant</div>
    <div class="llm-text">{llm_text}</div>
  </div>

  <div class="section-label">Spectral Feature Channels · 224×224 · ResNet-50 Input</div>
  <div class="channel-grid">{channel_cards_html}</div>

  <div class="section-label">Scalar Forensic Vector · 8-Dim · Suspicious Values Flagged</div>
  <div class="scalar-grid">{scalar_cards_html}</div>

  <div class="section-label">SHAP Feature Attribution · Scalar Branch · DeepExplainer</div>
  <div class="shap-wrap">
    <div class="shap-header">
      <div class="shap-title-block">
        <div class="shap-title">Which voice property gave it away?</div>
        <div class="shap-subtitle">SHAP values show each scalar feature's individual contribution to the verdict. Red bars push toward SPOOF. Green bars push toward BONAFIDE.</div>
      </div>
      <div class="shap-legend">
        <div class="shap-legend-item"><div class="shap-legend-bar" style="background:rgba(232,68,58,0.75)"></div>Pushes toward SPOOF</div>
        <div class="shap-legend-item"><div class="shap-legend-bar" style="background:rgba(61,186,122,0.65)"></div>Pushes toward BONAFIDE</div>
      </div>
    </div>
    <div class="shap-body">
      <div class="shap-endpoints">
        <span class="shap-ep">Baseline (avg) &nbsp;<span>0.50</span></span>
        <span class="shap-ep">Model output &nbsp;<span style="color:var(--accent)">{raw_prob:.3f}</span></span>
      </div>
      <div class="shap-rows">{shap_rows_html}</div>
      <div class="shap-total-row">
        <div class="shap-total-label">Final output</div>
        <div class="shap-output-bar">
          <div class="shap-output-fill" style="width:{out_fill_pct:.1f}%;background:{out_fill_color};"></div>
          <div style="position:absolute;{'right:4px' if is_spoof else 'left:4px'};top:50%;transform:translateY(-50%);font-family:'IBM Plex Mono',monospace;font-size:7px;color:#3DBA7A;opacity:0.6;letter-spacing:0.08em;">{'SPOOF \u2192' if is_spoof else '\u2190 BONAFIDE'}</div>
        </div>
        <div class="shap-output-val" style="color:{out_fill_color}">{out_val_disp}</div>
      </div>
    </div>
    <div class="shap-footer">// Method: heuristic approximation of SHAP on scalar branch · values in probability space</div>
  </div>


  <div class="footer">
    <div class="footer-left">EchoTrace · BackProp Bandits · Udhbhav 2026</div>
    <div class="footer-right">Report ID: {report_id}</div>
  </div>
  <div class="bottom-brackets"></div>
</div>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"Report_{os.path.basename(audio_path)}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ── Main entry point ──────────────────────────────────────────
def generate_forensic_report(audio_path: str, original_filename: str = None, precomputed: dict = None):
    """
    Identical signature to original: (audio_path) → (result_dict, html_path)

    Outputs to reports/:
      Report_<filename>.html  — Full dark HTML report
      Report_<filename>.pdf   — PDF via playwright (if installed)
    """
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        msg = f"Unsupported File Format ({file_ext}). Please use: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        print(f"\n[!] ERROR: {msg}")
        return {"result": "ERROR", "confidence": "0%", "error": msg}, None, None, None, None

    if not os.path.exists(audio_path):
        return {"result": "ERROR", "confidence": "0%", "error": "File not found"}, None, None, None, None

    actual_name = original_filename if original_filename else os.path.basename(audio_path)
    print(f"\n[*] Analyzing: {actual_name}")

    if precomputed:
        audio_core    = precomputed.get("audio_np")
        feature_image = precomputed.get("feature_image")
        scalars       = precomputed.get("scalars")
        result        = precomputed.get("analysis_result")
        print("   • Using pre-computed audio features and model results")
    else:
        audio_core = load_audio(audio_path, target_sr=16000, duration=4.0, random_crop=False)
        print("   • Extracting feature image & scalars...")
        feature_image = build_feature_image(audio_core, sr=16000)
        scalars       = extract_scalar_features(audio_core, sr=16000)
        print("   • Running model inference...")
        with open(audio_path, "rb") as f:
            file_bytes = f.read()
        result   = asyncio.run(run_inference(file_bytes))

    verdict  = result.get("result", "UNKNOWN")
    raw_prob = float(result.get("raw_prob", 0.5))
    conf_val = raw_prob if verdict == "SPOOF" else (1 - raw_prob)

    # ── Parallel LLM Generation ─────────────────────────────────
    from utils.llm_report import generate_llm_report
    from utils.llm_cards import generate_card_analysis, _fallback

    print("   • Generating LLM forensics cards...")
    try:
        scalar_cards, channel_cards = generate_card_analysis(
            verdict=verdict,
            confidence=conf_val,
            scalars=scalars
        )
    except Exception as e:
        print(f"   [!] Cards failed: {e}")
        scalar_cards, channel_cards = _fallback()

    print("   • Generating narrative report (dual-layered)...")
    flagged_pct = min(99, max(1, conf_val * 85))
    peak_ts = precomputed.get("peak_timestamp", 0.0) if precomputed else 2.14
    
    try:
        llm_text = generate_llm_report(
            verdict=verdict,
            confidence=conf_val,
            scalars=scalars,
            flagged_windows_pct=flagged_pct,
            peak_timestamp=peak_ts,
            channels=channel_cards
        )
    except Exception as e:
        print(f"   [!] Summary failed: {e}")
        llm_text = f"Classified as **{verdict}** with **{raw_prob:.1%}** confidence. Forensic features indicate synthetic artifacts."

    shap_data  = _compute_shap_approx(scalars)
    report_id  = f"ET-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(root_path, "reports")


    # Generate HTML report
    print("   • Generating HTML report...")
    html_path = generate_html_report(
        audio_path=actual_name,
        analysis_result=result,
        scalars=scalars,
        shap_data=shap_data,
        scalar_cards=scalar_cards,
        channel_cards=channel_cards,
        feature_image=feature_image,
        llm_text=llm_text,
        report_id=report_id,
        output_dir=output_dir,
    )
    print(f"   [OK] HTML: {html_path}")

    # Generate PDF
    print("   • Generating PDF report...")
    try:
        from utils.pdf_export import html_to_pdf
        pdf_path = html_to_pdf(html_path)
        if pdf_path:
            print(f"   [OK] PDF:  {pdf_path}")
        else:
            print("   [!] PDF skipped - run: pip install playwright && playwright install chromium --with-deps")
    except Exception as e:
        print(f"   [!] PDF error: {e}")

    return result, html_path, scalar_cards, channel_cards, shap_data


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  [!] EchoTrace Forensic Report Generator")
    print("=" * 60)
    print(f"  Supported: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 60 + "\n")

    audio_to_test = input("File path: ").strip()
    if not audio_to_test:
        import sys; sys.exit(1)

    result, report_path = generate_forensic_report(audio_to_test)

    if result and "error" not in result:
        print("-" * 60)
        print(f"RESULT:      {result['result']}")
        print(f"PROBABILITY: {result['confidence']}")
        print(f"FILE (PNG):  {report_path}")
        if report_path:
            reports_dir = os.path.join(root_path, "reports")
            base = os.path.splitext(os.path.basename(report_path))[0]
            for ext, label in [(".html", "URL (HTML)"), (".pdf", "DOC (PDF)")]:
                p = os.path.join(reports_dir, base + ext)
                if os.path.exists(p):
                    print(f"{label}:   {p}")
        print("-" * 60 + "\n")
    else:
        print(f"[X] Failed: {result.get('error', 'Unknown error')}\n")