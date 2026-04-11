"""
utils/llm_report.py — LLM forensic report generation
Primary  : Groq  (LLaMA 3.1 8B via API)
Fallback : Ollama (llama3.2:3b, local)
Switch   : LLM_BACKEND env var = "groq" | "ollama"
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.1-8b-instant"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_MODEL  = "llama3.2:3b"
OLLAMA_URL    = "http://localhost:11434/api/generate"

LLM_BACKEND   = os.getenv("LLM_BACKEND", "groq").lower()
TIMEOUT       = 30

# Per-feature forensic metadata: (name, susp_if_low, susp_if_high, forensic_note)
_FEATURE_META = [
    ("Spectral Flatness",  True,  False, "vocoders are over-tonal → low flatness is a red flag"),
    ("Zero Crossing Rate", True,  False, "synthetic speech has unnaturally consistent ZCR"),
    ("F1 Formant",         False, True,  "unnatural vowel-space placement in neural TTS"),
    ("F2 Formant",         False, True,  "over-smooth F2 transitions betray synthesis"),
    ("F3 Formant",         False, True,  "missing micro-variation in F3 is a vocoder signature"),
    ("Voiced Ratio",       False, False, "extreme values (too low/high) suggest tampering"),
    ("HNR",                False, True,  "vocoders produce unnaturally clean, noise-free output"),
    ("CPP",                False, True,  "over-regular cepstral peaks lack biological variance"),
]


def _build_prompt(
    verdict: str,
    confidence: float,
    scalars: list,
    flagged_windows_pct: float,
    peak_timestamp: float,
    channels: list = None,
) -> str:
    label    = "AI-GENERATED (SPOOF)" if verdict == "SPOOF" else "AUTHENTIC (BONAFIDE)"
    is_spoof = verdict == "SPOOF"

    feature_lines     = []
    suspicious_names  = []
    clean_names       = []

    for i, (name, susp_low, susp_high, note) in enumerate(_FEATURE_META):
        v = float(scalars[i])
        if i == 5:  # Voiced Ratio: flag extremes
            is_susp = v < 0.4 or v > 0.98
        else:
            is_susp = (susp_low and v < 0.15) or (susp_high and v > 0.70)

        if is_susp:
            flag = "SUSPICIOUS"
            suspicious_names.append(name)
        else:
            flag = "OK"
            clean_names.append(name)

        feature_lines.append(
            f"  [{i}] {name}: {v:.4f}  [{flag}]  — {note}"
        )

    feature_block = "\n".join(feature_lines)
    susp_str  = ", ".join(suspicious_names)  if suspicious_names  else "none"
    clean_str = ", ".join(clean_names)        if clean_names        else "none"

    channel_block = ""
    if channels:
        channel_block = "\nLAYER 2: 3-CHANNEL 2D FEATURE REPRESENTATIONS (SPATIAL & STRUCTURAL CONTEXT):\n"
        for ch in channels:
            channel_block += f"- {ch['badge']}: {ch.get('summary', 'Analysis pending.')}\n"

    # Nuanced instruction based on verdict + feature tension
    nuance = ""
    if not is_spoof and suspicious_names:
        nuance = (
            f"\nCRITICAL: The verdict is BONAFIDE but [{susp_str}] showed suspicious values. "
            f"You MUST: (a) name those features and explain what made them suspicious, "
            f"(b) explain why [{clean_str}] and the spatial consistency in the 2D channels were strong enough to override the suspicion."
        )
    elif is_spoof and clean_names:
        nuance = (
            f"\nCRITICAL: The verdict is SPOOF but [{clean_str}] appeared within normal range. "
            f"You MUST: (a) explain which specific features [{susp_str}] betrayed the synthesis, "
            f"(b) explain how the 2D channel data (spatial patterns/texture) revealed inconsistencies that the scalars missed."
        )

    return f"""You are EchoTrace, an AI audio forensics system. Your job is to provide a comprehensive, dual-layered forensic report explaining why a sample was classified as {label}. 

THE VERDICT: {label}
CONFIDENCE: {confidence:.1%}
KEY STATS: {flagged_windows_pct:.0f}% of the audio's windows were flagged. Peak anomaly at {peak_timestamp:.2f}s.

LAYER 1: 1D SCALAR FEATURES (SIMPLER, SUMMARY MEASUREMENTS)
{feature_block}
{channel_block if channel_block else ""}

YOUR TASK:
Explain the reasoning behind this classification by examining two distinct analytical pathways:
1. THE 2D CHANNEL ANALYSIS (DETAILED, PICTURE-LIKE VIEW): Discuss how the detailed, image-like spectral layers (Mel Spectrogram, MFCC dynamics, Contrast/Chroma) reveal patterns, textures, or spatial inconsistencies. Explain what about these visual-style representations led to the conclusion of {verdict}.
2. THE 1D SCALAR EVALUATION (STATISTICAL SUMMARY VIEW): Discuss which simple, single-number measurements (like Spectral Flatness, Formants, or HNR/CPP) were most decisive. Explain how these numerical summaries provide critical quantitative evidence that supports the visual findings.

WRITING GUIDELINES:
- Write 6-8 sentences in professional, flowing paragraph prose. No bullet points.
- Emphasize the two-step approach: Explain why the detailed "picture-like" information and the "quick summary" numerical measurements together make the classification stronger and more trustworthy.
- **Bold** feature names and the final verdict.
- Final sentence MUST start with "CONCLUSION:" and be a clear, jargon-free summary for a non-technical stakeholder (like a jury or journalist).

Write the report now:"""


def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 420,
        "temperature": 0.2,
    }
    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str) -> str:
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": 420},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["response"].strip()


def generate_llm_report(
    verdict: str,
    confidence: float,
    scalars=None,
    flagged_windows_pct: float = 0.0,
    peak_timestamp: float = 0.0,
    channels: list = None,
    # Legacy compat args (used only when scalars is None)
    f0_jitter: float = 0.0,
    spectral_contrast: float = 0.0,
    voiced_ratio: float = 1.0,
) -> str:
    """
    Generate a plain-English forensic report string.
    Returns a fallback rule-based summary if all LLM backends fail.

    Args:
        verdict              : "SPOOF" or "BONAFIDE"
        confidence           : model confidence as a float (0.0–1.0)
        scalars              : list/ndarray of 8 normalized scalar features [0, 1]
        flagged_windows_pct  : % of sliding windows classified SPOOF (0–100)
        peak_timestamp       : seconds where peak anomaly occurred
        channels             : Optional list of 3 channel card dicts
    """
    if scalars is not None:
        scalar_list = [float(x) for x in scalars]
        if len(scalar_list) < 8:
            scalar_list += [0.5] * (8 - len(scalar_list))
    else:
        # Legacy compat: reconstruct rough 8-dim vector
        scalar_list = [f0_jitter, 0.1, spectral_contrast, 0.5, 0.5,
                       voiced_ratio, f0_jitter, spectral_contrast]

    prompt = _build_prompt(verdict, confidence, scalar_list,
                           flagged_windows_pct, peak_timestamp, channels=channels)

    backend = LLM_BACKEND
    try:
        return _call_groq(prompt) if backend == "groq" else _call_ollama(prompt)
    except Exception as e_primary:
        print(f"[llm_report] Primary backend ({backend}) failed: {e_primary}")

    try:
        fallback = "ollama" if backend == "groq" else "groq"
        print(f"[llm_report] Trying fallback: {fallback}")
        return _call_groq(prompt) if fallback == "groq" else _call_ollama(prompt)
    except Exception as e_fallback:
        print(f"[llm_report] Fallback backend failed: {e_fallback}")

    return _rule_based_report(verdict, confidence, scalar_list,
                              flagged_windows_pct)


def _rule_based_report(
    verdict: str,
    confidence: float,
    scalars: list,
    flagged_windows_pct: float,
) -> str:
    sf, zcr, f1, f2, f3, vr, hnr, cpp = [float(x) for x in scalars]

    # Find the most suspicious features
    suspicious = []
    if sf < 0.15:
        suspicious.append(f"**Spectral Flatness** ({sf:.4f}) is abnormally tonal — consistent with vocoder synthesis")
    if zcr < 0.15:
        suspicious.append(f"**Zero Crossing Rate** ({zcr:.4f}) shows unnatural temporal consistency")
    if hnr > 0.70:
        suspicious.append(f"**HNR** ({hnr:.4f}) is unnaturally noise-free, a hallmark of neural TTS")
    if cpp > 0.70:
        suspicious.append(f"**CPP** ({cpp:.4f}) shows over-regular cepstral periodicity absent in real speech")
    if vr < 0.4 or vr > 0.98:
        suspicious.append(f"**Voiced Ratio** ({vr:.4f}) sits at an abnormal extreme")

    susp_str = "; ".join(suspicious) if suspicious else "no individual scalar features exceeded suspicion thresholds"

    if verdict == "SPOOF":
        return (
            f"Forensic analysis detected the following synthesis artifacts: {susp_str}. "
            f"**{flagged_windows_pct:.0f}%** of the temporal analysis windows were flagged as containing vocoder-origin patterns. "
            f"The convergence of anomalies across multiple scalar dimensions indicates neural TTS or voice conversion origin. "
            f"CONCLUSION: The sample is classified as **AI-GENERATED (SPOOF)** with **{confidence:.1%}** confidence."
        )
    else:
        tension = f"Although {susp_str}, " if suspicious else ""
        return (
            f"{tension}the overall scalar profile — **Spectral Flatness** ({sf:.4f}), **HNR** ({hnr:.4f}), and **CPP** ({cpp:.4f}) — "
            f"falls within ranges consistent with natural human phonation. "
            f"Only **{flagged_windows_pct:.0f}%** of temporal windows showed anomalous activity, insufficient to override authentic classification. "
            f"CONCLUSION: The sample is classified as **AUTHENTIC (BONAFIDE)** with **{confidence:.1%}** confidence."
        )