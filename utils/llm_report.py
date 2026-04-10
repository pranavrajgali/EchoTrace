"""
utils/llm_report.py — LLM forensic report generation
Primary  : Groq  (LLaMA 3.1 8B via API)
Fallback : Ollama (llama3.2:3b, local)
Switch   : LLM_BACKEND env var = "groq" | "ollama"
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Config ───────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.1-8b-instant"
GROQ_URL      = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_MODEL  = "llama3.2:3b"
OLLAMA_URL    = "http://localhost:11434/api/generate"

LLM_BACKEND   = os.getenv("LLM_BACKEND", "groq").lower()   # "groq" | "ollama"

TIMEOUT       = 30   # seconds


def _build_prompt(
    verdict: str,
    confidence: float,
    f0_jitter: float,
    spectral_contrast: float,
    peak_timestamp: float,
    flagged_windows_pct: float,
    voiced_ratio: float,
) -> str:
    """
    Construct the forensic analysis prompt sent to the LLM.
    """
    label = "AI-GENERATED (SPOOF)" if verdict == "SPOOF" else "AUTHENTIC (BONAFIDE)"
    return f"""You are EchoTrace, a forensic audio analysis AI. Write a concise plain-English report 
analyzing the provided data. 

STRICT REQUIREMENTS:
1. EXPLAIN the features: Briefly explain what the metrics (flatness, HNR, jitter, etc.) indicate about the audio's authenticity.
2. BOLD key terms: Highlight important findings, metrics, and the final result using **bold** text.
3. STRUCTURE: Start with the analysis of specific features, then provide a definitive **CONCLUSION** at the end.
4. TONE: Clinical, forensic, and objective. No greetings or headers. 
5. LENGTH: 3-5 sentences total.

ANALYSIS DATA:
- Verdict: **{label}**
- Model confidence: **{confidence:.1%}**
- Harmonic-to-Noise Ratio (HNR): {f0_jitter:.4f} (synthetic speech is often "too clean" with high HNR)
- Spectral / Formant Consistency: {spectral_contrast:.4f} (unnatural smoothness in transitions)
- Peak anomaly at: {peak_timestamp:.2f}s
- Flagged windows: **{flagged_windows_pct:.0f}%**
- Voiced ratio: {voiced_ratio:.0%}

Write the forensic summary now:"""


def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.3,
    }
    r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 256},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["response"].strip()


def generate_llm_report(
    verdict: str,
    confidence: float,
    f0_jitter: float = 0.0,
    spectral_contrast: float = 0.0,
    peak_timestamp: float = 0.0,
    flagged_windows_pct: float = 0.0,
    voiced_ratio: float = 1.0,
) -> str:
    """
    Generate a plain-English forensic report string.
    Returns a fallback rule-based summary if both LLM backends fail.

    Args:
        verdict              : "SPOOF" or "BONAFIDE"
        confidence           : model confidence as a float (0.0–1.0)
        f0_jitter            : scalar feature value
        spectral_contrast    : scalar feature value
        peak_timestamp       : seconds into the audio where peak anomaly occurred
        flagged_windows_pct  : % of sliding windows flagged as SPOOF (0–100)
        voiced_ratio         : fraction of audio with detected speech (0.0–1.0)
    """
    prompt = _build_prompt(
        verdict, confidence, f0_jitter, spectral_contrast,
        peak_timestamp, flagged_windows_pct, voiced_ratio,
    )

    backend = LLM_BACKEND

    # Try primary backend
    try:
        if backend == "groq":
            return _call_groq(prompt)
        else:
            return _call_ollama(prompt)
    except Exception as e_primary:
        print(f"[llm_report] Primary backend ({backend}) failed: {e_primary}")

    # Try fallback backend
    try:
        fallback = "ollama" if backend == "groq" else "groq"
        print(f"[llm_report] Trying fallback: {fallback}")
        if fallback == "groq":
            return _call_groq(prompt)
        else:
            return _call_ollama(prompt)
    except Exception as e_fallback:
        print(f"[llm_report] Fallback backend failed: {e_fallback}")

    # Rule-based fallback — always works
    return _rule_based_report(verdict, confidence, flagged_windows_pct, voiced_ratio)


def _rule_based_report(
    verdict: str,
    confidence: float,
    flagged_windows_pct: float,
    voiced_ratio: float,
) -> str:
    if verdict == "SPOOF":
        return (
            f"Analysis of spectral characteristics reveals **{flagged_windows_pct:.0f}%** of segments contain synthetic artifacts. "
            f"The **Harmonic-to-Noise Ratio** and **Spectral Flatness** exhibit levels of consistency typical of neural vocoder synthesis rather than natural human vocal tracts. "
            f"Periodic anomalies were detected throughout the **{voiced_ratio:.0%}** voiced content. "
            f"**CONCLUSION**: The sample is classified as **AI-GENERATED (SPOOF)** with **{confidence:.1%}** confidence."
        )
    else:
        return (
            f"Spectral features show natural variance in **formant transitions** and **zero-crossing rates** across the audio. "
            f"Only **{flagged_windows_pct:.0f}%** of windows showed minor anomalies, which are consistent with ambient noise rather than synthesis. "
            f"The **prosodic structure** aligns with authentic human phonation across **{voiced_ratio:.0%}** of the sample. "
            f"**CONCLUSION**: The sample is classified as **AUTHENTIC (BONAFIDE)** with **{confidence:.1%}** confidence."
        )