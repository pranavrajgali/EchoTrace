"""
utils/llm_cards.py — Groq-powered dynamic card text generation for EchoTrace reports.

Replaces ALL hardcoded descriptions in:
  - SCALAR_THRESHOLDS (8 scalar feature cards)
  - channel_labels (3 spectral channel cards)

One Groq call returns a JSON object with everything. Falls back to static
defaults if Groq is unreachable so the report always renders.
"""
import os
import json
import requests

# ── Groq config (mirrors utils/llm_report.py) ────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT      = 30


# ── Static fallbacks (used when Groq is unavailable) ─────────
_FALLBACK_SCALARS = [
    {
        "name": "Spectral Flatness",
        "idx": "[0]",
        "desc": "Wiener entropy measures tonality. Vocoders produce overly tonal output — unnaturally low flatness.",
        "susp_low": True,
        "susp_high": False,
        "low_status": "Abnormally tonal",
        "high_status": "Noise-dominated",
        "ok_status": "Within range",
    },
    {
        "name": "Zero Crossing Rate",
        "idx": "[1]",
        "desc": "Rate of signal sign changes. Synthetic speech has unnaturally consistent ZCR across frames.",
        "susp_low": True,
        "susp_high": False,
        "low_status": "Too consistent",
        "high_status": "Excessive noise",
        "ok_status": "Within range",
    },
    {
        "name": "F1 Formant",
        "idx": "[2]",
        "desc": "First formant frequency via LPC roots. Vocoders produce unrealistic vowel-space placement.",
        "susp_low": False,
        "susp_high": True,
        "low_status": "Compressed vowel space",
        "high_status": "Unnatural placement",
        "ok_status": "Within range",
    },
    {
        "name": "F2 Formant",
        "idx": "[3]",
        "desc": "Second formant frequency. Neural vocoders produce over-smooth F2 transitions.",
        "susp_low": False,
        "susp_high": True,
        "low_status": "Suppressed F2",
        "high_status": "Over-smooth",
        "ok_status": "Within range",
    },
    {
        "name": "F3 Formant",
        "idx": "[4]",
        "desc": "Third formant frequency. Synthetic speech lacks the natural micro-variation in F3.",
        "susp_low": False,
        "susp_high": True,
        "low_status": "Suppressed F3",
        "high_status": "Low variation",
        "ok_status": "Within range",
    },
    {
        "name": "Voiced Ratio",
        "idx": "[5]",
        "desc": "Fraction of frames containing detected voiced speech via energy thresholding.",
        "susp_low": False,
        "susp_high": False,
        "low_status": "Too little voicing",
        "high_status": "Abnormally high voicing",
        "ok_status": "Within range",
    },
    {
        "name": "HNR",
        "idx": "[6]",
        "desc": "Harmonic-to-Noise Ratio via autocorrelation. Vocoders produce unnaturally clean, noise-free output.",
        "susp_low": False,
        "susp_high": True,
        "low_status": "Noise-dominated",
        "high_status": "Unnaturally high",
        "ok_status": "Within range",
    },
    {
        "name": "CPP",
        "idx": "[7]",
        "desc": "Cepstral Peak Prominence. Vocoders produce too-regular cepstral peaks lacking biological variance.",
        "susp_low": False,
        "susp_high": True,
        "low_status": "Weak periodicity",
        "high_status": "Over-regular",
        "ok_status": "Within range",
    },
]

_FALLBACK_CHANNELS = [
    {
        "name": "Channel 1",
        "badge": "Mel Spectrogram",
        "tech": "// Ch1 · 128 mel bands · n_fft=2048 · hop=256",
        "summary": "Strong low-frequency energy with formant structure. Unnaturally periodic — consistent with neural vocoder synthesis rather than natural phonation.",
    },
    {
        "name": "Channel 2",
        "badge": "MFCC + Δ + Δ²",
        "tech": "// Ch2 · 40 MFCCs + 40 Δ + 40 ΔΔ = 120 rows",
        "summary": "Unusually uniform cepstral energy across all 120 rows. Natural speech shows higher delta variance — this flatness in temporal dynamics is a strong synthetic indicator.",
    },
    {
        "name": "Channel 3",
        "badge": "Contrast + Chroma",
        "tech": "// Ch3 · 7 contrast bands + 12 chroma CQT bins",
        "summary": "Sparse harmonic content with suppressed chroma variation across pitch classes. Authentic speech shows richer tonal distribution — sparsity here indicates vocoder spectral smoothing.",
    },
]


def _build_prompt(
    verdict: str,
    confidence: float,
    scalars: list,   # list of 8 floats
) -> str:
    scalar_lines = "\n".join(
        f"  [{i}] {name}: {scalars[i]:.4f}"
        for i, name in enumerate([
            "Spectral Flatness", "Zero Crossing Rate",
            "F1 Formant", "F2 Formant", "F3 Formant",
            "Voiced Ratio", "HNR", "CPP",
        ])
    )

    return f"""You are EchoTrace, a forensic audio AI. An audio sample was classified as {verdict} with {confidence:.1%} confidence.
The 8 normalized scalar features (all in range [0,1]) are:
{scalar_lines}

Return ONLY a valid JSON object — no markdown, no explanation, no preamble. The JSON must have exactly this structure:

{{
  "scalars": [
    {{
      "name": "Spectral Flatness",
      "idx": "[0]",
      "desc": "<one sentence: what this feature measures and what the value {scalars[0]:.4f} specifically tells a forensic analyst about this sample>",
      "low_status": "<3-5 word label for suspiciously low values>",
      "high_status": "<3-5 word label for suspiciously high values>",
      "ok_status": "Within range"
    }},
    {{
      "name": "Zero Crossing Rate",
      "idx": "[1]",
      "desc": "<one sentence specific to value {scalars[1]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "F1 Formant",
      "idx": "[2]",
      "desc": "<one sentence specific to value {scalars[2]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "F2 Formant",
      "idx": "[3]",
      "desc": "<one sentence specific to value {scalars[3]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "F3 Formant",
      "idx": "[4]",
      "desc": "<one sentence specific to value {scalars[4]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "Voiced Ratio",
      "idx": "[5]",
      "desc": "<one sentence specific to value {scalars[5]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "HNR",
      "idx": "[6]",
      "desc": "<one sentence specific to value {scalars[6]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }},
    {{
      "name": "CPP",
      "idx": "[7]",
      "desc": "<one sentence specific to value {scalars[7]:.4f}>",
      "low_status": "<3-5 words>",
      "high_status": "<3-5 words>",
      "ok_status": "Within range"
    }}
  ],
  "channels": [
    {{
      "name": "Channel 1",
      "badge": "Mel Spectrogram",
      "tech": "// Ch1 · 128 mel bands · n_fft=2048 · hop=256",
      "summary": "<2 sentences: what the mel spectrogram reveals about this specific {verdict} sample, referencing the scalar evidence>"
    }},
    {{
      "name": "Channel 2",
      "badge": "MFCC + Δ + Δ²",
      "tech": "// Ch2 · 40 MFCCs + 40 Δ + 40 ΔΔ = 120 rows",
      "summary": "<2 sentences: what the MFCC dynamics reveal about this specific sample>"
    }},
    {{
      "name": "Channel 3",
      "badge": "Contrast + Chroma",
      "tech": "// Ch3 · 7 contrast bands + 12 chroma CQT bins",
      "summary": "<2 sentences: what the spectral contrast and chroma reveal about this specific sample>"
    }}
  ]
}}"""


def generate_card_analysis(
    verdict: str,
    confidence: float,
    scalars,   # np.ndarray or list of 8 floats
) -> tuple:
    """
    Call Groq to generate dynamic, data-specific card text.

    Returns:
        scalar_cards : list of 8 dicts (each has name, idx, desc, susp_low,
                       susp_high, low_status, high_status, ok_status)
        channel_cards: list of 3 dicts (each has name, badge, tech, summary)

    Falls back to static defaults on any failure.
    """
    scalar_list = [float(x) for x in scalars]

    if not GROQ_API_KEY:
        print("[llm_cards] GROQ_API_KEY not set — using static fallbacks")
        return _fallback()

    prompt = _build_prompt(verdict, confidence, scalar_list)

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1800,
            "temperature": 0.25,   # low temp = consistent JSON structure
        }
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if the model wraps it anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        scalar_cards, channel_cards = _parse_response(data, scalar_list)
        print("[llm_cards] ✅ Dynamic card analysis generated via Groq")
        return scalar_cards, channel_cards

    except json.JSONDecodeError as e:
        print(f"[llm_cards] JSON parse error: {e} — using static fallbacks")
        return _fallback()
    except Exception as e:
        print(f"[llm_cards] Groq call failed: {e} — using static fallbacks")
        return _fallback()


def _parse_response(data: dict, scalar_list: list) -> tuple:
    """
    Parse Groq JSON response. Merge in the hardcoded susp_low/susp_high
    thresholds (those are model architecture decisions, not LLM territory).
    """
    # Suspicion direction is fixed by the feature physics — LLM doesn't decide this
    SUSP_DIRECTION = [
        (True,  False),  # [0] Spectral Flatness: low = spoof
        (True,  False),  # [1] ZCR: low = spoof
        (False, True),   # [2] F1: high = spoof
        (False, True),   # [3] F2: high = spoof
        (False, True),   # [4] F3: high = spoof
        (False, False),  # [5] Voiced Ratio: neither extreme is clearly spoof
        (False, True),   # [6] HNR: high = spoof
        (False, True),   # [7] CPP: high = spoof
    ]

    scalar_cards = []
    for i, item in enumerate(data.get("scalars", [])):
        susp_low, susp_high = SUSP_DIRECTION[i] if i < len(SUSP_DIRECTION) else (False, False)
        scalar_cards.append({
            "name":        item.get("name",       _FALLBACK_SCALARS[i]["name"]),
            "idx":         item.get("idx",        _FALLBACK_SCALARS[i]["idx"]),
            "desc":        item.get("desc",       _FALLBACK_SCALARS[i]["desc"]),
            "susp_low":    susp_low,
            "susp_high":   susp_high,
            "low_status":  item.get("low_status",  _FALLBACK_SCALARS[i]["low_status"]),
            "high_status": item.get("high_status", _FALLBACK_SCALARS[i]["high_status"]),
            "ok_status":   item.get("ok_status",  "Within range"),
        })

    # Pad if Groq returned fewer than 8
    while len(scalar_cards) < 8:
        i = len(scalar_cards)
        scalar_cards.append(_FALLBACK_SCALARS[i].copy())

    channel_cards = []
    for i, item in enumerate(data.get("channels", [])):
        channel_cards.append({
            "name":    item.get("name",    _FALLBACK_CHANNELS[i]["name"]),
            "badge":   item.get("badge",   _FALLBACK_CHANNELS[i]["badge"]),
            "tech":    item.get("tech",    _FALLBACK_CHANNELS[i]["tech"]),
            "summary": item.get("summary", _FALLBACK_CHANNELS[i]["summary"]),
        })

    while len(channel_cards) < 3:
        i = len(channel_cards)
        channel_cards.append(_FALLBACK_CHANNELS[i].copy())

    return scalar_cards, channel_cards


def _fallback() -> tuple:
    """Return static fallback data with susp_low/susp_high merged in."""
    scalar_cards = []
    for item in _FALLBACK_SCALARS:
        scalar_cards.append(item.copy())

    channel_cards = [item.copy() for item in _FALLBACK_CHANNELS]
    return scalar_cards, channel_cards
