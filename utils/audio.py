"""
utils/audio.py — VAD + audio validation for EchoTrace
Pure-Python voice activity detection using librosa RMS energy + ZCR.
No external C dependencies required.
"""
import io
import numpy as np
import librosa


class AudioValidationError(Exception):
    """Raised when audio fails pre-flight checks. Message is shown directly in UI."""
    pass


def validate_and_load(file_bytes: bytes, min_duration_sec: float = 2.0) -> tuple:
    """
    Load, validate, and prepare audio for EchoTrace inference.

    Returns:
        audio (np.ndarray): mono float32, 16kHz, peak-normalised
        voiced_ratio (float): fraction of frames containing speech (0.0–1.0)

    Raises:
        AudioValidationError: with a clean message suitable for display in the UI
    """
    # ── Load ──────────────────────────────────────────────────
    try:
        audio, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True, res_type="soxr_hq")
    except Exception as e:
        raise AudioValidationError(f"Could not read audio file: {e}")

    # ── Sample rate check ─────────────────────────────────────
    if sr < 16000:
        raise AudioValidationError(
            f"Sample rate too low ({sr} Hz). "
            "EchoTrace requires >= 16 kHz audio -- upsampling corrupts the upper "
            "half of the spectrogram and produces unreliable results. "
            "Please provide audio recorded at 16 kHz or higher."
        )

    # Resample to exactly 16 kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

    # ── Effective bandwidth check ──────────────────────────────
    # Catches audio that was upsampled from a lower rate (e.g. 8kHz→16kHz).
    # The file header says 16kHz, but the actual energy only reaches ~4kHz.
    # This produces an empty upper spectrogram → unreliable model output.
    effective_bw = _estimate_bandwidth(audio, sr)
    if effective_bw < 5000:
        raise AudioValidationError(
            f"Audio bandwidth too low (~{effective_bw:.0f} Hz effective). "
            "This appears to be low-rate audio (e.g. 8 kHz telephony) that was "
            "upsampled to a higher container rate. EchoTrace needs wideband "
            "(>= 8 kHz bandwidth) audio for reliable forensic analysis. "
            "Please provide a recording captured at 16 kHz or higher."
        )

    # ── Duration check ────────────────────────────────────────
    duration = len(audio) / sr
    if duration < min_duration_sec:
        raise AudioValidationError(
            f"Audio too short ({duration:.1f}s). "
            f"Minimum required duration is {min_duration_sec}s."
        )

    # ── Peak normalisation ────────────────────────────────────
    peak = np.max(np.abs(audio))
    if peak < 1e-7:
        raise AudioValidationError(
            "Audio appears to be silent (peak amplitude < 1e-7). "
            "Please check your recording."
        )
    audio = audio / peak

    # ── VAD ───────────────────────────────────────────────────
    voiced_ratio = _run_vad(audio, sr)

    if voiced_ratio < 0.05:
        raise AudioValidationError(
            "No speech detected in the audio. "
            "EchoTrace requires voice content for forensic analysis. "
            "Please ensure the recording contains actual speech and is not "
            "just silence, music, or background noise."
        )

    return audio, voiced_ratio


def _estimate_bandwidth(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Estimate the effective bandwidth of the audio using spectral rolloff.

    Returns the frequency (Hz) below which 95% of the spectral energy lies.
    For genuine 16kHz audio, this should be well above 5kHz.
    For upsampled 8kHz audio, this will be around 3.5–4kHz.
    """
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, roll_percent=0.95
    )[0]

    # Use the 75th percentile of rolloff values to be robust against
    # brief silent segments that would drag down the average
    if len(rolloff) == 0:
        return 0.0
    return float(np.percentile(rolloff, 75))


def _run_vad(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Pure-Python voice activity detection using RMS energy + zero-crossing rate.

    Strategy:
        1. Compute short-time RMS energy per frame (20ms frames, 10ms hop).
        2. Derive a dynamic threshold from the energy distribution.
        3. A frame is 'voiced' if its energy exceeds the threshold AND
           its zero-crossing rate is within a speech-like range
           (speech has moderate ZCR; noise/silence has very high or very low ZCR).

    Returns:
        float: fraction of frames classified as voiced (0.0 – 1.0)
    """
    frame_length = int(sr * 0.025)   # 25ms frames
    hop_length   = int(sr * 0.010)   # 10ms hop

    # ── RMS energy per frame ──
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    # ── Zero-crossing rate per frame ──
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    if len(rms) == 0:
        return 0.0

    # ── Dynamic energy threshold ──
    # Use the 15th percentile as floor estimate + scale factor
    energy_floor = np.percentile(rms, 15)
    energy_threshold = max(energy_floor * 4.0, np.mean(rms) * 0.35)

    # ── ZCR bounds for speech ──
    # Human speech typically has ZCR between 0.02 and 0.25
    # Pure noise tends to have very high ZCR; silence has near-zero
    zcr_low  = 0.01
    zcr_high = 0.30

    voiced_count = 0
    total_frames = len(rms)

    for i in range(total_frames):
        energy_ok = rms[i] > energy_threshold
        zcr_ok    = zcr_low <= zcr[i] <= zcr_high
        if energy_ok and zcr_ok:
            voiced_count += 1

    return voiced_count / total_frames