"""
utils/audio.py — VAD + audio validation for EchoTrace
Uses webrtcvad for voice activity detection.
"""
import io
import struct
import numpy as np
import librosa
import soundfile as sf

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("[audio] webrtcvad not installed — VAD disabled, install with: pip install webrtcvad")


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
            "EchoTrace requires ≥ 16 kHz audio — upsampling corrupts the upper "
            "half of the spectrogram and produces unreliable results. "
            "Please provide audio recorded at 16 kHz or higher."
        )

    # Resample to exactly 16 kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

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

    if voiced_ratio == 0.0:
        raise AudioValidationError(
            "No speech detected in the audio. "
            "Please ensure the recording contains a voice and is not muted."
        )

    return audio, voiced_ratio


def _run_vad(audio: np.ndarray, sr: int = 16000, aggressiveness: int = 2) -> float:
    """
    Run webrtcvad on 20ms frames and return the fraction of voiced frames.
    Falls back to 1.0 if webrtcvad is not available (assume voiced).
    """
    if not VAD_AVAILABLE:
        return 1.0

    vad = webrtcvad.Vad(aggressiveness)

    frame_ms   = 20           # webrtcvad supports 10, 20, 30 ms
    frame_len  = int(sr * frame_ms / 1000)   # samples per frame
    pcm_bytes  = _to_pcm16(audio)

    total_frames  = 0
    voiced_frames = 0

    for start in range(0, len(pcm_bytes) - frame_len * 2, frame_len * 2):
        frame = pcm_bytes[start : start + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            is_speech = vad.is_speech(frame, sr)
        except Exception:
            is_speech = False
        total_frames  += 1
        voiced_frames += int(is_speech)

    if total_frames == 0:
        return 0.0

    return voiced_frames / total_frames


def _to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to raw int16 PCM bytes."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    return struct.pack(f"{len(pcm_int16)}h", *pcm_int16)