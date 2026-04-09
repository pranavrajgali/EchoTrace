"""
core/preprocess.py — EchoTrace feature extraction and dataset classes
All paths use absolute /home/jovyan/work/data/ root.
No warm-start. Training from ImageNet init only.
"""
import os
import torch
import librosa
import numpy as np
import random
import glob
import csv
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# ── Absolute data root ────────────────────────────────────────
DATA_ROOT = "/home/jovyan/work/data"

ASV_PROTOCOL    = os.path.join(DATA_ROOT, "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
ASV_DIR         = os.path.join(DATA_ROOT, "LA/LA/ASVspoof2019_LA_train/flac")
WAVEFAKE_DIR    = os.path.join(DATA_ROOT, "wavefake-test/wavefake-test")
ITW_DIR         = os.path.join(DATA_ROOT, "release_in_the_wild/release_in_the_wild")
LIBRISPEECH_DIR = os.path.join(DATA_ROOT, "LibriSpeech")
MUSAN_DIR       = os.path.join(DATA_ROOT, "noise/musan")


# ── Audio Augmenter ───────────────────────────────────────────
class AudioAugmenter:
    """
    Real-world noise injection using MUSAN corpus.
    Falls back to white noise if MUSAN not found.
    """
    def __init__(self, p=0.3, musan_path=MUSAN_DIR):
        self.p = p
        self.musan_files = glob.glob(os.path.join(musan_path, "**/*.wav"), recursive=True)
        if not self.musan_files:
            print(f"[augment] WARNING: No MUSAN files found at {musan_path} — using white noise fallback")

    def add_noise(self, audio):
        if random.random() > self.p:
            return audio
        if self.musan_files:
            noise_path = random.choice(self.musan_files)
            try:
                noise, _ = librosa.load(noise_path, sr=16000)
                if len(noise) < len(audio):
                    noise = np.pad(noise, (0, len(audio) - len(noise)))
                else:
                    noise = noise[:len(audio)]
                snr_factor = random.uniform(0.01, 0.1)
                return audio + noise * snr_factor
            except Exception:
                pass
        # White noise fallback
        return audio + np.random.normal(0, 0.005, len(audio)).astype(np.float32)

    def apply_augmentations(self, audio):
        return self.add_noise(audio)


# ── Feature extraction ────────────────────────────────────────
def extract_scalar_features(audio, sr=16000):
    """
    8-dim forensically-targeted scalar feature vector for synthetic speech detection.
    
    Features (in order):
      [0] spectral_flatness   — noise-like vs tonal (vocoders are too tonal)
      [1] zcr                 — zero crossing rate (synthetic speech has unnatural consistency)
      [2] f1_formant          — F1 frequency (LPC-derived)
      [3] f2_formant          — F2 frequency (LPC-derived)
      [4] f3_formant          — F3 frequency (LPC-derived)
      [5] voiced_ratio        — fraction of voiced frames (vocoders have unnatural ratios)
      [6] hnr                 — harmonic-to-noise ratio in dB (vocoders are too clean)
      [7] cpp                 — cepstral peak prominence (vocoders too regular)
    
    Returns: float32 numpy array of shape (8,), normalized to [0, 1].
    """
    try:
        # [0] Spectral Flatness — Wiener entropy / spectral smoothness
        #   Theoretically [0, 1] but clip to handle numerical edge cases
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        flatness = np.clip(flatness, 0, 1)
        
        # [1] Zero Crossing Rate
        #   Normalized to [0, 1] by Nyquist in librosa, but clip for safety
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        zcr = np.clip(zcr, 0, 1)
        
        # [2-4] Formants via LPC (Linear Predictive Coding)
        #   Extract first 3 formants from LPC polynomial roots
        lpc_coefs = librosa.lpc(audio, order=8)
        roots = np.roots(lpc_coefs)
        angles = np.angle(roots)
        freqs = angles * (sr / (2 * np.pi))
        
        # Keep only positive frequencies below Nyquist
        formants = sorted([f for f in freqs if 0 < f < sr/2])
        
        # Pad or truncate to exactly 3 formants
        while len(formants) < 3:
            formants.append(sr / 2)  # Fallback to Nyquist
        formants = formants[:3]
        
        f1, f2, f3 = formants[0], formants[1], formants[2]
        
        # Normalize formants to [0, 1] by dividing by Nyquist
        nyquist = sr / 2
        f1_norm = f1 / nyquist
        f2_norm = f2 / nyquist
        f3_norm = f3 / nyquist
        
        # [5] Voiced Ratio — fraction of voiced frames via VAD / energy thresholding
        #   Use librosa.effects.split() to identify non-silent frames
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        energy = np.mean(S, axis=0)  # (n_frames,)
        threshold = np.mean(energy) * 0.4  # 40% of mean energy
        voiced_frames = np.sum(energy > threshold)
        voiced_ratio = voiced_frames / max(len(energy), 1)
        
        # [6] Harmonic-to-Noise Ratio (HNR) — via autocorrelation
        #   Compute autocorrelation and find fundamental period
        autocorr = np.correlate(audio, audio, mode='full')
        center = len(autocorr) // 2
        autocorr = autocorr[center:]  # Keep positive lags only
        autocorr = autocorr / (autocorr[0] + 1e-9)  # Normalize by lag-0
        
        # Find first peak after lag 0 (fundamental period indicator)
        min_lag = int(sr / 500)  # ~2ms (floor for pitch)
        max_lag = int(sr / 50)   # ~20ms (ceiling for pitch)
        if max_lag < len(autocorr) and min_lag < max_lag:
            peak_lag_idx = min_lag + np.argmax(autocorr[min_lag:max_lag])
            peak_autocorr = autocorr[peak_lag_idx]
        else:
            peak_autocorr = 0.5
        
        # HNR = ratio of harmonic energy to noise
        hnr_linear = peak_autocorr / (1.0 - peak_autocorr + 1e-7)
        hnr_db = 10.0 * np.log10(hnr_linear + 1e-7)
        # Normalize HNR to reasonable range [-20, 40] → [0, 1]
        hnr_norm = np.clip((hnr_db + 20) / 60, 0, 1)
        
        # [7] Cepstral Peak Prominence (CPP) — via cepstrum
        #   CPP = cepstrum peak - fitted regression line
        #   High CPP = voicing present; very stable CPP = synthetic
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=256))
        power_spectrum = np.mean(S, axis=1)  # Average over time
        power_spectrum = np.maximum(power_spectrum, 1e-10)
        log_spectrum = np.log(power_spectrum)
        
        # Cepstrum is inverse FFT of log spectrum
        cepstrum = np.fft.irfft(log_spectrum).real
        
        if len(cepstrum) > 20:
            # Find peak in quefrency range (pitch ~ 50-500 Hz)
            pitch_min = int(sr / 500)  # ~2ms
            pitch_max = int(sr / 50)   # ~20ms
            if pitch_max < len(cepstrum):
                cep_peak_idx = pitch_min + np.argmax(cepstrum[pitch_min:pitch_max])
                cep_peak = cepstrum[cep_peak_idx]
                
                # Fit regression line through cepstrum to estimate baseline
                x = np.arange(len(cepstrum))
                z = np.polyfit(x, cepstrum, 1)
                baseline = np.polyval(z, cep_peak_idx)
                
                cpp = cep_peak - baseline
            else:
                cpp = 0.0
        else:
            cpp = 0.0
        
        # Normalize CPP to [0, 1] — typical range is [-5, 20]
        cpp_norm = np.clip((cpp + 5) / 25, 0, 1)
        
        # Assemble 8-dim vector in the specified order
        scalars = np.array([
            flatness,      # [0] Spectral Flatness
            zcr,           # [1] Zero Crossing Rate
            f1_norm,       # [2] F1 Formant (normalized)
            f2_norm,       # [3] F2 Formant (normalized)
            f3_norm,       # [4] F3 Formant (normalized)
            voiced_ratio,  # [5] Voiced Ratio
            hnr_norm,      # [6] HNR (dB, normalized)
            cpp_norm,      # [7] CPP (normalized)
        ], dtype=np.float32)
        
    except Exception as e:
        # Fallback: return zeros if any computation fails
        print(f"[extract_scalar_features] Warning: exception during feature extraction: {e}")
        scalars = np.zeros(8, dtype=np.float32)

    # Guard against NaN/Inf
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Final clip to [0, 1]
    scalars = np.clip(scalars, 0, 1)
    
    return scalars


def build_feature_image(audio, sr=16000):
    """
    3-channel (224, 224, 3) feature image for ResNet50.
    Ch1: Mel spectrogram        — spectral shape
    Ch2: MFCC + delta + delta2  — cepstral dynamics (120 rows stacked)
    Ch3: Spectral contrast + chroma — harmonic structure (19 rows stacked)
    """
    # Ch1: Mel spectrogram
    mel  = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=256)
    ch1  = librosa.power_to_db(mel, ref=np.max)

    # Ch2: MFCC + delta + delta-delta (stacked → 120 rows)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    ch2  = np.vstack([mfcc,
                      librosa.feature.delta(mfcc),
                      librosa.feature.delta(mfcc, order=2)])

    # Ch3: Spectral contrast (7 rows) + Chroma (12 rows) → 19 rows
    # Chroma via filterbank (no STFT/CQT — avoids numba segfault)
    D    = np.abs(librosa.stft(audio, n_fft=2048, hop_length=256))
    sc   = librosa.feature.spectral_contrast(S=D, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=256)
    ch3  = np.vstack([sc, chroma])

    def _norm_resize(data):
        data = data.astype(np.float32)
        mn, mx = data.min(), data.max()
        data = (data - mn) / (mx - mn + 1e-6)
        img = Image.fromarray((data * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        return np.array(img)

    stacked = np.stack([_norm_resize(ch1),
                        _norm_resize(ch2),
                        _norm_resize(ch3)], axis=-1)   # (224, 224, 3) uint8
    return stacked


# ── Audio loader ──────────────────────────────────────────────
def load_audio(file_path, target_sr=16000, duration=4.0, random_crop=False):
    """Load audio, force mono, fixed duration, peak-normalise.
    
    Args:
        file_path   : Path to audio file
        target_sr   : Target sample rate (Hz)
        duration    : Target duration (seconds)
        random_crop : If True and audio is longer than target, randomly crop.
                      If False, always take first 4s.
    """
    target_len = int(target_sr * duration)
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"[load_audio] Failed to load {file_path}: {e}")
        return np.zeros(target_len, dtype=np.float32)

    # Crop to target duration
    if len(audio) > target_len:
        if random_crop:
            # Random crop: pick a random starting point in the valid range
            max_start = len(audio) - target_len
            start = random.randint(0, max_start) if max_start > 0 else 0
            audio = audio[start : start + target_len]
        else:
            # Fixed crop: always take first 4s
            audio = audio[:target_len]
    elif len(audio) < target_len:
        # Pad if too short
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 1e-7:
        audio = audio / peak
    return audio.astype(np.float32)


# ── Shared transform ──────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _to_tensor(audio):
    """Convert audio → (tensor, scalars_tensor)."""
    img     = Image.fromarray(build_feature_image(audio))
    tensor  = _TRANSFORM(img)
    scalars = torch.tensor(extract_scalar_features(audio), dtype=torch.float32)
    return tensor, scalars


# ── ASVspoof 2019 LA ─────────────────────────────────────────
class ASVDataset(Dataset):
    """
    ASVspoof 2019 LA dataset.
    Protocol line format: SPEAKER_ID FILE_ID - SYSTEM_ID LABEL
    Label: 'bonafide' → 0 (real), 'spoof' → 1 (fake)
    Audio: DATA_ROOT/asvspoof2019/LA/ASVspoof2019_LA_train/flac/<FILE_ID>.flac
    """
    def __init__(self,
                 protocol_file=ASV_PROTOCOL,
                 data_dir=ASV_DIR,
                 subset_size=None,
                 augment=False,
                 augment_prob=0.3):
        self.data_dir  = data_dir
        self.augment   = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None
        self.files, self.labels = [], []

        if not os.path.exists(protocol_file):
            raise FileNotFoundError(f"ASV protocol not found: {protocol_file}")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"ASV data dir not found: {data_dir}")

        with open(protocol_file, "r") as f:
            lines = f.readlines()

        random.shuffle(lines)
        if subset_size:
            lines = lines[:subset_size]

        missing = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label   = 0 if parts[4] == "bonafide" else 1
            path    = os.path.join(data_dir, file_id + ".flac")
            if os.path.exists(path):
                self.files.append(path)
                self.labels.append(label)
            else:
                missing += 1

        if missing > 0:
            print(f"[ASVDataset] WARNING: {missing} files missing from disk")

        real = sum(1 for l in self.labels if l == 0)
        fake = len(self.labels) - real
        print(f"[ASVDataset] Loaded {len(self.files)} files | real={real} fake={fake}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio(self.files[idx])
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        tensor, scalars = _to_tensor(audio)
        return tensor, scalars, self.labels[idx]


# ── WaveFake ──────────────────────────────────────────────────
class WaveFakeDataset(Dataset):
    """
    WaveFake dataset.
    Real  : wavefake-test/the-LJSpeech-1.1/wavs/*.wav
             + wavefake-test/jsut_ver1.1/**/*.wav  (if present)
    Fake  : wavefake-test/generated_audio/**/*.wav  (ALL subfolders)
    """
    def __init__(self,
                 data_dir=WAVEFAKE_DIR,
                 subset_size=None,
                 augment=False,
                 augment_prob=0.3):
        self.augment   = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"WaveFake dir not found: {data_dir}")

        # Real files
        real_files = glob.glob(os.path.join(data_dir, "the-LJSpeech-1.1/wavs/*.wav"))
        jsut_files = glob.glob(os.path.join(data_dir, "jsut_ver1.1/basic5000/wav/*.wav"))
        real_files = real_files + jsut_files

        # Fake files — everything under generated_audio/
        fake_files = glob.glob(os.path.join(data_dir, "generated_audio/**/*.wav"), recursive=True)

        if not real_files:
            print(f"[WaveFakeDataset] WARNING: No real files found under {data_dir}")
        if not fake_files:
            print(f"[WaveFakeDataset] WARNING: No fake files found under {data_dir}/generated_audio/")

        if subset_size:
            random.shuffle(real_files)
            random.shuffle(fake_files)
            real_files = real_files[:subset_size // 2]
            fake_files = fake_files[:subset_size // 2]

        self.all_files = real_files + fake_files
        self.labels    = [0] * len(real_files) + [1] * len(fake_files)
        print(f"[WaveFakeDataset] Loaded {len(self.all_files)} files | real={len(real_files)} fake={len(fake_files)}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio = load_audio(self.all_files[idx])
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        tensor, scalars = _to_tensor(audio)
        return tensor, scalars, self.labels[idx]


# ── InTheWild ─────────────────────────────────────────────────
class InTheWildDataset(Dataset):
    """
    In-The-Wild dataset.
    Structure: release_in_the_wild/{train,val,test}/{real,fake}/*.wav
    Label: real/ → 0, fake/ → 1
    """
    def __init__(self,
                 data_dir=ITW_DIR,
                 subset="train",
                 subset_size=None,
                 augment=False,
                 augment_prob=0.3):
        self.augment   = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None

        split_dir = os.path.join(data_dir, subset)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"InTheWild split dir not found: {split_dir}")

        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        real_files = glob.glob(os.path.join(real_dir, "*.wav")) if os.path.isdir(real_dir) else []
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav")) if os.path.isdir(fake_dir) else []

        if not real_files:
            print(f"[InTheWildDataset] WARNING: No real files found at {real_dir}")
        if not fake_files:
            print(f"[InTheWildDataset] WARNING: No fake files found at {fake_dir}")

        if subset_size:
            random.shuffle(real_files)
            random.shuffle(fake_files)
            real_files = real_files[:subset_size // 2]
            fake_files = fake_files[:subset_size // 2]

        self.all_files = real_files + fake_files
        self.labels    = [0] * len(real_files) + [1] * len(fake_files)
        print(f"[InTheWildDataset] ({subset}) Loaded {len(self.all_files)} files | real={len(real_files)} fake={len(fake_files)}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio = load_audio(self.all_files[idx])
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        tensor, scalars = _to_tensor(audio)
        return tensor, scalars, self.labels[idx]


# ── MultiDataset (kept for import compatibility) + build_combined_dataset ─
from torch.utils.data import ConcatDataset

def build_combined_dataset(asv, wavefake, wild, librispeech=None):
    """
    Replaces MultiDataset. Uses ConcatDataset for deterministic,
    non-duplicating indexing. DistributedSampler works correctly with this.
    Every sample is seen exactly once per epoch.
    """
    datasets = [asv, wavefake, wild]
    if librispeech is not None:
        datasets.append(librispeech)
    combined = ConcatDataset(datasets)
    total = sum(len(d) for d in datasets)
    print(f"[CombinedDataset] Total samples: {total} | Sources: {len(datasets)}")
    return combined


# ── LibriSpeech ───────────────────────────────────────────────
class LibriSpeechDataset(Dataset):
    """
    LibriSpeech train-other-500 dataset.
    All audio is real (bonafide).
    Label: 0 (real)
    
    Features:
    - 500+ speakers (high speaker diversity)
    - Varied acoustic conditions (noisy/reverberant)
    - Prevents model learning "clean = real" shortcut
    """
    def __init__(self, data_dir=LIBRISPEECH_DIR, subset_size=None, augment=False, augment_prob=0.5):
        self.augment   = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None
        
        # Find all .flac files in train-other-500
        self.files = glob.glob(f"{data_dir}/**/*.flac", recursive=True)
        
        if not self.files:
            raise FileNotFoundError(f"No FLAC files found at {data_dir}. Expected format: {data_dir}/**/*.flac")
        
        # Optional: subsample for faster training
        if subset_size:
            random.shuffle(self.files)
            self.files = self.files[:subset_size]
        
        # All LibriSpeech samples are real (bonafide)
        self.labels = [0] * len(self.files)
        print(f"[LibriSpeechDataset] Loaded {len(self.files)} real files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio(self.files[idx], random_crop=True)  # Random crop prevents memorization
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        tensor, scalars = _to_tensor(audio)
        return tensor, scalars, self.labels[idx]