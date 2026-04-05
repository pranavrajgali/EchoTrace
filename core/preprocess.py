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
    8-dim forensic scalar vector. Stable features only (no pyin for DDP speed).
    Returns float32 numpy array, clipped to [0, 1].
    """
    try:
        zcr      = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff  = librosa.feature.spectral_rolloff(y=audio, sr=sr)

        scalars = np.array([
            zcr,
            flatness,
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.std(rolloff),
            0.0,  # placeholder — jitter reserved for inference
            0.0,  # placeholder — shimmer reserved for inference
        ], dtype=np.float32)
    except Exception:
        scalars = np.zeros(8, dtype=np.float32)

    scalars = np.nan_to_num(scalars, nan=0.0, posinf=1.0, neginf=0.0)
    max_val = np.max(np.abs(scalars))
    return np.clip(scalars / (max_val + 1e-9), 0, 1)


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


# ── MultiDataset ──────────────────────────────────────────────
class MultiDataset(Dataset):
    """
    Round-robin across ASVspoof, WaveFake, InTheWild, and optionally LibriSpeech.
    Every __getitem__ returns (tensor, scalars, label).
    """
    def __init__(self, asv, wavefake, wild, librispeech=None):
        datasets = [asv, wavefake, wild]
        if librispeech is not None:
            datasets.append(librispeech)
        
        if any(d is None for d in [asv, wavefake, wild]):
            raise ValueError("ASV, WaveFake, and InTheWild datasets are mandatory.")
        
        self.datasets  = datasets
        self.total_len = sum(len(d) for d in self.datasets)
        num_sources = len(self.datasets)
        print(f"[MultiDataset] Total samples: {self.total_len} | Sources: {num_sources}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds_idx      = idx % len(self.datasets)
        local_idx   = np.random.randint(0, len(self.datasets[ds_idx]))
        return self.datasets[ds_idx][local_idx]


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