# preprocess.py - Optimized for Real-World Robustness and ResNet50 Fine-Tuning
import os
import torch
import librosa
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import scipy.signal
from scipy.io import wavfile
import soundfile as sf
from typing import List, Tuple, Optional
import glob

class AudioAugmenter:
    """Audio augmentation class to simulate domain shift and real-world conditions"""

    def __init__(self, p=0.3): # Lowered default probability to 0.3 to prevent signal destruction
        self.p = p  

    def add_noise(self, audio, noise_type='white', snr_range=(10, 30)):
        """Add background noise at random SNR - adjusted for realistic noise levels"""
        if random.random() > self.p:
            return audio
        if noise_type == 'white':
            noise = np.random.normal(0, 1, len(audio))
        elif noise_type == 'pink':
            white = np.random.normal(0, 1, len(audio))
            noise = np.convolve(white, [1, 1], mode='same')
            noise = noise / np.max(np.abs(noise))
        else:
            return audio
        snr_db = random.uniform(*snr_range)
        audio_rms = np.sqrt(np.mean(audio**2))
        noise_rms = audio_rms / (10**(snr_db/20))
        noise = noise * noise_rms / (np.sqrt(np.mean(noise**2)) + 1e-9)
        return audio + noise

    def change_speed(self, audio, speed_range=(0.95, 1.05)):
        if random.random() > self.p: return audio
        speed = random.uniform(*speed_range)
        return librosa.effects.time_stretch(audio, rate=speed)

    def change_pitch(self, audio, pitch_range=(-1, 1)):
        if random.random() > self.p: return audio
        pitch_steps = random.uniform(*pitch_range)
        return librosa.effects.pitch_shift(audio, sr=16000, n_steps=pitch_steps)

    def add_reverb(self, audio, reverb_amount=0.2):
        if random.random() > self.p: return audio
        delay_samples = int(16000 * 0.03)
        decay = 0.3
        reverb_audio = audio.copy()
        for i in range(2):
            delay = delay_samples * (i + 1)
            if delay < len(audio):
                reverb_audio[delay:] += audio[:-delay] * (decay ** (i + 1))
        return audio * (1 - reverb_amount) + reverb_audio * reverb_amount

    def simulate_mp3_compression(self, audio):
        """Simulates social media compression artifacts"""
        if random.random() > self.p: return audio
        bitrate = random.choice([32, 64, 128]) # Focus on low bitrates
        cutoff_freq = min(7500, bitrate * 80) 
        sos = scipy.signal.butter(4, cutoff_freq, 'lowpass', fs=16000, output='sos')
        return scipy.signal.sosfilt(sos, audio)

    def apply_augmentations(self, audio):
        """Apply random combination of augmentations - excludes Mandatory Normalization"""
        augmentations = [
            lambda x: self.add_noise(x),
            lambda x: self.change_speed(x),
            lambda x: self.change_pitch(x),
            lambda x: self.add_reverb(x),
            lambda x: self.simulate_mp3_compression(x)
        ]
        # Apply 1-2 random augmentations to maintain recognizable signal
        num_augs = random.randint(1, 2)
        selected_augs = random.sample(augmentations, num_augs)
        augmented_audio = audio.copy()
        for aug in selected_augs:
            augmented_audio = aug(augmented_audio)
        
        # Consistent length maintenance
        if len(augmented_audio) > 64000:
            augmented_audio = augmented_audio[:64000]
        elif len(augmented_audio) < 64000:
            augmented_audio = np.pad(augmented_audio, (0, 64000 - len(augmented_audio)))
        return augmented_audio

def load_audio_multiformat(file_path: str, target_sr: int = 16000, duration: float = 4.0) -> np.ndarray:
    """Load audio with high-quality resampling using soxr_hq."""
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, duration=duration, res_type='soxr_hq')
    except Exception:
        try:
            audio, sr = sf.read(file_path, dtype='float32')
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")
    
    # Standardize to fixed length
    max_samples = int(target_sr * duration)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]
    return audio

class ASVDataset(Dataset):
    def __init__(self, protocol_file, data_dir, subset_size=1000, augment=False, augment_prob=0.3):
        self.data_dir = os.path.abspath(data_dir)
        self.files, self.labels = [], []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None

        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.files.append(parts[1])
                    self.labels.append(0 if parts[4] == 'bonafide' else 1)

        idx_0 = [i for i, l in enumerate(self.labels) if l == 0]
        idx_1 = [i for i, l in enumerate(self.labels) if l == 1]
        n_samples = min(subset_size, len(idx_0), len(idx_1))
        selected = random.sample(idx_0, n_samples) + random.sample(idx_1, n_samples)
        self.files = [self.files[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        base_path = os.path.join(self.data_dir, self.files[idx])
        audio_path = next((base_path + ext for ext in ['.flac', '.wav', '.mp3'] if os.path.exists(base_path + ext)), None)
        if not audio_path: raise FileNotFoundError(f"Missing: {self.files[idx]}")

        # 1. Load 
        audio = load_audio_multiformat(audio_path)

        # 2. MANDATORY PEAK NORMALIZATION (Critical for multi-dataset training)
        peak = np.max(np.abs(audio))
        if peak > 1e-7: audio = audio / peak

        # 3. Augment (Training only)
        if self.augment and self.augmenter:
            audio = self.augmenter.apply_augmentations(audio)

        # 4. Mel-Spectrogram (Forced n_mels=224 for ResNet)
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=224, n_fft=1024)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # 5. Image Conversion
        img = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img).convert("RGB")
        return self.transform(img_pil), self.labels[idx]

class InTheWildDataset(Dataset):
    def __init__(self, data_dir, subset='train', subset_size=1000, augment=False, augment_prob=0.3):
        self.data_dir = os.path.abspath(data_dir)
        self.files, self.labels = [], []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None

        fake_dir = os.path.join(self.data_dir, subset, 'fake')
        real_dir = os.path.join(self.data_dir, subset, 'real')
        
        fakes = glob.glob(os.path.join(fake_dir, "*.*"))
        reals = glob.glob(os.path.join(real_dir, "*.*"))
        
        n = min(subset_size, len(fakes), len(reals))
        self.files = random.sample(fakes, n) + random.sample(reals, n)
        self.labels = [1]*n + [0]*n

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio_multiformat(self.files[idx])
        
        # Mandatory Normalization
        peak = np.max(np.abs(audio))
        if peak > 1e-7: audio = audio / peak

        if self.augment and self.augmenter:
            audio = self.augmenter.apply_augmentations(audio)

        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=224, n_fft=1024)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        img = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img).convert("RGB")
        return self.transform(img_pil), self.labels[idx]

class MultiDataset(Dataset):
    def __init__(self, asv_dataset, wild_dataset):
        self.asv = asv_dataset
        self.wild = wild_dataset
        self.length = min(len(asv_dataset), len(wild_dataset)) * 2

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.asv[random.randint(0, len(self.asv)-1)]
        return self.wild[random.randint(0, len(self.wild)-1)]