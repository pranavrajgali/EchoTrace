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


class AudioAugmenter:
    """
    Handles robust audio augmentation using the MUSAN noise corpus.
    Using real-world noise (speech/music/noise) instead of synthetic white noise 
    prevents the model from overfitting to clean digital signals.
    """
    def __init__(self, p=0.3, musan_path='../data/noise/musan/noise/'):
        self.p = p
        # Load real noise clips from the MUSAN directory
        self.musan_files = glob.glob(os.path.join(musan_path, "**/*.wav"), recursive=True)

    def add_noise(self, audio):
        """Injects real background noise at random volumes."""
        if random.random() > self.p:
            return audio
        
        if self.musan_files:
            noise_path = random.choice(self.musan_files)
            noise, _ = librosa.load(noise_path, sr=16000)
            
            # Match noise length to the input audio
            if len(noise) < len(audio):
                noise = np.pad(noise, (0, len(audio) - len(noise)))
            else:
                noise = noise[:len(audio)]
            
            # Apply at a random intensity to simulate different environments
            snr_factor = random.uniform(0.01, 0.1)
            return audio + noise * snr_factor
        
        # Fallback to standard white noise if MUSAN is not found
        return audio + np.random.normal(0, 0.005, len(audio))

    def apply_augmentations(self, audio):
        """Entry point for training-time augmentations."""
        return self.add_noise(audio)

def extract_scalar_features(audio, sr=16000):
    """
    Stabilized forensic scalar extraction.
    Removed complex pyin/jitter for stability in this DDP environment.
    """
    try:
        # Fast, stable features only
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        scalars = np.array([
            zcr, flatness,
            np.mean(centroid), np.std(centroid),
            np.mean(rolloff), np.std(rolloff),
            0.0, 0.0 # Placeholders for jitter/shimmer
        ], dtype=np.float32)
    except:
        scalars = np.zeros(8, dtype=np.float32)
    
    # Replace any NaN/Inf from edge-case audio (silent, single-pitch, etc.)
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=1.0, neginf=0.0)
    # Normalize to [0,1] range for network stability
    return np.clip(scalars / (np.max(np.abs(scalars)) + 1e-9), 0, 1)

def build_feature_image(audio, sr=16000):
    """
    Stacks three spectral views into a (224, 224, 3) feature image for ResNet50.
    Ch1: Mel-Spectrogram (Spectral shape).
    Ch2: MFCCs + Delta + Delta-Delta (120 rows for speech dynamics).
    Ch3: Spectral Contrast + Chroma (Harmonic/Artifact structure).
    """
    # Channel 1: Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=224, n_fft=2048, hop_length=256)
    ch1 = librosa.power_to_db(mel, ref=np.max)
    
    # Channel 2: MFCCs with dynamics (Stacked to 120 rows)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    ch2 = np.vstack([mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)])
    
    # Channel 3: Spectral Contrast (7) + Chroma (12) (Stacked to 19 rows)
    ch3 = np.vstack([librosa.feature.spectral_contrast(y=audio, sr=sr), 
                     librosa.feature.chroma_stft(y=audio, sr=sr)])
    
    def normalize_and_resize(data):
        # Normalize local data to uint8 scale
        data = (data - data.min()) / (data.max() - data.min() + 1e-6)
        img = Image.fromarray((data * 255).astype(np.uint8)).resize((224, 224))
        return np.array(img)

    # Stack into RGB-style image
    return np.stack([normalize_and_resize(ch1), 
                     normalize_and_resize(ch2), 
                     normalize_and_resize(ch3)], axis=-1)

def load_audio_multiformat(file_path, target_sr=16000, duration=4.0):
    """Loads audio and enforces a fixed duration for batching."""
    audio, _ = librosa.load(file_path, sr=target_sr, duration=duration)
    target_len = int(target_sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]

class ASVDataset(Dataset):
    """Dataset class for ASVspoof 2019 LA files."""
    def __init__(self, protocol_file, data_dir, subset_size=None, augment=False, augment_prob=0.3):
        self.data_dir = data_dir
        self.files, self.labels = [], []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None

        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            if subset_size:
                lines = lines[:subset_size]
            for line in lines:
                parts = line.strip().split()
                self.files.append(parts[1])
                self.labels.append(0 if parts[4] == 'bonafide' else 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx] + ".flac")
        audio = load_audio_multiformat(path)
        # Mandatory Peak Normalization
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        
        img = Image.fromarray(build_feature_image(audio))
        scalars = extract_scalar_features(audio)
        return self.transform(img), torch.tensor(scalars, dtype=torch.float32), self.labels[idx]

class WaveFakeDataset(Dataset):
    """
    Targeted dataset for modern neural vocoders (HiFi-GAN, MelGAN, etc.).
    All folders in 'generated_audio' are treated as FAKE.
    """
    def __init__(self, data_dir, subset_size=None, augment=False, augment_prob=0.3):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None
        
        # Real: LJSpeech-1.1 wavs + JSUT original Japanese speech
        self.real_files = glob.glob(os.path.join(data_dir, "the-LJSpeech-1.1/wavs/*.wav"))
        self.real_files += glob.glob(os.path.join(data_dir, "jsut_ver1.1/**/*.wav"), recursive=True)
        
        # Fake: All recursive wavs in generated_audio
        self.fake_files = glob.glob(os.path.join(data_dir, "generated_audio/**/*.wav"), recursive=True)
        
        if subset_size:
            random.shuffle(self.real_files)
            random.shuffle(self.fake_files)
            self.real_files = self.real_files[:subset_size // 2]
            self.fake_files = self.fake_files[:subset_size // 2]
        
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]
        
        audio = load_audio_multiformat(file_path)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        
        img = Image.fromarray(build_feature_image(audio))
        scalars = extract_scalar_features(audio)
        return self.transform(img), torch.tensor(scalars, dtype=torch.float32), label


class InTheWildDataset(Dataset):
    """
    Handles the 'In-The-Wild' dataset.
    Globs all .wav files, reads meta.csv (columns: file, label) to split into real/fake.
    Falls back to filename heuristics if no CSV exists.
    """
    def __init__(self, data_dir, subset_size=None, augment=False, augment_prob=0.3):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment
        self.augmenter = AudioAugmenter(p=augment_prob) if augment else None
        
        # Glob all .wav files in the directory
        all_wav_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
        
        # Try to read meta.csv for labels
        csv_path = os.path.join(data_dir, "meta.csv")
        self.real_files = []
        self.fake_files = []
        
        if os.path.exists(csv_path):
            # Read CSV: columns are file, label (where label is "real" or "fake")
            file_to_label = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_to_label[row['file']] = row['label']
            
            for wav_file in all_wav_files:
                basename = os.path.basename(wav_file)
                if basename in file_to_label:
                    if file_to_label[basename] == 'real':
                        self.real_files.append(wav_file)
                    else:
                        self.fake_files.append(wav_file)
        else:
            # Fallback: files containing "real" in filename are real, others are fake
            for wav_file in all_wav_files:
                if 'real' in wav_file.lower():
                    self.real_files.append(wav_file)
                else:
                    self.fake_files.append(wav_file)
        
        # Apply subset_size if set
        if subset_size:
            random.shuffle(self.real_files)
            random.shuffle(self.fake_files)
            self.real_files = self.real_files[:subset_size // 2]
            self.fake_files = self.fake_files[:subset_size // 2]
        
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]
        
        audio = load_audio_multiformat(file_path)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        if self.augment:
            audio = self.augmenter.apply_augmentations(audio)
        
        img = Image.fromarray(build_feature_image(audio))
        scalars = extract_scalar_features(audio)
        return self.transform(img), torch.tensor(scalars, dtype=torch.float32), label



class MultiDataset(Dataset):
    """
    Combines multiple datasets by round-robin index mapping.
    """
    def __init__(self, asv, wavefake, wild):
        self.datasets = [asv, wavefake, wild]
        self.lengths = [len(d) for d in self.datasets]
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Round-robin: idx % 3 picks source (0=asv, 1=wavefake, 2=wild)
        ds_idx = idx % 3
        selected_ds = self.datasets[ds_idx]
        local_idx = np.random.randint(0, len(selected_ds))
        return selected_ds[local_idx]