"""
core/preprocess_gpu.py — GPU-accelerated feature extraction using torchaudio
Replaces librosa entirely. All mel/mfcc computed on CUDA.
"""
import os, glob, random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# ── Paths ─────────────────────────────────────────────────────
DATA_ROOT    = "/home/jovyan/work/data"
ASV_PROTOCOL = f"{DATA_ROOT}/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
ASV_DIR      = f"{DATA_ROOT}/asvspoof2019/LA/ASVspoof2019_LA_train/flac"
WAVEFAKE_DIR = f"{DATA_ROOT}/wavefake/wavefake-test"
ITW_DIR      = f"{DATA_ROOT}/in_the_wild/release_in_the_wild"

SR           = 16000
DURATION     = 4.0
TARGET_LEN   = int(SR * DURATION)

# ImageNet normalisation constants
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── GPU Transform Pipeline ────────────────────────────────────
class GPUFeatureExtractor(torch.nn.Module):
    """
    All transforms live on GPU.
    Input  : (B, T) float32 waveform, already loaded & padded
    Output : (B, 3, 224, 224) normalised tensor
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

        # Ch1: Mel spectrogram (128 mels)
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=256,
            n_mels=128, power=2.0
        ).to(device)
        self.db = T.AmplitudeToDB(top_db=80).to(device)

        # Ch2: MFCC (40 coeffs)
        self.mfcc = T.MFCC(
            sample_rate=SR,
            n_mfcc=40,
            melkwargs={"n_fft": 2048, "hop_length": 256, "n_mels": 128}
        ).to(device)

        # Ch3: Spectrogram for contrast proxy (magnitude)
        self.spec = T.Spectrogram(n_fft=2048, hop_length=256, power=1.0).to(device)

        self.resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.mean   = MEAN.to(device)
        self.std    = STD.to(device)

    def _norm(self, x):
        """Normalise a (B, H, W) feature map to [0, 1] then to 3-channel."""
        mn = x.amin(dim=(-2, -1), keepdim=True)
        mx = x.amax(dim=(-2, -1), keepdim=True)
        return (x - mn) / (mx - mn + 1e-6)

    def forward(self, waveform):
        # waveform: (B, T)

        # Ch1: mel → dB → norm → resize
        mel  = self.db(self.mel(waveform))          # (B, 128, T)
        ch1  = self._norm(mel)
        ch1  = self.resize(ch1.unsqueeze(1)).squeeze(1)   # (B, 224, 224)

        # Ch2: MFCC → norm → resize
        mfcc = self.mfcc(waveform)                  # (B, 40, T)
        ch2  = self._norm(mfcc)
        ch2  = self.resize(ch2.unsqueeze(1)).squeeze(1)

        # Ch3: magnitude spectrogram → norm → resize (proxy for contrast/chroma)
        spec = self.spec(waveform)                  # (B, F, T)
        ch3  = self._norm(spec)
        ch3  = self.resize(ch3.unsqueeze(1)).squeeze(1)

        # Stack → (B, 3, 224, 224)
        out = torch.stack([ch1, ch2, ch3], dim=1)

        # ImageNet normalise
        out = (out - self.mean) / self.std
        return out


def _extract_scalars_gpu(waveform):
    """
    8 scalar features computed entirely in torch (no librosa).
    waveform: (B, T) on GPU
    Returns : (B, 8) float32
    """
    # ZCR
    signs   = torch.sign(waveform)
    zcr     = (signs[:, 1:] != signs[:, :-1]).float().mean(dim=1, keepdim=True)

    # RMS energy
    rms     = waveform.pow(2).mean(dim=1, keepdim=True).sqrt()

    # Spectral centroid proxy via FFT
    fft     = torch.fft.rfft(waveform, dim=1).abs()          # (B, T//2+1)
    freqs   = torch.linspace(0, SR / 2, fft.shape[1], device=waveform.device)
    cent    = (fft * freqs).sum(dim=1) / (fft.sum(dim=1) + 1e-8)
    cent    = cent.unsqueeze(1) / (SR / 2)                   # normalise to [0,1]

    # Spectral spread
    spread  = ((fft * (freqs - cent * SR / 2).pow(2)).sum(dim=1) /
               (fft.sum(dim=1) + 1e-8)).sqrt()
    spread  = spread.unsqueeze(1) / (SR / 2)

    # Spectral rolloff (85th percentile energy)
    cumsum  = fft.cumsum(dim=1)
    total   = cumsum[:, -1:] + 1e-8
    rolloff_idx = (cumsum < 0.85 * total).sum(dim=1).float() / fft.shape[1]
    rolloff = rolloff_idx.unsqueeze(1)

    # Spectral flatness (geometric/arithmetic mean ratio)
    log_fft  = fft.clamp(min=1e-8).log().mean(dim=1, keepdim=True)
    lin_fft  = fft.mean(dim=1, keepdim=True).clamp(min=1e-8).log()
    flatness = (log_fft - lin_fft).exp().clamp(0, 1)

    # Peak amplitude
    peak    = waveform.abs().amax(dim=1, keepdim=True)

    # Crest factor
    crest   = (peak / (rms + 1e-8)).clamp(0, 10) / 10

    scalars = torch.cat([zcr, rms, cent, spread, rolloff, flatness, peak, crest], dim=1)
    return scalars.clamp(0, 1)


# ── Audio loader (CPU, fast) ──────────────────────────────────
def load_audio_fast(path):
    """Load with torchaudio (faster than librosa), return (T,) float32."""
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.squeeze(0)                       # (T,)
        if len(wav) < TARGET_LEN:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LEN - len(wav)))
        else:
            wav = wav[:TARGET_LEN]
        peak = wav.abs().max()
        if peak > 1e-7:
            wav = wav / peak
        return wav
    except Exception as e:
        return torch.zeros(TARGET_LEN)


# ── Datasets ──────────────────────────────────────────────────
class FastASVDataset(Dataset):
    def __init__(self, protocol_file=ASV_PROTOCOL, data_dir=ASV_DIR, subset_size=None):
        self.files, self.labels = [], []
        with open(protocol_file) as f:
            lines = f.readlines()
        random.shuffle(lines)
        if subset_size:
            lines = lines[:subset_size]
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            label = 0 if parts[4] == "bonafide" else 1
            path  = os.path.join(data_dir, parts[1] + ".flac")
            if os.path.exists(path):
                self.files.append(path)
                self.labels.append(label)
        real = sum(1 for l in self.labels if l == 0)
        print(f"[FastASV] {len(self.files)} files | real={real} fake={len(self.files)-real}")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return load_audio_fast(self.files[idx]), self.labels[idx]


class FastWaveFakeDataset(Dataset):
    def __init__(self, data_dir=WAVEFAKE_DIR, subset_size=None):
        real = (glob.glob(f"{data_dir}/the-LJSpeech-1.1/wavs/*.wav") +
                glob.glob(f"{data_dir}/jsut_ver1.1/basic5000/wav/*.wav"))
        fake = glob.glob(f"{data_dir}/generated_audio/**/*.wav", recursive=True)
        if subset_size:
            random.shuffle(real); random.shuffle(fake)
            real = real[:subset_size//2]; fake = fake[:subset_size//2]
        self.files  = real + fake
        self.labels = [0]*len(real) + [1]*len(fake)
        print(f"[FastWaveFake] {len(self.files)} files | real={len(real)} fake={len(fake)}")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return load_audio_fast(self.files[idx]), self.labels[idx]


class FastITWDataset(Dataset):
    def __init__(self, data_dir=ITW_DIR, subset="train", subset_size=None):
        split = os.path.join(data_dir, subset)
        real  = glob.glob(f"{split}/real/*.wav")
        fake  = glob.glob(f"{split}/fake/*.wav")
        if subset_size:
            random.shuffle(real); random.shuffle(fake)
            real = real[:subset_size//2]; fake = fake[:subset_size//2]
        self.files  = real + fake
        self.labels = [0]*len(real) + [1]*len(fake)
        print(f"[FastITW] {len(self.files)} files | real={len(real)} fake={len(fake)}")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        return load_audio_fast(self.files[idx]), self.labels[idx]


class FastMultiDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.total    = sum(len(d) for d in datasets)
        # flat index
        self.index = []
        for di, ds in enumerate(datasets):
            for si in range(len(ds)):
                self.index.append((di, si))
        random.shuffle(self.index)
        print(f"[FastMulti] Total: {self.total}")

    def __len__(self): return self.total
    def __getitem__(self, idx):
        di, si = self.index[idx]
        return self.datasets[di][si]
