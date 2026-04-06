# EchoTrace Dataset Forensic Report
**Version:** 1.0  
**Status:** Comprehensive Analysis  

This report details the composition, organizational structure, and statistical distribution of the datasets used to train and evaluate the EchoTrace deepfake detection system.

---

## 1. ASVspoof 2019 (Logical Access)
The foundational dataset for synthetic speech detection, focusing on state-of-the-art Text-to-Speech (TTS) and Voice Conversion (VC).

### Directory Structure
```text
/home/jovyan/work/data/LA/LA/
├── ASVspoof2019_LA_train/          # Main training partition
│   └── flac/                       # 25,380 files
├── ASVspoof2019_LA_dev/            # Development/Validation
│   └── flac/                       # 24,844 files
├── ASVspoof2019_LA_eval/           # Blind evaluation partition
│   └── flac/                       # 71,237 files
└── ASVspoof2019_LA_cm_protocols/   # Metadata & Ground Truth
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### Sample Breakdown
| Partition | Real (Bonafide) | Fake (Spoof) | Total |
| :--- | :--- | :--- | :--- |
| **Train** | 2,580 | 22,800 | 25,380 |
| **Dev** | 2,548 | 22,296 | 24,844 |
| **Eval** | 7,355 | 63,882 | 71,237 |

### Forensic Metadata
*   **Spoofing Attacks:** Includes 19 different systems (A01-A19).
*   **Sample Rate:** 16 kHz, 16-bit FLAC.
*   **Labels:** Found in protocol files (5th column: `bonafide` or `spoof`).

---

## 2. WaveFake
A large-scale dataset featuring modern neural vocoders (MelGAN, Parallel WaveGAN, Multi-band MelGAN, etc.) to evaluate generalization across different generation architectures.

### Directory Structure
```text
/home/jovyan/work/data/wavefake/wavefake-test/
├── the-LJSpeech-1.1/               # Core Real Source (English)
│   └── wavs/*.wav                  # 13,100 files
├── jsut_ver1.1/                    # Real Source (Japanese)
│   └── basic5000/wav/*.wav         # 5,000 files
└── generated_audio/                # ALL Fake Samples
    ├── ljspeech_melgan/
    ├── ljspeech_parallel_wavegan/
    ├── ljspeech_multi_band_melgan/
    ├── ljspeech_full_band_melgan/
    ├── ljspeech_hifi_gan/
    └── ljspeech_waveglow/
```

### Sample Breakdown
| Category | Source | Count |
| :--- | :--- | :--- |
| **Real** | LJSpeech + JSUT | 18,100 |
| **Fake** | Generated (All Vocoders) | 134,266 |
| **Total** | | **152,366** |

### Forensic Metadata
*   **Diversity:** Covers multi-lingual real audio (English/Japanese).
*   **Tech Stack:** Represents "In-the-wild" synthesis where standard protocols like ASVspoof might not apply.

---

## 3. In-The-Wild
A dataset consisting of real-world "deepfake" samples collected from social media and news reports, specifically designed to test the model against actual manipulated content found online.

### Directory Structure
```text
/home/jovyan/work/data/release_in_the_wild/release_in_the_wild/
├── train/
│   ├── real/                       # 13,974 files
│   └── fake/                       # 8,271 files
├── val/
│   ├── real/                       # 3,992 files
│   └── fake/                       # 2,363 files
└── test/
    ├── real/                       # 1,997 files
    └── fake/                       # 1,182 files
```

### Sample Breakdown
| Split | Real Samples | Fake Samples | Total |
| :--- | :--- | :--- | :--- |
| **Train** | 13,974 | 8,271 | 22,245 |
| **Val** | 3,992 | 2,363 | 6,355 |
| **Test** | 1,997 | 1,182 | 3,179 |

### Forensic Metadata
*   **Source:** Reality-based audio clips.
*   **Challenge:** Highly variable noise levels, codecs (MP3/AAC/OGG conversions), and recording environments.

---

## 4. LibriSpeech (Supplementary Real Data)
Used primarily to enhance speaker diversity and prevent the model from identifying "high-quality audio" as a proxy for "fake audio."

### Directory Structure
```text
/home/jovyan/work/data/LibriSpeech/
└── train-other-500/                # Partition used for robustness
    ├── [Speaker_ID]/               # ~500 Speaker Folders
    │   └── [Chapter_ID]/
    │       └── *.flac              # Processed FLAC segments
```

### Sample Breakdown
| Source | Type | Total Count |
| :--- | :--- | :--- |
| **LibriSpeech** | **Real (Bonafide)** | **148,688** |

### Forensic Metadata
*   **Diversity:** 500+ speakers.
*   **Acoustics:** Varied environments (some noisy/reverberant), forcing the model to learn deep voice features rather than just silence/noise profile differences.

---

## Summary Distribution
| Status | Total Samples |
| :--- | :--- |
| **Total Real (Bonafide)** | **185,273** |
| **Total Fake (Spoof)** | **233,184** |
| **Grand Total** | **418,457** |

> [!NOTE]
> The EchoTrace pipeline uses a balanced sampling strategy during training to manage the inherent imbalance in some sub-datasets (notably WaveFake's and ASVspoof's high fake-to-real ratios).
