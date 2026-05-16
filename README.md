# EchoTrace | State-of-the-Art Deepfake Audio Forensics

**EchoTrace** is a forensic-grade audio analysis suite designed to detect synthetic speech and AI-generated voice clones with high precision. Built by the **BackProp Bandits**, this tool leverages a Dual-Stream Fusion Model combining a fine-tuned ResNet-50 vision backbone and a Biometric Multilayer Perceptron. Our approach identifies subtle artifacts in the spectral domain and physical vocal tract anomalies that are invisible to the human ear.

---

## Key Features

*   **Dual-Stream Fusion Analysis**: Combines spectral texture (Vision Stream) with biometric physics (8-Dimensional Scalar Vector Stream).
*   **Explainable AI Layer**: Uses **SHAP (DeepExplainer)** to generate mathematical feature attributions, paired with **LLM Forensic Reasoning** (via LLaMA 3.1 8B) for plain-English analysis reports.
*   **5-Mode Developer Audit Suite**: High-performance server evaluation tool for benchmarking across ASVspoof, In-The-Wild, LibriSpeech, and WaveFake datasets.
*   **Sliding-Window Ensemble**: Moves a 2-second window with 500ms overlap to identify localized spoofing artifacts in long recordings.
*   **Forensic Report Generation**: High-resolution spectral analysis reports including time-domain waveforms, 3-channel forensic images, and Grad-CAM heatmaps.
*   **Hardware-Optimized Training**: Built on a 4-GPU Distributed Data Parallel (DDP) pipeline with Automatic Mixed Precision (AMP) and SyncBatchNorm.

---

## Performance & Evaluation

EchoTrace is trained and validated on a massive, balanced corpus of **220,380 samples**:
*   **ASVspoof 2019**: 25,380 samples (Laboratory attacks)
*   **WaveFake**: 70,000 samples (Neural vocoders: MelGAN, Parallel WaveGAN)
*   **In-The-Wild**: 25,000 samples (Real-world deepfakes from YouTube/Social Media)
*   **LibriSpeech**: 100,000 samples (Clean "Real" anchor dataset)

### Benchmarks
| Dataset | EER | ROC-AUC | Balanced Accuracy |
| :--- | :--- | :--- | :--- |
| **ASVspoof 2019 Dev** | 1.22% | 0.9994 | 98.29% |
| **In-The-Wild Test** | **0.86%** | **0.9992** | **99.24%** |

*Note: EchoTrace prioritizes "Physical Consistency" over lab-specific watermarks, making it significantly more robust to real-world, compressed, and noisy audio.*

---

## Technical Architecture

EchoTrace treats deepfake detection as a hybrid Computer Vision and biological Resonant Physics problem.

### 1. The 3-Channel Forensic Image (Vision Stream)
We pack three distinct forensic representations into a single RGB image (224x224x3):
*   **Channel 1 (Mel Spectrogram)**: Captures general timbre and spectral energy.
*   **Channel 2 (MFCC + Deltas)**: Captures cepstral dynamics where AI often leaves micro-discontinuities.
*   **Channel 3 (Spectral Contrast + Chroma CQT)**: Detects harmonic phase shifts unnatural to human vocal folds.

### 2. The 8 Biometric Scalars (Physics Stream)
We extract physical features that capture the unique biological fingerprint of a human vocal tract:
*   **Formants (F1, F2, F3)**: Captures resonance of the throat and oral cavity.
*   **CPP & HNR**: Measures voice quality and biological purity.
*   **Spectral Flatness**: Identifies unnatural "mathematically perfect" frequency gaps common in AI.

### 3. Training Infrastructure
*   **Distributed Strategy**: 4-GPU DDP cluster using the NCCL backend.
*   **Freeze Strategy**: Layers 1-3 of ResNet-50 are frozen; only Layer 4 and the fusion head are fine-tuned.
*   **Loss Function**: **Focal Loss** (`gamma=2.0`, `alpha=0.5`) to focus the model on "hard" spoof samples.

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pranavrajgali/EchoTrace.git
   cd EchoTrace
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS or Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Weights**
   Ensure `ensemble_model.pth` is present in the root directory.

---

## Usage

### Interactive Web Dashboard (Streamlit)
The primary interface for uploading audio, live recording, and viewing SHAP explanations.
```bash
streamlit run streamlit_app.py
```

### High-Performance Audit (Server)
Run the 5-mode stress test to benchmark the model against all forensic datasets (ASVspoof, In-The-Wild, WaveFake, LibriSpeech):
```bash
python scripts/evaluate_server.py --checkpoint ensemble_model.pth --mode 5
```

### Automated Forensic Reports
Generate a detailed PDF/Image report for a single audio file with Grad-CAM visualization:
```bash
python tests/single_example_report_generator.py
```

---

## Project Structure
```text
.
├── streamlit_app.py         # Main Web Application UI
├── ensemble_model.pth       # Pre-trained Model Weights
├── core/                    # Engine and Intelligence
│   ├── model.py             # ResNet-50 Architecture and Biometric MLP
│   ├── preprocess.py        # DDP Datasets and Audio Pipelines
│   └── inference.py         # Signal Processing and Prediction
├── scripts/                 # Execution Pipelines
│   ├── train_ddp.py         # Multi-GPU Training Script
│   ├── evaluate_server.py   # 5-Mode Developer Audit Suite
│   └── evaluate_pc.py       # Local Performance Evaluation
├── docs/                    # Technical Documentation
│   ├── EchoTrace_Technical_Report.md  # Architectural Breakdown
│   ├── TRAINING_GUIDE.md    # Guide to running the DDP Pipeline
│   └── FUTURE_CHANGES.md    # Development Roadmap
├── tests/
│   └── single_example_report_generator.py  # Standalone Forensic Script
├── utils/                   # Shared Utilities
└── requirements.txt         # Project Dependencies
```

---

**Built by BackProp Bandits**  
*EchoTrace is a forensic tool intended for investigative and educational purposes. While highly accurate, no detection system is infallible. Always use EchoTrace results as one part of a broader verification workflow.*
