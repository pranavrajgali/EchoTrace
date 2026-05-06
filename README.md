# EchoTrace | AI-Powered Deepfake Audio Detection

**EchoTrace** is a forensic-grade audio analysis suite designed to detect synthetic speech and AI-generated voice clones. Built by the **BackProp Bandits**, this tool leverages a Dual-Stream Fusion Model combining a fine-tuned ResNet-50 architecture and a Biometric Multilayer Perceptron. This approach identifies subtle artifacts in the spectral domain and physical biometric anomalies that are invisible to the human ear.

***

## Key Features

* **Dual-Stream Analysis**: Combines spectral texture (Vision Stream) with biometric physics (8-Dimensional Scalar Vector Stream).
* **Multi-Source Analysis**: Upload .wav, .mp3, or .flac files or record live audio directly in the browser.
* **Sliding-Window Ensemble Inference**: Moves a 2-second window with 500ms overlap to calculate a probability for every window, ensuring localized spoofed segments are caught instead of relying on a single snapshot.
* **Peak-Window Targeted Analysis**: Scans the entire file to identify the peak suspicion timestamp, then extracts high-resolution 4-second forensic features centered on that spike.
* **Effective Bandwidth Estimator**: Uses spectral rolloff to prevent false positives from upsampled low-quality audio (e.g., telephony 8kHz audio).
* **Pure-Python Voice Activity Detection**: Custom Voice Activity Detection using root mean square energy distributions and zero-crossing rate bounds for zero-dependency deployment.
* **Explainable AI Layer**: Uses SHAP (DeepExplainer) to generate feature attributions, creating a mathematical receipt for decisions, combined with LLM Forensic Reasoning (via LLaMA 3.1 8B) for plain-English reports.
* **Forensic Report Generation**: High-resolution spectral analysis reports including time-domain waveform, 3-channel forensic image, and Grad-CAM heatmaps.
* **Premium Sleek Interface**: A dark-mode, high-fidelity UI built with Streamlit.

***

## Performance & Evaluation

The underlying model has been rigorously validated across diverse datasets (ASVspoof, WaveFake, InTheWild, LibriSpeech) containing over 240,480 samples.
* **ASVspoof 2019 Dev (Standard Lab Test)**: EER: 1.22%, ROC-AUC: 0.9994, Balanced Accuracy: 98.29%
* **In-The-Wild (Real-World Evidence Test)**: EER: 0.86%, ROC-AUC: 0.9992, Balanced Accuracy: 99.24%
* **Overall Deepfake Recall**: 99.7%

EchoTrace achieves better results in the wild than in specific lab benchmarks because it focuses on the fundamental biology of speech rather than overfitting to specific laboratory attack watermarks.

***

## Technical Architecture & Pipeline

EchoTrace uses a Late Fusion approach, treating audio classification as both an image recognition task and a physics-based biometric task. 

### 1. The Processing Pipeline
Every audio sample follows a strict "Pre-Flight to Verdict" workflow:
1. **Validation Gate**: Rejects invalid sample rates, durations, or silence.
2. **Spectral Bandwidth Check**: Detects upsampled 8kHz audio to avoid ghost predictions.
3. **Voice Activity Detection**: Verifies human speech presence (Voiced Ratio > 5%).
4. **Sliding Window Inference**: Extracts features over moving segments.
5. **Ensemble Probability**: Calculates soft-weighted ensemble calculations.
6. **Feature Attribution & LLM Reasoning**: Generates reasoning via SHAP and LLaMA 3.1 8B.

### 2. Feature Engineering Deep Dive

**A. The 3-Channel Forensic Image (Vision Stream)**
Instead of standard grayscale spectrograms, EchoTrace packs three distinct forensic representations into a single RGB image (224x224x3):
* **Channel 1 (Mel Spectrogram)**: Captures general timbre and spectral energy.
* **Channel 2 (MFCC + Deltas)**: Captures cepstral dynamics. AI often leaves micro-jerks in these transitions.
* **Channel 3 (Spectral Contrast + Chroma CQT)**: Captures harmonic stability. AI vocoders often shift harmonic energy unnaturally.

**B. The 8 Biometric Scalars (Physics Stream)**
* **Spectral Flatness**: Detects unnatural peaks or gaps in the voice frequency.
* **Zero Crossing Rate**: Identifies frequency fluctuations indicating synthetic artifacts.
* **F1 Formant**: Captures lower vocal tract resonance (throat and mouth opening).
* **F2 Formant**: Captures oral cavity resonance (tongue position).
* **F3 Formant**: Captures fine articulation and lip rounding details.
* **Voiced Ratio**: Measures the presence of biological vocal fold vibration.
* **Harmonic-to-Noise Ratio**: Detects underlying noise floors; AI is often too clean.
* **Cepstral Peak Prominence**: A high-precision measure of biological voice quality.

### 3. Core Architecture
* **Vision Branch (ResNet-50)**: Initialized with ImageNet-1K weights. Layers 1 through 3 are frozen, Layer 4 is unfrozen and fine-tuned with a differential learning rate (1e-5).
* **Biometric Branch**: A Multi-Layer Perceptron that processes the 8-dimensional vector of physical speech characteristics.
* **Late Fusion Head**: Outputs are concatenated into a 2056-dimensional combined vector before hitting the final forensic classifier head. This allows the model to veto a visual prediction if the physiological scalars indicate synthetic speech.
* **Physics Priority Module**: Acts as a signal booster for biological voice traits.

### 4. Training Infrastructure and Technical Overcomes
* **Hardware setup**: Trained on a 4-GPU Distributed Data Parallel cluster.
* **Optimization**: Uses Automatic Mixed Precision (AMP/float16) for 2x faster math, and a CosineAnnealingLR scheduler.
* **Focal Loss for Class Imbalance**: Penalizes missing easy real samples and focuses on hard edge cases.
* **Deterministic Dataset Concatenation**: Utilized PyTorch's ConcatDataset to fix DDP dataset ordering issues and ensure all GPUs heartbeat together, preventing DDP deadlocks.
* **Data Augmentation**: Employs MUSAN noise injection (background chatter, street noise, music) for real-world robustness.
* **CPU Thread Thrashing Optimization**: Strict thread cap for data workers (NUMBA_NUM_THREADS=1) resulting in a massive increase in data throughput.

***

## Pipeline & Architecture Diagram

![EchoTrace Pipeline And Architecture](<img width="1321" height="731" alt="image" src="https://github.com/user-attachments/assets/185e6051-df9d-4e5c-a40c-6efff41fc57d" />
)

***

## Installation & Setup

Ensure you have Python 3.9+ installed on your system (Windows, macOS, or Linux).

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/BackProp-Bandits-main.git
   cd BackProp-Bandits-main
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
   Ensure `deepfake_detector.pth` or `ensemble_model.pth` is present in the root directory.

***

## Usage

### Option 1: Web Interface (Streamlit)
The recommended way for interactive use and live recording.
```bash
streamlit run streamlit_app.py
```

### Option 2: Highest-Accuracy Reporting (Forensics)
For the most reliable Grad-CAM extraction and forensic-grade image output, use the dedicated standalone script. This generates a detailed report in the reports folder.
```bash
python tests/single_example_report_generator.py
```

***

## Project Structure

```text
.
├── streamlit_app.py         # Main Web Application
├── ensemble_model.pth       # Pre-trained Model Weights
├── core/                    # Engine and Intelligence
│   ├── model.py             # ResNet-50 Architecture and Biometric MLP
│   ├── inference.py         # Signal Processing and Prediction
│   └── ...
├── tests/
│   └── single_example_report_generator.py  # Forensic Script
├── utils/                   # Shared Utilities
├── reports/                 # Auto-generated Forensic Logs
└── requirements.txt         # Project Dependencies
```

***

## Disclaimer
EchoTrace is a forensic tool intended for investigative and educational purposes. While highly accurate, no detection system is infallible. Always use EchoTrace results as one part of a broader verification workflow.

**Built by BackProp Bandits**
