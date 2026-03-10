# 🎙️ EchoTrace | AI-Powered Deepfake Audio Detection

**EchoTrace** is a forensic-grade audio analysis suite designed to detect synthetic speech and AI-generated voice clones. Built by the **BackProp Bandits**, this tool leverages a fine-tuned ResNet-50 architecture to identify subtle artifacts in the spectral domain that are invisible to the human ear.

---

## 🚀 Key Features

*   **Multi-Source Analysis**: Upload `.wav`, `.mp3`, or `.flac` files or record live audio directly in the browser.
*   **Forensic Report Generation**: High-resolution spectral analysis reports including:
    *   **Time-Domain Waveform**: Visualizing amplitude over time.
    *   **Spectral Fingerprinting**: 224x224 Mel-spectrogram analysis.
    *   **Grad-CAM Heatmaps**: AI-driven "attention" maps highlighting exactly where the model detected synthetic traits.
*   **Interactive Confidence Timeline**: (New in v3.0) Sliding-window analysis with interactive Plotly graph showing spoof confidence over time.
*   **Real-time Inference**: Swift processing with confidence scoring.
*   **Premium Sleek Interface**: A dark-mode, high-fidelity UI built with Streamlit.

---

## 📊 Performance & Evaluation

The underlying model has been rigorously validated across diverse datasets:
- **Samples Evaluated**: 4,000 (Lab) + 2,000 (Real-world).
- **Exceptional Generalization**: High accuracy even on degraded, noisy, or compressed audio streams.
- **Sensor**: ResNet-50 Fine-Tuned for Audio Forensic Analysis.

---

## 🏗️ Technical Architecture & Pipeline

EchoTrace uses a "Spectro-Visual" approach to deepfake detection, treating audio classification as an image recognition task.

### 1. The Processing Pipeline
1.  **Signal Normalization**: Audio is resampled to **16kHz**, peak-normalized to match training distribution, and standardized to a **4-second** duration.
2.  **STFT & Mel-Scaling**: The raw signal is transformed into a **224x224 Mel-Spectrogram**. This converts the time-series data into a frequency-domain image that represents the spectral energy of the voice.
3.  **RGB Transformation**: The grayscale spectrogram is mapped to the RGB color space and normalized using ImageNet statistics to prepare it for the ResNet backbone.

### 2. Core Architecture & Transfer Learning
*   **Backbone**: A fine-tuned **ResNet-50** deep convolutional neural network (CNN).
*   **Transfer Learning Strategy**: 
    *   **Base Weights**: The model is initialized with ImageNet-1K weights, benefiting from pre-trained feature extractors that recognize edges, textures, and shapes.
    *   **Layer Freezing**: Layers 1 through 3 are "frozen" to preserve general image-recognition patterns, preventing the gradients from destroying the low-level features during training.
    *   **Strategic Unfreezing**: The final residual block (**Layer 4**) is **unfrozen** and fine-tuned. This allows the model to adapt its high-level feature extraction specifically to the unique geometric properties of audio Mel-spectrograms.
    *   **Differential Learning Rates**: We employ a specialized optimizer setup where the backbone (Layer 4) learns at a slower rate (`1e-5`) than the custom classifier head (`1e-4`), ensuring stable convergence.
*   **Classifier Head**: The default 1,000-class head is replaced with a high-capacity binary classification block (512-unit FC -> ReLU -> Dropout -> Sigmoid Output), specifically tuned for the "Bonafide vs. Spoof" task.
*   **Explainable AI (XAI)**: We implement **Grad-CAM** (Gradient-weighted Class Activation Mapping). By tracking the gradients flowing into the final convolutional layer, the system generates a heatmap showing precisely which frequency bands or temporal moments triggered the "Fake" verdict.

---

## 📋 Pipeline & Architecture Diagram

![EchoTrace_Pipeline And_Architecture](https://github.com/user-attachments/assets/62d449e6-53e0-4ca5-ba52-086f61e55381)

---

## 🛠️ Installation & Setup

Ensure you have Python 3.9+ installed on your system (Windows/macOS/Linux).

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
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Weights**
   Ensure `deepfake_detector.pth` is present in the root directory.

---

## 🕹️ Usage

### Option 1: Web Interface (Streamlit)
The recommended way for interactive use and live recording.
```bash
streamlit run streamlit_app.py
```

### Option 2: Highest-Accuracy Reporting (Forensics) 🧪
For the **most reliable Grad-CAM extraction** and **forensic-grade image output**, use the dedicated standalone script. This generates a detailed `.png` report in the `reports/` folder.
```bash
python tests/single_example_report_generator.py
```

---

## 📂 Project Structure

```text
.
├── streamlit_app.py         # Main Web Application
├── deepfake_detector.pth    # Pre-trained Model Weights
├── core/                    # Engine & Intelligence
│   ├── model.py             # ResNet-50 Architecture
│   ├── inference.py         # Signal Processing & Prediction
│   └── ...
├── tests/
│   └── single_example_report_generator.py  # Forensic Script
├── utils/                   # Shared Utilities
├── reports/                 # Auto-generated Forensic Logs
└── requirements.txt         # Project Dependencies
```

---

## ⚠️ Disclaimer
EchoTrace is a forensic tool intended for investigative and educational purposes. While highly accurate, no detection system is infallible. Always use EchoTrace results as one part of a broader verification workflow.

**Built by BackProp Bandits**

