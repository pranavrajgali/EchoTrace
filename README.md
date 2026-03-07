# EchoTrace Backend Engine 

**Forensic-grade Deepfake Audio Detection API**  
Built with **FastAPI** + **PyTorch** — the analytical core powering the EchoTrace frontend.  
Processes audio streams, runs neural inference, generates **Grad-CAM heatmaps** and **forensic PDF reports**.

##  Tech Stack

- **Framework**: FastAPI (Python)
- **ML Backend**: PyTorch + Torchaudio
- **Server**: Uvicorn
- **Audio Processing**: Librosa

##  Performance & Validation

The EchoTrace model has been **rigorously tested** on both controlled lab datasets and challenging real-world ("in-the-wild") audio, showing strong robustness against social media compression, background noise, and acoustic variations.

### Core Metrics

| Metric                  | Lab (ASVspoof Dev) | Real-world (InTheWild Val) |
|-------------------------|--------------------|-----------------------------|
| **Overall Accuracy**    | 97.58%            | **98.70%**                 |
| **Bonafide (Real) Acc** | 98.25%            | 98.90%                     |
| **Spoof (Fake) Acc**    | 96.90%            | 98.50%                     |
| **ROC AUC**             | 0.9973            | **0.9992**                 |
| **PR AUC**              | 0.9975            | **0.9992**                 |
| **Equal Error Rate**    | —                 | **1.0000%**                |

- **Samples evaluated**: 4,000 (lab) + 2,000 (real-world)
- Exceptional generalization on degraded / compressed audio

##  Highest-Accuracy Reporting (Recommended for Forensics) 


For the **most reliable Grad-CAM extraction** and **forensic-grade PDF output**, use the dedicated script:

```bash
python tests/single_example_report_generator.py
```

##  Prerequisites

- Python 3.9 or higher
- (Recommended) NVIDIA GPU + CUDA for significantly faster inference

##  Installation & Setup

```bash
# Clone the repository
git clone https://github.com/CRRAO-Internal-Hackathon-2026/BackProp-Bandits.git
cd EchoTrace-Backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# or on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt




