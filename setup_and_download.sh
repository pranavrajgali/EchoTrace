#!/bin/bash
# Setup Kaggle credentials and download datasets

set -e  # Exit on error

echo "=========================================="
echo "🔐 Setting up Kaggle credentials..."
echo "=========================================="

# Install kaggle if not present
pip install kaggle --quiet

# Create .kaggle directory
mkdir -p ~/.kaggle

# Create kaggle.json config
cat > ~/.kaggle/kaggle.json << 'EOF'
{
  "username": "abhinavmucharla",
  "key": "f609fd8d5d04b5c10e934f11c8a8c801"
}
EOF

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json

echo "✅ Kaggle credentials configured"

echo ""
echo "=========================================="
echo "📥 Starting dataset downloads..."
echo "=========================================="
echo "This will take several hours (86 GB total)"
echo "Datasets will be saved to: /home/jovyan/work/data/"
echo ""

# Run download script
cd /home/jovyan/work/EchoTrace
python3 download_datasets.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ALL DATASETS DOWNLOADED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Ready to train:"
    echo "  cd /home/jovyan/work/EchoTrace"
    echo "  python train_ddp.py"
else
    echo ""
    echo "❌ Download failed. Check logs above."
    exit 1
fi
