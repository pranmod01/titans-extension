#!/bin/bash
# Colab setup script for Multi-Signal Titans
# Run this after mounting Google Drive

set -e

# Clone the repository (replace with your actual repo URL)
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/titans-extension.git}"

if [ -d "/content/titans-extension" ]; then
    echo "Repository already exists, pulling latest..."
    cd /content/titans-extension
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" /content/titans-extension
fi

cd /content/titans-extension

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create checkpoint directory
mkdir -p /content/drive/MyDrive/titans_checkpoints

echo ""
echo "Setup complete!"
echo "You can now run experiments with:"
echo "  python -m multi_signal_titans.experiments --experiment 1"
echo ""
echo "Or import the module:"
echo "  from multi_signal_titans import create_model, Config"
