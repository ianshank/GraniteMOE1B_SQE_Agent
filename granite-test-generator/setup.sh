#!/bin/bash
# Granite Test Generator - Quick Setup Script

echo "Setting up Granite Test Generator..."
echo "======================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n granite-moe python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate granite-moe

# Install PyTorch
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies (optional)
read -p "Do you want to install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/training
mkdir -p data/requirements  
mkdir -p data/user_stories
mkdir -p output
mkdir -p models/fine_tuned_granite
mkdir -p cache
mkdir -p logs

# Setup pre-commit hooks (if dev dependencies installed)
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy config/model_config.yaml.example to config/model_config.yaml"
echo "2. Update the configuration with your API credentials"
echo "3. Run: conda activate granite-moe"
echo "4. Run: python main.py"
echo ""
echo "For more information, see README.md"
