#!/bin/bash
# Setup script for Unattended Object Detection System

echo "🚀 Setting up Unattended Object Detection System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python version OK: $python_version"
else
    echo "❌ Python 3.8+ required, found: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python test_system.py

echo "✅ Setup complete!"
echo ""
echo "To use the system:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the system: python app.py --video your_video.mp4"
echo "3. See README.md for detailed usage instructions"
