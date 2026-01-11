#!/bin/bash
# Setup script for PaddleOCR with Python 3.12 compatibility

echo "🔍 Setting up PaddleOCR for OCR Q&A Segmentation"
echo "================================================"

# Check Python version
echo "📌 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Current version: $PYTHON_VERSION"

# Extract major.minor version
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1-2 | tr -d '.')

if [ "$PYTHON_MAJOR_MINOR" -ge 313 ]; then
    echo ""
    echo "⚠️  WARNING: Python $PYTHON_VERSION is too new for PaddleOCR"
    echo "   PaddleOCR requires Python < 3.13"
    echo ""
    echo "🔧 SOLUTIONS:"
    echo "   Option 1 (Recommended): Install Python 3.12"
    echo "      brew install python@3.12"
    echo "      python3.12 -m venv venv"
    echo "      source venv/bin/activate"
    echo "      pip install -r requirements.txt"
    echo ""
    echo "   Option 2: Use Tesseract only (already configured as fallback)"
    echo "      The system will automatically use Tesseract if PaddleOCR is unavailable"
    echo ""
    read -p "Press Enter to continue with Tesseract fallback setup..."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Try to install PaddleOCR
echo "📥 Attempting to install dependencies..."
pip install --upgrade pip

# Try PaddleOCR installation
if pip install paddleocr paddlepaddle 2>&1 | grep -q "ERROR"; then
    echo "⚠️  PaddleOCR installation failed (expected on Python 3.14)"
    echo "   Installing Tesseract dependencies instead..."
    pip install pytesseract opencv-python pillow numpy
    echo "✅ Tesseract setup complete (fallback mode)"
else
    echo "✅ PaddleOCR installed successfully!"
fi

# Install remaining dependencies
echo "📥 Installing remaining dependencies..."
pip install -r requirements.txt 2>&1 | grep -v "already satisfied"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
