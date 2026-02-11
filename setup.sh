#!/bin/bash

echo "Setting up Adaptive Reasoning Agent..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if sentence-transformers is installed
python -c "import sentence_transformers" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ sentence-transformers installed successfully"
else
    echo "❌ sentence-transformers installation failed"
    echo "Trying to install separately..."
    pip install sentence-transformers
fi

# Check if torch is installed
python -c "import torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ torch installed successfully"
else
    echo "❌ torch installation failed"
    echo "Trying to install separately..."
    pip install torch
fi

echo ""
echo "Setup complete! Run the app with:"
echo "streamlit run app.py"
