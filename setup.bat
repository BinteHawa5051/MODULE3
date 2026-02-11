@echo off
echo Setting up Adaptive Reasoning Agent...

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Checking sentence-transformers...
python -c "import sentence_transformers" 2>nul
if %errorlevel% equ 0 (
    echo [OK] sentence-transformers installed successfully
) else (
    echo [ERROR] sentence-transformers installation failed
    echo Trying to install separately...
    pip install sentence-transformers
)

echo.
echo Checking torch...
python -c "import torch" 2>nul
if %errorlevel% equ 0 (
    echo [OK] torch installed successfully
) else (
    echo [ERROR] torch installation failed
    echo Trying to install separately...
    pip install torch
)

echo.
echo Setup complete! Run the app with:
echo streamlit run app.py
pause
