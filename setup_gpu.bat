@echo off
echo ========================================
echo    GPU Acceleration Setup Script
echo    Install CUDA version PyTorch for RTX GPU
echo ========================================
echo.

echo Checking virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found. Please run run.bat first
    pause
    exit /b 1
)

echo.
echo Checking current PyTorch version...
venv\Scripts\python.exe -c "import torch; print('Current version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.

echo ========================================
echo Will perform the following operations:
echo 1. Uninstall CPU version of PyTorch
echo 2. Install CUDA 12.4 version of PyTorch (compatible with CUDA 12.x)
echo 3. Verify GPU availability
echo ========================================
echo.

set /p confirm="Continue with installation? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Installation cancelled
    pause
    exit /b 0
)

echo.
echo [1/3] Uninstalling CPU version of PyTorch...
venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Installing CUDA version of PyTorch (this may take several minutes)...
echo Note: Using CUDA 12.4 build which is compatible with your CUDA 12.9 driver
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [3/3] Verifying GPU availability...
venv\Scripts\python.exe -c "import torch; print(''); print('='*50); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); cuda_available = torch.cuda.is_available(); print('GPU Model:', torch.cuda.get_device_name(0) if cuda_available else 'Not detected'); print('GPU Memory:', str(torch.cuda.get_device_properties(0).total_memory / 1024**3) + ' GB' if cuda_available else ''); print('='*50); print('')"

echo.
echo ========================================
echo Installation complete!
echo.
echo Please restart the program by running run.bat
echo The program will automatically use GPU acceleration
echo ========================================
pause
