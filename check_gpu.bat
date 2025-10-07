@echo off
echo ========================================
echo    GPU Detection Tool
echo ========================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

echo Detecting GPU and CUDA environment...
echo.

venv\Scripts\python.exe -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); cuda_available = torch.cuda.is_available(); print('CUDA Version:', torch.version.cuda if torch.version.cuda else 'CPU version'); print('GPU Count:', torch.cuda.device_count() if cuda_available else 0); print('GPU Name:', torch.cuda.get_device_name(0) if cuda_available else 'No GPU detected'); print('GPU Memory:', str(torch.cuda.get_device_properties(0).total_memory / 1024**3) + ' GB' if cuda_available else '')"

echo.
echo ========================================

venv\Scripts\python.exe -c "import torch; cuda_available = torch.cuda.is_available(); print('Recommendation:' if not cuda_available else 'GPU Available!'); print('You have CPU version PyTorch. Run setup_gpu.bat to install GPU version' if not cuda_available else 'GPU acceleration available. Performance will be 3-10x faster!')"

echo ========================================
echo.
pause
