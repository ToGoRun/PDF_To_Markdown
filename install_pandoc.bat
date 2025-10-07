@echo off
echo ========================================
echo    Pandoc Installation Script
echo    Required for Word export functionality
echo ========================================
echo.

echo Pandoc is required for converting Markdown to Word (.docx) files.
echo.
echo This script will help you install Pandoc in one of two ways:
echo   1. Using Chocolatey (automatic installation)
echo   2. Manual download from Pandoc website
echo.

REM Check if Pandoc is already installed
where pandoc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Pandoc is already installed!
    pandoc --version
    echo.
    echo You can now use Word export in the PDF converter.
    pause
    exit /b 0
)

echo Pandoc is not currently installed.
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Chocolatey is detected. Would you like to install Pandoc using Chocolatey?
    set /p use_choco="Install with Chocolatey? (Y/N): "
    if /i "%use_choco%"=="Y" (
        echo.
        echo Installing Pandoc via Chocolatey...
        choco install pandoc -y

        echo.
        echo ========================================
        echo Verifying installation...
        where pandoc >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo SUCCESS! Pandoc installed successfully.
            pandoc --version
            echo.
            echo Please restart your terminal and run the PDF converter.
        ) else (
            echo.
            echo Installation may require administrator privileges.
            echo Please run this script as Administrator or install manually.
        )
        echo ========================================
        pause
        exit /b 0
    )
)

echo.
echo ========================================
echo Manual Installation Instructions:
echo ========================================
echo.
echo 1. Visit: https://github.com/jgm/pandoc/releases/latest
echo.
echo 2. Download: pandoc-X.XX.X-windows-x86_64.msi
echo    (Look for the .msi installer file)
echo.
echo 3. Run the installer and follow the installation wizard
echo.
echo 4. After installation, restart your terminal
echo.
echo 5. Run 'pandoc --version' to verify installation
echo.
echo ========================================
echo.

set /p open_browser="Open Pandoc download page in browser? (Y/N): "
if /i "%open_browser%"=="Y" (
    start https://github.com/jgm/pandoc/releases/latest
)

echo.
echo Alternative: Use python-docx fallback
echo ========================================
echo If you don't want to install Pandoc, the program will automatically
echo use python-docx as a fallback. This works but has simpler formatting.
echo.
pause
