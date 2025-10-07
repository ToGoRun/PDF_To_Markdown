# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF to Markdown converter with GUI preview, powered by marker-pdf library. Converts PDF documents to Markdown format with support for text, images, and formatting. Features side-by-side PDF and Markdown preview with page navigation.

## Architecture

### Core Components

- **PDFtoMarkdownConverter** ([PDF_Change.py:30](PDF_Change.py#L30)): Main application class, manages GUI, conversion pipeline, and device detection
- **Device Detection** ([PDF_Change.py:54](PDF_Change.py#L54)): Auto-detects CUDA GPU availability for hardware acceleration
- **Conversion Pipeline** ([PDF_Change.py:260](PDF_Change.py#L260)): Threaded PDF→Markdown conversion using marker-pdf library
- **Image Extraction** ([PDF_Change.py:359](PDF_Change.py#L359)): Extracts embedded images from PDFs and updates references in Markdown

### Key Technical Details

**Virtual Environment Enforcement**: Script auto-restarts in venv if not already running there ([PDF_Change.py:14](PDF_Change.py#L14-18))

**Model Caching**: Models downloaded to `./models/` directory with environment variables set for HuggingFace and PyTorch caches ([PDF_Change.py:280](PDF_Change.py#L280-286))

**GPU Acceleration**: Uses TORCH_DEVICE environment variable to route processing to CUDA when available ([PDF_Change.py:271](PDF_Change.py#L271))

**Performance Modes** ([PDF_Change.py:328-374](PDF_Change.py#L328-374)):
- **Fast Mode (default)**: Uses `disable_ocr=True` to extract embedded PDF text only (~30 seconds typical)
- **OCR Mode**: Enables OCR for scanned PDFs with `force_ocr=True` and optimized batch sizes
- **High Quality Mode**: Uses smaller batch sizes for better accuracy
- User toggles via checkboxes in toolbar ([PDF_Change.py:118-120](PDF_Change.py#L118-120))

**Image Handling**:
- Images extracted as base64/bytes from marker output
- Saved with block_id as filename (e.g., `_page_X_Picture_Y.png`)
- Markdown image references updated with relative paths ([PDF_Change.py:437](PDF_Change.py#L437-504))

**Export Formats**:
- Markdown: Direct export with extracted images in `{filename}_images/` folder
- Word: Uses pypandoc (primary) or python-docx (fallback) for DOCX conversion ([PDF_Change.py:606](PDF_Change.py#L606-643))

### GUI Architecture

Built with tkinter, three main panels:
- Left: PDF canvas preview (PyMuPDF/fitz rendering)
- Right: Markdown text preview (ScrolledText widget)
- Bottom: Page navigation controls

Page-to-Markdown mapping uses simple line-based splitting ([PDF_Change.py:341](PDF_Change.py#L341-357))

## Development Commands

### Setup and Running
```bash
# Install dependencies (creates venv automatically on first run)
.\run.bat                    # Windows batch
.\run.ps1                    # PowerShell

# Manual venv setup if needed
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python PDF_Change.py
```

### GPU Acceleration Setup
```bash
# Check GPU availability
.\check_gpu.bat

# Install CUDA-enabled PyTorch (for RTX/NVIDIA GPUs)
.\setup_gpu.bat              # Installs PyTorch with CUDA 12.4 support
```

### Word Export Setup (Optional)
```bash
# Install Pandoc for high-quality Word export
.\install_pandoc.bat

# Or install manually:
# 1. Download from https://github.com/jgm/pandoc/releases/latest
# 2. Run the .msi installer
# 3. Restart terminal

# Note: python-docx fallback will be used if Pandoc is not installed
```

### Testing Workflow
```bash
# Run application
.\run.bat

# Test conversion with a PDF file
# 1. Click "选择PDF文件" to select PDF
# 2. Click "开始转换" to convert
# 3. Use navigation buttons to preview pages
# 4. Export using "导出Markdown" or "导出Word"
```

## Dependencies

Core libraries in [requirements.txt](requirements.txt):
- `marker-pdf`: PDF to Markdown conversion engine
- `PyMuPDF` (fitz): PDF rendering and preview
- `Pillow`: Image processing

Optional (for Word export):
- `pypandoc`: Preferred Markdown→DOCX converter (requires Pandoc binary)
- `python-docx`: Fallback DOCX generator

PyTorch: Auto-installed by marker-pdf (CPU by default, use setup_gpu.bat for CUDA version)

## Important Notes

- Application requires virtual environment - will auto-restart if not in venv
- First run downloads marker-pdf models (~1-2GB) to `./models/` directory
- GPU acceleration provides 3-10x speedup on NVIDIA RTX cards
- Page splitting is approximate (line-based distribution across pages)
- Word export requires either Pandoc or will fall back to basic python-docx formatting
- Image paths in exported Markdown use relative paths (`{filename}_images/`)
- Model cache persists between runs to avoid re-downloading

## Common Issues

**GPU not detected**: Run `check_gpu.bat` - if shows CPU-only, run `setup_gpu.bat` to install CUDA PyTorch

**Word export fails**: Run `install_pandoc.bat` to install Pandoc, or download from https://github.com/jgm/pandoc/releases/latest. If Pandoc is not installed, the program will automatically use python-docx fallback (simpler formatting)

**Models download slowly**: First conversion downloads ~1-2GB models - subsequent runs use cached models in `./models/`
