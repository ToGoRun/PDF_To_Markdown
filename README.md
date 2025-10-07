# PDF to Markdown Converter

A powerful PDF to Markdown converter with GUI preview, powered by the [marker-pdf](https://github.com/VikParuchuri/marker) library. Features GPU acceleration, side-by-side PDF and Markdown preview, and export to both Markdown and Word formats.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## ✨ Features

- 📄 **High-Quality Conversion**: Converts PDF documents to Markdown with excellent formatting preservation
- 🖼️ **Image Extraction**: Automatically extracts and saves embedded images from PDFs
- 👁️ **Live Preview**: Side-by-side PDF and Markdown preview with page navigation
- 🚀 **GPU Acceleration**: 3-10x faster conversion on NVIDIA GPUs (RTX series)
- 📝 **Multiple Export Formats**: Export to Markdown (.md) or Word (.docx)
- 🎯 **User-Friendly GUI**: Simple tkinter-based interface for easy operation

## 🖥️ Screenshots

The application features three main panels:
- **Left**: PDF canvas preview
- **Right**: Markdown text preview
- **Bottom**: Page navigation controls

## 🚀 Quick Start

### Prerequisites

- Windows OS (other platforms may work with minor modifications)
- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ToGoRun/PDF_To_Markdown.git
   cd PDF_To_Markdown
   ```

2. **Run the application** (automatic setup)
   ```bash
   .\run.bat
   ```

   The script will automatically:
   - Create a virtual environment
   - Install all required dependencies
   - Launch the application

### GPU Acceleration (Optional but Recommended)

For NVIDIA GPU users, enable GPU acceleration for 3-10x faster conversion:

```bash
# Check if GPU is detected
.\check_gpu.bat

# Install CUDA-enabled PyTorch
.\setup_gpu.bat
```

### Word Export Setup (Optional)

For high-quality Word export, install Pandoc:

```bash
# Run the installation helper
.\install_pandoc.bat

# Or download manually from:
# https://github.com/jgm/pandoc/releases/latest
```

**Note**: If Pandoc is not installed, the application will automatically use python-docx as a fallback.

## 📖 Usage

1. **Launch the application**
   ```bash
   .\run.bat
   ```

2. **Convert a PDF**
   - Click "选择PDF文件" to select your PDF file
   - Click "开始转换" to start conversion
   - Use navigation buttons to preview different pages
   - Export using "导出Markdown" or "导出Word"

3. **View Results**
   - Markdown files are saved with extracted images in `{filename}_images/` folder
   - Word documents include all content and images

## 🛠️ Technical Details

### Architecture

- **Conversion Engine**: [marker-pdf](https://github.com/VikParuchuri/marker) - State-of-the-art PDF to Markdown conversion
- **PDF Rendering**: PyMuPDF (fitz) for fast PDF preview
- **GUI Framework**: tkinter (built-in Python GUI library)
- **GPU Acceleration**: PyTorch with CUDA support

### Model Caching

On first run, the application downloads AI models (~1-2GB) to the `./models/` directory. These models are cached for future use, so subsequent conversions are much faster.

### Device Detection

The application automatically detects available compute devices:
- ✅ CUDA GPU (NVIDIA)
- ✅ CPU fallback

GPU information is displayed in the status bar when a GPU is detected.

## 📦 Dependencies

Core libraries:
- `marker-pdf` - PDF to Markdown conversion engine
- `PyMuPDF` - PDF rendering and preview
- `Pillow` - Image processing
- `python-docx` - Word document generation
- `pypandoc` - Enhanced Word export (requires Pandoc binary)

PyTorch is automatically installed by marker-pdf. For GPU acceleration, run `setup_gpu.bat` to install the CUDA version.

## 🐛 Troubleshooting

### GPU Not Detected

**Problem**: Application shows "CPU" instead of your GPU

**Solution**:
```bash
.\check_gpu.bat    # Check current status
.\setup_gpu.bat    # Install CUDA-enabled PyTorch
```

### Word Export Fails

**Problem**: "Export failed, need to install Pandoc" error

**Solution**:
- Run `.\install_pandoc.bat` for guided installation
- Or download Pandoc manually from https://github.com/jgm/pandoc/releases/latest
- The application will use python-docx fallback if Pandoc is unavailable

### Models Download Slowly

**Problem**: First conversion takes a long time

**Explanation**: The first run downloads ~1-2GB of AI models. This is normal and only happens once. Subsequent conversions use cached models and are much faster.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [marker-pdf](https://github.com/VikParuchuri/marker) - Excellent PDF to Markdown conversion library
- [PyMuPDF](https://pymupdf.readthedocs.io/) - Fast PDF rendering
- [Pandoc](https://pandoc.org/) - Universal document converter

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For issues and questions, please use the [GitHub Issues](https://github.com/ToGoRun/PDF_To_Markdown/issues) page.

---

**Note**: This tool is designed for Windows. For Linux/Mac users, you may need to modify the batch scripts (.bat) to shell scripts (.sh).
