# OCR Projects

This repository contains two different OCR (Optical Character Recognition) projects:

1. **Direct OCR**: Performs OCR directly on input images
2. **Processed OCR**: Applies image processing techniques before performing OCR

## Project Structure

```
.
├── direct_ocr/           # Direct OCR project
│   ├── src/             # Source code
│   ├── tests/           # Test files
│   ├── data/            # Data directory
│   └── config/          # Configuration files
│
├── processed_ocr/        # Processed OCR project
│   ├── src/             # Source code
│   ├── tests/           # Test files
│   ├── data/            # Data directory
│   └── config/          # Configuration files
│
└── common/              # Shared utilities and configurations
    ├── utils/           # Common utility functions
    └── config/          # Shared configuration files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

## Projects

### Direct OCR
Performs OCR directly on input images without any preprocessing. This is useful for high-quality images where preprocessing might not be necessary.

### Processed OCR
Applies various image processing techniques (like denoising, contrast enhancement, etc.) before performing OCR. This is useful for low-quality images or images with specific issues that need to be addressed before OCR.

## Contributing
Feel free to submit issues and enhancement requests! 