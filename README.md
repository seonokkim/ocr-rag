# OCR-RAG: OCR Evaluation and Analysis Framework

This repository contains a comprehensive OCR (Optical Character Recognition) evaluation framework that compares different OCR approaches and preprocessing methods.

## Features

- **Multiple OCR Engines**: Support for Tesseract, PaddleOCR, and EasyOCR
- **Different Processing Methods**:
  - Direct OCR: Performs OCR directly on input images
  - Preprocessed OCR: Applies various image processing techniques before OCR
  - YOLO + OCR: Uses YOLO for text detection before OCR
- **Comprehensive Evaluation**: 
  - Processing time analysis
  - Success rate comparison
  - Accuracy, precision, and recall metrics
  - Dataset analysis and visualization

## Project Structure

```
.
├── analyze_dataset.py      # Dataset analysis and visualization
├── direct_ocr.py          # Direct OCR implementation
├── preprocessed_ocr.py    # Preprocessed OCR implementation
├── yolo_ocr.py           # YOLO + OCR implementation
├── run_evaluation.py      # Main evaluation script
├── evaluate_ocr.py        # OCR evaluation utilities
├── requirements.txt       # Project dependencies
├── setup.py              # Project setup script
│
├── data/                 # Dataset directory
├── processed_ocr/        # Processed OCR results
├── direct_ocr/          # Direct OCR results
└── common/              # Shared utilities and configurations
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

1. Run the evaluation:
```bash
python run_evaluation.py
```

This will:
- Analyze the dataset characteristics
- Run evaluations for all OCR methods
- Generate comprehensive reports and visualizations
- Save results in a timestamped directory

2. View results:
- Check the generated `evaluation_results_YYYYMMDD_HHMMSS` directory
- Review `evaluation_results.csv` for detailed metrics
- View `evaluation_report.png` for visual comparisons
- Read `evaluation_summary.txt` for performance analysis

## Evaluation Methods

### Direct OCR
Performs OCR directly on input images without any preprocessing. Best for high-quality images.

### Preprocessed OCR
Applies various image processing techniques:
- Grayscale conversion
- Adaptive thresholding
- Gaussian blur
- Image sharpening
- Denoising

### YOLO + OCR
Uses YOLO for text detection before performing OCR, which can improve accuracy for complex images.

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details. 