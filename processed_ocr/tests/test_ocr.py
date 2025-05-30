import pytest
from src.ocr import ProcessedOCR
import numpy as np
from PIL import Image, ImageFont
from common.config.paths import IMAGE_DIR, LABEL_DIR
import os
import glob

@pytest.fixture
def ocr():
    return ProcessedOCR()

def test_ocr_initialization():
    ocr = ProcessedOCR()
    assert ocr.lang == 'kor'
    
    ocr_eng = ProcessedOCR(lang='eng')
    assert ocr_eng.lang == 'eng'

def test_preprocess_image(ocr):
    # Create a test image
    img = Image.new('RGB', (100, 100), color='white')
    img_array = np.array(img)
    
    # Test preprocessing
    processed = ocr.preprocess_image(img_array)
    
    # Check if the output is a numpy array
    assert isinstance(processed, np.ndarray)
    
    # Check if the image is grayscale
    assert len(processed.shape) == 2

def test_perform_ocr_on_real_data(ocr):
    image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    if not image_files:
        pytest.skip("No image files in IMAGE_DIR")
    image_path = image_files[0]
    result = ocr.perform_ocr(image_path)
    assert isinstance(result, str)
    assert len(result.strip()) > 0

def test_process_directory(ocr):
    # Skip if no images in directory
    if not os.listdir(IMAGE_DIR):
        pytest.skip("No images in directory")
        
    results = ocr.process_directory()
    assert isinstance(results, dict) 