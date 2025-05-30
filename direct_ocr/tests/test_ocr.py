import pytest
from src.ocr import DirectOCR
import os
from common.config.paths import IMAGE_DIR, LABEL_DIR
from PIL import ImageFont
import glob

@pytest.fixture
def ocr():
    return DirectOCR()

def test_ocr_initialization():
    ocr = DirectOCR()
    assert ocr.lang == 'kor'
    
    ocr_eng = DirectOCR(lang='eng')
    assert ocr_eng.lang == 'eng'

def test_perform_ocr_on_real_data(ocr):
    # 실제 데이터 폴더에서 첫 번째 이미지를 골라 OCR 테스트
    image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    if not image_files:
        pytest.skip("No image files in IMAGE_DIR")
    image_path = image_files[0]
    result = ocr.perform_ocr(image_path)
    assert isinstance(result, str)
    assert len(result.strip()) > 0

def test_process_directory(ocr):
    if not os.listdir(IMAGE_DIR):
        pytest.skip("No images in directory")
    results = ocr.process_directory()
    assert isinstance(results, dict) 