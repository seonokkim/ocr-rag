import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
import easyocr
import time
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self):
        self.preprocessing_methods = {
            'grayscale': self._grayscale,
            'adaptive_threshold': self._adaptive_threshold,
            'gaussian_blur': self._gaussian_blur,
            'sharpen': self._sharpen,
            'denoise': self._denoise
        }
    
    def _grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _adaptive_threshold(self, image):
        gray = self._grayscale(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    def _gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def _sharpen(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _denoise(self, image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

class PreprocessedOCR:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.models = {
            'tesseract': self._tesseract_ocr,
            'paddle': PaddleOCR(use_angle_cls=True, lang='ko'),
            'easyocr': easyocr.Reader(['ko'])
        }
    
    def _tesseract_ocr(self, image):
        return pytesseract.image_to_string(image, lang='kor')
    
    def process_image(self, image_path, model_name, preprocessing_method):
        """
        Process a single image with preprocessing and OCR
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Apply preprocessing
            preprocessed = self.preprocessor.preprocessing_methods[preprocessing_method](image)
            
            start_time = time.time()
            
            if model_name == 'tesseract':
                result = self.models[model_name](preprocessed)
            elif model_name == 'paddle':
                result = self.models[model_name].ocr(preprocessed, cls=True)
                result = '\n'.join([line[1][0] for line in result[0]]) if result[0] else ''
            else:  # easyocr
                result = self.models[model_name].readtext(preprocessed)
                result = '\n'.join([line[1] for line in result])
            
            processing_time = time.time() - start_time
            
            return {
                'text': result,
                'processing_time': processing_time
            }
        except Exception as e:
            print(f"Error processing {image_path} with {model_name} and {preprocessing_method}: {str(e)}")
            return None

    def evaluate_model(self, dataset_path, model_name, preprocessing_method, ground_truth=None):
        """
        Evaluate OCR model performance with preprocessing
        """
        dataset_path = Path(dataset_path)
        results = []
        
        for image_path in tqdm(list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png'))):
            result = self.process_image(image_path, model_name, preprocessing_method)
            if result:
                results.append({
                    'image_path': str(image_path),
                    'text': result['text'],
                    'processing_time': result['processing_time']
                })
        
        # Calculate metrics
        metrics = {
            'total_images': len(results),
            'avg_processing_time': np.mean([r['processing_time'] for r in results]),
            'success_rate': len(results) / len(list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png')))
        }
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f'preprocessed_ocr_{model_name}_{preprocessing_method}_results.csv', index=False)
        
        return metrics

def main():
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    ocr = PreprocessedOCR()
    
    preprocessing_methods = ['grayscale', 'adaptive_threshold', 'gaussian_blur', 'sharpen', 'denoise']
    models = ['tesseract', 'paddle', 'easyocr']
    
    # Evaluate each combination of model and preprocessing method
    for model_name in models:
        for preprocessing_method in preprocessing_methods:
            print(f"\nEvaluating {model_name} with {preprocessing_method}...")
            metrics = ocr.evaluate_model(dataset_path, model_name, preprocessing_method)
            print(f"Results for {model_name} with {preprocessing_method}:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    main() 