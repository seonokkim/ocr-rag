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

class DirectOCR:
    def __init__(self):
        self.models = {
            'tesseract': self._tesseract_ocr,
            'paddle': PaddleOCR(use_angle_cls=True, lang='ko'),
            'easyocr': easyocr.Reader(['ko'])
        }
        
    def _tesseract_ocr(self, image):
        return pytesseract.image_to_string(image, lang='kor')
    
    def process_image(self, image_path, model_name):
        """
        Process a single image with specified model
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            start_time = time.time()
            
            if model_name == 'tesseract':
                result = self.models[model_name](image)
            elif model_name == 'paddle':
                result = self.models[model_name].ocr(image, cls=True)
                result = '\n'.join([line[1][0] for line in result[0]]) if result[0] else ''
            else:  # easyocr
                result = self.models[model_name].readtext(image)
                result = '\n'.join([line[1] for line in result])
            
            processing_time = time.time() - start_time
            
            return {
                'text': result,
                'processing_time': processing_time
            }
        except Exception as e:
            print(f"Error processing {image_path} with {model_name}: {str(e)}")
            return None

    def evaluate_model(self, dataset_path, model_name, ground_truth=None):
        """
        Evaluate OCR model performance
        """
        dataset_path = Path(dataset_path)
        results = []
        
        for image_path in tqdm(list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png'))):
            result = self.process_image(image_path, model_name)
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
        df.to_csv(f'direct_ocr_{model_name}_results.csv', index=False)
        
        return metrics

def main():
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    ocr = DirectOCR()
    
    # Evaluate each model
    for model_name in ['tesseract', 'paddle', 'easyocr']:
        print(f"\nEvaluating {model_name}...")
        metrics = ocr.evaluate_model(dataset_path, model_name)
        print(f"Results for {model_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 