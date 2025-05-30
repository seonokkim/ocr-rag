import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import time
from tqdm import tqdm
from ultralytics import YOLO
import pytesseract
from paddleocr import PaddleOCR
import easyocr

class YOLOOCR:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        self.ocr_models = {
            'tesseract': self._tesseract_ocr,
            'paddle': PaddleOCR(use_angle_cls=True, lang='ko'),
            'easyocr': easyocr.Reader(['ko'])
        }
    
    def _tesseract_ocr(self, image):
        return pytesseract.image_to_string(image, lang='kor')
    
    def detect_text_regions(self, image):
        """
        Detect text regions using YOLO
        """
        results = self.yolo_model(image)
        text_regions = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # Assuming class 0 is text
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    text_regions.append((x1, y1, x2, y2))
        
        return text_regions
    
    def process_image(self, image_path, ocr_model_name):
        """
        Process a single image with YOLO detection and OCR
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Detect text regions
            text_regions = self.detect_text_regions(image)
            
            start_time = time.time()
            all_text = []
            
            # Process each text region
            for x1, y1, x2, y2 in text_regions:
                region = image[y1:y2, x1:x2]
                
                if ocr_model_name == 'tesseract':
                    text = self.ocr_models[ocr_model_name](region)
                elif ocr_model_name == 'paddle':
                    result = self.ocr_models[ocr_model_name].ocr(region, cls=True)
                    text = '\n'.join([line[1][0] for line in result[0]]) if result[0] else ''
                else:  # easyocr
                    result = self.ocr_models[ocr_model_name].readtext(region)
                    text = '\n'.join([line[1] for line in result])
                
                all_text.append(text)
            
            processing_time = time.time() - start_time
            
            return {
                'text': '\n'.join(all_text),
                'processing_time': processing_time,
                'num_regions': len(text_regions)
            }
        except Exception as e:
            print(f"Error processing {image_path} with {ocr_model_name}: {str(e)}")
            return None

    def evaluate_model(self, dataset_path, ocr_model_name):
        """
        Evaluate YOLO + OCR model performance
        """
        dataset_path = Path(dataset_path)
        results = []
        
        for image_path in tqdm(list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png'))):
            result = self.process_image(image_path, ocr_model_name)
            if result:
                results.append({
                    'image_path': str(image_path),
                    'text': result['text'],
                    'processing_time': result['processing_time'],
                    'num_regions': result['num_regions']
                })
        
        # Calculate metrics
        metrics = {
            'total_images': len(results),
            'avg_processing_time': np.mean([r['processing_time'] for r in results]),
            'avg_regions_per_image': np.mean([r['num_regions'] for r in results]),
            'success_rate': len(results) / len(list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.png')))
        }
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f'yolo_ocr_{ocr_model_name}_results.csv', index=False)
        
        return metrics

def main():
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    yolo_ocr = YOLOOCR()
    
    # Evaluate each OCR model with YOLO detection
    for model_name in ['tesseract', 'paddle', 'easyocr']:
        print(f"\nEvaluating YOLO + {model_name}...")
        metrics = yolo_ocr.evaluate_model(dataset_path, model_name)
        print(f"Results for YOLO + {model_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 