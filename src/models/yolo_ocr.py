import numpy as np
from ultralytics import YOLO
import easyocr
from .base import BaseOCRModel
import logging
import cv2

class YOLOOCRModel(BaseOCRModel):
    """YOLO 기반 텍스트 검출 + EasyOCR 인식기 조합 모델"""
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize YOLO OCR model.
        
        Args:
            model_path (str): Path to YOLO model weights
        """
        self.model = YOLO(model_path)
        
        # EasyOCR 초기화
        try:
            self.ocr = easyocr.Reader(['ko'], gpu=True)
            logging.info("EasyOCR initialized with Korean language support")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {str(e)}")
            raise
        
        # 바운딩 박스 필터링을 위한 설정
        self.min_box_size = 20  # 최소 박스 크기
        self.max_box_size = 500  # 최대 박스 크기
        self.confidence_threshold = 0.5  # YOLO 검출 신뢰도 임계값
        self.ocr_confidence_threshold = 0.5  # OCR 신뢰도 임계값
        
        # 이미지 전처리 설정
        self.min_crop_size = 10  # 최소 크롭 크기
        self.padding = 5  # 바운딩 박스 패딩

    def __call__(self, image: np.ndarray) -> list:
        """Perform OCR on the input image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            list: List of (text, bbox) tuples
        """
        # Run YOLO detection
        results = self.model(image)
        
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                conf = box.conf[0].cpu().numpy()
                
                # Get class name
                cls = box.cls[0].cpu().numpy()
                text = result.names[int(cls)]
                
                # Add to predictions if confidence is high enough
                if conf > 0.5:
                    predictions.append((text, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
        
        return predictions

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 이미지 크기 확인
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # 이미지 정규화
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        return image

    def _filter_boxes(self, boxes, confidences):
        """바운딩 박스 필터링"""
        filtered_boxes = []
        filtered_confidences = []
        
        for box, conf in zip(boxes, confidences):
            if conf < self.confidence_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            
            # 박스 크기 필터링
            if width < self.min_box_size or height < self.min_box_size or \
               width > self.max_box_size or height > self.max_box_size:
                continue
                
            filtered_boxes.append(box)
            filtered_confidences.append(conf)
            
        return filtered_boxes, filtered_confidences

    def _process_crop(self, image, box):
        """크롭된 이미지 처리"""
        x1, y1, x2, y2 = map(int, box)
        
        # 패딩 추가
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(image.shape[1], x2 + self.padding)
        y2 = min(image.shape[0], y2 + self.padding)
        
        # 이미지 크롭
        crop = image[y1:y2, x1:x2]
        
        # 크롭된 이미지가 너무 작은 경우 스킵
        if crop.size == 0 or crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
            return None, None
            
        return crop, (x1, y1, x2, y2)

    def predict(self, processed_image: np.ndarray):
        """텍스트 검출 및 인식"""
        try:
            # 1. YOLO로 텍스트 영역 검출
            results = self.model(processed_image)
            
            # 검출 결과 추출
            predictions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence score
                    conf = box.conf[0].cpu().numpy()
                    
                    # Get class name
                    cls = box.cls[0].cpu().numpy()
                    text = result.names[int(cls)]
                    
                    # Add to predictions if confidence is high enough
                    if conf > 0.5:
                        predictions.append((text, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error in YOLO OCR prediction: {str(e)}")
            return []

    def postprocess(self, prediction_result):
        """후처리"""
        if not prediction_result:
            return []
            
        # 중복 제거 및 정렬
        unique_predictions = []
        seen_boxes = set()
        
        for text, box in prediction_result:
            box_tuple = tuple(map(tuple, box))
            if box_tuple not in seen_boxes:
                seen_boxes.add(box_tuple)
                unique_predictions.append((text, box))
        
        return unique_predictions 