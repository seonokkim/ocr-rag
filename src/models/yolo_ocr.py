import numpy as np
from ultralytics import YOLO
import easyocr
from .base import BaseOCRModel
import logging
import cv2

class YOLOOCRModel(BaseOCRModel):
    """YOLO 기반 텍스트 검출 + EasyOCR 인식기 조합 모델"""
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml", 
                 yolo_model_path: str = "yolov8n.pt"):
        super().__init__(config_path)
        self.device = "cuda" if use_gpu else "cpu"
        
        # YOLO 모델 초기화
        try:
            self.yolo = YOLO(yolo_model_path)
            logging.info(f"YOLO model loaded from {yolo_model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            raise
        
        # EasyOCR 초기화
        try:
            self.ocr = easyocr.Reader(['ko'], gpu=use_gpu)
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
            results = self.yolo.predict(processed_image, device=self.device, verbose=False)
            
            # 검출 결과 추출
            boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
            confidences = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else []
            
            # 바운딩 박스 필터링
            filtered_boxes, filtered_confidences = self._filter_boxes(boxes, confidences)
            
            # 2. 각 박스별로 EasyOCR 인식
            predictions = []
            for box, conf in zip(filtered_boxes, filtered_confidences):
                # 이미지 크롭 및 처리
                crop, (x1, y1, x2, y2) = self._process_crop(processed_image, box)
                if crop is None:
                    continue
                
                # OCR 수행
                try:
                    ocr_results = self.ocr.readtext(crop)
                    if ocr_results:
                        # 가장 신뢰도 높은 결과 선택
                        ocr_results.sort(key=lambda x: x[2], reverse=True)
                        text, ocr_conf = ocr_results[0][1], ocr_results[0][2]
                        
                        # OCR 신뢰도 필터링
                        if ocr_conf > self.ocr_confidence_threshold:
                            predictions.append((text, [float(x1), float(y1), float(x2), float(y2)]))
                except Exception as e:
                    logging.warning(f"OCR failed for a crop: {str(e)}")
                    continue
            
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
            box_tuple = tuple(box)
            if box_tuple not in seen_boxes:
                seen_boxes.add(box_tuple)
                unique_predictions.append((text, box))
        
        return unique_predictions 