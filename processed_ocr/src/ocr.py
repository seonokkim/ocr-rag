import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import os
from common.config.paths import IMAGE_DIR, LABEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedOCR:
    def __init__(self, lang='kor'):
        """
        Initialize ProcessedOCR with specified language.
        
        Args:
            lang (str): Language for OCR (default: 'kor')
        """
        self.lang = lang
        
    def preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
        
    def perform_ocr(self, image_path):
        """
        Perform OCR on the preprocessed image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Extracted text from the image
        """
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Convert back to PIL Image for pytesseract
            processed_pil = Image.fromarray(processed_image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_pil, lang=self.lang)
            
            logger.info(f"Successfully performed OCR on {image_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            raise
            
    def process_directory(self):
        """
        Process all images in the configured image directory.
        
        Returns:
            dict: Dictionary mapping image filenames to their OCR results
        """
        results = {}
        
        for filename in os.listdir(IMAGE_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(IMAGE_DIR, filename)
                try:
                    text = self.perform_ocr(image_path)
                    results[filename] = text
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    results[filename] = None
                    
        return results 