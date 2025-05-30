import pytesseract
from PIL import Image
import logging
import os
from common.config.paths import IMAGE_DIR, LABEL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectOCR:
    def __init__(self, lang='kor'):
        """
        Initialize DirectOCR with specified language.
        
        Args:
            lang (str): Language for OCR (default: 'kor')
        """
        self.lang = lang
        
    def perform_ocr(self, image_path):
        """
        Perform OCR directly on the input image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Extracted text from the image
        """
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang)
            
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