import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'sample-data')

# Image and label paths
IMAGE_DIR = os.path.join(DATA_DIR, 'images', '5350224', '1994')
LABEL_DIR = os.path.join(DATA_DIR, 'labels', '5350224', '1994')

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True) 