import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def analyze_dataset(dataset_path):
    """
    Analyze the dataset structure and characteristics
    """
    dataset_path = Path(dataset_path)
    
    # Initialize metrics
    metrics = {
        'total_images': 0,
        'image_sizes': [],
        'aspect_ratios': [],
        'file_types': {},
        'avg_brightness': [],
        'avg_contrast': []
    }
    
    # Walk through the dataset
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                metrics['total_images'] += 1
                file_path = os.path.join(root, file)
                
                # Get file type
                ext = os.path.splitext(file)[1].lower()
                metrics['file_types'][ext] = metrics['file_types'].get(ext, 0) + 1
                
                # Read image
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        # Get image size
                        height, width = img.shape[:2]
                        metrics['image_sizes'].append((width, height))
                        metrics['aspect_ratios'].append(width / height)
                        
                        # Calculate brightness and contrast
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        metrics['avg_brightness'].append(np.mean(gray))
                        metrics['avg_contrast'].append(np.std(gray))
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Calculate statistics
    stats = {
        'Total Images': metrics['total_images'],
        'File Types': metrics['file_types'],
        'Average Image Size': np.mean(metrics['image_sizes'], axis=0),
        'Average Aspect Ratio': np.mean(metrics['aspect_ratios']),
        'Average Brightness': np.mean(metrics['avg_brightness']),
        'Average Contrast': np.mean(metrics['avg_contrast'])
    }
    
    return stats

def plot_dataset_characteristics(metrics):
    """
    Plot various characteristics of the dataset
    """
    plt.figure(figsize=(15, 10))
    
    # Plot image sizes
    plt.subplot(2, 2, 1)
    sizes = np.array(metrics['image_sizes'])
    plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.5)
    plt.title('Image Sizes Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Plot aspect ratios
    plt.subplot(2, 2, 2)
    plt.hist(metrics['aspect_ratios'], bins=30)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    
    # Plot brightness
    plt.subplot(2, 2, 3)
    plt.hist(metrics['avg_brightness'], bins=30)
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Count')
    
    # Plot contrast
    plt.subplot(2, 2, 4)
    plt.hist(metrics['avg_contrast'], bins=30)
    plt.title('Contrast Distribution')
    plt.xlabel('Contrast')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.close()

if __name__ == "__main__":
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    stats = analyze_dataset(dataset_path)
    
    # Print statistics
    print("\nDataset Analysis Results:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save results to CSV
    pd.DataFrame([stats]).to_csv('dataset_analysis.csv', index=False) 