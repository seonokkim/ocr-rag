import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Configure matplotlib for Korean font
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False # Allow negative signs in plots

from direct_ocr.src.ocr import DirectOCR
from processed_ocr.src.ocr import ProcessedOCR
from common.config.paths import IMAGE_DIR, LABEL_DIR

def load_test_data(image_dir, label_dir, max_samples=100):
    """테스트 데이터 로드"""
    image_files = []
    label_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                label_path = os.path.join(label_dir, file.rsplit('.', 1)[0] + '.json')
                
                if os.path.exists(label_path):
                    image_files.append(image_path)
                    label_files.append(label_path)
                    
                    if len(image_files) >= max_samples:
                        break
    
    return image_files, label_files

def evaluate_ocr(ocr_class, image_files, label_files):
    """OCR 성능 평가"""
    results = []
    ocr = ocr_class()
    
    for img_path, label_path in tqdm(zip(image_files, label_files), total=len(image_files)):
        # 실제 텍스트 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            ground_truth = label_data.get('text', '')
        
        # OCR 수행 및 시간 측정
        start_time = time.time()
        result = ocr.perform_ocr(img_path)
        processing_time = time.time() - start_time
        
        # 결과 저장
        results.append({
            'image': img_path,
            'ground_truth': ground_truth,
            'predicted': result,
            'processing_time': processing_time
        })
    
    return pd.DataFrame(results)

def calculate_metrics(results):
    """OCR 성능 지표 계산"""
    # 정확도 계산 (정확히 일치하는 경우)
    exact_matches = (results['ground_truth'] == results['predicted']).mean()
    
    # 평균 처리 시간
    avg_time = results['processing_time'].mean()
    
    return {
        'exact_match_rate': exact_matches,
        'avg_processing_time': avg_time
    }

def analyze_errors(results):
    """OCR 오류 분석"""
    # 오류가 있는 케이스만 필터링
    errors = results[results['ground_truth'] != results['predicted']]
    
    # 오류 케이스 출력
    for _, row in errors.iterrows():
        print(f"\n이미지: {os.path.basename(row['image'])}")
        print(f"실제 텍스트: {row['ground_truth']}")
        print(f"예측 텍스트: {row['predicted']}")
        print("-" * 50)

def main():
    # 테스트 데이터 로드
    print("테스트 데이터 로드 중...")
    image_files, label_files = load_test_data(IMAGE_DIR, LABEL_DIR)
    print(f"로드된 테스트 데이터: {len(image_files)}개")

    # DirectOCR 평가
    print("\nDirectOCR 평가 중...")
    direct_results = evaluate_ocr(DirectOCR, image_files, label_files)

    # ProcessedOCR 평가
    print("\nProcessedOCR 평가 중...")
    processed_results = evaluate_ocr(ProcessedOCR, image_files, label_files)

    # 성능 지표 계산
    direct_metrics = calculate_metrics(direct_results)
    processed_metrics = calculate_metrics(processed_results)

    # 결과 출력
    print("\nDirectOCR 성능:")
    print(f"정확도: {direct_metrics['exact_match_rate']:.2%}")
    print(f"평균 처리 시간: {direct_metrics['avg_processing_time']:.2f}초")

    print("\nProcessedOCR 성능:")
    print(f"정확도: {processed_metrics['exact_match_rate']:.2%}")
    print(f"평균 처리 시간: {processed_metrics['avg_processing_time']:.2f}초")

    # 성능 비교 시각화
    plt.figure(figsize=(12, 5))

    # 정확도 비교
    plt.subplot(1, 2, 1)
    accuracy_data = pd.DataFrame({
        'OCR Type': ['DirectOCR', 'ProcessedOCR'],
        'Accuracy': [direct_metrics['exact_match_rate'], processed_metrics['exact_match_rate']]
    })
    sns.barplot(data=accuracy_data, x='OCR Type', y='Accuracy')
    plt.title('OCR 정확도 비교')
    plt.ylim(0, 1)

    # 처리 시간 비교
    plt.subplot(1, 2, 2)
    time_data = pd.DataFrame({
        'OCR Type': ['DirectOCR', 'ProcessedOCR'],
        'Processing Time': [direct_metrics['avg_processing_time'], processed_metrics['avg_processing_time']]
    })
    sns.barplot(data=time_data, x='OCR Type', y='Processing Time')
    plt.title('평균 처리 시간 비교 (초)')

    plt.tight_layout()
    plt.savefig('ocr_performance_comparison.png')
    print("\n성능 비교 그래프가 'ocr_performance_comparison.png'로 저장되었습니다.")

    # 오류 분석
    print("\nDirectOCR 오류 분석:")
    analyze_errors(direct_results)

    print("\nProcessedOCR 오류 분석:")
    analyze_errors(processed_results)

if __name__ == "__main__":
    main() 