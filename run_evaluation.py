import os
import pandas as pd
import matplotlib.pyplot as plt
from analyze_dataset import analyze_dataset, plot_dataset_characteristics
from direct_ocr import DirectOCR
from preprocessed_ocr import PreprocessedOCR
from yolo_ocr import YOLOOCR

def run_all_evaluations(dataset_path):
    """
    Run all evaluations and generate comprehensive report
    """
    print("1. Analyzing dataset...")
    dataset_stats = analyze_dataset(dataset_path)
    plot_dataset_characteristics(dataset_stats)
    
    print("\n2. Running direct OCR evaluation...")
    direct_ocr = DirectOCR()
    direct_results = {}
    for model_name in ['tesseract', 'paddle', 'easyocr']:
        print(f"\nEvaluating {model_name}...")
        direct_results[model_name] = direct_ocr.evaluate_model(dataset_path, model_name)
    
    print("\n3. Running preprocessed OCR evaluation...")
    preprocessed_ocr = PreprocessedOCR()
    preprocessed_results = {}
    preprocessing_methods = ['grayscale', 'adaptive_threshold', 'gaussian_blur', 'sharpen', 'denoise']
    for model_name in ['tesseract', 'paddle', 'easyocr']:
        preprocessed_results[model_name] = {}
        for method in preprocessing_methods:
            print(f"\nEvaluating {model_name} with {method}...")
            preprocessed_results[model_name][method] = preprocessed_ocr.evaluate_model(
                dataset_path, model_name, method)
    
    print("\n4. Running YOLO + OCR evaluation...")
    yolo_ocr = YOLOOCR()
    yolo_results = {}
    for model_name in ['tesseract', 'paddle', 'easyocr']:
        print(f"\nEvaluating YOLO + {model_name}...")
        yolo_results[model_name] = yolo_ocr.evaluate_model(dataset_path, model_name)
    
    # Generate comprehensive report
    generate_report(direct_results, preprocessed_results, yolo_results)

def generate_report(direct_results, preprocessed_results, yolo_results):
    """
    Generate comprehensive evaluation report
    """
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot processing time comparison
    plt.subplot(2, 2, 1)
    models = list(direct_results.keys())
    direct_times = [direct_results[m]['avg_processing_time'] for m in models]
    yolo_times = [yolo_results[m]['avg_processing_time'] for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], direct_times, width, label='Direct OCR')
    plt.bar([i + width/2 for i in x], yolo_times, width, label='YOLO + OCR')
    plt.xticks(x, models)
    plt.title('Processing Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Average Processing Time (s)')
    plt.legend()
    
    # Plot success rate comparison
    plt.subplot(2, 2, 2)
    direct_success = [direct_results[m]['success_rate'] for m in models]
    yolo_success = [yolo_results[m]['success_rate'] for m in models]
    
    plt.bar([i - width/2 for i in x], direct_success, width, label='Direct OCR')
    plt.bar([i + width/2 for i in x], yolo_success, width, label='YOLO + OCR')
    plt.xticks(x, models)
    plt.title('Success Rate Comparison')
    plt.xlabel('Model')
    plt.ylabel('Success Rate')
    plt.legend()
    
    # Plot preprocessing method comparison for each model
    plt.subplot(2, 2, 3)
    methods = list(preprocessed_results[models[0]].keys())
    model_times = []
    for model in models:
        times = [preprocessed_results[model][m]['avg_processing_time'] for m in methods]
        model_times.append(times)
    
    for i, model in enumerate(models):
        plt.plot(methods, model_times[i], marker='o', label=model)
    
    plt.title('Preprocessing Method Impact on Processing Time')
    plt.xlabel('Preprocessing Method')
    plt.ylabel('Average Processing Time (s)')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_report.png')
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Model': models * 2,
        'Method': ['Direct'] * len(models) + ['YOLO'] * len(models),
        'Processing Time': direct_times + yolo_times,
        'Success Rate': direct_success + yolo_success
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    print("\nDirect OCR Results:")
    for model, metrics in direct_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nYOLO + OCR Results:")
    for model, metrics in yolo_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nBest performing model (by success rate):")
    best_direct = max(direct_results.items(), key=lambda x: x[1]['success_rate'])
    best_yolo = max(yolo_results.items(), key=lambda x: x[1]['success_rate'])
    print(f"Direct OCR: {best_direct[0]} ({best_direct[1]['success_rate']:.2%})")
    print(f"YOLO + OCR: {best_yolo[0]} ({best_yolo[1]['success_rate']:.2%})")

if __name__ == "__main__":
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    run_all_evaluations(dataset_path) 