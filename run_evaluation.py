import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from analyze_dataset import analyze_dataset, plot_dataset_characteristics
from direct_ocr import DirectOCR
from preprocessed_ocr import PreprocessedOCR
from yolo_ocr import YOLOOCR

def create_evaluation_directory():
    """
    Create a timestamped directory for storing evaluation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(f"evaluation_results_{timestamp}")
    eval_dir.mkdir(exist_ok=True)
    return eval_dir

def run_all_evaluations(dataset_path):
    """
    Run all evaluations and generate comprehensive report
    """
    # Create evaluation directory
    eval_dir = create_evaluation_directory()
    print(f"Storing results in: {eval_dir}")
    
    print("1. Analyzing dataset...")
    dataset_stats = analyze_dataset(dataset_path)
    plot_dataset_characteristics(dataset_stats)
    plt.savefig(eval_dir / 'dataset_analysis.png')
    plt.close()
    
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
    generate_report(direct_results, preprocessed_results, yolo_results, eval_dir)

def generate_report(direct_results, preprocessed_results, yolo_results, eval_dir):
    """
    Generate comprehensive evaluation report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
    plt.savefig(eval_dir / 'evaluation_report.png')
    plt.close()
    
    # Prepare detailed results DataFrame
    results_data = []
    
    # Add direct OCR results
    for model in models:
        results_data.append({
            'Timestamp': timestamp,
            'Model': model,
            'Method': 'Direct',
            'Processing Time': direct_results[model]['avg_processing_time'],
            'Success Rate': direct_results[model]['success_rate'],
            'Accuracy': direct_results[model].get('accuracy', None),
            'Precision': direct_results[model].get('precision', None),
            'Recall': direct_results[model].get('recall', None)
        })
    
    # Add YOLO OCR results
    for model in models:
        results_data.append({
            'Timestamp': timestamp,
            'Model': model,
            'Method': 'YOLO',
            'Processing Time': yolo_results[model]['avg_processing_time'],
            'Success Rate': yolo_results[model]['success_rate'],
            'Accuracy': yolo_results[model].get('accuracy', None),
            'Precision': yolo_results[model].get('precision', None),
            'Recall': yolo_results[model].get('recall', None)
        })
    
    # Add preprocessed OCR results
    for model in models:
        for method in methods:
            results_data.append({
                'Timestamp': timestamp,
                'Model': model,
                'Method': f'Preprocessed_{method}',
                'Processing Time': preprocessed_results[model][method]['avg_processing_time'],
                'Success Rate': preprocessed_results[model][method]['success_rate'],
                'Accuracy': preprocessed_results[model][method].get('accuracy', None),
                'Precision': preprocessed_results[model][method].get('precision', None),
                'Recall': preprocessed_results[model][method].get('recall', None)
            })
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(eval_dir / 'evaluation_results.csv', index=False)
    
    # Generate and save summary report
    with open(eval_dir / 'evaluation_summary.txt', 'w') as f:
        f.write(f"Evaluation Summary (Generated at {timestamp})\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Direct OCR Results:\n")
        for model, metrics in direct_results.items():
            f.write(f"\n{model}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
        
        f.write("\nYOLO + OCR Results:\n")
        for model, metrics in yolo_results.items():
            f.write(f"\n{model}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
        
        f.write("\nBest performing models:\n")
        best_direct = max(direct_results.items(), key=lambda x: x[1]['success_rate'])
        best_yolo = max(yolo_results.items(), key=lambda x: x[1]['success_rate'])
        f.write(f"Direct OCR: {best_direct[0]} ({best_direct[1]['success_rate']:.2%})\n")
        f.write(f"YOLO + OCR: {best_yolo[0]} ({best_yolo[1]['success_rate']:.2%})\n")
        
        # Add preprocessing method comparison
        f.write("\nPreprocessing Method Comparison:\n")
        for model in models:
            f.write(f"\n{model}:\n")
            for method in methods:
                success_rate = preprocessed_results[model][method]['success_rate']
                f.write(f"  {method}: {success_rate:.2%}\n")
    
    print(f"\nEvaluation results have been saved to: {eval_dir}")
    print("Files generated:")
    print(f"- evaluation_results.csv: Detailed results with timestamps")
    print(f"- evaluation_report.png: Visual comparison plots")
    print(f"- evaluation_summary.txt: Summary report with best performing models")
    print(f"- dataset_analysis.png: Dataset characteristics analysis")

if __name__ == "__main__":
    dataset_path = r"G:\내 드라이브\2025_Work\ocr\repo\kor-ocr2db\data"
    run_all_evaluations(dataset_path) 