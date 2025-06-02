import os
import time
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from Levenshtein import distance as levenshtein_distance
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse

# Attempt to import PaddleOCR module
try:
    from models import EasyOCRModel, PaddleOCRModel, YOLOOCRModel
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"\nWarning: Could not import PaddleOCR module - {str(e)}")
    print("PaddleOCR model will be excluded from evaluation.")
    from models import EasyOCRModel, YOLOOCRModel
    PADDLEOCR_AVAILABLE = False

from preprocessing import (
    SharpeningPreprocessor,
    DenoisingPreprocessor,
    BinarizationPreprocessor
)
from utils.evaluation_utils import (
    create_evaluation_config,
    save_evaluation_results,
    load_all_results,
    generate_performance_report
)
from models import YOLOOCRModel, TesseractModel

def bbox_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def convert_bbox_to_x1y1x2y2(bbox, fmt='easyocr'):
    """Convert bounding box format to [x1, y1, x2, y2]."""
    if fmt == 'easyocr':
        # EasyOCR format is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] or similar quadrilateral
        # We need the min/max x and y
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif fmt == 'json':
        # JSON format is [x, y, width, height]
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    else:
        raise ValueError(f"Unknown bounding box format: {fmt}")

def load_test_data(config: Dict[str, Any]) -> tuple:
    """Load test data (including subfolders, loads all annotations)."""
    images = []
    ground_truth_annotations = [] # Store list of all annotations instead of list of texts
    
    data_dir = config['data']['test_dir']
    label_dir = config['data']['label_dir']

    # Match images and label files (explore subfolders)
    for root, _, files in os.walk(Path(data_dir) / 'images'): # Explore from images/ subfolder
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                
                # Get relative path from images/5350224/
                relative_img_path = img_path.relative_to(Path(data_dir) / 'images' / '5350224')
                
                # Construct corresponding label path
                json_path = Path(label_dir) / 'labels' / '5350224' / relative_img_path.parent / file.replace('.jpg', '.json')

                if json_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                        
                        # Store list of all annotations
                        ground_truth_annotations.append(label_data.get('annotations', []))
                        images.append(img)
                        
                        # Return after finding first valid image
                        return images, ground_truth_annotations
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: No corresponding JSON file found for {img_path.name} at {json_path}")
    
    return images, ground_truth_annotations

def get_preprocessing_pipeline(steps: List[str]) -> List[Any]:
    """Creates a preprocessing pipeline."""
    pipeline = []
    for step in steps:
        if step == 'sharpening':
            pipeline.append(SharpeningPreprocessor())
        elif step == 'denoising':
            pipeline.append(DenoisingPreprocessor())
        elif step == 'binarization':
            pipeline.append(BinarizationPreprocessor())
    return pipeline

def evaluate_combination(
    model,
    images: List[np.ndarray],
    ground_truth: List[List[Dict[str, Any]]],
    preprocessing_steps: List[str]
) -> Dict[str, Any]:
    """Evaluates a specific model and preprocessing combination."""
    start_time = time.time()
    all_predictions = []
    
    # Create preprocessing pipeline
    pipeline = get_preprocessing_pipeline(preprocessing_steps)
    
    for img in images:
        # Apply preprocessing
        processed_img = img
        for preprocessor in pipeline:
            processed_img = preprocessor(processed_img)
        
        # Perform prediction
        pred = model(processed_img)
        all_predictions.append(pred)
    
    inference_time = time.time() - start_time
    
    # Calculate accuracy (bounding box based matching)
    total_items = 0
    matched_items = 0
    total_chars = 0
    matched_chars = 0
    
    # Counters for additional metrics
    type_metrics = {}  # Accuracy by text type
    region_metrics = {}  # Accuracy by location (top/middle/bottom)
    length_metrics = {}  # Accuracy by text length (short/medium/long)
    size_metrics = {}  # Accuracy by bounding box size (small/medium/large)
    
    # Variables for full text comparison
    total_levenshtein_distance = 0
    total_gt_length = 0
    rouge = Rouge()
    total_rouge_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    total_rouge_count = 0
    
    # Variables for BLEU score
    total_bleu_score = 0
    total_bleu_count = 0
    smoothing = SmoothingFunction().method1
    
    for pred_list, gt_annotations in zip(all_predictions, ground_truth):
        # Extract text and bounding boxes from Ground Truth annotations
        gt_texts = [anno.get('annotation.text', '') for anno in gt_annotations]
        gt_boxes = [convert_bbox_to_x1y1x2y2(anno.get('annotation.bbox', []), fmt='json') 
                   for anno in gt_annotations]
        gt_types = [anno.get('annotation.ttype', 'unknown') for anno in gt_annotations]
        
        # Generate strings for full text comparison
        gt_full_text = ' '.join(gt_texts)
        pred_full_text = ' '.join([text for text, _ in pred_list])
        
        # Calculate Levenshtein distance
        if gt_full_text and pred_full_text:
            total_levenshtein_distance += levenshtein_distance(gt_full_text, pred_full_text)
            total_gt_length += len(gt_full_text)
        
        # Calculate ROUGE score
        try:
            if gt_full_text and pred_full_text:
                rouge_scores = rouge.get_scores(pred_full_text, gt_full_text)[0]
                for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                    total_rouge_scores[metric] += rouge_scores[metric]['f']
                total_rouge_count += 1
        except Exception as e:
            print(f"Warning: ROUGE score calculation failed: {str(e)}")
        
        # Calculate BLEU score
        try:
            if gt_full_text and pred_full_text:
                bleu_score = sentence_bleu([gt_full_text.split()], pred_full_text.split(), 
                                         smoothing_function=smoothing)
                total_bleu_score += bleu_score
                total_bleu_count += 1
        except Exception as e:
            print(f"Warning: BLEU score calculation failed: {str(e)}")
        
        # Match predictions with ground truth
        matched_gt_indices = set()
        for pred_text, pred_box in pred_list:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_text, gt_box) in enumerate(zip(gt_texts, gt_boxes)):
                if gt_idx in matched_gt_indices:
                    continue
                    
                iou = bbox_iou(gt_box, pred_box)
                if iou > best_iou and iou > 0.5:  # IoU 임계값
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                matched_items += 1
                
                # 문자 단위 정확도 계산
                gt_text = gt_texts[best_gt_idx]
                total_chars += len(gt_text)
                matched_chars += sum(1 for c1, c2 in zip(gt_text, pred_text) if c1 == c2)
                
                # 추가 메트릭 업데이트
                gt_type = gt_types[best_gt_idx]
                type_metrics[gt_type] = type_metrics.get(gt_type, 0) + 1
                
                # 위치 기반 메트릭
                y_center = (gt_boxes[best_gt_idx][1] + gt_boxes[best_gt_idx][3]) / 2
                img_height = img.shape[0]
                if y_center < img_height / 3:
                    region = 'top'
                elif y_center < 2 * img_height / 3:
                    region = 'middle'
                else:
                    region = 'bottom'
                region_metrics[region] = region_metrics.get(region, 0) + 1
                
                # 길이 기반 메트릭
                text_length = len(gt_text)
                if text_length < 5:
                    length = 'short'
                elif text_length < 10:
                    length = 'medium'
                else:
                    length = 'long'
                length_metrics[length] = length_metrics.get(length, 0) + 1
                
                # 크기 기반 메트릭
                box_width = gt_boxes[best_gt_idx][2] - gt_boxes[best_gt_idx][0]
                box_height = gt_boxes[best_gt_idx][3] - gt_boxes[best_gt_idx][1]
                box_size = box_width * box_height
                if box_size < 1000:
                    size = 'small'
                elif box_size < 5000:
                    size = 'medium'
                else:
                    size = 'large'
                size_metrics[size] = size_metrics.get(size, 0) + 1
        
        total_items += len(gt_texts)
    
    # Calculate final metrics
    metrics = {
        'item_accuracy': matched_items / total_items if total_items > 0 else 0,
        'char_accuracy': matched_chars / total_chars if total_chars > 0 else 0,
        'inference_time': inference_time,
    }
    
    # Add type-specific metrics
    for ttype in type_metrics:
        metrics[f'type_{ttype}'] = type_metrics[ttype] / total_items if total_items > 0 else 0
    
    # Add region-specific metrics
    for region in region_metrics:
        metrics[f'region_{region}'] = region_metrics[region] / total_items if total_items > 0 else 0
    
    # Add length-specific metrics
    for length in length_metrics:
        metrics[f'length_{length}'] = length_metrics[length] / total_items if total_items > 0 else 0
    
    # Add size-specific metrics
    for size in size_metrics:
        metrics[f'size_{size}'] = size_metrics[size] / total_items if total_items > 0 else 0
    
    # Add text similarity metrics
    if total_gt_length > 0:
        metrics['normalized_levenshtein'] = 1 - (total_levenshtein_distance / total_gt_length)
    else:
        metrics['normalized_levenshtein'] = 0
    
    if total_bleu_count > 0:
        metrics['bleu_score'] = total_bleu_score / total_bleu_count
    else:
        metrics['bleu_score'] = 0
    
    if total_rouge_count > 0:
        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            metrics[f'rouge_{metric}'] = total_rouge_scores[metric] / total_rouge_count
    else:
        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            metrics[f'rouge_{metric}'] = 0
    
    return {
        'metrics': metrics,
        'predictions': all_predictions
    }

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None, help='List of models to evaluate (tesseract, easyocr, yolo, paddleocr)')
    parser.add_argument('--test_data_limit', type=int, default=None, help='Number of test data to use')
    args = parser.parse_args()

    # Load configuration
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Use argument if provided, else config
    test_data_limit = args.test_data_limit if args.test_data_limit is not None else config.get('evaluation', {}).get('test_data_limit', 1)
    # Patch config for load_test_data
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['test_data_limit'] = test_data_limit

    # Determine models to evaluate
    model_names = args.models if args.models is not None else config.get('evaluation', {}).get('models', ['tesseract', 'easyocr', 'yolo', 'paddleocr'])

    # Load test data
    test_images, ground_truth = load_test_data(config)
    print(f"Loaded {len(test_images)} test image{'s' if len(test_images) > 1 else ''} for evaluation.")

    # Model mapping
    model_map = {
        'tesseract': TesseractModel,
        'easyocr': EasyOCRModel,
        'yolo': YOLOOCRModel,
        'paddleocr': PaddleOCRModel if PADDLEOCR_AVAILABLE else None
    }

    # Initialize models
    evaluation_targets = {}
    for name in model_names:
        if name == 'paddleocr' and not PADDLEOCR_AVAILABLE:
            print('PaddleOCR is not available and will be skipped.')
            continue
        model_cls = model_map.get(name)
        if model_cls is not None:
            try:
                evaluation_targets[f'base_{name}'] = model_cls()
            except Exception as e:
                print(f"Warning: Could not initialize {name}: {e}")
        else:
            print(f"Warning: Unknown model name '{name}' - skipping.")

    if not evaluation_targets:
        print("No models available for evaluation. Exiting script.")
        return

    # Generate preprocessing combinations
    preprocessing_combinations = [
        [],  # No preprocessing
        ['sharpening'],
        ['denoising'],
        ['binarization'],
        ['sharpening', 'denoising'],
        ['sharpening', 'binarization'],
        ['denoising', 'binarization'],
        ['sharpening', 'denoising', 'binarization']
    ]

    # Perform evaluation for all model and preprocessing combinations
    for target_name, model in evaluation_targets.items():
        print(f"\nEvaluating model: {target_name}")
        
        # Skip certain preprocessing for specific models if needed
        model_preprocessing_combinations = preprocessing_combinations
        if 'yolo' in target_name.lower():
            # YOLO might not need some preprocessing
            model_preprocessing_combinations = [[], ['sharpening']]
            
        for preprocess_steps in model_preprocessing_combinations:
            print(f"Preprocessing steps: {preprocess_steps if preprocess_steps else 'None'}")
            # Create evaluation config
            eval_config = create_evaluation_config(
                model_name=target_name,
                preprocessing_steps=preprocess_steps,
                use_gpu=config['hardware']['use_gpu']
            )
            # Perform evaluation
            results = evaluate_combination(
                model=model,
                images=test_images,
                ground_truth=ground_truth,
                preprocessing_steps=preprocess_steps
            )
            # Save results
            save_evaluation_results(results, eval_config)
            # Print intermediate results
            print(f"Item Accuracy: {results['metrics']['item_accuracy']:.4f}")
            print(f"Character Accuracy: {results['metrics']['char_accuracy']:.4f}")
            print(f"Inference Time: {results['metrics']['inference_time']:.2f} seconds")

    # Analyze overall results and generate report
    print("\nAnalyzing overall results...")
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("Evaluation complete! Results can be found in the results directory.")

if __name__ == "__main__":
    main() 