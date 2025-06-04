#!/bin/bash

# Define models to evaluate
models=("tesseract" "easyocr" "paddleocr" "yolo_ocr" "azure_read" "azure_layout" "azure_prebuilt_read")

echo "Starting model evaluations..."

# Evaluate each model one by one
for model in "${models[@]}"; do
    echo "\n=== Evaluating $model ==="
    python src/run_evaluation.py \
        --models $model \
        --test_data_limit 1
    echo "=== Evaluation of $model complete ===\n"
done

echo "All model evaluations completed!"
