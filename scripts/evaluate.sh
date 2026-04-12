#!/bin/bash
# Evaluate YOLOv5 model

echo "============================================================"
echo "Starting YOLOv5 Evaluation"
echo "============================================================"

# Default parameters
WEIGHTS=${1:-runs/train/exp/weights/best.pt}
DATA=${2:-/workspace/yolov5-implementation/data/dataset.yaml}
DEVICE=${3:-0}

echo "Weights: $WEIGHTS"
echo "Data: $DATA"
echo "Device: $DEVICE"
echo ""

# Run evaluation
cd /workspace/yolov5-implementation
python src/evaluation/evaluate.py \
    --weights $WEIGHTS \
    --data $DATA \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Evaluation Complete!"
echo "============================================================"
