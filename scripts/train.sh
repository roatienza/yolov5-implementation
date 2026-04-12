#!/bin/bash
# Train YOLOv5 model on COCO dataset

echo "============================================================"
echo "Starting YOLOv5 Training"
echo "============================================================"

# Default parameters
MODEL=${1:-yolov5s}
EPOCHS=${2:-100}
BATCH=${3:-16}
DEVICE=${4:-0}

echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH"
echo "Device: $DEVICE"
echo ""

# Run training
cd /workspace/yolov5-implementation
python src/training/train.py \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch $BATCH \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
