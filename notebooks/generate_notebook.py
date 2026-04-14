"""
Generate Jupyter notebook with embedded detection images.
This script creates a notebook with actual image data embedded as base64.
"""

import json
import base64
import cv2
import random
import numpy as np
from pathlib import Path
import os

# Configuration
DATA_DIR = '/data/memory/coco'
WEIGHTS_PATH = '/workspace/yolov5-implementation/runs/train/exp_s/weights/best.pt'
TEST_IMAGES_DIR = f'{DATA_DIR}/images/val2017'
OUTPUT_DIR = '/workspace/yolov5-implementation/runs/evaluate/visualization'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'basebaseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color map
np.random.seed(42)
CLASS_COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

random.seed(42)
np.random.seed(42)

print("Loading model...")
from ultralytics import YOLO
model = YOLO(WEIGHTS_PATH)
model.eval()
print("Model loaded!")

# Get random images
image_files = list(Path(TEST_IMAGES_DIR).glob('*.jpg'))
sample_images = random.sample(image_files, 4)

print(f"Processing {len(sample_images)} images...")

def visualize_detections(image_path, model, conf_threshold=0.25):
    img = cv2.imread(str(image_path))
    results = model(img, conf=conf_threshold, iou=0.45, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())
            detections.append({
                'box': xyxy.tolist(),
                'confidence': float(conf),
                'class_id': cls,
                'class_name': COCO_CLASSES[cls]
            })
    
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det['box']]
        class_id = det['class_id']
        class_name = det['class_name']
        conf = det['confidence']
        color = tuple(CLASS_COLORS[class_id % 80].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{class_name}: {conf:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img, detections

# Process images and collect results
all_detections = []
results_data = []
image_outputs = []

for i, img_path in enumerate(sample_images, 1):
    print(f"Processing {i}/4: {img_path.name}")
    annotated_img, detections = visualize_detections(img_path, model)
    
    # Save annotated image
    output_path = Path(OUTPUT_DIR) / f'{img_path.stem}_detections.png'
    cv2.imwrite(str(output_path), annotated_img)
    
    # Convert to base64 for embedding
    _, buffer = cv2.imencode('.png', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    image_outputs.append({
        'filename': img_path.name,
        'detections': detections,
        'base64': img_base64
    })
    
    all_detections.extend(detections)
    for det in detections:
        results_data.append({
            'image_name': img_path.name,
            'class_id': det['class_id'],
            'class_name': det['class_name'],
            'confidence': det['confidence'],
            'x1': det['box'][0],
            'y1': det['box'][1],
            'x2': det['box'][2],
            'y2': det['box'][3]
        })

# Save CSV
import pandas as pd
df = pd.DataFrame(results_data)
csv_path = Path(OUTPUT_DIR) / 'detection_results.csv'
df.to_csv(str(csv_path), index=False)
print(f"CSV saved to: {csv_path}")

# Build notebook with embedded images
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# YOLOv5 Evaluation Visualization\n",
                "\n",
                "This notebook visualizes the trained YOLOv5 model's performance on 4 random images from the COCO validation split.\n",
                "\n",
                "**Status**: ✅ Completed - Uses `best.pt` checkpoint"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Results Summary\n",
                f"\n",
                "- **Total images processed**: 4\n",
                f"- **Total detections**: {len(all_detections)}\n",
                f"- **Average detections per image**: {len(all_detections)/4:.2f}\n",
                "\n",
                "Below are the detection results with bounding boxes and labels embedded directly in this notebook."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add image cells
for i, img_data in enumerate(image_outputs, 1):
    detections_summary = ", ".join([f"{d['class_name']} ({d['confidence']:.2f})" for d in img_data['detections']])
    
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"### Image {i}: {img_data['filename']}\n",
            f"\n",
            f"**Detections**: {len(img_data['detections'])} objects\n",
            f"- {detections_summary}"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": i,
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "image/png": img_data['base64']
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            f"# Display detection results for {img_data['filename']}\n",
            f"from IPython.display import Image, display\n",
            f"display(Image(filename='{img_data['filename'].replace('.jpg', '_detections.png')}'))"
        ]
    })

# Add summary cell
class_counts = {}
for det in all_detections:
    class_name = det['class_name']
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Detection Summary\n",
        "\n",
        "| Class | Count |\n",
        "|-------|-------|\n"
    ]
})

for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    notebook["cells"][-1]["source"].append(f"| {class_name} | {count} |\n")

notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Output Files\n",
        "\n",
        "All visualizations are saved to: `/workspace/yolov5-implementation/runs/evaluate/visualization/`\n",
        "- Detection images: `{img}_detections.png`\n",
        "- Detection results: `detection_results.csv`"
    ]
})

# Save notebook
notebook_path = Path('/workspace/yolov5-implementation/notebooks/evaluation_visualization.ipynb')
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"\n✅ Notebook saved to: {notebook_path}")
print(f"✅ Total cells: {len(notebook['cells'])}")
print(f"✅ Embedded images: {len(image_outputs)}")
