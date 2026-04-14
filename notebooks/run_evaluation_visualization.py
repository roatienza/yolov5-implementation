#!/usr/bin/env python3
"""
YOLOv5 Evaluation Visualization Script
Uses best.pt to run inference on 4 random images from the test split.
Applies bounding boxes and labels for easy visualization.
"""

import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
DATA_DIR = '/data/memory/coco'
WEIGHTS_PATH = '/workspace/yolov5-implementation/runs/train/exp_s/weights/best.pt'
TEST_IMAGES_DIR = f'{DATA_DIR}/images/val2017'  # Use val2017 for evaluation
OUTPUT_DIR = '/workspace/yolov5-implementation/runs/evaluate/visualization'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color map for visualization
np.random.seed(42)
CLASS_COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)


def load_model(weights_path):
    """Load trained YOLOv5 model."""
    print(f"Loading model from: {weights_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Please train the model first or update the path."
        )
    
    # Load model using ultralytics
    from ultralytics import YOLO
    model = YOLO(weights_path)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    return model


def get_random_test_images(images_dir, num_samples=4):
    """Get random sample of test images."""
    image_files = list(Path(images_dir).glob('*.jpg')) + \
                  list(Path(images_dir).glob('*.jpeg')) + \
                  list(Path(images_dir).glob('*.png'))
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Select random sample
    num_samples = min(num_samples, len(image_files))
    sample_images = random.sample(image_files, num_samples)
    
    print(f"Selected {num_samples} random images for visualization")
    return sample_images


def visualize_detections(image_path, model, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run inference on an image and visualize detections.
    
    Args:
        image_path: Path to image file
        model: Trained YOLOv5 model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        annotated_image: Image with bounding boxes drawn
        detections: List of detection results
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_img = img.copy()
    img_height, img_width = img.shape[:2]
    
    # Run inference using ultralytics
    results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # Extract predictions
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
            
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = boxes.conf[i].cpu().numpy()   # confidence
            cls = boxes.cls[i].cpu().numpy()     # class index
            
            x1, y1, x2, y2 = xyxy.astype(int)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f'class_{int(cls)}'
            })
    
    # Draw bounding boxes on image
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']
        
        # Get color for this class
        color = tuple(CLASS_COLORS[class_id % 80].tolist())
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f'{class_name}: {conf:.2f}'
        
        # Calculate label background size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        # Draw label background
        cv2.rectangle(img, (x1, y1 - label_height - 10), 
                     (x1 + label_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img, detections


def plot_detections(image_path, annotated_img, detections, title=None, save_path=None):
    """
    Create a matplotlib plot of the detection results.
    
    Args:
        image_path: Original image path
        annotated_img: Image with bounding boxes
        detections: List of detection results
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    
    if title is None:
        title = f'Detections: {image_path.name}'
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    plt.close()
    
    # Print detection summary
    print(f"\n{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"Total detections: {len(detections)}")
    print(f"{'='*60}")
    
    if len(detections) > 0:
        print(f"{'Class':<20} {'Confidence':<15} {'Box Coordinates'}")
        print("-" * 60)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            print(f"{det['class_name']:<20} {det['confidence']:<15.2f} "
                  f"({x1}, {y1}) - ({x2}, {y2})")
    else:
        print("No detections found for this image.")
    
    print(f"{'='*60}\n")


def run_evaluation():
    """Main evaluation function."""
    print("\n" + "="*60)
    print("YOLOv5 EVALUATION VISUALIZATION")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(WEIGHTS_PATH)
    
    # Get random test images (4 images as requested)
    sample_images = get_random_test_images(TEST_IMAGES_DIR, num_samples=4)
    
    # Process each sample image
    print("\n" + "="*60)
    print("RUNNING INFERENCE ON 4 RANDOM TEST IMAGES")
    print("="*60 + "\n")
    
    all_detections = []
    results_data = []
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n[{i}/{len(sample_images)}] Processing: {img_path.name}")
        
        try:
            # Run inference and visualize
            annotated_img, detections = visualize_detections(
                img_path, model, conf_threshold=0.25
            )
            
            # Save annotated image
            output_path = Path(OUTPUT_DIR) / f'{img_path.stem}_detections.png'
            cv2.imwrite(str(output_path), annotated_img)
            print(f"Saved annotated image to: {output_path}")
            
            # Create and save plot
            plot_path = Path(OUTPUT_DIR) / f'{img_path.stem}_plot.png'
            plot_detections(img_path, annotated_img, detections, save_path=plot_path)
            
            # Collect results
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
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary Statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total images processed: {len(sample_images)}")
    print(f"Total detections: {len(all_detections)}")
    if len(sample_images) > 0:
        print(f"Average detections per image: {len(all_detections) / len(sample_images):.2f}")
    
    # Class counts
    class_counts = {}
    for det in all_detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print(f"\nDetected classes:")
        print(f"{'Class':<20} {'Count':<10}")
        print("-" * 30)
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{class_name:<20} {count:<10}")
    
    # Save detection results to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = Path(OUTPUT_DIR) / 'detection_results.csv'
        df.to_csv(str(csv_path), index=False)
        print(f"\nDetection results saved to: {csv_path}")
        print(f"Total records: {len(df)}")
    else:
        print("\nNo detections to save.")
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print(f"Total files saved: {len(list(Path(OUTPUT_DIR).glob('*.png')))}")
    print(f"{'='*60}")
    
    return OUTPUT_DIR


if __name__ == '__main__':
    output_dir = run_evaluation()
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
