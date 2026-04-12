"""
Generate validation report and class distribution statistics for COCO dataset.
"""

import json
from pathlib import Path
from typing import Dict
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def generate_validation_report(images_dir: str, labels_dir: str, output_path: str) -> Dict:
    """
    Generate a comprehensive validation report for the dataset.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_path: Path to save the JSON report
    
    Returns:
        validation_report: Dictionary with validation results
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Statistics
    stats = {
        'total_images': 0,
        'readable_images': 0,
        'corrupted_images': 0,
        'images_with_labels': 0,
        'images_without_labels': 0,
        'total_annotations': 0,
        'class_distribution': {},
        'coordinate_errors': 0,
        'class_id_errors': 0,
        'validation_status': 'VALID'
    }
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
    
    stats['total_images'] = len(image_files)
    
    # Class counter
    class_counter = Counter()
    
    # Validate each image
    for img_path in image_files:
        img_name = img_path.stem
        
        # Check for corresponding label file
        label_path = labels_dir / f"{img_name}.txt"
        
        if not label_path.exists():
            stats['images_without_labels'] += 1
            continue
        
        stats['images_with_labels'] += 1
        
        # Parse label file
        try:
            content = label_path.read_text().strip()
            if content:
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        stats['total_annotations'] += 1
                        class_counter[class_id] += 1
                        
                        # Check class ID range
                        if class_id < 0 or class_id > 79:
                            stats['class_id_errors'] += 1
                        
                        # Check coordinate range
                        coords = [x_center, y_center, width, height]
                        for coord in coords:
                            if coord < 0 or coord > 1:
                                stats['coordinate_errors'] += 1
                                break
        except Exception as e:
            stats['corrupted_images'] += 1
    
    # Update class distribution
    stats['class_distribution'] = dict(sorted(class_counter.items()))
    
    # Determine validation status
    if stats['corrupted_images'] > 0 or stats['coordinate_errors'] > 0 or stats['class_id_errors'] > 0:
        stats['validation_status'] = 'INVALID'
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Validation report saved: {output_path}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Validation status: {stats['validation_status']}")
    
    return stats


def generate_class_distribution_histogram(labels_dir: str, output_path: str, num_classes: int = 80) -> None:
    """
    Generate a class distribution histogram.
    
    Args:
        labels_dir: Directory containing YOLO label files
        output_path: Path to save the histogram image
        num_classes: Number of classes (default: 80 for COCO)
    """
    labels_dir = Path(labels_dir)
    
    # Count annotations per class
    class_counter = Counter()
    
    for label_path in labels_dir.glob('*.txt'):
        try:
            content = label_path.read_text().strip()
            if content:
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
        except Exception:
            continue
    
    # Prepare data for plotting
    class_ids = list(range(num_classes))
    counts = [class_counter.get(cid, 0) for cid in class_ids]
    
    # COCO class names
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    bars = ax.bar(class_ids, counts, color=colors, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Number of Annotations', fontsize=12)
    ax.set_title('COCO Dataset Class Distribution', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks with class names (show every 5th class to avoid overcrowding)
    tick_positions = list(range(0, num_classes, 5))
    tick_labels = [class_names[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add total count annotation
    total_count = sum(counts)
    ax.text(0.02, 0.98, f'Total Annotations: {total_count:,}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add average count annotation
    avg_count = np.mean(counts)
    ax.text(0.02, 0.92, f'Average per Class: {avg_count:,.0f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution histogram saved: {output_path}")


if __name__ == '__main__':
    # Default paths for COCO dataset in /data
    data_root = '/data/memory/coco'
    output_dir = '/data/memory/coco'
    
    # Generate validation report for train set
    print("=" * 60)
    print("Generating validation report for train2017")
    print("=" * 60)
    generate_validation_report(
        images_dir=f'{data_root}/images/train2017',
        labels_dir=f'{data_root}/labels/train2017',
        output_path=f'{output_dir}/validation_report_train.json'
    )
    
    # Generate validation report for val set
    print("\n" + "=" * 60)
    print("Generating validation report for val2017")
    print("=" * 60)
    generate_validation_report(
        images_dir=f'{data_root}/images/val2017',
        labels_dir=f'{data_root}/labels/val2017',
        output_path=f'{output_dir}/validation_report_val.json'
    )
    
    # Generate class distribution histogram
    print("\n" + "=" * 60)
    print("Generating class distribution histogram")
    print("=" * 60)
    generate_class_distribution_histogram(
        labels_dir=f'{data_root}/labels/train2017',
        output_path='/workspace/yolov5-implementation/data/visualizations/class_dist.png'
    )
