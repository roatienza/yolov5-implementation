"""
Visualize COCO dataset samples with YOLO annotations.
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


# COCO class colors (80 classes)
def get_color(class_id: int) -> tuple:
    """Get a consistent color for a class ID."""
    np.random.seed(class_id)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color


# COCO class names
COCO_CLASS_NAMES = [
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


class DatasetVisualizer:
    """Visualize COCO dataset with YOLO annotations."""
    
    def __init__(self, images_dir: str, labels_dir: str, output_dir: str):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO label files
            output_dir: Output directory for visualizations
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def draw_box(self, image: np.ndarray, x_center: float, y_center: float,
                 width: float, height: float, class_id: int,
                 conf: float = None) -> None:
        """
        Draw a bounding box on the image.
        
        Args:
            image: Image array
            x_center, y_center: Normalized center coordinates
            width, height: Normalized width and height
            class_id: Class ID
            conf: Confidence score (optional)
        """
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        
        # Clamp coordinates
        x_min = max(0, min(x_min, w))
        y_min = max(0, min(y_min, h))
        x_max = max(0, min(x_max, w))
        y_max = max(0, min(y_max, h))
        
        # Get color for class
        color = get_color(class_id)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Get class name
        class_name = COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else f'class_{class_id}'
        
        # Create label text
        if conf is not None:
            label = f'{class_name}: {conf:.2f}'
        else:
            label = class_name
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - text_h - 5), (x_min + text_w, y_min), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def visualize_image(self, img_path: Path, output_path: Path) -> bool:
        """
        Visualize a single image with its annotations.
        
        Args:
            img_path: Path to image file
            output_path: Path to save visualization
        
        Returns:
            True if successful, False otherwise
        """
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            return False
        
        # Load labels
        label_name = img_path.stem + '.txt'
        label_path = self.labels_dir / label_name
        
        if label_path.exists():
            # Parse and draw labels
            for line in label_path.read_text().strip().split('\n'):
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    self.draw_box(image, x_center, y_center, width, height, class_id)
        
        # Save visualization
        cv2.imwrite(str(output_path), image)
        return True
    
    def visualize_samples(self, num_samples: int = 10) -> int:
        """
        Visualize a random sample of images.
        
        Args:
            num_samples: Number of samples to visualize
        
        Returns:
            Number of successfully visualized images
        """
        # Get all image files
        image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"No images found in {self.images_dir}")
            return 0
        
        # Shuffle and select samples
        np.random.shuffle(image_files)
        samples = image_files[:min(num_samples, len(image_files))]
        
        print(f"Visualizing {len(samples)} samples...")
        print(f"  Input: {self.images_dir}")
        print(f"  Output: {self.output_dir}")
        
        success_count = 0
        for i, img_path in enumerate(samples):
            output_path = self.output_dir / f"sample_{i:03d}.jpg"
            if self.visualize_image(img_path, output_path):
                success_count += 1
        
        print(f"Visualized {success_count}/{len(samples)} images")
        return success_count


def visualize_dataset(images_dir: str, labels_dir: str, output_dir: str,
                     num_samples: int = 10) -> int:
    """
    Visualize a sample of images from the dataset.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    
    Returns:
        Number of successfully visualized images
    """
    visualizer = DatasetVisualizer(images_dir, labels_dir, output_dir)
    return visualizer.visualize_samples(num_samples)


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Default paths for COCO dataset in /data
    data_root = '/data/memory/coco'
    output_dir = '/workspace/yolov5-implementation/data/visualizations'
    
    # Visualize train set
    print("=" * 60)
    print("Visualizing COCO train2017 samples")
    print("=" * 60)
    visualize_dataset(
        images_dir=f'{data_root}/images/train2017',
        labels_dir=f'{data_root}/labels/train2017',
        output_dir=f'{output_dir}/train',
        num_samples=10
    )
    
    # Visualize val set
    print("\n" + "=" * 60)
    print("Visualizing COCO val2017 samples")
    print("=" * 60)
    visualize_dataset(
        images_dir=f'{data_root}/images/val2017',
        labels_dir=f'{data_root}/labels/val2017',
        output_dir=f'{output_dir}/val',
        num_samples=10
    )
