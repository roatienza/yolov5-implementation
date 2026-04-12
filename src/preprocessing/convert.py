"""
Convert COCO JSON annotations to YOLO format.
IMPORTANT: All paths should reference /data directory in sandbox.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm


# COCO has 80 classes with non-contiguous IDs
# YOLO requires contiguous IDs from 0 to 79
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 90
]

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


class COCOtoYOLOConverter:
    """Convert COCO JSON annotations to YOLO txt format."""
    
    def __init__(self, coco_json_path: str, images_dir: str, output_labels_dir: str):
        """
        Args:
            coco_json_path: Path to COCO annotation JSON file
            images_dir: Directory containing COCO images
            output_labels_dir: Output directory for YOLO label files
        """
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # Create output directory
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO annotations
        with open(self.coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create category ID to class index mapping
        self.category_map = {cat_id: idx for idx, cat_id in enumerate(COCO_CATEGORY_IDS)}
        
        # Create image ID to file path mapping
        self.image_id_to_path = {}
        for img in self.coco_data['images']:
            self.image_id_to_path[img['id']] = img['file_name']
    
    def convert(self) -> Dict:
        """
        Convert all annotations from COCO to YOLO format.
        
        Returns:
            Dictionary with conversion statistics
        """
        print(f"Converting COCO annotations to YOLO format")
        print(f"  Input: {self.coco_json_path}")
        print(f"  Images: {self.images_dir}")
        print(f"  Output: {self.output_labels_dir}")
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Convert each image's annotations
        stats = {
            'total_images': len(self.coco_data['images']),
            'images_with_annotations': 0,
            'images_without_annotations': 0,
            'total_annotations': 0,
            'failed_conversions': 0
        }
        
        for img in tqdm(self.coco_data['images'], desc="Converting annotations"):
            img_id = img['id']
            img_filename = self.image_id_to_path[img_id]
            img_width = img['width']
            img_height = img['height']
            
            # Get annotations for this image
            annotations = annotations_by_image.get(img_id, [])
            
            if len(annotations) == 0:
                stats['images_without_annotations'] += 1
                # Create empty label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = self.output_labels_dir / label_filename
                label_path.write_text('')
                continue
            
            stats['images_with_annotations'] += 1
            
            try:
                # Convert annotations to YOLO format
                yolo_lines = []
                for ann in annotations:
                    # Get class index
                    cat_id = ann['category_id']
                    if cat_id not in self.category_map:
                        continue
                    
                    class_idx = self.category_map[cat_id]
                    
                    # Get bounding box: COCO format [x_min, y_min, width, height]
                    x_min, y_min, box_w, box_h = ann['bbox']
                    
                    # Convert to YOLO format: [x_center, y_center, width, height] normalized
                    x_center = (x_min + box_w / 2) / img_width
                    y_center = (y_min + box_h / 2) / img_height
                    norm_w = box_w / img_width
                    norm_h = box_h / img_height
                    
                    # Clamp values to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))
                    
                    yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    stats['total_annotations'] += 1
                
                # Write YOLO label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = self.output_labels_dir / label_filename
                label_path.write_text('\n'.join(yolo_lines))
                
            except Exception as e:
                print(f"Error converting {img_filename}: {e}")
                stats['failed_conversions'] += 1
        
        print(f"\nConversion complete!")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images with annotations: {stats['images_with_annotations']}")
        print(f"  Images without annotations: {stats['images_without_annotations']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Failed conversions: {stats['failed_conversions']}")
        
        return stats
    
    def get_class_mapping(self) -> Dict[int, str]:
        """
        Returns:
            Dictionary mapping COCO category IDs to YOLO class indices
        """
        return {cat_id: COCO_CLASS_NAMES[idx] for idx, cat_id in enumerate(COCO_CATEGORY_IDS)}


def convert_coco_to_yolo(coco_json_path: str, output_labels_dir: str, images_dir: str) -> Dict:
    """
    Converts COCO JSON annotations to YOLO txt format.
    
    COCO format: [x_min, y_min, width, height] in pixels
    YOLO format: [x_center, y_center, width, height] normalized to [0, 1]
    
    Args:
        coco_json_path: Path to COCO annotation JSON file
        output_labels_dir: Output directory for YOLO label files
        images_dir: Directory containing COCO images
    
    Returns:
        class_mapping: Dictionary mapping COCO category IDs to YOLO class indices
    """
    converter = COCOtoYOLOConverter(coco_json_path, images_dir, output_labels_dir)
    stats = converter.convert()
    return converter.get_class_mapping()


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Default paths for COCO dataset in /data
    data_root = '/data/memory/coco'
    
    # Convert train set
    print("=" * 60)
    print("Converting COCO train2017")
    print("=" * 60)
    convert_coco_to_yolo(
        coco_json_path=f'{data_root}/annotations/annotations/instances_train2017.json',
        output_labels_dir=f'{data_root}/labels/train2017',
        images_dir=f'{data_root}/images/train2017'
    )
    
    # Convert val set
    print("\n" + "=" * 60)
    print("Converting COCO val2017")
    print("=" * 60)
    convert_coco_to_yolo(
        coco_json_path=f'{data_root}/annotations/annotations/instances_val2017.json',
        output_labels_dir=f'{data_root}/labels/val2017',
        images_dir=f'{data_root}/images/val2017'
    )
