"""
Validate COCO dataset and converted YOLO format.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm


class DatasetValidator:
    """Validates COCO dataset integrity and YOLO conversion."""
    
    def __init__(self, images_dir: str, labels_dir: str):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO label files
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        # Valid class IDs for COCO (0-79)
        self.min_class_id = 0
        self.max_class_id = 79
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'readable_images': 0,
            'corrupted_images': 0,
            'images_with_labels': 0,
            'images_without_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'empty_labels': 0,
            'total_annotations': 0,
            'coordinate_errors': 0,
            'class_id_errors': 0
        }
    
    def validate(self) -> Dict:
        """
        Validates dataset integrity.
        
        Checks:
        - All images are readable (no corruption)
        - Label file exists for each image
        - Label coordinates are within [0, 1] range
        - No empty labels (unless image has no objects)
        - Class IDs are within valid range (0-79)
        
        Returns:
            validation_report: Dictionary with validation results
        """
        print(f"Validating dataset:")
        print(f"  Images: {self.images_dir}")
        print(f"  Labels: {self.labels_dir}")
        print()
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f'*{ext}'))
        
        self.stats['total_images'] = len(image_files)
        
        if self.stats['total_images'] == 0:
            print(f"ERROR: No images found in {self.images_dir}")
            return self.stats
        
        # Validate each image
        for img_path in tqdm(image_files, desc="Validating images"):
            img_name = img_path.stem
            
            # Check if image is readable
            try:
                img = Image.open(img_path)
                img.verify()
                self.stats['readable_images'] += 1
            except Exception as e:
                print(f"Corrupted image: {img_path} - {e}")
                self.stats['corrupted_images'] += 1
                continue
            
            # Check for corresponding label file
            label_path = self.labels_dir / f"{img_name}.txt"
            
            if not label_path.exists():
                self.stats['images_without_labels'] += 1
                continue
            
            self.stats['images_with_labels'] += 1
            
            # Validate label file
            label_valid = self._validate_label_file(label_path)
            
            if label_valid:
                self.stats['valid_labels'] += 1
            else:
                self.stats['invalid_labels'] += 1
        
        # Print summary
        self._print_report()
        
        return self.stats
    
    def _validate_label_file(self, label_path: Path) -> bool:
        """
        Validate a single YOLO label file.
        
        Args:
            label_path: Path to label file
        
        Returns:
            True if valid, False otherwise
        """
        try:
            content = label_path.read_text().strip()
            
            # Check for empty label
            if not content:
                self.stats['empty_labels'] += 1
                return True  # Empty labels are valid (no objects)
            
            lines = content.split('\n')
            is_valid = True
            
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) < 5:
                    print(f"Invalid label format in {label_path}: {line}")
                    is_valid = False
                    continue
                
                # Parse values
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                self.stats['total_annotations'] += 1
                
                # Check class ID range
                if class_id < self.min_class_id or class_id > self.max_class_id:
                    print(f"Invalid class ID {class_id} in {label_path}")
                    self.stats['class_id_errors'] += 1
                    is_valid = False
                
                # Check coordinate range [0, 1]
                coords = [x_center, y_center, width, height]
                for coord in coords:
                    if coord < 0 or coord > 1:
                        print(f"Coordinate out of range in {label_path}: {line}")
                        self.stats['coordinate_errors'] += 1
                        is_valid = False
            
            return is_valid
            
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            return False
    
    def _print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nImage Statistics:")
        print(f"  Total images: {self.stats['total_images']}")
        print(f"  Readable images: {self.stats['readable_images']}")
        print(f"  Corrupted images: {self.stats['corrupted_images']}")
        
        print(f"\nLabel Statistics:")
        print(f"  Images with labels: {self.stats['images_with_labels']}")
        print(f"  Images without labels: {self.stats['images_without_labels']}")
        print(f"  Valid labels: {self.stats['valid_labels']}")
        print(f"  Invalid labels: {self.stats['invalid_labels']}")
        print(f"  Empty labels: {self.stats['empty_labels']}")
        
        print(f"\nAnnotation Statistics:")
        print(f"  Total annotations: {self.stats['total_annotations']}")
        print(f"  Coordinate errors: {self.stats['coordinate_errors']}")
        print(f"  Class ID errors: {self.stats['class_id_errors']}")
        
        # Overall status
        is_valid = (
            self.stats['corrupted_images'] == 0 and
            self.stats['invalid_labels'] == 0 and
            self.stats['coordinate_errors'] == 0 and
            self.stats['class_id_errors'] == 0
        )
        
        print(f"\nOverall Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print("=" * 60)


def validate_coco_dataset(images_dir: str, labels_dir: str) -> Dict:
    """
    Validates COCO dataset integrity and YOLO conversion.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
    
    Returns:
        validation_report: Dictionary with validation results
    """
    validator = DatasetValidator(images_dir, labels_dir)
    return validator.validate()


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Default paths for COCO dataset in /data
    data_root = '/data/memory/coco'
    
    # Validate train set
    print("=" * 60)
    print("Validating COCO train2017")
    print("=" * 60)
    validate_coco_dataset(
        images_dir=f'{data_root}/images/train2017',
        labels_dir=f'{data_root}/labels/train2017'
    )
    
    # Validate val set
    print("\n" + "=" * 60)
    print("Validating COCO val2017")
    print("=" * 60)
    validate_coco_dataset(
        images_dir=f'{data_root}/images/val2017',
        labels_dir=f'{data_root}/labels/val2017'
    )
