"""
Data augmentation utilities for YOLOv5.
Note: YOLOv5 handles augmentation on-the-fly during training.
This module provides optional offline augmentation utilities.
"""

import os
import random
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import cv2


class DataAugmenter:
    """Provides optional offline data augmentation utilities."""
    
    def __init__(self, images_dir: str, labels_dir: str, output_dir: str):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO label files
            output_dir: Output directory for augmented data
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    def horizontal_flip(self, image: np.ndarray, labels: List[Tuple]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Apply horizontal flip augmentation.
        
        Args:
            image: Image array
            labels: List of (class_id, x_center, y_center, width, height)
        
        Returns:
            Augmented image and updated labels
        """
        # Flip image horizontally
        flipped_image = cv2.flip(image, 1)
        
        # Update label coordinates
        flipped_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label
            # Flip x_center: x' = 1 - x
            flipped_x = 1.0 - x_center
            flipped_labels.append((class_id, flipped_x, y_center, width, height))
        
        return flipped_image, flipped_labels
    
    def rotate(self, image: np.ndarray, labels: List[Tuple], angle: float) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Apply rotation augmentation.
        
        Args:
            image: Image array
            labels: List of (class_id, x_center, y_center, width, height)
            angle: Rotation angle in degrees
        
        Returns:
            Augmented image and updated labels
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix
        matrix[0, 2] += new_w // 2 - w // 2
        matrix[1, 2] += new_h // 2 - h // 2
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, matrix, (new_w, new_h))
        
        # Update label coordinates (simplified - doesn't account for image resize)
        # For production use, implement proper bounding box rotation
        rotated_labels = labels  # Placeholder
        
        return rotated_image, rotated_labels
    
    def mosaic(self, images: List[np.ndarray], labels_list: List[List[Tuple]]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Apply Mosaic augmentation (combines 4 images).
        
        Args:
            images: List of 4 image arrays
            labels_list: List of label lists for each image
        
        Returns:
            Combined mosaic image and updated labels
        """
        if len(images) < 4:
            # Pad with copies if less than 4 images
            while len(images) < 4:
                images.append(images[-1].copy())
                labels_list.append(labels_list[-1].copy())
        
        # Get image dimensions
        h, w = images[0].shape[:2]
        
        # Create mosaic image (2x2 grid)
        mosaic = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
        
        # Combine images
        mosaic[:h, :w] = images[0]
        mosaic[:h, w:] = images[1]
        mosaic[h:, :w] = images[2]
        mosaic[h:, w:] = images[3]
        
        # Update label coordinates
        mosaic_labels = []
        for i, labels in enumerate(labels_list):
            for label in labels:
                class_id, x_center, y_center, width, height = label
                
                # Scale coordinates to mosaic size
                new_x = x_center / 2.0
                new_y = y_center / 2.0
                new_w = width / 2.0
                new_h = height / 2.0
                
                # Offset based on position in mosaic
                if i == 1:  # Top-right
                    new_x += 0.5
                elif i == 2:  # Bottom-left
                    new_y += 0.5
                elif i == 3:  # Bottom-right
                    new_x += 0.5
                    new_y += 0.5
                
                mosaic_labels.append((class_id, new_x, new_y, new_w, new_h))
        
        return mosaic, mosaic_labels


def apply_offline_augmentation(images_dir: str, labels_dir: str, output_dir: str,
                               augmentation_type: str = 'flip',
                               augmentation_ratio: float = 1.0) -> None:
    """
    Apply offline augmentation to create additional training samples.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_dir: Output directory for augmented data
        augmentation_type: Type of augmentation ('flip', 'rotate', 'mosaic')
        augmentation_ratio: Ratio of images to augment (0.0 to 1.0)
    """
    print(f"Applying {augmentation_type} augmentation...")
    print(f"  Input: {images_dir}")
    print(f"  Output: {output_dir}")
    
    # Get all image files
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.jpeg'))
    
    # Shuffle and select subset
    random.shuffle(image_files)
    num_to_augment = int(len(image_files) * augmentation_ratio)
    image_files = image_files[:num_to_augment]
    
    augmenter = DataAugmenter(images_dir, labels_dir, output_dir)
    
    augmented_count = 0
    for img_path in image_files:
        img_name = img_path.stem
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Load labels
        label_path = Path(labels_dir) / f"{img_name}.txt"
        if not label_path.exists():
            continue
        
        labels = []
        for line in label_path.read_text().strip().split('\n'):
            if line:
                parts = line.split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
        
        # Apply augmentation
        if augmentation_type == 'flip':
            aug_image, aug_labels = augmenter.horizontal_flip(image, labels)
        elif augmentation_type == 'rotate':
            aug_image, aug_labels = augmenter.rotate(image, labels, angle=15)
        else:
            continue
        
        # Save augmented image
        aug_img_path = Path(output_dir) / 'images' / f"{img_name}_aug.jpg"
        cv2.imwrite(str(aug_img_path), aug_image)
        
        # Save augmented labels
        aug_label_path = Path(output_dir) / 'labels' / f"{img_name}_aug.txt"
        label_text = '\n'.join([f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in aug_labels])
        aug_label_path.write_text(label_text)
        
        augmented_count += 1
    
    print(f"Augmented {augmented_count} images")
