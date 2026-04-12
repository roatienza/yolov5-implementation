"""
Training pipeline for YOLOv5 on COCO dataset.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
import torch
from ultralytics import YOLO


class YOLOv5Trainer:
    """Train YOLOv5 model on COCO dataset."""
    
    def __init__(self, model_type: str = 'yolov5s', data_yaml: str = None,
                 epochs: int = 100, batch_size: int = 16, img_size: int = 640,
                 device: str = '0', workers: int = 8):
        """
        Args:
            model_type: YOLOv5 model variant (n, s, m, l, x)
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size per GPU
            img_size: Input image size
            device: GPU device ID(s)
            workers: Number of data loading workers
        """
        self.model_type = model_type
        self.data_yaml = data_yaml or '/workspace/yolov5-implementation/data/dataset.yaml'
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.workers = workers
        
        # Initialize model
        self.model = YOLO(f'yolov5{model_type}.pt')
        
        # Training arguments
        self.train_args = {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'workers': self.workers,
            'patience': 50,  # Early stopping patience
            'save': True,
            'save_period': -1,  # Save checkpoint every epoch
            'cache': False,
            'image_weights': False,
            'optimizer': 'SGD',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': [0],  # Freeze first layer
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'plots': True,
            'source': '',
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': True,
            'opset': None,
            'workspace': None,
            'nms': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'overlap': True,
            'mask': False,
            'no_autoscale': False,
            'cfg': None,
            'weights': None,
            'project': 'runs/train',
            'name': f'exp_{model_type}',
            'exist_ok': False,
            'entity': None,
            'upload_dataset': False,
            'bbox_interval': -1,
            'artifact_alias': 'latest',
            'save_dir': None
        }
    
    def train(self) -> Dict:
        """
        Train the YOLOv5 model.
        
        Returns:
            Dictionary with training results
        """
        print("=" * 60)
        print("Starting YOLOv5 Training")
        print("=" * 60)
        print(f"Model: {self.model_type}")
        print(f"Data: {self.data_yaml}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Train model
        results = self.model.train(**self.train_args)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return results
    
    def validate(self, data: str = None) -> Dict:
        """
        Validate the trained model.
        
        Args:
            data: Path to dataset YAML (optional, uses training data if not specified)
        
        Returns:
            Dictionary with validation results
        """
        val_data = data or self.data_yaml
        
        print(f"Validating model on {val_data}")
        results = self.model.val(data=val_data)
        
        return results


def train_yolov5(model_type: str = 'yolov5s', epochs: int = 100,
                 batch_size: int = 16, device: str = '0') -> Dict:
    """
    Train YOLOv5 model on COCO dataset.
    
    Args:
        model_type: YOLOv5 model variant (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        device: GPU device ID(s)
    
    Returns:
        Dictionary with training results
    """
    trainer = YOLOv5Trainer(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )
    return trainer.train()


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv5 on COCO dataset')
    parser.add_argument('--model', type=str, default='yolov5s',
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                        help='YOLOv5 model variant')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID(s)')
    
    args = parser.parse_args()
    
    # Extract model type (e.g., 's' from 'yolov5s')
    model_type = args.model.replace('yolov5', '')
    
    train_yolov5(
        model_type=model_type,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
