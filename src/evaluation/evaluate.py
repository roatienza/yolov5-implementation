"""
Evaluation module for YOLOv5 model.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import json
import torch
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class YOLOv5Evaluator:
    """Evaluate YOLOv5 model on COCO dataset."""
    
    def __init__(self, model_path: str, data_yaml: str = None, device: str = '0'):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            data_yaml: Path to dataset YAML configuration
            device: GPU device ID
        """
        self.model_path = model_path
        self.data_yaml = data_yaml or '/workspace/yolov5-implementation/data/dataset.yaml'
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
    
    def evaluate(self, split: str = 'val', conf: float = 0.001, iou: float = 0.65) -> Dict:
        """
        Evaluate the model on the specified split.
        
        Args:
            split: Dataset split to evaluate ('val' or 'test')
            conf: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("=" * 60)
        print("Starting YOLOv5 Evaluation")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_yaml}")
        print(f"Split: {split}")
        print(f"Confidence: {conf}")
        print(f"IoU: {iou}")
        print("=" * 60)
        
        # Run evaluation
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            conf=conf,
            iou=iou,
            device=self.device,
            save=True,
            plots=True
        )
        
        # Extract metrics
        metrics = {
            'model_path': self.model_path,
            'split': split,
            'metrics': results.box.map,
            'map50': results.box.map50,
            'map75': results.box.map75,
            'map50_95': results.box.map50_95,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'classes': len(results.names),
            'total_images': results.stats.total[0]
        }
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("=" * 60)
        
        return metrics
    
    def benchmark(self, images_dir: str = None, batch_size: int = 1, num_images: int = 100) -> Dict:
        """
        Benchmark inference speed.
        
        Args:
            images_dir: Directory containing test images
            batch_size: Batch size for inference
            num_images: Number of images to process
        
        Returns:
            Dictionary with benchmark results
        """
        if images_dir is None:
            # Use validation images
            data = yaml.safe_load(open(self.data_yaml))
            images_dir = Path(data['path']) / data['val']
        
        images_dir = Path(images_dir)
        image_files = list(images_dir.glob('*.jpg'))[:num_images]
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Benchmarking inference on {len(image_files)} images...")
        
        # Warmup
        self.model.predict(source=str(image_files[0]), imgsz=640, verbose=False)
        
        # Benchmark
        import time
        start_time = time.time()
        
        results = self.model.predict(
            source=[str(img) for img in image_files],
            imgsz=640,
            batch_size=batch_size,
            verbose=False
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        fps = len(image_files) / total_time
        latency_ms = (total_time / len(image_files)) * 1000
        
        benchmark_results = {
            'num_images': len(image_files),
            'batch_size': batch_size,
            'total_time_seconds': total_time,
            'fps': fps,
            'latency_ms': latency_ms
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  FPS: {fps:.2f}")
        print(f"  Average latency: {latency_ms:.2f} ms")
        
        return benchmark_results
    
    def export_metrics(self, output_path: str) -> None:
        """
        Export evaluation metrics to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        # Get latest metrics from runs
        runs_dir = Path(self.model_path).parent.parent.parent / 'val'
        
        if runs_dir.exists():
            # Load results.csv
            csv_path = runs_dir / 'results.csv'
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                latest_metrics = df.iloc[-1].to_dict()
                
                with open(output_path, 'w') as f:
                    json.dump(latest_metrics, f, indent=2)
                
                print(f"Metrics exported to: {output_path}")


def evaluate_model(model_path: str, data_yaml: str = None, device: str = '0') -> Dict:
    """
    Evaluate a trained YOLOv5 model.
    
    Args:
        model_path: Path to trained model checkpoint
        data_yaml: Path to dataset YAML
        device: GPU device ID
    
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = YOLOv5Evaluator(model_path, data_yaml, device)
    return evaluator.evaluate()


if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='/workspace/yolov5-implementation/data/dataset.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference speed benchmark')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images for benchmark')
    
    args = parser.parse_args()
    
    evaluator = YOLOv5Evaluator(args.weights, args.data, args.device)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark = evaluator.benchmark(num_images=args.num_images)
    
    # Export metrics
    output_path = Path(args.weights).parent / 'evaluation_metrics.json'
    evaluator.export_metrics(str(output_path))
