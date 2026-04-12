"""
Inference module for YOLOv5 model.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from ultralytics import YOLO
import cv2
import numpy as np


class YOLOv5Detector:
    """Run inference with YOLOv5 model."""
    
    def __init__(self, model_path: str, device: str = '0', img_size: int = 640):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: GPU device ID
            img_size: Input image size
        """
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Set image size
        self.model.imgsz = img_size
    
    def predict(self, source: Union[str, List[str], np.ndarray], 
                conf: float = 0.25, iou: float = 0.45,
                save: bool = False, output_dir: str = None) -> List[Dict]:
        """
        Run inference on images or video.
        
        Args:
            source: Image path, list of image paths, or numpy array
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save output images
            output_dir: Directory to save output images
        
        Returns:
            List of detection results
        """
        # Run inference
        results = self.model.predict(
            source=source,
            imgsz=self.img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            save=save,
            save_dir=output_dir if save else None,
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            detection = {
                'boxes': result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else None,
                'confidence': result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else None,
                'class': result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else None,
                'names': result.names,
                'path': result.path
            }
            detections.append(detection)
        
        return detections
    
    def detect_image(self, image_path: str, conf: float = 0.25, 
                     iou: float = 0.45, draw: bool = True) -> Dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            draw: Whether to draw bounding boxes
        
        Returns:
            Dictionary with detection results
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            imgsz=self.img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else None
        confidence = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else None
        class_ids = result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else None
        
        # Draw bounding boxes if requested
        if draw and boxes is not None:
            image = cv2.imread(image_path)
            image = self.draw_boxes(image, boxes, confidence, class_ids, result.names)
        
        return {
            'image_path': image_path,
            'boxes': boxes,
            'confidence': confidence,
            'class_ids': class_ids,
            'names': result.names,
            'num_detections': len(result.boxes) if len(result.boxes) > 0 else 0,
            'image': image if draw else None
        }
    
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, 
                   confidence: np.ndarray, class_ids: np.ndarray,
                   class_names: Dict[int, str]) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image (BGR format)
            boxes: Bounding boxes [x1, y1, x2, y2]
            confidence: Confidence scores
            class_ids: Class IDs
            class_names: Dictionary mapping class IDs to names
        
        Returns:
            Image with drawn bounding boxes
        """
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidence, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            cls_name = class_names.get(int(cls_id), f'class_{int(cls_id)}')
            label = f'{cls_name}: {conf:.2f}'
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1)
        
        return image
    
    def verify_model(self) -> Dict:
        """
        Verify that the model can run a forward pass.
        
        Returns:
            Dictionary with verification results
        """
        print(f"Verifying model: {self.model_path}")
        
        # Check if file exists
        if not Path(self.model_path).exists():
            return {'status': 'FAILED', 'error': 'Model file not found'}
        
        # Check device
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"Device: {device_name}")
        
        # Run a test inference
        try:
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = self.model.predict(
                source=dummy_image,
                imgsz=640,
                conf=0.25,
                verbose=False
            )
            
            return {
                'status': 'SUCCESS',
                'model_path': self.model_path,
                'device': device_name,
                'num_classes': len(results[0].names),
                'test_passed': True
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }


def run_inference(model_path: str, source: str, output_dir: str = None,
                  conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
    """
    Run inference with a trained YOLOv5 model.
    
    Args:
        model_path: Path to trained model checkpoint
        source: Image path, directory, or video path
        output_dir: Directory to save output images
        conf: Confidence threshold
        iou: IoU threshold for NMS
    
    Returns:
        List of detection results
    """
    detector = YOLOv5Detector(model_path)
    
    save = output_dir is not None
    detections = detector.predict(source, conf=conf, iou=iou, save=save, output_dir=output_dir)
    
    return detections


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLOv5 inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--source', type=str, required=True,
                        help='Image path, directory, or video path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--verify', action='store_true',
                        help='Verify model instead of running inference')
    
    args = parser.parse_args()
    
    detector = YOLOv5Detector(args.weights)
    
    if args.verify:
        result = detector.verify_model()
        print(f"Verification result: {result}")
    else:
        detections = run_inference(
            args.weights, args.source, args.output_dir, args.conf, args.iou
        )
        print(f"Processed {len(detections)} images")
        for i, det in enumerate(detections):
            if det['boxes'] is not None:
                print(f"  Image {i+1}: {len(det['boxes'])} detections")
            else:
                print(f"  Image {i+1}: 0 detections")
