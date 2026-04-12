"""
Create dataset YAML configuration for YOLOv5.
"""

import yaml
from pathlib import Path


def create_dataset_yaml(data_root: str, output_path: str) -> None:
    """
    Create dataset YAML configuration file for YOLOv5.
    
    Args:
        data_root: Root directory of the dataset (e.g., /data/memory/coco)
        output_path: Path to save the YAML file
    """
    data_root = Path(data_root)
    
    # Create YAML configuration
    dataset_config = {
        'path': str(data_root),  # Dataset root dir
        'train': 'images/train2017',  # train images (relative to 'path')
        'val': 'images/val2017',  # val images (relative to 'path')
        'test': 'images/test2017',  # test images (optional)
        'nc': 80,  # number of classes
        'names': [
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
    }
    
    # Save YAML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Dataset YAML created: {output_path}")
    print(f"Content:\n{yaml.dump(dataset_config, default_flow_style=False)}")


if __name__ == '__main__':
    # Example usage
    data_root = '/data/memory/coco'
    output_path = '/workspace/yolov5-implementation/data/dataset.yaml'
    
    create_dataset_yaml(data_root, output_path)
