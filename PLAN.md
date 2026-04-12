# YOLOv5 Object Detector - Implementation Plan

## Phase 1: Repository Setup

### 1.0 GPU Environment Verification

**CRITICAL**: All operations MUST use GPU acceleration. Before starting any phase, verify GPU availability:

```bash
# Verify GPU is available in sandbox
nvidia-smi
# OR
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"
```

**Requirements:**
- GPU MUST be available and accessible
- All training and inference operations MUST run on GPU
- If GPU is not available, the agent MUST report an error and stop

### 1.1 Repository Structure
```
yolov5-implementation/
├── data/
│   └── dataset.yaml
├── src/
│   ├── preprocessing/
│   │   ├── download.py
│   │   ├── augment.py
│   │   ├── convert.py
│   │   └── visualize.py
│   ├── training/
│   │   ├── train.py
│   │   └── config.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── inference/
│       ├── detect.py
│       └── export.py
├── configs/
│   ├── preprocessing.yaml
│   └── training.yaml
├── scripts/
│   ├── setup.sh
│   ├── train.sh
│   └── evaluate.sh
├── docs/
│   └── README.md
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── .gitignore
```

### 1.2 Initial Setup Tasks
- Create directory structure
- Initialize Git repository
- Write requirements.txt
- Create .gitignore

### 1.3 Phase 1 Completion Targets ✅

**Before moving to Phase 2, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| GPU verified and accessible | `nvidia-smi` shows GPU available | ⬜ |
| Repository created at `roatienza/yolov5-implementation` | GitHub API check | ⬜ |
| All directories in structure exist | `ls -R` command | ⬜ |
| `requirements.txt` contains all dependencies | File content check | ⬜ |
| `.gitignore` excludes `__pycache__`, `.pt`, `.pth`, `*.onnx`, `data/` | File content check | ⬜ |
| Initial commit pushed to `main` branch | Git log verification | ⬜ |
| Repository is private | GitHub settings check | ⬜ |

**Success Criteria:** All 7 targets marked as ✅ completed

---

## Phase 2: Data Preprocessing Pipeline (COCO-Focused)

### 2.1 Data Directory Configuration

**IMPORTANT**: The agent MUST operate in the sandbox environment and use the `/data` directory (or wherever the data folder is mounted in the sandbox) for all data operations.

- **Data Directory**: `/data` (sandbox-mounted data folder)
- **All downloads and preprocessing**: Must be performed in `/data`
- **Do NOT use local filesystem**: Always reference `/data` paths

### 2.2 COCO Dataset Structure

The COCO 2017 dataset is split as follows:
- **Train2017**: 118,287 images with annotations
- **Val2017**: 5,000 images with annotations
- **Test2017**: 40,670 images (test-dev: 20,288 images for evaluation)

**Dataset Size**: ~20.1 GB (images only)

**Storage Location**: `/data/coco/`

### 2.2 Download Module (`src/preprocessing/download.py`)

```python
# Download COCO dataset automatically
# IMPORTANT: Always use /data directory in sandbox environment
DATA_ROOT = '/data/coco'  # Sandbox-mounted data directory

def download_coco_dataset(output_dir=DATA_ROOT):
    """
    Downloads COCO 2017 dataset with standard splits.
    ALWAYS operates in sandbox /data directory.
    
    Args:
        output_dir: Root directory for dataset storage (default: /data/coco)
    
    Structure created in /data/coco/:
        /data/coco/
        ├── images/
        │   ├── train2017/
        │   ├── val2017/
        │   └── test2017/
        └── annotations/
            ├── instances_train2017.json
            ├── instances_val2017.json
            └── image_info_test-dev2017.json
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    urls = [
        'http://images.cocodataset.org/zips/train2017.zip',   # 19G, 118k images
        'http://images.cocodataset.org/zips/val2017.zip',     # 1G, 5k images
        'http://images.cocodataset.org/zips/test2017.zip'     # 7G, 41k images
    ]
    annotation_urls = [
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'http://images.cocodataset.org/annotations/image_info_test-dev2017.zip'
    ]
```

### 2.3 COCO to YOLO Conversion Module (`src/preprocessing/convert.py`)

```python
# Convert COCO JSON annotations to YOLO format
# IMPORTANT: All paths should reference /data directory in sandbox

def convert_coco_to_yolo(coco_json_path, output_labels_dir, images_dir):
    """
    Converts COCO JSON annotations to YOLO txt format.
    Operates on data stored in /data/coco/
    
    COCO format: [x_min, y_min, width, height] in pixels
    YOLO format: [x_center, y_center, width, height] normalized to [0, 1]
    
    Args:
        coco_json_path: Path to COCO annotation JSON file (e.g., /data/coco/annotations/instances_train2017.json)
        output_labels_dir: Output directory for YOLO label files (e.g., /data/coco/labels/train2017/)
        images_dir: Directory containing COCO images (e.g., /data/coco/images/train2017/)
    
    Returns:
        class_mapping: Dictionary mapping COCO category IDs to YOLO class indices
    """
    # COCO has 80 classes with non-contiguous IDs
    # YOLO requires contiguous IDs from 0 to 79
    coco_category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 
                         18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 
                         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 
                         50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 
                         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
                         82, 84, 85, 86, 87, 88, 90]
    
    # Convert bounding box: COCO [x_min, y_min, w, h] → YOLO [x_center, y_center, w, h]
    # Normalize to [0, 1] range based on image dimensions
```

### 2.4 Validation Module (`src/preprocessing/validate.py`)

```python
# Validate COCO dataset and converted YOLO format
def validate_coco_dataset(images_dir, labels_dir):
    """
    Validates COCO dataset integrity and YOLO conversion.
    
    Checks:
    - All images are readable (no corruption)
    - Label file exists for each image
    - Label coordinates are within [0, 1] range
    - No empty labels (unless image has no objects)
    - Class IDs are within valid range (0-79)
    
    Returns:
        validation_report: Dictionary with validation results
    """
```

### 2.5 Dataset Split Verification (`src/preprocessing/split.py`)

```python
# Verify COCO standard splits
def verify_coco_splits():
    """
    COCO 2017 uses pre-defined splits (no random splitting needed):
    
    - train2017.txt: 118,287 images
    - val2017.txt: 5,000 images
    - test-dev2017.txt: 20,288 images (for evaluation submission)
    
    This function verifies the splits are correct and creates image list files.
    """
```

### 2.6 Augmentation Module (`src/preprocessing/augment.py`)

```python
# Augmentation applied during training (not preprocessing)
# YOLOv5 handles augmentation on-the-fly during training
# This module provides optional offline augmentation for data augmentation strategies

def apply_offline_augmentation(images_dir, labels_dir, output_dir):
    """
    Optional: Apply offline augmentation to create additional training samples.
    
    Augmentations:
    - Mosaic (combines 4 images)
    - MixUp (blends 2 images)
    - Random horizontal flip
    - Random rotation (±15°)
    - Random brightness/contrast/hue adjustments
    - Random scaling (0.5x - 1.5x)
    
    Note: YOLOv5 applies most augmentations on-the-fly during training.
    """
```

### 2.7 Visualization Module (`src/preprocessing/visualize.py`)

```python
# Visualize COCO dataset statistics
def visualize_coco_dataset(images_dir, labels_dir, output_dir):
    """
    Generates visualization reports for COCO dataset:
    
    - Class distribution histogram (80 classes)
    - Bounding box aspect ratio distribution
    - Object count per image distribution
    - Image size distribution
    - Sample annotated images with bounding boxes
    - Label quality report (JSON)
    """
```

### 2.8 Dataset YAML Configuration

After preprocessing, create `data/coco.yaml` with

### 2.9 Phase 2 Completion Targets ✅

**Before moving to Phase 3, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| COCO dataset downloaded to `/data/coco/` | `ls -la /data/coco/` | ⬜ |
| All 3 splits present (train2017, val2017, test2017) | Directory structure check | ⬜ |
| Annotations converted to YOLO format | `ls /data/coco/labels/` | ⬜ |
| Validation report generated with 0 critical errors | `cat /data/coco/validation_report.json` | ⬜ |
| Class distribution histogram generated | File exists: `visualizations/class_dist.png` | ⬜ |
| `data/coco.yaml` created with 80 classes | File content check | ⬜ |
| All preprocessing code committed to repo | Git log verification | ⬜ |

**Success Criteria:** All 7 targets marked as ✅ completed paths pointing to `/data/coco/`:

```yaml
# COCO 2017 dataset configuration for YOLOv5
# IMPORTANT: All paths reference /data directory in sandbox
path: /data/coco  # dataset root directory (sandbox-mounted)
train: images/train2017  # train images (relative to 'path')
val: images/val2017      # val images
test: images/test2017    # test images (optional)

# 80 COCO classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

### 2.9 Preprocessing Pipeline Execution (`scripts/preprocess_coco.sh`)

```bash
#!/bin/bash
# Complete COCO preprocessing pipeline

# Step 1: Download COCO dataset
python src/preprocessing/download.py --dataset coco --output data/coco

# Step 2: Convert COCO JSON to YOLO format
python src/preprocessing/convert.py \
  --input data/coco/annotations/instances_train2017.json \
  --images data/coco/images/train2017 \
  --labels data/coco/labels/train2017

# Step 3: Validate conversion
python src/preprocessing/validate.py \
  --images data/coco/images/train2017 \
  --labels data/coco/labels/train2017 \
  --report data/validation_report.json

# Step 4: Generate visualizations
python src/preprocessing/visualize.py \
  --images data/coco/images/train2017 \
  --labels data/coco/labels/train2017 \
  --output data/visualizations/

# Step 5: Create dataset YAML
python -c "
import yaml
config = {
    'path': '../datasets/coco',
    'train': 'images/train2017',
    'val': 'images/val2017',
    'test': 'images/test2017',
    'names': [list of 80 class names]
}
with open('data/coco.yaml', 'w') as f:
    yaml.dump(config, f)
"
```

### 2.10 Deliverables

- ✅ COCO dataset downloaded and organized in `/data/coco/`
- ✅ Annotations converted from COCO JSON to YOLO txt format
- ✅ `data/coco.yaml` configuration file (with /data paths)
- ✅ Validation report (`/data/validation_report.json`)
- ✅ Visualization plots (`/data/visualizations/`)
- ✅ Ready-to-train dataset structure in sandbox:

```
/data/coco/
├── images/
│   ├── train2017/  (118,287 images)
│   ├── val2017/    (5,000 images)
│   └── test2017/   (40,670 images)
├── labels/
│   ├── train2017/  (118,287 .txt files)
│   └── val2017/    (5,000 .txt files)
└── coco.yaml       (dataset configuration)
```

**IMPORTANT**: All data operations MUST occur in the `/data` directory (sandbox-mounted). Do NOT store data in the local filesystem.

---

## Phase 3: Model Creation and Training

**CRITICAL**: All training operations MUST use GPU acceleration. The sandbox environment has GPU support enabled.

### 3.0 GPU Verification (Required Before Training)

**Before starting training, verify GPU availability:**

```python
# src/training/config.py
import torch

def verify_gpu_available():
    """Verify GPU is available before training - MUST fail if no GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available! Training requires GPU acceleration.")
    
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    return True

# Call this before any training operation
verify_gpu_available()
```

**GPU Configuration:**
```python
# Always use GPU device
DEVICE = 'cuda:0'  # First GPU
FORCE_GPU = True  # Force GPU usage
```

### 3.1 Existing Sandbox Packages (No Installation Needed)

| Package | Version | Status |
|---------|---------|--------|
| torch | 2.11.0 | ✅ Already installed (with CUDA) |
| opencv-python | 4.13.0.92 | ✅ Already installed |
| Pillow | 12.2.0 | ✅ Already installed |
| numpy | 2.4.4 | ✅ Already installed |
| pandas | 3.0.2 | ✅ Already installed |
| matplotlib | 3.10.8 | ✅ Already installed |
| tqdm | 4.67.3 | ✅ Already installed |

### 3.2 Packages to Install (If Not Present)

```bash
# Only install if missing in sandbox
pip install ultralytics>=8.0.0  # YOLOv5/v8 library
pip install tensorboard>=2.4.1  # Training visualization
pip install albumentations>=1.0.3  # Advanced augmentations
pip install pyyaml>=6.0  # YAML configuration
```

### 3.3 YOLOv5 Model Architecture Overview

YOLOv5 models are available in 5 sizes, all using the same architecture with different depths and widths:

| Model | Size | Parameters | FLOPs | Architecture Details |
|-------|------|------------|-------|---------------------|
| **YOLOv5n** | Nano | 1.9M | 4.5G | Smallest, fastest |
| **YOLOv5s** | Small | 7.2M | 16.5G | Best speed/accuracy tradeoff |
| **YOLOv5m** | Medium | 21.2M | 49.0G | Balanced performance |
| **YOLOv5l** | Large | 46.5M | 109.1G | High accuracy |
| **YOLOv5x** | X-Large | 86.7M | 205.7G | Maximum accuracy |

**Architecture Components**:
- **Backbone**: CSPDarknet with Focus layers
- **Neck**: PANet (Path Aggregation Network)
- **Head**: YOLO detection head with 3 output scales

### 3.3 Model Creation Script (`src/training/create_model.py`)

```python
# Create YOLOv5 model instance
# IMPORTANT: Always operate in sandbox environment

from ultralytics import YOLO
import torch

def create_yolov5_model(model_size='s', pretrained=True, device='cuda:0'):
    """
    Creates a YOLOv5 model instance for training.
    
    Args:
        model_size: 'n', 's', 'm', 'l', or 'x' (default: 's')
        pretrained: Load COCO pretrained weights (default: True)
        device: Device to use ('cuda:0' for GPU, 'cpu' for CPU)
    
    Returns:
        model: YOLOv5 model instance ready for training
    
    Example:
        >>> model = create_yolov5_model(model_size='s', pretrained=True)
        >>> model.info()  # Display model architecture
    """
    # Model name mapping
    model_names = {
        'n': 'yolov5n.pt',
        's': 'yolov5s.pt',
        'm': 'yolov5m.pt',
        'l': 'yolov5l.pt',
        'x': 'yolov5x.pt'
    }
    
    model_name = model_names.get(model_size, 'yolov5s.pt')
    
    # Load pretrained model (automatically downloads if not present)
    if pretrained:
        model = YOLO(model_name)
    else:
        # Create model from scratch
        model = YOLO(f'yolov5{model_size}.yaml')
    
    # Move to device
    model.to(device)
    
    return model

def display_model_architecture(model):
    """
    Display detailed model architecture information.
    
    Shows:
    - Layer types and dimensions
    - Number of parameters
    - FLOPs
    - Input/output shapes
    """
    model.info(verbose=True)

def export_model_structure(model, output_path):
    """
    Export model structure to YAML for documentation.
    
    Args:
        model: YOLOv5 model instance
        output_path: Path to save YAML (e.g., /data/models/yolov5s_structure.yaml)
    """
    model.export(format='torchscript')  # Can also export to ONNX
```

### 3.4 Training Configuration (`configs/training.yaml`)

```yaml
# Training configuration for YOLOv5
# IMPORTANT: All paths reference /data directory in sandbox
# CRITICAL: GPU MUST be used for all training operations

# Model settings
model_size: 's'  # n, s, m, l, x
pretrained: true  # Use COCO pretrained weights

# Training parameters
epochs: 100
batch_size: 16
img_size: 640
device: 'cuda:0'  # Use GPU in sandbox - REQUIRED

# Data settings (sandbox paths)
data: '/data/coco.yaml'  # Dataset configuration
workers: 8  # Data loading workers

# Optimization
optimizer: 'SGD'  # SGD or Adam
lr0: 0.01  # Initial learning rate
lrf: 0.1  # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# GPU Settings
force_gpu: true  # Force GPU usage
gpu_memory_fraction: 0.9  # Use 90% of GPU memory
```

# Augmentation
hyp: 'hyp.scratch-low.yaml'  # Hyperparameters
augment: true  # Enable mosaic and mixup

# Checkpointing
save_period: 10  # Save checkpoint every N epochs
patience: 50  # Early stopping patience

# Output
project: '/data/runs'  # Output directory in sandbox
name: 'experiment_001'
exist_ok: true  # Overwrite existing runs
```

### 3.5 Training Script (`src/training/train.py`)

```python
# Train YOLOv5 model
# IMPORTANT: All operations in sandbox environment
# CRITICAL: GPU MUST be used for training

from ultralytics import YOLO
import torch
import yaml

def verify_gpu_before_training():
    """Verify GPU is available before training - MUST fail if no GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available! Training requires GPU acceleration.")
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def train_yolov5(model, data_path, epochs=100, batch_size=16, img_size=640):
    """
    Trains a YOLOv5 model on the specified dataset.
    ALL TRAINING MUST USE GPU.
    
    Args:
        model: YOLOv5 model instance (from create_yolov5_model)
        data_path: Path to dataset YAML (e.g., /data/coco.yaml)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size (default: 640)
    
    Returns:
        results: Training results dictionary
    
    Example:
        >>> model = create_yolov5_model(model_size='s')
        >>> results = train_yolov5(model, data_path='/data/coco.yaml', epochs=100)
    """
    # CRITICAL: Verify GPU before training
    verify_gpu_before_training()
    
    # Train the model (GPU is automatically used by ultralytics if available)
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='0',  # Use GPU (device='0' = first GPU)
        workers=8,
        project='/data/runs',  # Sandbox output directory
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        verbose=True
    )
    
    return results

def monitor_training(model, results):
    """
    Monitor training progress and log metrics.
    
    Tracks:
    - Loss curves (box, obj, cls)
    - mAP@0.5 and mAP@0.5:0.95
    - Precision and Recall
    - Learning rate schedule
    """
    # Results are automatically logged to /data/runs/train/
    # TensorBoard logs available at /data/runs/train/results
```

### 3.6 Training Execution (`scripts/train.sh`)

```bash
#!/bin/bash
# Train YOLOv5 model
# IMPORTANT: Run in sandbox environment

# Check if ultralytics is installed, install if needed
python -c "import ultralytics" 2>/dev/null || pip install ultralytics>=8.0.0

# Create model and train
python << EOF
from src.training.create_model import create_yolov5_model
from src.training.train import train_yolov5

# Create YOLOv5s model with pretrained weights
model = create_yolov5_model(model_size='s', pretrained=True)

# Display model architecture
model.info(verbose=True)

# Train on COCO dataset
results = train_yolov5(
    model=model,
    data_path='/data/coco.yaml',
    epochs=100,
    batch_size=16,
    img_size=640
)

# Save best model
model.save('/data/runs/train/weights/best.pt')
EOF
```

### 3.7 Model Checkpointing Strategy

```python
# Automatic checkpointing during training
# Checkpoints saved to /data/runs/train/weights/

checkpoint_schedule = {
    'every_n_epochs': 10,  # Save every 10 epochs
    'best_model': True,  # Save best mAP model
    'last_model': True,  # Save last epoch model
    'final_model': True  # Save final model after training
}

# Checkpoint files:
# - /data/runs/train/weights/epoch_10.pt
# - /data/runs/train/weights/epoch_20.pt
# - ...
# - /data/runs/train/weights/best.pt (best mAP)
# - /data/runs/train/weights/last.pt (last epoch)
# - /data/runs/train/weights/final.pt (after training)
```

### 3.8 Training Monitoring and Logging

```python
# Training metrics logged to /data/runs/train/

# Files created:
# - results.csv: Training metrics per epoch
# - results.png: Loss and metric curves
# - confusion_matrix.png: Confusion matrix
# - precision_recall_curve.png: PR curves
# - tensorboard logs: /data/runs/train/results/

# Key metrics tracked:
metrics = {
    'train/box_loss': float,
    'train/obj_loss': float,
    'train/cls_loss': float,
    'val/mAP_0.5': float,
    'val/mAP_0.5:0.95': float,
    'val/precision': float,
    'val/recall': float,
    'val/F1': float
}
```

### 3.9 Deliverables

- ✅ Model creation script (`src/training/create_model.py`)
- ✅ Training script (`src/training/train.py`)
- ✅ Training configuration (`configs/training.yaml`)
- ✅ Training execution script (`scripts/train.sh`)
- ✅ Trained model weights (`/data/runs/train/weights/best.pt`)
- ✅ Training logs and TensorBoard logs (`/data/runs/train/`)
- ✅ Training metrics CSV (`/data/runs/train/results.csv`)
- ✅ Model architecture documentation

**IMPORTANT**: All training operations MUST occur in the sandbox. Model weights and logs are saved to `/data/runs/` directory.

### 3.10 Phase 3 Completion Targets ✅

**Before moving to Phase 4, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| GPU verified and training used GPU | Logs show "GPU Available" message | ⬜ |
| Required packages installed (`ultralytics`, `tensorboard`, `albumentations`, `pyyaml`) | `pip list` verification | ⬜ |
| YOLOv5 model architecture created and documented | File exists: `docs/model_architecture.txt` | ⬜ |
| Training completed for 100 epochs | Checkpoint exists: `/data/runs/train/weights/last.pt` | ⬜ |
| Best model saved with validation metrics | File exists: `/data/runs/train/weights/best.pt` | ⬜ |
| TensorBoard logs generated | Directory exists: `/data/runs/train/*/results` | ⬜ |
| Training results.csv contains all epochs | File size > 10KB | ⬜ |
| All training code committed to repo | Git log verification | ⬜ |

**Success Criteria:** All 8 targets marked as ✅ completed

**Success Criteria:** All 7 targets marked as ✅ completed

---

## Phase 4: Model Evaluation (Detailed)

**IMPORTANT**: All evaluation operations MUST occur in the sandbox environment using the `/data` directory.

### 4.1 Metrics Module (`src/evaluation/metrics.py`)

```python
# Calculate comprehensive evaluation metrics
# IMPORTANT: All paths should reference /data directory in sandbox

def calculate_metrics(predictions, targets, iou_thresholds=None):
    """
    Calculate all evaluation metrics for object detection.
    
    Metrics computed:
    - mAP@0.5: Mean Average Precision at IoU=0.5
    - mAP@0.5:0.95: Mean Average Precision across IoU thresholds 0.5 to 0.95
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    - Per-class mAP, Precision, Recall
    - False Positive Rate, False Negative Rate
    
    Args:
        predictions: List of detected boxes [x1, y1, x2, y2, confidence, class]
        targets: List of ground truth boxes [x1, y1, x2, y2, class]
        iou_thresholds: List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])
    
    Returns:
        metrics_dict: Dictionary with all computed metrics
    """
    
def calculate_inference_speed(model, img_size=640, num_iterations=100):
    """
    Measure inference speed in FPS (Frames Per Second).
    
    Args:
        model: Trained YOLOv5 model
        img_size: Image size for inference (default: 640)
        num_iterations: Number of iterations for averaging (default: 100)
    
    Returns:
        fps: Average frames per second
        latency_ms: Average latency in milliseconds
    """
    
def calculate_model_size(model_path):
    """
    Calculate model size metrics.
    
    Returns:
        - Parameters: Total number of parameters
        - FLOPs: Floating point operations
        - Model size in MB
    """
```

### 4.2 Evaluation Script (`src/evaluation/evaluate.py`)

```python
# Comprehensive evaluation pipeline
# IMPORTANT: Operates in sandbox /data directory

def evaluate_model(weights_path, data_yaml_path, split='val'):
    """
    Full evaluation pipeline for trained YOLOv5 model.
    
    Args:
        weights_path: Path to trained model weights (e.g., /data/runs/train/weights/best.pt)
        data_yaml_path: Path to dataset configuration (e.g., /data/coco.yaml)
        split: Dataset split to evaluate (train, val, or test)
    
    Process:
    1. Load trained model from weights_path
    2. Load dataset from data_yaml_path
    3. Run inference on all images in specified split
    4. Compute metrics using calculate_metrics()
    5. Generate confusion matrix
    6. Create precision-recall curves per class
    7. Export evaluation report
    
    Returns:
        evaluation_results: Dictionary with all metrics and results
    """
    
    # Save results to /data/runs/evaluate/
    results_dir = '/data/runs/evaluate/'
    os.makedirs(results_dir, exist_ok=True)
    
    # Export:
    # - evaluation_report.md: Full evaluation summary
    # - metrics.csv: All metrics in CSV format
    # - confusion_matrix.png: Visual confusion matrix
    # - pr_curves.png: Precision-recall curves
```

### 4.3 Error Analysis (`src/evaluation/error_analysis.py`)

```python
# Analyze model errors and identify improvement areas

def analyze_errors(predictions, targets, images_dir, output_dir='/data/runs/evaluate/error_analysis/'):
    """
    Perform detailed error analysis on model predictions.
    
    Analysis includes:
    - Identify misclassified samples (save images)
    - Analyze false positives (visualize with bounding boxes)
    - Analyze false negatives (show missed detections)
    - Detect difficult cases:
      * Occluded objects
      * Small objects (<32x32 pixels)
      * Low confidence predictions
      * Edge cases
    - Class-specific performance breakdown
    - IoU threshold sensitivity analysis
    
    Args:
        predictions: Model predictions
        targets: Ground truth annotations
        images_dir: Directory with test images (e.g., /data/coco/images/val2017/)
        output_dir: Output directory for error analysis (default: /data/runs/evaluate/error_analysis/)
    
    Returns:
        error_report: Dictionary with error statistics and sample images
    """
```

### 4.4 Visualization Jupyter Notebook (`notebooks/evaluation_visualization.ipynb`)

**IMPORTANT**: This notebook MUST be executed in the sandbox environment.

```python
# evaluation_visualization.ipynb
# Interactive visualization of model performance on random test images
# IMPORTANT: All paths must reference /data directory in sandbox

import torch
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration - MUST use sandbox /data directory
DATA_DIR = '/data/coco'
WEIGHTS_PATH = '/data/runs/train/weights/best.pt'
TEST_IMAGES_DIR = f'{DATA_DIR}/images/val2017'  # Use val2017 for evaluation
OUTPUT_DIR = '/data/runs/evaluate/visualization'

# Load model
model = torch.load(WEIGHTS_PATH, map_location='cpu')
model.eval()

# Get random test images
image_files = list(Path(TEST_IMAGES_DIR).glob('*.jpg'))
random.seed(42)
sample_images = random.sample(image_files, min(10, len(image_files)))

# Run inference and visualize
for img_path in sample_images:
    # Load image
    img = cv2.imread(str(img_path))
    
    # Run inference
    results = model(img, size=640)
    
    # Visualize detections
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Plot bounding boxes
    for pred in results[0].boxes:
        x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
        conf = pred.conf[0].cpu().numpy()
        cls = pred.cls[0].cpu().numpy()
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        label = f'Class {int(cls)}: {conf:.2f}'
        plt.text(x1, y1-10, label, color='green', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Detections: {img_path.name}')
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(OUTPUT_DIR) / f'{img_path.stem}_detections.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.show()

# Summary statistics
print(f"Visualized {len(sample_images)} random test images")
print(f"Results saved to: {OUTPUT_DIR}")
```

### 4.5 Evaluation Execution (`scripts/evaluate.sh`)

```bash
#!/bin/bash
# Evaluate trained YOLOv5 model
# IMPORTANT: All operations in sandbox /data directory

# Set paths (sandbox-mounted)
WEIGHTS="/data/runs/train/weights/best.pt"
DATA_YAML="/data/coco.yaml"
OUTPUT_DIR="/data/runs/evaluate"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python src/evaluation/evaluate.py \
  --weights $WEIGHTS \
  --data $DATA_YAML \
  --split val \
  --output $OUTPUT_DIR

# Run error analysis
python src/evaluation/error_analysis.py \
  --weights $WEIGHTS \
  --data $DATA_YAML \
  --split val \
  --output $OUTPUT_DIR/error_analysis

# Generate visualization (run Jupyter notebook in sandbox)
# Note: Execute notebook in sandbox environment
echo "To visualize results, run: jupyter notebook notebooks/evaluation_visualization.ipynb"
echo "IMPORTANT: Must execute in sandbox with /data directory mounted"
```

### 4.6 Evaluation Deliverables

- ✅ **Evaluation Report** (`/data/runs/evaluate/evaluation_report.md`)
  - Overall mAP@0.5 and mAP@0.5:0.95
  - Precision, Recall, F1-score
  - Per-class metrics table
  - Confusion matrix
  - Inference speed (FPS)
  - Model size metrics

- ✅ **Metrics CSV** (`/data/runs/evaluate/metrics.csv`)
  - All metrics in machine-readable format
  - Per-class breakdown
  - Epoch-wise comparison (if multiple checkpoints)

- ✅ **Visualization Plots** (`/data/runs/evaluate/plots/`)
  - Confusion matrix heatmap
  - Precision-recall curves (per class and overall)
  - ROC curves (optional)
  - Loss curves from training

- ✅ **Error Analysis** (`/data/runs/evaluate/error_analysis/`)
  - False positive samples (images with bounding boxes)
  - False negative samples (missed detections)
  - Difficult cases (occluded, small objects)
  - Class-specific error breakdown

- ✅ **Jupyter Notebook** (`notebooks/evaluation_visualization.ipynb`)
  - Interactive visualization of random test images
  - Bounding box overlay with confidence scores
  - Class distribution visualization
  - **IMPORTANT**: Must be executed in sandbox environment

- ✅ **Sample Detection Images** (`/data/runs/evaluate/visualization/`)
  - 10 random test images with detection results
  - High-resolution PNG files (150 DPI)

**IMPORTANT**: All evaluation outputs are saved to `/data/runs/evaluate/` in the sandbox. The Jupyter notebook must be executed in the sandbox environment with `/data` directory mounted.

### 4.7 Phase 4 Completion Targets ✅

**Before moving to Phase 5, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| Evaluation script executed successfully | Exit code 0 | ⬜ |
| Evaluation report generated (`evaluation_report.md`) | File exists and > 5KB | ⬜ |
| Metrics CSV contains all 80 classes | `wc -l metrics.csv` ≥ 81 | ⬜ |
| Confusion matrix plot generated | File exists: `plots/confusion_matrix.png` | ⬜ |
| Error analysis completed | Directory exists: `/data/runs/evaluate/error_analysis/` | ⬜ |
| Jupyter notebook executed with 10 sample images | 10 PNG files in `/data/runs/evaluate/visualization/` | ⬜ |
| mAP@0.5:0.95 ≥ 60% (target benchmark) | Check `metrics.csv` | ⬜ |
| All evaluation code committed to repo | Git log verification | ⬜ |

**Success Criteria:** All 8 targets marked as ✅ completed

---

## Phase 5: Model Export & Inference

### 5.1 Model Checkpoint Management

**IMPORTANT**: The agent MUST operate in the sandbox environment and use the `/data` directory for all model storage.

- **Model Storage Location**: `/data/runs/train/weights/`
- **Checkpoint Format**: Plain PyTorch `.pt` files (no ONNX or TorchScript conversion needed)
- **Do NOT use local filesystem**: Always reference `/data` paths

### 5.2 Model Checkpoint Types

YOLOv5 automatically saves the following checkpoints during training:

| Checkpoint | Location | Description |
|------------|----------|-------------|
| `best.pt` | `/data/runs/train/weights/best.pt` | Best model based on mAP@0.5:0.95 |
| `last.pt` | `/data/runs/train/weights/last.pt` | Last epoch checkpoint |
| `final.pt` | `/data/runs/train/weights/final.pt` | Final training checkpoint |

### 5.3 Export Module (`src/inference/export.py`)

```python
# Model checkpoint management - NO ONNX/TorchScript conversion needed
# IMPORTANT: All paths should reference /data directory in sandbox

def verify_model_checkpoint(checkpoint_path='/data/runs/train/weights/best.pt'):
    """
    Verifies the trained model checkpoint is valid and loadable.
    Operates on data stored in /data/runs/
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (default: /data/runs/train/weights/best.pt)
    
    Returns:
        verification_report: Dictionary with checkpoint validation results
    """
    import torch
    
    # Load the model checkpoint
    model = torch.load(checkpoint_path, map_location='cpu')
    
    # Verify model structure
    assert 'model' in model, "Checkpoint missing model weights"
    assert 'epoch' in model, "Checkpoint missing epoch information"
    
    # Verify model can perform forward pass
    model['model'].eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        output = model['model'](dummy_input)
    
    print(f"✓ Model checkpoint verified successfully: {checkpoint_path}")
    print(f"  - Epoch: {model['epoch']}")
    print(f"  - mAP@0.5:0.95: {model.get('metrics', {}).get('map50_95', 'N/A')}")
    
    return {
        'status': 'valid',
        'checkpoint_path': checkpoint_path,
        'epoch': model['epoch'],
        'map50_95': model.get('metrics', {}).get('map50_95', 'N/A')
    }
```

### 5.4 Detection Module (`src/inference/detect.py`)

```python
# Inference with plain PyTorch checkpoint - NO export conversion needed
# IMPORTANT: All paths should reference /data directory in sandbox

def run_inference(checkpoint_path='/data/runs/train/weights/best.pt', 
                  image_path=None, batch_images=None):
    """
    Runs inference using the trained YOLOv5 model checkpoint.
    Operates on data stored in /data/runs/
    
    Args:
        checkpoint_path: Path to the trained model checkpoint (default: /data/runs/train/weights/best.pt)
        image_path: Single image path for inference
        batch_images: List of image paths for batch inference
    
    Returns:
        detections: List of detection results with bounding boxes and confidence scores
    """
    import torch
    from ultralytics.yolo.engine.model import YOLO
    
    # Load the trained model directly from checkpoint
    model = YOLO(checkpoint_path)
    
    # Run inference
    if image_path:
        results = model.predict(source=image_path, imgsz=640, conf=0.25)
    elif batch_images:
        results = model.predict(source=batch_images, imgsz=640, conf=0.25)
    
    return results
```

### 5.5 Inference Execution (`scripts/inference.sh`)

```bash
#!/bin/bash
# Run inference with trained YOLOv5 model
# IMPORTANT: All paths reference /data directory in sandbox

CHECKPOINT="/data/runs/train/weights/best.pt"
TEST_IMAGES="/data/coco/images/val2017"
OUTPUT_DIR="/data/runs/inference/results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference on validation set (sample of 100 images)
python src/inference/detect.py \
  --weights "$CHECKPOINT" \
  --source "$TEST_IMAGES" \
  --batch-size 16 \
  --img-size 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf \
  --project "$OUTPUT_DIR"

echo "Inference completed. Results saved to: $OUTPUT_DIR"
```

### 5.6 Deliverables

| File | Location | Description |
|------|----------|-------------|
| `best.pt` | `/data/runs/train/weights/best.pt` | Best trained model checkpoint |
| `last.pt` | `/data/runs/train/weights/last.pt` | Last epoch checkpoint |
| `final.pt` | `/data/runs/train/weights/final.pt` | Final training checkpoint |
| Inference results | `/data/runs/inference/results/` | Detection outputs (images, txt files) |
| Verification report | `/data/runs/inference/verification_report.json` | Model checkpoint validation |

### 5.7 Phase 5 Completion Targets ✅

**Before moving to Phase 6, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| Model checkpoint `best.pt` exists | File exists: `/data/runs/train/weights/best.pt` | ⬜ |
| Model checkpoint verified (forward pass works on GPU) | No errors in verification script, GPU used | ⬜ |
| Inference script tested with sample images | Detection results in `/data/runs/inference/results/` | ⬜ |
| Inference speed ≥ 30 FPS on GPU | Benchmark test results (GPU required) | ⬜ |
| Verification report generated | File exists: `/data/runs/inference/verification_report.json` | ⬜ |
| All inference code committed to repo | Git log verification | ⬜ |

**Success Criteria:** All 6 targets marked as ✅ completed

---

## Phase 6: Documentation & Testing

### 6.1 Documentation (`docs/README.md`)
- Project overview
- Installation instructions
- Usage examples
- API reference

### 6.2 Tests (`tests/test_pipeline.py`)
- Test data validation functions
- Test label conversion
- Test model loading
- Test inference pipeline

### 6.3 Test Execution
```bash
python -m pytest tests/ -v
```

### 6.4 Phase 6 Completion Targets ✅

**Before project completion, these targets must be achieved:**

| Target | Verification Method | Status |
|--------|-------------------|--------|
| README.md created with project overview | File exists and > 2KB | ⬜ |
| Installation instructions documented | Section exists in README | ⬜ |
| Usage examples provided | Code snippets in README | ⬜ |
| All tests pass (pytest exit code 0) | `pytest tests/ -v` | ⬜ |
| Test coverage ≥ 70% | `pytest --cov` report | ⬜ |
| All code committed to repo | Git log verification | ⬜ |
| Final commit message: "Complete YOLOv5 implementation" | Git log check | ⬜ |

**Success Criteria:** All 7 targets marked as ✅ completed

---

## Project Completion Checklist ✅

**The entire project is complete when ALL phase targets are achieved:**

| Phase | Target Count | Status |
|-------|-------------|--------|
| Phase 1: Repository Setup | 7 targets | ⬜ |
| Phase 2: Data Preprocessing | 7 targets | ⬜ |
| Phase 3: Model Training | 8 targets | ⬜ |
| Phase 4: Model Evaluation | 8 targets | ⬜ |
| Phase 5: Model Export | 6 targets | ⬜ |
| Phase 6: Documentation & Testing | 7 targets | ⬜ |
| **TOTAL** | **43 targets** | ⬜ |

**Final Success Criteria:** All 43 targets across all 6 phases marked as ✅ completed

**CRITICAL REMINDER:** All operations MUST use GPU acceleration throughout the entire project.

---

## Target Benchmark Scores

### Official YOLOv5 COCO Benchmarks (Image Size: 640x640)

Based on the official YOLOv5 repository benchmarks, the following are the target performance metrics for each model variant:

| Model | Parameters | FLOPs | mAP@0.5:0.95 | mAP@0.5 | Inference Speed (V100) |
|-------|------------|-------|--------------|---------|------------------------|
| YOLOv5n | 1.9M | 4.5G | **28.4%** | 47.9% | 83 ms |
| YOLOv5s | 7.2M | 16.5G | **37.4%** | 52.9% | 99 ms |
| YOLOv5m | 21.2M | 49.0G | **45.4%** | 61.6% | 159 ms |
| YOLOv5l | 46.5M | 109.1G | **49.0%** | 65.5% | 232 ms |
| YOLOv5x | 86.7M | 205.7G | **50.7%** | 67.0% | 302 ms |

### Target Scores for Custom Dataset Training

When training on a custom dataset, the agent should aim to achieve:

| Metric | Target (Small Dataset <1K images) | Target (Medium Dataset 1K-10K images) | Target (Large Dataset >10K images) |
|--------|----------------------------------|--------------------------------------|-----------------------------------|
| mAP@0.5 | ≥ 70% | ≥ 80% | ≥ 85% |
| mAP@0.5:0.95 | ≥ 50% | ≥ 60% | ≥ 65% |
| Precision | ≥ 0.70 | ≥ 0.80 | ≥ 0.85 |
| Recall | ≥ 0.65 | ≥ 0.75 | ≥ 0.80 |
| F1-Score | ≥ 0.67 | ≥ 0.77 | ≥ 0.82 |

### Performance Validation Criteria

The trained model is considered successful if it meets:
- **mAP@0.5:0.95** ≥ 60% (for datasets with >1000 images)
- **Inference Speed** ≥ 30 FPS on NVIDIA GPU
- **Model Size** < 50MB for YOLOv5s variant

### Notes
- Benchmarks are based on COCO val2017 dataset
- Actual performance depends on dataset quality, class balance, and object complexity
- Training for 100-300 epochs is recommended for optimal results
- Use YOLOv5s as the default model for best speed/accuracy tradeoff

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Setup repository structure | 30 minutes |
| Phase 2 | Build data preprocessing pipeline | 2-3 hours |
| Phase 3 | Implement training pipeline | 1-2 hours |
| Phase 4 | Build evaluation suite | 1-2 hours |
| Phase 5 | Export and inference | 30 minutes |
| Phase 6 | Documentation and tests | 1 hour |
| **Total** | | **6-9 hours** |

*Note: Actual training time depends on dataset size and hardware (typically 4-24 hours for 100 epochs)*

---

## Commit Strategy

```bash
# Phase 1
git add . && git commit -m "Initial repository structure"

# Phase 2
git add . && git commit -m "Add data preprocessing pipeline"

# Phase 3
git add . && git commit -m "Implement training pipeline"

# Phase 4
git add . && git commit -m "Add evaluation suite"

# Phase 5
git add . && git commit -m "Add model export and inference"

# Phase 6
git add . && git commit -m "Add documentation and tests"
```

---

## Summary

This plan provides a streamlined approach for an autonomous agent to build a complete YOLOv5 object detection system:

1. **Phase 1**: Repository structure setup
2. **Phase 2**: Data preprocessing (download, validate, augment, convert, split, visualize)
3. **Phase 3**: Training pipeline with YOLOv5
4. **Phase 4**: Evaluation with comprehensive metrics
5. **Phase 5**: Model export (ONNX, TorchScript) and inference
6. **Phase 6**: Documentation and basic testing

### Key Deliverables:
- ✅ Complete codebase in `roatienza/yolov5-implementation`
- ✅ Data preprocessing pipeline
- ✅ Training pipeline
- ✅ Evaluation suite
- ✅ Model export functionality
- ✅ Documentation

### Estimated Execution Time: 6-9 hours (excluding training time)
