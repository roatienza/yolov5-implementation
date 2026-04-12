"""
Metrics computation for YOLOv5 evaluation.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_map(predictions: List[Dict], ground_truth: List[Dict], 
                iou_thresholds: np.ndarray = None) -> Dict:
    """
    Compute mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of prediction dictionaries with 'bbox', 'score', 'class'
        ground_truth: List of ground truth dictionaries with 'bbox', 'class'
        iou_thresholds: Array of IoU thresholds (default: 0.5 to 0.95)
    
    Returns:
        Dictionary with mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Get unique classes
    all_classes = set()
    for pred in predictions:
        all_classes.add(pred['class'])
    for gt in ground_truth:
        all_classes.add(gt['class'])
    
    all_classes = sorted(list(all_classes))
    
    # Compute AP for each class
    ap_per_class = {}
    for cls in all_classes:
        cls_preds = [p for p in predictions if p['class'] == cls]
        cls_gts = [g for g in ground_truth if g['class'] == cls]
        
        ap = compute_ap(cls_preds, cls_gts, iou_thresholds)
        ap_per_class[cls] = ap
    
    # Compute mAP
    map_val = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    
    return {
        'map': map_val,
        'ap_per_class': ap_per_class
    }


def compute_ap(predictions: List[Dict], ground_truth: List[Dict],
               iou_thresholds: np.ndarray) -> float:
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: List of predictions for the class
        ground_truth: List of ground truth boxes for the class
        iou_thresholds: Array of IoU thresholds
    
    Returns:
        Average Precision value
    """
    if not predictions or not ground_truth:
        return 0.0
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Compute IoU for all prediction-GT pairs
    ious = []
    for pred in predictions:
        max_iou = 0.0
        for gt in ground_truth:
            iou = compute_iou(pred['bbox'], gt['bbox'])
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
    
    # Compute precision-recall curve
    ious = np.array(ious)
    
    ap_values = []
    for iou_thresh in iou_thresholds:
        # True positives: IoU >= threshold
        tp = (ious >= iou_thresh).astype(int)
        
        # Cumulative sums
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        
        # Precision and Recall
        recall = cum_tp / len(ground_truth)
        precision = cum_tp / (cum_tp + cum_fp)
        
        # Compute AP using interpolation
        ap = compute_interpolated_ap(recall, precision)
        ap_values.append(ap)
    
    return np.mean(ap_values)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union for two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute interpolated Average Precision.
    
    Args:
        recall: Array of recall values
        precision: Array of precision values
    
    Returns:
        Interpolated AP
    """
    # Sort by recall
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    
    # Compute interpolated precision
    max_precision = np.maximum.accumulate(precision[::-1])[::-1]
    
    # Compute AP as area under curve
    ap = np.trapz(max_precision, recall)
    
    return ap


def compute_precision_recall(predictions: List[Dict], ground_truth: List[Dict],
                             iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Compute precision and recall for object detection.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for matching
    
    Returns:
        Tuple of (precision, recall)
    """
    if not predictions:
        return 0.0, 0.0
    
    # Match predictions to ground truth
    tp = 0
    matched_gts = set()
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gts:
                continue
            if pred['class'] != gt['class']:
                continue
            
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gts.add(best_gt_idx)
    
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
    
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
