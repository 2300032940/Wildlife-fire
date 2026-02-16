"""
Utility functions for Wildlife Camera Trap Detection System
Includes visualization, metrics calculation, and helper functions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from PIL import Image

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image with labels and confidence scores
    
    Args:
        image: Input image (numpy array)
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of class labels (indices)
        scores: List of confidence scores
        class_names: List of class names (optional)
        color: BGR color for boxes
        thickness: Line thickness
    
    Returns:
        Image with drawn bounding boxes
    """
    img = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if class_names:
            class_name = class_names[int(label)]
        else:
            class_name = f"Class {int(label)}"
        
        label_text = f"{class_name}: {score:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU score (0 to 1)
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def plot_training_history(
    history_file: Path,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot training history (loss, mAP, precision, recall)
    
    Args:
        history_file: Path to training history CSV
        save_path: Path to save plot (optional)
    """
    import pandas as pd
    
    # Read history
    df = pd.read_csv(history_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot loss
    if 'train_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot mAP
    if 'mAP50' in df.columns:
        axes[0, 1].plot(df['epoch'], df['mAP50'], label='mAP@0.5', linewidth=2)
        if 'mAP50-95' in df.columns:
            axes[0, 1].plot(df['epoch'], df['mAP50-95'], label='mAP@0.5:0.95', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot precision
    if 'precision' in df.columns:
        axes[1, 0].plot(df['epoch'], df['precision'], label='Precision', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall
    if 'recall' in df.columns:
        axes[1, 1].plot(df['epoch'], df['recall'], label='Recall', linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot class distribution as bar chart
    
    Args:
        class_counts: Dictionary of class names and counts
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def resize_image(
    image: np.ndarray,
    target_size: int = 640,
    keep_ratio: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width/height)
        keep_ratio: Keep aspect ratio
    
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    
    if keep_ratio:
        # Calculate scale factor
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        if new_w < target_size or new_h < target_size:
            # Create black canvas
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            # Calculate padding
            top = (target_size - new_h) // 2
            left = (target_size - new_w) // 2
            # Place resized image on canvas
            canvas[top:top+new_h, left:left+new_w] = resized
            resized = canvas
    else:
        # Simple resize without keeping ratio
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        scale = target_size / max(h, w)
    
    return resized, scale


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (BGR)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def save_image(image: np.ndarray, save_path: Path) -> None:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        save_path: Path to save image
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def create_mosaic(images: List[np.ndarray], grid_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Create mosaic of images
    
    Args:
        images: List of images
        grid_size: Grid size (rows, cols)
    
    Returns:
        Mosaic image
    """
    rows, cols = grid_size
    n_images = rows * cols
    
    # Ensure we have enough images
    if len(images) < n_images:
        images = images + [np.zeros_like(images[0])] * (n_images - len(images))
    
    # Get image size
    h, w = images[0].shape[:2]
    
    # Create mosaic
    mosaic = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images[:n_images]):
        row = idx // cols
        col = idx % cols
        mosaic[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    return mosaic


def print_detection_stats(detections: List[Dict]) -> None:
    """
    Print detection statistics
    
    Args:
        detections: List of detection dictionaries
    """
    print("\n" + "="*50)
    print("Detection Statistics")
    print("="*50)
    print(f"Total detections: {len(detections)}")
    
    if detections:
        # Count by class
        from collections import Counter
        class_counts = Counter([d['class'] for d in detections])
        
        print("\nDetections by class:")
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
        
        # Average confidence
        avg_conf = np.mean([d['confidence'] for d in detections])
        print(f"\nAverage confidence: {avg_conf:.3f}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Wildlife Detection Utilities Module")
    print("This module provides helper functions for the detection system")
