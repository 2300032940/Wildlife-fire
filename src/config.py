"""
Configuration settings for Wildlife Camera Trap Detection System
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# ==================== PATHS ====================
# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RUNS_DIR = MODELS_DIR / "runs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_IMAGES_DIR, 
                  CHECKPOINTS_DIR, RUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATASET ====================
# Kaggle dataset
KAGGLE_DATASET = "andrewmvd/wildlife-camera-traps"

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# Image settings
IMG_SIZE = 640  # YOLOv8 default input size
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# ==================== MODEL ====================
# YOLOv8 model variants: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
# n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate)
MODEL_VARIANT = "yolov8s"  # Good balance of speed and accuracy

# Pretrained weights
PRETRAINED_WEIGHTS = f"{MODEL_VARIANT}.pt"

# ==================== TRAINING ====================
# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Image augmentation
AUGMENT = True
HSV_H = 0.015  # Image HSV-Hue augmentation (fraction)
HSV_S = 0.7    # Image HSV-Saturation augmentation (fraction)
HSV_V = 0.4    # Image HSV-Value augmentation (fraction)
DEGREES = 0.0  # Image rotation (+/- deg)
TRANSLATE = 0.1  # Image translation (+/- fraction)
SCALE = 0.5    # Image scale (+/- gain)
SHEAR = 0.0    # Image shear (+/- deg)
PERSPECTIVE = 0.0  # Image perspective (+/- fraction)
FLIPUD = 0.0   # Image flip up-down (probability)
FLIPLR = 0.5   # Image flip left-right (probability)
MOSAIC = 1.0   # Image mosaic (probability)
MIXUP = 0.0    # Image mixup (probability)

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # Options: linear, cosine
COS_LR = True

# Early stopping
PATIENCE = 20  # Epochs to wait for improvement before stopping

# ==================== INFERENCE ====================
# Confidence threshold
CONF_THRESHOLD = 0.5  # Minimum confidence for detection

# IoU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.45

# Maximum detections per image
MAX_DET = 300

# ==================== DEVICE ====================
# Device configuration (auto-detect GPU)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Automatic Mixed Precision for faster training

# ==================== LOGGING ====================
# Verbose output
VERBOSE = True

# Save period (save checkpoint every N epochs)
SAVE_PERIOD = 10

# ==================== CLASSES ====================
# Animal classes (will be populated from dataset)
# This is a placeholder - actual classes will be extracted from annotations
CLASS_NAMES = []

# ==================== VISUALIZATION ====================
# Bounding box colors (BGR format for OpenCV)
BBOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
BBOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# ==================== WEB APP ====================
# Streamlit configuration
APP_TITLE = "Wildlife Camera Trap Detection"
APP_ICON = "🦁"
MAX_UPLOAD_SIZE = 10  # MB

# ==================== HELPER FUNCTIONS ====================
def get_config_dict():
    """Return configuration as dictionary"""
    config = {
        'data': {
            'raw_dir': str(RAW_DATA_DIR),
            'processed_dir': str(PROCESSED_DATA_DIR),
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
        },
        'model': {
            'variant': MODEL_VARIANT,
            'pretrained': PRETRAINED_WEIGHTS,
        },
        'training': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'img_size': IMG_SIZE,
            'lr': LEARNING_RATE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE,
            'device': DEVICE,
        },
        'inference': {
            'conf_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'max_det': MAX_DET,
        }
    }
    return config

def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("Wildlife Detection System Configuration")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_VARIANT}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
