"""
Training module for Wildlife Camera Trap Detection
Implements YOLOv8 training pipeline with evaluation metrics
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config


class WildlifeTrainer:
    """Trainer for wildlife detection model"""
    
    def __init__(
        self,
        model_variant: str = None,
        device: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model_variant: YOLOv8 model variant (n/s/m/l/x)
            device: Device to use (cuda/cpu)
        """
        self.model_variant = model_variant or config.MODEL_VARIANT
        self.device = device or config.DEVICE
        
        print("=" * 60)
        print("Wildlife Detection Trainer")
        print("=" * 60)
        print(f"Model: {self.model_variant}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Initialize model
        self.model = None
        
    def load_model(self, pretrained: bool = True, weights_path: str = None):
        """
        Load YOLOv8 model
        
        Args:
            pretrained: Load pretrained weights
            weights_path: Path to custom weights
        """
        print("\n📦 Loading model...")
        
        if weights_path:
            # Load custom weights
            self.model = YOLO(weights_path)
            print(f"✓ Loaded custom weights from {weights_path}")
        elif pretrained:
            # Load pretrained weights
            weights = f"{self.model_variant}.pt"
            self.model = YOLO(weights)
            print(f"✓ Loaded pretrained weights: {weights}")
        else:
            # Load architecture only
            weights = f"{self.model_variant}.yaml"
            self.model = YOLO(weights)
            print(f"✓ Loaded model architecture: {weights}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = None,
        batch_size: int = None,
        img_size: int = None,
        lr: float = None,
        patience: int = None,
        save_dir: str = None,
        **kwargs
    ):
        """
        Train the model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            lr: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save results
            **kwargs: Additional training arguments
        """
        # Use config defaults if not specified
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        img_size = img_size or config.IMG_SIZE
        lr = lr or config.LEARNING_RATE
        patience = patience or config.PATIENCE
        save_dir = save_dir or str(config.RUNS_DIR)
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Learning rate: {lr}")
        print(f"Patience: {patience}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': lr,
            'patience': patience,
            'device': self.device,
            'project': save_dir,
            'name': f'wildlife_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'verbose': config.VERBOSE,
            'save': True,
            'save_period': config.SAVE_PERIOD,
            'plots': True,
            'val': True,
            
            # Data augmentation
            'augment': config.AUGMENT,
            'hsv_h': config.HSV_H,
            'hsv_s': config.HSV_S,
            'hsv_v': config.HSV_V,
            'degrees': config.DEGREES,
            'translate': config.TRANSLATE,
            'scale': config.SCALE,
            'shear': config.SHEAR,
            'perspective': config.PERSPECTIVE,
            'flipud': config.FLIPUD,
            'fliplr': config.FLIPLR,
            'mosaic': config.MOSAIC,
            'mixup': config.MIXUP,
            
            # Learning rate scheduler
            'cos_lr': config.COS_LR,
            
            # Mixed precision
            'amp': config.USE_AMP,
        }
        
        # Update with any additional arguments
        train_args.update(kwargs)
        
        # Train the model
        print("\n🚀 Training started...\n")
        
        try:
            results = self.model.train(**train_args)
            
            print("\n" + "=" * 60)
            print("✓ Training Complete!")
            print("=" * 60)
            
            # Print final metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                print("\n📊 Final Metrics:")
                print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
                print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
                print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
            
            # Save best model to checkpoints
            best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_weights.exists():
                import shutil
                dest = config.CHECKPOINTS_DIR / 'best.pt'
                shutil.copy2(best_weights, dest)
                print(f"\n✓ Best model saved to {dest}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
    
    def validate(self, data_yaml: str, weights_path: str = None):
        """
        Validate the model
        
        Args:
            data_yaml: Path to dataset YAML file
            weights_path: Path to model weights
        """
        print("\n" + "=" * 60)
        print("Validating Model")
        print("=" * 60)
        
        if weights_path:
            self.load_model(pretrained=False, weights_path=weights_path)
        
        results = self.model.val(
            data=data_yaml,
            device=self.device,
            plots=True
        )
        
        print("\n📊 Validation Metrics:")
        print(f"  mAP@0.5: {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(self, weights_path: str, format: str = 'onnx'):
        """
        Export model to different format
        
        Args:
            weights_path: Path to model weights
            format: Export format (onnx, torchscript, etc.)
        """
        print(f"\n📤 Exporting model to {format}...")
        
        self.load_model(pretrained=False, weights_path=weights_path)
        self.model.export(format=format)
        
        print(f"✓ Model exported successfully")


def main():
    """Main function for training"""
    parser = argparse.ArgumentParser(description="Wildlife Detection Training")
    
    # Model arguments
    parser.add_argument('--model', type=str, default=config.MODEL_VARIANT,
                       help='YOLOv8 model variant (n/s/m/l/x)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to pretrained weights')
    
    # Dataset arguments
    parser.add_argument('--data', type=str,
                       default=str(config.PROCESSED_DATA_DIR / 'dataset.yaml'),
                       help='Path to dataset YAML file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=config.IMG_SIZE,
                       help='Input image size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=config.PATIENCE,
                       help='Early stopping patience')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device to use (cuda/cpu)')
    
    # Mode arguments
    parser.add_argument('--validate', action='store_true',
                       help='Run validation only')
    parser.add_argument('--export', type=str, default=None,
                       help='Export model to format (onnx/torchscript)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = WildlifeTrainer(
        model_variant=args.model,
        device=args.device
    )
    
    if args.validate:
        # Validation mode
        trainer.validate(
            data_yaml=args.data,
            weights_path=args.weights
        )
    elif args.export:
        # Export mode
        if not args.weights:
            print("❌ Error: --weights required for export")
            return
        trainer.export_model(
            weights_path=args.weights,
            format=args.export
        )
    else:
        # Training mode
        trainer.load_model(pretrained=True, weights_path=args.weights)
        trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            patience=args.patience
        )


if __name__ == "__main__":
    main()
