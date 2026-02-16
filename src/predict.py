"""
Prediction/Inference module for Wildlife Camera Trap Detection
Performs object detection on images and visualizes results
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.utils import draw_bounding_boxes, save_image, load_image, print_detection_stats


class WildlifeDetector:
    """Wildlife detection inference class"""
    
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = None,
        iou_threshold: float = None,
        device: str = None
    ):
        """
        Initialize detector
        
        Args:
            weights_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use (cuda/cpu)
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold or config.CONF_THRESHOLD
        self.iou_threshold = iou_threshold or config.IOU_THRESHOLD
        self.device = device or config.DEVICE
        
        print("=" * 60)
        print("Wildlife Detector")
        print("=" * 60)
        print(f"Weights: {self.weights_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Load model
        self.model = self._load_model()
        
        # Get class names
        self.class_names = self.model.names
        print(f"\n✓ Model loaded with {len(self.class_names)} classes")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        
        model = YOLO(str(self.weights_path))
        return model
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> Dict:
        """
        Predict on single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image
            show: Show image with detections
        
        Returns:
            Dictionary with detection results
        """
        image_path = Path(image_path)
        
        # Load image
        image = load_image(image_path)
        
        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        boxes = []
        labels = []
        scores = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Store detection
                detections.append({
                    'class': self.class_names[cls],
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': xyxy.tolist()
                })
                
                boxes.append(xyxy)
                labels.append(cls)
                scores.append(conf)
        
        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(
            image,
            boxes,
            labels,
            scores,
            class_names=self.class_names
        )
        
        # Save annotated image
        if save_path:
            save_image(annotated_image, save_path)
            print(f"✓ Saved annotated image to {save_path}")
        
        # Show image
        if show:
            cv2.imshow('Detection Results', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        result = {
            'image_path': str(image_path),
            'num_detections': len(detections),
            'detections': detections,
            'annotated_image': annotated_image
        }
        
        return result
    
    def predict_batch(
        self,
        image_dir: Union[str, Path],
        save_dir: Optional[Path] = None,
        extensions: List[str] = None
    ) -> List[Dict]:
        """
        Predict on batch of images
        
        Args:
            image_dir: Directory containing images
            save_dir: Directory to save annotated images
            extensions: Image file extensions to process
        
        Returns:
            List of detection results
        """
        image_dir = Path(image_dir)
        extensions = extensions or config.IMG_EXTENSIONS
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
        
        print(f"\n📊 Processing {len(image_files)} images...")
        
        # Create save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        all_results = []
        
        for img_file in tqdm(image_files, desc="Detecting"):
            # Determine save path
            if save_dir:
                save_path = save_dir / f"detected_{img_file.name}"
            else:
                save_path = None
            
            # Predict
            result = self.predict_image(
                image_path=img_file,
                save_path=save_path,
                show=False
            )
            
            all_results.append(result)
        
        # Print summary statistics
        total_detections = sum(r['num_detections'] for r in all_results)
        images_with_detections = sum(1 for r in all_results if r['num_detections'] > 0)
        
        print("\n" + "=" * 60)
        print("Batch Detection Summary")
        print("=" * 60)
        print(f"Total images: {len(all_results)}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(all_results):.2f}")
        print("=" * 60)
        
        return all_results
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        save_path: Optional[Path] = None,
        show: bool = False
    ) -> None:
        """
        Predict on video (optional feature)
        
        Args:
            video_path: Path to input video
            save_path: Path to save annotated video
            show: Show video with detections
        """
        video_path = Path(video_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Create video writer
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run inference on frame
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )[0]
                
                # Get annotated frame
                annotated_frame = results.plot()
                
                # Save frame
                if save_path:
                    out.write(annotated_frame)
                
                # Show frame
                if show:
                    cv2.imshow('Video Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {frame_count} frames")
        if save_path:
            print(f"✓ Saved annotated video to {save_path}")


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="Wildlife Detection Inference")
    
    # Model arguments
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    
    # Input arguments
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to input directory or video')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--show', action='store_true',
                       help='Show detection results')
    
    # Detection arguments
    parser.add_argument('--conf', type=float, default=config.CONF_THRESHOLD,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=config.IOU_THRESHOLD,
                       help='IoU threshold for NMS')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = WildlifeDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        # Single image prediction
        print(f"\n🔍 Detecting animals in {args.image}...")
        
        result = detector.predict_image(
            image_path=args.image,
            save_path=save_dir / f"detected_{Path(args.image).name}",
            show=args.show
        )
        
        # Print results
        print(f"\n✓ Found {result['num_detections']} animal(s)")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class']}: {det['confidence']:.3f}")
    
    elif args.source:
        source_path = Path(args.source)
        
        if source_path.is_dir():
            # Batch prediction
            detector.predict_batch(
                image_dir=source_path,
                save_dir=save_dir
            )
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video prediction
            detector.predict_video(
                video_path=source_path,
                save_path=save_dir / f"detected_{source_path.name}",
                show=args.show
            )
        else:
            print(f"❌ Unsupported source: {source_path}")
    
    else:
        parser.print_help()
        print("\n❌ Error: Please specify --image or --source")


if __name__ == "__main__":
    main()
