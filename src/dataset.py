"""
Dataset handling module for Wildlife Camera Trap Detection
Handles dataset download, preprocessing, and YOLO format conversion
"""

import os
import sys
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.utils import plot_class_distribution


class WildlifeDatasetHandler:
    """Handler for wildlife camera trap dataset"""
    
    def __init__(self):
        self.raw_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.kaggle_dataset = config.KAGGLE_DATASET
        
        # Class mapping (will be populated from dataset)
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def download_dataset(self) -> None:
        """Download dataset from Kaggle"""
        print("=" * 60)
        print("Downloading Wildlife Camera Trap Dataset from Kaggle")
        print("=" * 60)
        
        try:
            import kaggle
            
            # Download dataset
            print(f"\nDownloading {self.kaggle_dataset}...")
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=self.raw_dir,
                unzip=True
            )
            
            print(f"✓ Dataset downloaded to {self.raw_dir}")
            
        except Exception as e:
            print(f"\n❌ Error downloading dataset: {e}")
            print("\nPlease ensure:")
            print("1. You have a Kaggle account")
            print("2. Kaggle API token (kaggle.json) is set up")
            print("3. Token is in the correct location:")
            print("   - Windows: C:\\Users\\<Username>\\.kaggle\\kaggle.json")
            print("   - Linux/Mac: ~/.kaggle/kaggle.json")
            print("\nTo get your API token:")
            print("1. Go to https://www.kaggle.com/")
            print("2. Click on your profile → Account")
            print("3. Scroll to API section → Create New API Token")
            sys.exit(1)
    
    def analyze_dataset_structure(self) -> Dict:
        """Analyze dataset structure and return statistics"""
        print("\n" + "=" * 60)
        print("Analyzing Dataset Structure")
        print("=" * 60)
        
        # Find all images and annotations
        image_files = []
        annotation_files = []
        
        for ext in config.IMG_EXTENSIONS:
            image_files.extend(list(self.raw_dir.rglob(f"*{ext}")))
        
        annotation_files = list(self.raw_dir.rglob("*.xml"))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Total annotations: {len(annotation_files)}")
        
        # Parse annotations to get class information
        class_counts = Counter()
        total_objects = 0
        
        for ann_file in tqdm(annotation_files, desc="Parsing annotations"):
            try:
                tree = ET.parse(ann_file)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    class_counts[name] += 1
                    total_objects += 1
            except Exception as e:
                print(f"Warning: Failed to parse {ann_file}: {e}")
        
        print(f"\n  Total objects: {total_objects}")
        print(f"  Number of classes: {len(class_counts)}")
        print(f"\n  Class distribution:")
        for class_name, count in class_counts.most_common():
            print(f"    {class_name}: {count}")
        
        # Create class mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_counts.keys()))}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Save class mapping
        class_mapping_file = self.processed_dir / "class_mapping.json"
        with open(class_mapping_file, 'w') as f:
            json.dump({
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }, f, indent=2)
        
        print(f"\n✓ Class mapping saved to {class_mapping_file}")
        
        # Update config
        config.CLASS_NAMES = list(self.class_to_idx.keys())
        
        stats = {
            'num_images': len(image_files),
            'num_annotations': len(annotation_files),
            'num_objects': total_objects,
            'num_classes': len(class_counts),
            'class_counts': dict(class_counts)
        }
        
        return stats
    
    def parse_voc_annotation(self, xml_file: Path) -> Tuple[List, Tuple[int, int]]:
        """
        Parse PASCAL VOC format XML annotation
        
        Args:
            xml_file: Path to XML file
        
        Returns:
            List of objects and image size
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Parse objects
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            objects.append({
                'class': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return objects, (img_width, img_height)
    
    def convert_to_yolo_format(
        self,
        bbox: List[int],
        img_size: Tuple[int, int]
    ) -> List[float]:
        """
        Convert PASCAL VOC bbox to YOLO format
        
        VOC format: [xmin, ymin, xmax, ymax]
        YOLO format: [x_center, y_center, width, height] (normalized 0-1)
        
        Args:
            bbox: Bounding box in VOC format
            img_size: Image size (width, height)
        
        Returns:
            Bounding box in YOLO format
        """
        xmin, ymin, xmax, ymax = bbox
        img_width, img_height = img_size
        
        # Calculate center, width, height
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Normalize to 0-1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def convert_dataset_to_yolo(self) -> None:
        """Convert entire dataset to YOLO format"""
        print("\n" + "=" * 60)
        print("Converting Dataset to YOLO Format")
        print("=" * 60)
        
        # Create output directories
        images_dir = self.processed_dir / "images"
        labels_dir = self.processed_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Find all annotation files
        annotation_files = list(self.raw_dir.rglob("*.xml"))
        
        print(f"\nProcessing {len(annotation_files)} annotations...")
        
        converted_count = 0
        skipped_count = 0
        
        for ann_file in tqdm(annotation_files, desc="Converting"):
            try:
                # Parse annotation
                objects, img_size = self.parse_voc_annotation(ann_file)
                
                # Find corresponding image
                img_name = ann_file.stem
                img_file = None
                for ext in config.IMG_EXTENSIONS:
                    potential_img = ann_file.parent / f"{img_name}{ext}"
                    if potential_img.exists():
                        img_file = potential_img
                        break
                
                if img_file is None:
                    skipped_count += 1
                    continue
                
                # Copy image to processed directory
                dest_img = images_dir / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Create YOLO format label file
                label_file = labels_dir / f"{img_name}.txt"
                
                with open(label_file, 'w') as f:
                    for obj in objects:
                        class_idx = self.class_to_idx[obj['class']]
                        yolo_bbox = self.convert_to_yolo_format(obj['bbox'], img_size)
                        
                        # Write: class_idx x_center y_center width height
                        f.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")
                
                converted_count += 1
                
            except Exception as e:
                print(f"\nWarning: Failed to process {ann_file}: {e}")
                skipped_count += 1
        
        print(f"\n✓ Conversion complete!")
        print(f"  Converted: {converted_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Images saved to: {images_dir}")
        print(f"  Labels saved to: {labels_dir}")
    
    def create_train_val_split(self) -> None:
        """Create train/validation split"""
        print("\n" + "=" * 60)
        print("Creating Train/Validation Split")
        print("=" * 60)
        
        images_dir = self.processed_dir / "images"
        labels_dir = self.processed_dir / "labels"
        
        # Get all image files
        image_files = sorted(list(images_dir.glob("*.*")))
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * config.TRAIN_RATIO)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"\n📊 Split Statistics:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Training: {len(train_files)} ({config.TRAIN_RATIO*100:.0f}%)")
        print(f"  Validation: {len(val_files)} ({config.VAL_RATIO*100:.0f}%)")
        
        # Create train/val directories
        for split in ['train', 'val']:
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Move files
        print("\nMoving files to train/val directories...")
        
        for img_file in tqdm(train_files, desc="Train"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            shutil.copy2(img_file, self.processed_dir / 'train' / 'images' / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, self.processed_dir / 'train' / 'labels' / label_file.name)
        
        for img_file in tqdm(val_files, desc="Val"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            shutil.copy2(img_file, self.processed_dir / 'val' / 'images' / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, self.processed_dir / 'val' / 'labels' / label_file.name)
        
        print(f"\n✓ Split complete!")
        print(f"  Train: {self.processed_dir / 'train'}")
        print(f"  Val: {self.processed_dir / 'val'}")
    
    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml file for YOLOv8"""
        print("\n" + "=" * 60)
        print("Creating dataset.yaml for YOLOv8")
        print("=" * 60)
        
        yaml_content = f"""# Wildlife Camera Trap Dataset Configuration
path: {self.processed_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(self.class_to_idx)}

# Class names
names:
"""
        
        for idx, name in self.idx_to_class.items():
            yaml_content += f"  {idx}: {name}\n"
        
        yaml_file = self.processed_dir / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ dataset.yaml created at {yaml_file}")
    
    def verify_dataset(self) -> None:
        """Verify dataset integrity"""
        print("\n" + "=" * 60)
        print("Verifying Dataset")
        print("=" * 60)
        
        for split in ['train', 'val']:
            images_dir = self.processed_dir / split / 'images'
            labels_dir = self.processed_dir / split / 'labels'
            
            images = list(images_dir.glob("*.*"))
            labels = list(labels_dir.glob("*.txt"))
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
            
            # Check for missing labels
            missing_labels = 0
            for img in images:
                label_file = labels_dir / f"{img.stem}.txt"
                if not label_file.exists():
                    missing_labels += 1
            
            if missing_labels > 0:
                print(f"  ⚠ Missing labels: {missing_labels}")
            else:
                print(f"  ✓ All images have labels")
        
        print("\n" + "=" * 60)


def main():
    """Main function for dataset preparation"""
    parser = argparse.ArgumentParser(description="Wildlife Dataset Preparation")
    parser.add_argument('--download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--convert', action='store_true', help='Convert to YOLO format')
    parser.add_argument('--verify', action='store_true', help='Verify dataset')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    handler = WildlifeDatasetHandler()
    
    if args.all or args.download:
        handler.download_dataset()
        stats = handler.analyze_dataset_structure()
    
    if args.all or args.convert:
        # Load class mapping if not already loaded
        class_mapping_file = config.PROCESSED_DATA_DIR / "class_mapping.json"
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r') as f:
                mapping = json.load(f)
                handler.class_to_idx = mapping['class_to_idx']
                handler.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        
        handler.convert_dataset_to_yolo()
        handler.create_train_val_split()
        handler.create_dataset_yaml()
    
    if args.all or args.verify:
        handler.verify_dataset()
    
    if not any([args.download, args.convert, args.verify, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
