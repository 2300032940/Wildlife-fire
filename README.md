# Wildlife Camera Trap Detection System

A production-level deep learning system for detecting and localizing animals in forest camera trap images using YOLOv8 object detection.

## 📋 Project Overview

This project implements an AI-powered wildlife detection system that:
- Detects and localizes animals in camera trap images
- Uses YOLOv8 for state-of-the-art object detection
- Handles multiple animals in a single image
- Provides bounding boxes, class labels, and confidence scores
- Includes a web interface for easy deployment

## 🎯 Features

- **Object Detection**: YOLOv8-based detection with bounding boxes
- **Multi-Animal Support**: Detects multiple animals in one image
- **Transfer Learning**: Fine-tuned on wildlife dataset
- **Web Interface**: Streamlit app for easy image upload and inference
- **Production Ready**: Clean, modular, well-documented code
- **GPU Accelerated**: Automatic GPU detection and utilization

## 📊 Dataset

**Source**: [Wildlife Camera Traps - Kaggle](https://www.kaggle.com/datasets/andrewmvd/wildlife-camera-traps)

The dataset contains camera trap images with bounding box annotations for various wildlife species.

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Dataset**: Kaggle API

## 📁 Project Structure

```
wildlife_detection/
│
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   ├── processed/              # Processed YOLO format
│   └── sample_images/          # Test images
│
├── models/
│   ├── checkpoints/            # Saved model weights
│   └── runs/                   # Training logs and results
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── dataset.py              # Dataset handling
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Inference module
│   └── utils.py                # Utility functions
│
├── notebooks/
│   └── data_exploration.ipynb  # Dataset analysis
│
├── app.py                      # Streamlit web app
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd wildlife_detection
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Kaggle API

To download the dataset, you need Kaggle API credentials:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json`
4. Place it in the appropriate location:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
5. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 📥 Dataset Preparation

### Download and Process Dataset

```bash
# Download dataset from Kaggle
python -m src.dataset --download

# Convert annotations to YOLO format
python -m src.dataset --convert

# Verify dataset
python -m src.dataset --verify
```

This will:
- Download the wildlife camera trap dataset
- Parse XML annotations (PASCAL VOC format)
- Convert to YOLO format (class x_center y_center width height)
- Create train/validation split (80/20)
- Generate dataset statistics

## 🏋️ Training

### Basic Training

```bash
python -m src.train --epochs 50 --batch-size 16 --img-size 640
```

### Advanced Training Options

```bash
python -m src.train \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --model yolov8s \
  --lr 0.01 \
  --patience 20 \
  --device cuda
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--model`: YOLOv8 model variant (n/s/m/l/x, default: s)
- `--lr`: Learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 20)
- `--device`: Device to use (cuda/cpu, default: auto-detect)

### Training Output

Training will generate:
- Model checkpoints in `models/checkpoints/`
- Training logs in `models/runs/`
- Performance curves (loss, mAP, precision, recall)
- Validation results

## 🔮 Inference

### Single Image Prediction

```bash
python -m src.predict \
  --image data/sample_images/test.jpg \
  --weights models/checkpoints/best.pt \
  --conf 0.5
```

### Batch Prediction

```bash
python -m src.predict \
  --source data/sample_images/ \
  --weights models/checkpoints/best.pt \
  --conf 0.5 \
  --save-dir results/
```

**Parameters:**
- `--image` or `--source`: Input image or directory
- `--weights`: Path to trained model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--save-dir`: Directory to save results

## 🌐 Web Application

### Launch Streamlit App

```bash
streamlit run app.py
```

The web app will open in your browser at `http://localhost:8501`

### Features:
- Upload wildlife images (JPG, PNG)
- Real-time detection with bounding boxes
- Adjustable confidence threshold
- Detection statistics and results table
- Download annotated images

## 📈 Model Performance

Expected performance metrics (after fine-tuning):

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.65-0.75 |
| Precision | 0.70-0.80 |
| Recall | 0.65-0.75 |
| Inference Speed | ~50-100 FPS (GPU) |

## 🧪 Evaluation Metrics

The system computes:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **IoU (Intersection over Union)**: Bounding box overlap
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model yolov8n`
- Reduce image size: `--img-size 416`

### Kaggle API Issues
- Verify `kaggle.json` is in correct location
- Check file permissions (Linux/Mac)
- Ensure Kaggle account is verified

### Low Performance
- Train for more epochs
- Use larger model (yolov8m or yolov8l)
- Increase image augmentation
- Check dataset quality and annotations

## 🔄 Future Improvements

- [ ] Add video inference support
- [ ] Implement species classification
- [ ] Add behavior recognition
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Create mobile app version
- [ ] Add real-time camera integration
- [ ] Implement tracking across frames

## 📝 License

This project is for educational purposes. Please check the dataset license on Kaggle.

## 🙏 Acknowledgments

- **Dataset**: Wildlife Camera Traps dataset on Kaggle
- **Model**: Ultralytics YOLOv8
- **Framework**: PyTorch

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for wildlife conservation**
