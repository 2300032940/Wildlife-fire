# 🚀 Quick Start Guide - Wildlife Detection System

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Kaggle account (free)
- 8GB+ RAM recommended
- GPU optional (but recommended for training)

## Installation

### 1. Navigate to Project
```bash
cd c:\Users\SREER\Documents\Fire\wildlife_detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Kaggle API Setup (Required)

### Step 1: Get API Token
1. Go to https://www.kaggle.com/
2. Sign in or create account
3. Click your profile picture → Account
4. Scroll to "API" section
5. Click "Create New API Token"
6. Download `kaggle.json`

### Step 2: Place Token
- **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- **Linux/Mac**: `~/.kaggle/kaggle.json`

### Step 3: Verify
```bash
kaggle datasets list
```

If this works, you're ready!

## Dataset Preparation

Run this single command to download and prepare everything:

```bash
python -m src.dataset --all
```

This will:
- ✅ Download wildlife dataset from Kaggle (~500MB)
- ✅ Parse annotations
- ✅ Convert to YOLO format
- ✅ Create train/val split
- ✅ Generate dataset.yaml

**Time**: ~5-10 minutes

## Training

### Quick Training (Testing)
```bash
python -m src.train --epochs 5 --batch-size 8
```
**Time**: ~10-15 minutes (GPU) or ~1 hour (CPU)

### Full Training (Recommended)
```bash
python -m src.train --epochs 50 --batch-size 16
```
**Time**: ~2-3 hours (GPU) or ~10-15 hours (CPU)

### Training Tips
- Start with 5 epochs to test everything works
- Then run full 50+ epochs for best results
- Model saves to `models/checkpoints/best.pt`

## Web Application

Launch the web interface:

```bash
streamlit run app.py
```

Then:
1. Open browser at http://localhost:8501
2. Upload a wildlife image
3. See detection results!

## Testing Inference

Test on a single image:

```bash
python -m src.predict \
  --image data/sample_images/test.jpg \
  --weights models/checkpoints/best.pt \
  --conf 0.5
```

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install -r requirements.txt
```

### "Could not find kaggle.json"
- Check file location: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- Ensure file exists and is named exactly `kaggle.json`

### "CUDA out of memory"
```bash
# Reduce batch size
python -m src.train --batch-size 4 --img-size 416
```

### "No trained models found" (Web App)
```bash
# Train a model first
python -m src.train --epochs 5 --batch-size 8
```

## Next Steps

1. ✅ Set up Kaggle API
2. ✅ Download dataset
3. ✅ Train model (start with 5 epochs)
4. ✅ Test web app
5. ✅ Run full training (50+ epochs)
6. ✅ Deploy and use!

## Full Documentation

- [README.md](README.md) - Complete documentation
- [Implementation Plan](../brain/implementation_plan.md) - Technical details
- [Walkthrough](../brain/walkthrough.md) - Detailed guide

## Support

For detailed instructions, see the main [README.md](README.md)

---

**Happy Wildlife Detection! 🦁🐻🦌**
