"""
Streamlit Web Application for Wildlife Camera Trap Detection
Upload images and get real-time animal detection with bounding boxes
"""

import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src import config
from src.predict import WildlifeDetector


# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .detection-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(weights_path, conf_threshold, iou_threshold):
    """Load detector model (cached)"""
    try:
        detector = WildlifeDetector(
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        return detector
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🦁 Wildlife Camera Trap Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Animal Detection in Forest Camera Trap Images</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Configuration")
        
        # Find available model weights
        checkpoint_dir = config.CHECKPOINTS_DIR
        available_weights = list(checkpoint_dir.glob("*.pt"))
        
        if not available_weights:
            st.error("❌ No trained models found!")
            st.info("Please train a model first using:\n```\npython -m src.train --epochs 50\n```")
            st.stop()
        
        # Select weights
        weights_options = {w.name: str(w) for w in available_weights}
        selected_weights_name = st.selectbox(
            "Select Model Weights",
            options=list(weights_options.keys()),
            index=0
        )
        weights_path = weights_options[selected_weights_name]
        
        # Detection parameters
        st.subheader("Detection Parameters")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.CONF_THRESHOLD,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.0,
            max_value=1.0,
            value=config.IOU_THRESHOLD,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        # About section
        st.subheader("ℹ️ About")
        st.info(
            "This application uses YOLOv8 deep learning model to detect "
            "and localize animals in camera trap images. Upload an image "
            "to see the detection results with bounding boxes and confidence scores."
        )
        
        # Model info
        with st.expander("📊 Model Information"):
            st.write(f"**Model:** {config.MODEL_VARIANT}")
            st.write(f"**Device:** {config.DEVICE}")
            st.write(f"**Image Size:** {config.IMG_SIZE}px")
    
    # Load detector
    detector = load_detector(weights_path, conf_threshold, iou_threshold)
    
    if detector is None:
        st.stop()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Statistics", "ℹ️ Instructions"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Wildlife Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a camera trap image for animal detection"
        )
        
        if uploaded_file is not None:
            # Create columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                # Load image
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # Image info
                st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                # Run detection
                with st.spinner("🔍 Detecting animals..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    # Predict
                    result = detector.predict_image(
                        image_path=tmp_path,
                        save_path=None,
                        show=False
                    )
                    
                    # Display annotated image
                    annotated_image = result['annotated_image']
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, use_container_width=True)
                    
                    # Detection count
                    num_detections = result['num_detections']
                    if num_detections > 0:
                        st.success(f"✅ Found {num_detections} animal(s)!")
                    else:
                        st.warning("⚠️ No animals detected. Try adjusting the confidence threshold.")
            
            # Detection details
            if result['num_detections'] > 0:
                st.subheader("📋 Detection Details")
                
                # Create metrics row
                cols = st.columns(min(4, result['num_detections']))
                
                for i, detection in enumerate(result['detections'][:4]):
                    with cols[i]:
                        st.metric(
                            label=f"Detection {i+1}",
                            value=detection['class'],
                            delta=f"{detection['confidence']:.2%}"
                        )
                
                # Detection table
                st.subheader("📊 Detailed Results")
                
                # Create table data
                table_data = []
                for i, det in enumerate(result['detections'], 1):
                    bbox = det['bbox']
                    table_data.append({
                        '#': i,
                        'Animal': det['class'],
                        'Confidence': f"{det['confidence']:.3f}",
                        'Bbox (x1, y1, x2, y2)': f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
                    })
                
                st.table(table_data)
                
                # Download button for annotated image
                st.subheader("💾 Download Results")
                
                # Convert to bytes
                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                if is_success:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                
                # Download JSON results
                json_data = {
                    'num_detections': result['num_detections'],
                    'detections': [
                        {
                            'class': d['class'],
                            'confidence': d['confidence'],
                            'bbox': d['bbox']
                        }
                        for d in result['detections']
                    ]
                }
                
                st.download_button(
                    label="📥 Download Detection Data (JSON)",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"detections_{uploaded_file.name}.json",
                    mime="application/json"
                )
    
    with tab2:
        st.subheader("📊 Model Statistics")
        
        # Load class mapping if available
        class_mapping_file = config.PROCESSED_DATA_DIR / "class_mapping.json"
        
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r') as f:
                class_mapping = json.load(f)
            
            st.write("**Detected Classes:**")
            
            # Display classes in columns
            classes = sorted(class_mapping['class_to_idx'].keys())
            
            # Create columns for classes
            num_cols = 3
            cols = st.columns(num_cols)
            
            for i, class_name in enumerate(classes):
                with cols[i % num_cols]:
                    st.write(f"• {class_name}")
            
            st.write(f"\n**Total Classes:** {len(classes)}")
        else:
            st.info("No class mapping found. Please prepare the dataset first.")
        
        # Model performance (if available)
        st.subheader("🎯 Model Performance")
        st.info(
            "Model performance metrics will be displayed here after training. "
            "Typical metrics include mAP@0.5, Precision, Recall, and F1 Score."
        )
    
    with tab3:
        st.subheader("📖 How to Use")
        
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Upload Image**
           - Click on "Upload Wildlife Image" button
           - Select a camera trap image (JPG, PNG, or BMP)
           - The image will be displayed on the left
        
        2. **Adjust Settings** (Optional)
           - Use the sidebar to adjust detection parameters
           - **Confidence Threshold**: Higher values = fewer but more confident detections
           - **IoU Threshold**: Controls overlap tolerance for multiple detections
        
        3. **View Results**
           - Detected animals will be shown with bounding boxes
           - Each detection includes:
             - Animal class/species
             - Confidence score
             - Bounding box coordinates
        
        4. **Download Results**
           - Download the annotated image with bounding boxes
           - Download detection data in JSON format
        
        ### Tips for Best Results
        
        - ✅ Use clear, well-lit images
        - ✅ Ensure animals are visible and not heavily occluded
        - ✅ Adjust confidence threshold if too many/few detections
        - ✅ Try different model weights if available
        
        ### Troubleshooting
        
        - **No detections?** Try lowering the confidence threshold
        - **Too many false positives?** Increase the confidence threshold
        - **Model not loading?** Ensure you have trained a model first
        """)
        
        st.subheader("🔧 Training Your Own Model")
        
        st.code("""
# 1. Download and prepare dataset
python -m src.dataset --all

# 2. Train model
python -m src.train --epochs 50 --batch-size 16

# 3. Run this web app
streamlit run app.py
        """, language="bash")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">'
        'Built with ❤️ for Wildlife Conservation | Powered by YOLOv8'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
