import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .confidence-bar {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🖼️ Image Classification App</h1>', unsafe_allow_html=True)

# Project description
st.markdown("""
<div class="description-box">
    <h3>📋 About This Project</h3>
    <p>This application uses a pre-trained deep learning model to classify uploaded images. 
    The model was trained using Aqsha's model_best and can predict the category of your uploaded images 
    with confidence scores.</p>
    <p><strong>Features:</strong></p>
    <ul>
        <li>Upload and classify images in real-time</li>
        <li>View prediction results with confidence scores</li>
        <li>Automatic image preprocessing and normalization</li>
        <li>User-friendly interface</li>
    </ul>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        # Try to load model_best from common locations
        model_paths = [
            "model_best",
            "model_best.h5",
            "model_best.keras",
            "models/model_best",
            "models/model_best.h5",
            "models/model_best.keras"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path)
                    st.success(f"✅ Model loaded successfully from: {path}")
                    return model
                except Exception as e:
                    st.warning(f"⚠️ Could not load model from {path}: {str(e)}")
                    continue
        
        if model is None:
            st.error("❌ Model not found. Please ensure model_best is in the project directory.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def get_class_labels():
    """Define class labels for CIFAR-10 dataset"""
    return [
        "airplane",    # Class 0
        "automobile",  # Class 1
        "bird",        # Class 2
        "cat",         # Class 3
        "deer",        # Class 4
        "dog",         # Class 5
        "frog",        # Class 6
        "horse",       # Class 7
        "ship",        # Class 8
        "truck"        # Class 9
    ]

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Display model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📊 Model Input Shape: {model.input_shape}")
        st.info(f"📊 Model Output Shape: {model.output_shape}")
    
    with col2:
        st.info(f"📊 Total Parameters: {model.count_params():,}")
        st.info(f"📊 Model Layers: {len(model.layers)}")
    
    st.divider()
    
    # File upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file to classify"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size:,} bytes")
        
        with col2:
            st.subheader("🔍 Preprocessed Image")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Display preprocessed image
                display_img = processed_image[0]  # Remove batch dimension
                st.image(display_img, caption="Preprocessed Image", use_column_width=True)
                
                # Make prediction
                with st.spinner("🔄 Making prediction..."):
                    try:
                        predictions = model.predict(processed_image, verbose=0)
                        
                        # Get class labels
                        class_labels = get_class_labels()
                        
                        # Get top prediction
                        top_prediction_idx = np.argmax(predictions[0])
                        top_confidence = predictions[0][top_prediction_idx]
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.subheader("🎯 Prediction Results")
                        
                        st.write(f"**Predicted Class:** {class_labels[top_prediction_idx]}")
                        st.write(f"**Confidence:** {top_confidence:.4f} ({top_confidence*100:.2f}%)")
                        
                        # Confidence bar
                        confidence_percent = top_confidence * 100
                        st.write(f"**Confidence Level:**")
                        st.progress(float(top_confidence))
                        
                        # Top 5 predictions
                        st.subheader("📈 Top 5 Predictions")
                        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                        
                        for i, idx in enumerate(top_5_indices):
                            confidence = predictions[0][idx]
                            st.write(f"{i+1}. **{class_labels[idx]}**: {confidence:.4f} ({confidence*100:.2f}%)")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
            else:
                st.error("❌ Failed to preprocess image")

if __name__ == "__main__":
    main()
