import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Try to import OpenCV, fallback to PIL if not available
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    st.warning("⚠️ OpenCV not available, using PIL for image processing")

# Page configuration
st.set_page_config(
    page_title="Image Classifier",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
<style>
    /* Simple styling - no fancy gradients or effects */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1000px;
    }
    
    /* Clean header */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #333;
        text-align: center;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #ddd;
        padding-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Simple cards */
    .custom-card {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Simple prediction result */
    .prediction-result {
        background: #007bff;
        color: white;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-result h2 {
        color: white;
        margin: 0.25rem 0;
        font-size: 1.5rem;
    }
    
    .prediction-result h3 {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 1rem;
        font-weight: normal;
    }
    
    /* Simple prediction items */
    .prediction-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    
    .prediction-item:first-child {
        background: #f8f9fa;
        border-left: 3px solid #007bff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image to classify it using machine learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Model Information")
    
    # Model stats will be populated here
    model_stats_container = st.container()
    
    st.markdown("---")
    
    st.markdown("### Supported Classes")
    
    # Class labels will be displayed here
    classes_container = st.container()

# Simple description
st.markdown("""
<div class="custom-card">
    <p style="text-align: center; margin: 0;">
        Upload an image below to classify it. The model can identify 10 different object categories.
    </p>
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
        # Convert to RGB if needed (handle RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image using PIL
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image_resized)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_array.astype(np.float32) / 255.0
        
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

def create_confidence_chart(predictions, class_labels, top_n=5):
    """Create a simple confidence chart"""
    # Get top N predictions
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_confidences = predictions[0][top_indices]
    top_labels = [class_labels[i].title() for i in top_indices]
    
    # Create simple horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_labels,
        x=top_confidences,
        orientation='h',
        marker=dict(color='#007bff'),
        text=[f'{conf:.1%}' for conf in top_confidences],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Confidence",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    fig.update_xaxes(range=[0, 1], tickformat='.0%')
    
    return fig

def create_confidence_gauge(confidence):
    """Create a simple confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence", 'font': {'size': 16, 'color': '#2d3748'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#64748b'},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "#f1f5f9"},
                {'range': [50, 80], 'color': "#e2e8f0"}
            ],
            'threshold': {
                'line': {'color': "#667eea", 'width': 3},
                'thickness': 0.8,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Update sidebar with model information
    with model_stats_container:
        st.markdown(f"**Input Shape:** {model.input_shape}")
        st.markdown(f"**Output Shape:** {model.output_shape}")
        st.markdown(f"**Parameters:** {model.count_params():,}")
        st.markdown(f"**Layers:** {len(model.layers)}")
    
    # Display class labels in sidebar
    with classes_container:
        class_labels = get_class_labels()
        for i, label in enumerate(class_labels):
            st.markdown(f"{i}. {label.title()}")
    
    # File upload section
    st.markdown("### Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file to classify",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file to classify"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Create main layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("#### Preprocessed Image")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Display preprocessed image
                display_img = processed_image[0]  # Remove batch dimension
                st.image(display_img, caption="Preprocessed Image", use_column_width=True)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    try:
                        predictions = model.predict(processed_image, verbose=0)
                        
                        # Get class labels
                        class_labels = get_class_labels()
                        
                        # Get top prediction
                        top_prediction_idx = np.argmax(predictions[0])
                        top_confidence = predictions[0][top_prediction_idx]
                        
                        # Main prediction result
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>{class_labels[top_prediction_idx].title()}</h2>
                            <h3>{top_confidence*100:.1f}% Confidence</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Simple confidence bar
                        st.progress(float(top_confidence))
                        
                        # Top 5 predictions in a simple list
                        st.markdown("#### Top 5 Predictions")
                        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                        
                        for i, idx in enumerate(top_5_indices):
                            confidence = predictions[0][idx]
                            st.markdown(f"""
                            <div class="prediction-item">
                                <div>
                                    <strong>{i+1}. {class_labels[idx].title()}</strong>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-weight: bold;">{confidence*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Optional: Show chart if user wants more details
                        if st.checkbox("Show detailed chart"):
                            st.plotly_chart(create_confidence_chart(predictions, class_labels), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Failed to preprocess image")

if __name__ == "__main__":
    main()
