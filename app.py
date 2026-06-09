import sys
import time
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add project root to path so we can import src
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import preprocess_from_array, apply_ssr, to_uint8
from src.enhancement import apply_enhancement
from src.segmentation import segment_fruit
from src.features import extract_features
from src.pipeline import process_image, image_to_cnn_input
from src.evaluate import make_gradcam_heatmap, plot_gradcam

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Fruit Quality Analyzer & Model Comparison",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8533 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        border-left: 5px solid #FF4B4B;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Model Card Container */
    .model-card {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.4);
    }
    
    .model-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    
    .badge-fresh {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .badge-rotten {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metrics grid layout */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 15px;
    }
    
    /* DIP Pipeline steps cards */
    .dip-step-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    
    .dip-step-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Model Loading Helper Functions (Cached)
# ----------------------------------------------------
@st.cache_resource
def load_mobilenet_model():
    model_path = project_root / "results" / "models" / "mobilenetv2_s10_stage2.h5"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load MobileNetV2 model: {e}")
        return None

@st.cache_resource
def load_svm_model():
    model_path = project_root / "results" / "models" / "svm_scenario_05.pkl"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load SVM model: {e}")
        return None

@st.cache_resource
def load_rf_model():
    model_path = project_root / "results" / "models" / "rf_scenario_09.pkl"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load Random Forest model: {e}")
        return None

# Load models
model_cnn = load_mobilenet_model()
model_svm = load_svm_model()
model_rf = load_rf_model()

# ----------------------------------------------------
# Dashboard Layout
# ----------------------------------------------------
st.markdown('<div class="main-title">Fruit Quality Model Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Compare classical Machine Learning vs Deep Learning on the Digital Image Processing (DIP) Pipeline</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/apple.png", width=70)
st.sidebar.markdown("### Image Processing Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload a fruit image...", 
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Support images: Apple, Banana, Tomato (Fresh or Rotten)"
)

# Main container
if uploaded_file is None:
    # ----------------------------------------------------
    # Landing / Introduction Section
    # ----------------------------------------------------
    st.info("💡 **Silakan unggah gambar buah di panel kiri untuk memulai analisis.**")
    
    st.markdown('<div class="section-header">Performance Overview (Historical Test Set)</div>', unsafe_allow_html=True)
    
    # Load and display pre-computed test metrics
    col_metric_1, col_metric_2 = st.columns([1, 1])
    
    try:
        metrics_dir = project_root / "results" / "metrics"
        svm_csv = metrics_dir / "scenario_05.csv"
        rf_csv = metrics_dir / "scenario_09.csv"
        cnn_csv = metrics_dir / "scenario_10.csv"
        
        metrics_list = []
        if svm_csv.exists():
            metrics_list.append(pd.read_csv(svm_csv))
        if rf_csv.exists():
            metrics_list.append(pd.read_csv(rf_csv))
        if cnn_csv.exists():
            metrics_list.append(pd.read_csv(cnn_csv))
            
        if metrics_list:
            df_metrics = pd.concat(metrics_list, ignore_index=True)
            # Clean dataframe for display
            df_disp = df_metrics[[
                "model", "accuracy", "f1_weighted", "inference_time_ms_per_image", "n_test_samples"
            ]].copy()
            df_disp.columns = ["Model", "Test Accuracy", "Weighted F1-Score", "Latency per Image (ms)", "Test Samples"]
            df_disp["Test Accuracy"] = df_disp["Test Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            df_disp["Weighted F1-Score"] = df_disp["Weighted F1-Score"].apply(lambda x: f"{x:.4f}")
            df_disp["Latency per Image (ms)"] = df_disp["Latency per Image (ms)"].apply(lambda x: f"{x:.3f} ms")
            
            with col_metric_1:
                st.markdown("##### Historical Test Set Results")
                st.dataframe(df_disp, use_container_width=True, hide_index=True)
                
            with col_metric_2:
                st.markdown("##### Accuracy & Latency Tradeoff")
                fig, ax1 = plt.subplots(figsize=(6, 3))
                
                models = df_metrics["model"].tolist()
                accs = [x * 100 for x in df_metrics["accuracy"]]
                latencies = df_metrics["inference_time_ms_per_image"].tolist()
                
                # Plot Accuracy
                color = '#FF4B4B'
                ax1.set_xlabel('Model')
                ax1.set_ylabel('Accuracy (%)', color=color)
                bars = ax1.bar(models, accs, color=color, alpha=0.6, width=0.4, label='Accuracy')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(80, 100)
                
                # Instantiate a second axes that shares the same x-axis
                ax2 = ax1.twinx()  
                color = '#007FFF'
                ax2.set_ylabel('Latency (ms)', color=color)
                ax2.plot(models, latencies, color=color, marker='o', linewidth=2, label='Latency')
                ax2.tick_params(axis='y', labelcolor=color)
                
                fig.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No pre-computed metrics files found in results/metrics/.")
    except Exception as e:
        st.error(f"Error loading historical performance: {e}")
        
    st.markdown("---")
    st.markdown("""
    ### Tentang Sistem Perbandingan Ini
    Proyek Pengolahan Citra Digital (PCD) ini mengimplementasikan pipeline pemrosesan lengkap untuk mendeteksi kesegaran buah (Fresh / Rotten):
    1. **Restorasi Citra**: Menggunakan Single-Scale Retinex (SSR) pada kanal L (LAB) untuk koreksi pencahayaan.
    2. **Peningkatan Kontras**: Menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk mengoptimalkan detail.
    3. **Segmentasi Otsu**: Memisahkan buah dari latar belakang secara adaptif.
    4. **Ekstraksi Fitur Klasik**: Mengekstrak 220 dimensi fitur warna (hsv histogram & moments), tekstur (GLCM & LBP), dan bentuk.
    5. **Klasifikasi**: Membandingkan **SVM** (klasik), **Random Forest** (klasik), dan **MobileNetV2** (Deep Learning).
    """)

else:
    # Read uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("Error: Gambar tidak dapat dimuat. Pastikan file adalah gambar valid.")
    else:
        # Save BGR image for processing
        h, w = img_bgr.shape[:2]
        
        # ----------------------------------------------------
        # 1. DIP Pipeline Visualizer
        # ----------------------------------------------------
        st.markdown('<div class="section-header">1. Digital Image Processing (DIP) Pipeline Steps</div>', unsafe_allow_html=True)
        
        # Extract individual steps
        img_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to standard input shape (224, 224)
        img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Step 2: SSR
        img_ssr = apply_ssr(img_resized)
        img_ssr_rgb = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
        
        # Step 3: CLAHE
        img_clahe = apply_enhancement(img_ssr, "clahe")
        img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        
        # Step 4: Segmented
        img_segmented, mask, obj_ratio, fallback = segment_fruit(img_clahe)
        img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
        
        # Display steps in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">Original (Resized)</div></div>', unsafe_allow_html=True)
            st.image(img_resized_rgb, use_container_width=True)
            st.caption(f"Dim: {w}x{h} → 224x224")
            
        with col2:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">SSR Illumination</div></div>', unsafe_allow_html=True)
            st.image(img_ssr_rgb, use_container_width=True)
            st.caption("Retinex on L Channel")
            
        with col3:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">CLAHE Enhanced</div></div>', unsafe_allow_html=True)
            st.image(img_clahe_rgb, use_container_width=True)
            st.caption("Adaptive Contrast")
            
        with col4:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">Otsu Fruit Mask</div></div>', unsafe_allow_html=True)
            st.image(mask, use_container_width=True, cmap="gray")
            st.caption(f"Obj ratio: {obj_ratio:.2%}")
            
        with col5:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">Segmented Image</div></div>', unsafe_allow_html=True)
            st.image(img_segmented_rgb, use_container_width=True)
            st.caption("Foreground Output")
            
        # ----------------------------------------------------
        # 2. Perform Model Inference
        # ----------------------------------------------------
        st.markdown('<div class="section-header">2. Multi-Model Inference & Prediction</div>', unsafe_allow_html=True)
        
        # We need features for SVM and RF
        with st.spinner("Extracting features and running classification..."):
            # Extract features from the segmented CLAHE image
            # Features = color + texture + shape (220 dim)
            try:
                features = extract_features(img_clahe, mask, feature_groups="all", segmented=True)
                features_input = features.reshape(1, -1)
            except Exception as e:
                st.error(f"Gagal mengekstrak fitur manual: {e}")
                features_input = None
                
            # Process CNN input
            cnn_input = image_to_cnn_input(img_segmented)
            
            # Predict MobileNetV2
            cnn_pred_class = "N/A"
            cnn_confidence = 0.0
            cnn_time = 0.0
            if model_cnn is not None:
                t0 = time.perf_counter()
                cnn_probs = model_cnn.predict(cnn_input, verbose=0)[0]
                cnn_time = (time.perf_counter() - t0) * 1000
                cnn_idx = np.argmax(cnn_probs)
                cnn_pred_class = "fresh" if cnn_idx == 0 else "rotten"
                cnn_confidence = cnn_probs[cnn_idx]
                
            # Predict SVM
            svm_pred_class = "N/A"
            svm_time = 0.0
            svm_decision = 0.0
            if model_svm is not None and features_input is not None:
                t0 = time.perf_counter()
                svm_idx = model_svm.predict(features_input)[0]
                svm_time = (time.perf_counter() - t0) * 1000
                svm_pred_class = "fresh" if svm_idx == 0 else "rotten"
                try:
                    svm_decision = model_svm.decision_function(features_input)[0]
                except Exception:
                    svm_decision = 0.0
                    
            # Predict Random Forest
            rf_pred_class = "N/A"
            rf_confidence = 0.0
            rf_time = 0.0
            if model_rf is not None and features_input is not None:
                t0 = time.perf_counter()
                rf_idx = model_rf.predict(features_input)[0]
                rf_probs = model_rf.predict_proba(features_input)[0]
                rf_time = (time.perf_counter() - t0) * 1000
                rf_pred_class = "fresh" if rf_idx == 0 else "rotten"
                rf_confidence = rf_probs[rf_idx]
                
        # Display inference results in 3 cards
        m_col1, m_col2, m_col3 = st.columns(3)
        
        # Card 1: CNN
        with m_col1:
            badge_class = "badge-fresh" if cnn_pred_class == "fresh" else "badge-rotten"
            st.markdown(f"""
            <div class="model-card">
                <div class="model-title">MobileNetV2 (CNN)</div>
                <div class="badge {badge_class}">{cnn_pred_class}</div>
                <div class="metrics-grid">
                    <div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{cnn_confidence:.2%}</div>
                    </div>
                    <div>
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">{cnn_time:.2f} ms</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Card 2: SVM
        with m_col2:
            badge_class = "badge-fresh" if svm_pred_class == "fresh" else "badge-rotten"
            # Format decision score representation
            decision_text = f"{svm_decision:.3f}"
            st.markdown(f"""
            <div class="model-card">
                <div class="model-title">Support Vector Machine (SVM)</div>
                <div class="badge {badge_class}">{svm_pred_class}</div>
                <div class="metrics-grid">
                    <div>
                        <div class="metric-label">Decision Score</div>
                        <div class="metric-value">{decision_text}</div>
                    </div>
                    <div>
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">{svm_time:.2f} ms</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Card 3: Random Forest
        with m_col3:
            badge_class = "badge-fresh" if rf_pred_class == "fresh" else "badge-rotten"
            st.markdown(f"""
            <div class="model-card">
                <div class="model-title">Random Forest (RF)</div>
                <div class="badge {badge_class}">{rf_pred_class}</div>
                <div class="metrics-grid">
                    <div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{rf_confidence:.2%}</div>
                    </div>
                    <div>
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">{rf_time:.2f} ms</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # ----------------------------------------------------
        # 3. Grad-CAM Visualization
        # ----------------------------------------------------
        if model_cnn is not None:
            st.markdown('<div class="section-header">3. Deep Learning Explanation (Grad-CAM)</div>', unsafe_allow_html=True)
            
            with st.spinner("Generating Grad-CAM heatmap..."):
                try:
                    heatmap = make_gradcam_heatmap(model_cnn, cnn_input)
                    
                    # Create custom overlay
                    heatmap_resized = cv2.resize(heatmap, (img_segmented.shape[1], img_segmented.shape[0]))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    
                    img_u8 = to_uint8(img_segmented)
                    overlay = cv2.addWeighted(img_u8, 0.6, heatmap_color, 0.4, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    
                    # Display Side-by-Side
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        st.markdown("##### Preprocessed Image (Input to CNN)")
                        st.image(img_segmented_rgb, use_container_width=True)
                    with g_col2:
                        st.markdown("##### Grad-CAM Focus Area (Where CNN Looks)")
                        st.image(overlay_rgb, use_container_width=True)
                        st.caption("Daerah berwarna merah mengindikasikan fitur/area yang paling menentukan keputusan klasifikasi model CNN.")
                except Exception as e:
                    st.warning(f"Grad-CAM could not be generated: {e}")
