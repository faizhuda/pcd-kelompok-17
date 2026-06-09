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
    page_title="Fruit Quality AI Analytics Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-inspired SaaS Dashboard with Orange Accent
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Clean white card styles */
    .saas-card {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 10px 15px -3px rgba(0, 0, 0, 0.03);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 24px;
    }
    .saas-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    /* Sidebar Overrides */
    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Accent Header styling */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #fff0eb;
    }
    
    /* Accuracy badges in sidebar */
    .sidebar-badge {
        background-color: #fff0eb;
        color: #FF6B4A;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #ffe2d9;
        display: inline-block;
    }
    
    /* Prediction status badges */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        font-size: 0.9rem;
        font-weight: 800;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-fresh {
        background-color: #e6fffa;
        color: #0d9488;
        border: 1px solid #b2f5ea;
    }
    .badge-rotten {
        background-color: #fff5f5;
        color: #e53e3e;
        border: 1px solid #fed7d7;
    }
    
    /* Customize native streamlit file uploader to have dashed orange border */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #FF6B4A;
        border-radius: 16px;
        padding: 24px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    /* Zebra-striped comparison table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 0.95rem;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #f1f5f9;
        background-color: #ffffff;
    }
    .styled-table th {
        background-color: #f8fafc;
        color: #475569;
        text-align: left;
        font-weight: 700;
        padding: 14px 16px;
        border-bottom: 2px solid #f1f5f9;
    }
    .styled-table td {
        padding: 14px 16px;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f8fafc;
    }
    .styled-table tbody tr:hover {
        background-color: #f1f5f9;
        transition: background-color 0.2s ease;
    }
    
    /* Pipeline cards layout */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        padding: 16px;
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #f1f5f9;
        margin-bottom: 15px;
    }
    .pipeline-step {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    .pipeline-arrow {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    /* Text layout overrides */
    h1, h2, h3 {
        color: #0f172a !important;
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
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception:
        return None

@st.cache_resource
def load_svm_model():
    model_path = project_root / "results" / "models" / "svm_scenario_05.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

@st.cache_resource
def load_rf_model():
    model_path = project_root / "results" / "models" / "rf_scenario_09.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

# Load models
model_cnn = load_mobilenet_model()
model_svm = load_svm_model()
model_rf = load_rf_model()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
        <span style="font-size: 3.5rem;">🍎</span>
        <h2 style="margin-top: 10px; font-weight: 800; font-size: 1.4rem; color: #0f172a; letter-spacing: -0.5px;">CV Analytics</h2>
        <span style="color: #64748b; font-size: 0.8rem; font-weight: 500;">Academic Demo Platform</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Models Evaluated")
    
    # CNN Card
    st.markdown("""
    <div style="background-color: white; padding: 16px; border-radius: 12px; border: 1px solid #f1f5f9; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
        <h4 style="margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">MobileNetV2 (CNN)</h4>
        <p style="margin: 0 0 10px 0; font-size: 0.75rem; color: #64748b; line-height: 1.3;">Scenario 10: Full Image Processing pipeline + Deep Learning Classification</p>
        <span class="sidebar-badge">98.29% ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)
    
    # SVM Card
    st.markdown("""
    <div style="background-color: white; padding: 16px; border-radius: 12px; border: 1px solid #f1f5f9; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
        <h4 style="margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">Support Vector Machine</h4>
        <p style="margin: 0 0 10px 0; font-size: 0.75rem; color: #64748b; line-height: 1.3;">Scenario 5: 220 manual features (HSV, GLCM, LBP) + SVM Classifier</p>
        <span class="sidebar-badge">95.29% ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)
    
    # RF Card
    st.markdown("""
    <div style="background-color: white; padding: 16px; border-radius: 12px; border: 1px solid #f1f5f9; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
        <h4 style="margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 700; color: #0f172a;">Random Forest</h4>
        <p style="margin: 0 0 10px 0; font-size: 0.75rem; color: #64748b; line-height: 1.3;">Scenario 9: 220 manual features + Random Forest Classifier</p>
        <span class="sidebar-badge">93.79% ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN CONTENT
# ----------------------------------------------------

# Hero Section
st.markdown("""
<div style="margin-top: 10px; margin-bottom: 30px;">
    <h1 style="font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 8px; letter-spacing: -1px;">Fruit Quality Analysis Dashboard</h1>
    <h3 style="font-size: 1.15rem; font-weight: 400; color: #64748b; margin: 0; line-height: 1.5; max-width: 900px;">
        Comparative Evaluation of Classical Machine Learning and Deep Learning Models for Fruit Quality Classification
    </h3>
</div>
""", unsafe_allow_html=True)

# Top KPI Cards
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown("""
    <div class="saas-card">
        <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Best Accuracy</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #FF6B4A; margin-top: 6px; letter-spacing: -1px;">98.29%</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px; font-weight: 500;">MobileNetV2 (Scenario 10)</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col2:
    st.markdown("""
    <div class="saas-card">
        <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Fastest Model</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 6px; letter-spacing: -1px;">Random Forest</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px; font-weight: 500;">0.013 ms / Image latency</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col3:
    st.markdown("""
    <div class="saas-card">
        <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Feature Extraction</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 6px; letter-spacing: -1px;">220 Dimensions</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px; font-weight: 500;">HSV Colors, GLCM Texture, Shape</div>
    </div>
    """, unsafe_allow_html=True)

# Image Upload Section
st.markdown('<div class="section-header">Image Upload & Analysis</div>', unsafe_allow_html=True)

# Render upload widget inside center column for alignment
col_up_1, col_up_2, col_up_3 = st.columns([1, 4, 1])
with col_up_2:
    uploaded_file = st.file_uploader(
        "Unggah gambar buah untuk analisis kualitas...", 
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Mendukung format gambar populer. File akan diproses secara paralel oleh seluruh skenario model."
    )

if uploaded_file is None:
    # ----------------------------------------------------
    # LANDING VIEW: Performance Analytics & Pipeline Info
    # ----------------------------------------------------
    
    # 1. Pipeline Visualizations
    st.markdown('<div class="section-header">Image Processing Pipelines</div>', unsafe_allow_html=True)
    
    col_p_1, col_p_2 = st.columns(2)
    with col_p_1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight: 700; color:#0f172a;'>Classical Machine Learning Pipeline (S5 & S9)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-bottom: 15px;'>Citra mengalami koreksi iluminasi, segmentasi Otsu, ekstraksi manual (220 dimensi), dan klasifikasi menggunakan model SVM / RF.</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-container">
            <span class="pipeline-step">Input Image</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">SSR + CLAHE</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Otsu Segment</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Feature Extr.</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">SVM / RF</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Prediction</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_p_2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight: 700; color:#0f172a;'>Deep Learning Pipeline (S10)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-bottom: 15px;'>Citra mengalami pra-proses (SSR, CLAHE, segmentasi) sebelum ditransfer ke CNN MobileNetV2 dengan fine-tuning 20 layer terakhir.</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-container">
            <span class="pipeline-step">Input Image</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">SSR + CLAHE</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Otsu Segment</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">MobileNetV2</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Fine Tuning</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Prediction</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Performance Analytics
    st.markdown('<div class="section-header">Performance Analytics (Historical Test Set)</div>', unsafe_allow_html=True)
    
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
        
        # Performance Charts Column
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom: 15px;'>Model Performance Comparison Charts</h4>", unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        sns.set_theme(style="white")
        accent_color = "#FF6B4A"
        
        # 1. Accuracy
        sns.barplot(x="model", y="accuracy", data=df_metrics, ax=axes[0], color=accent_color, alpha=0.9, width=0.45)
        axes[0].set_title("Test Accuracy Comparison", fontsize=11, fontweight="bold", pad=15)
        axes[0].set_ylabel("Accuracy", fontsize=9)
        axes[0].set_xlabel("", fontsize=9)
        axes[0].set_ylim(0.80, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height()*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() - 0.05),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9, fontweight="bold", color="white")
        
        # 2. F1-Score
        sns.barplot(x="model", y="f1_weighted", data=df_metrics, ax=axes[1], color="#334155", alpha=0.9, width=0.45)
        axes[1].set_title("Weighted F1-Score Comparison", fontsize=11, fontweight="bold", pad=15)
        axes[1].set_ylabel("F1-Score", fontsize=9)
        axes[1].set_xlabel("", fontsize=9)
        axes[1].set_ylim(0.80, 1.0)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.05),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9, fontweight="bold", color="white")
        
        # 3. Latency
        sns.barplot(x="model", y="inference_time_ms_per_image", data=df_metrics, ax=axes[2], color="#cbd5e1", alpha=0.9, width=0.45)
        axes[2].set_title("Inference Latency per Image (ms)", fontsize=11, fontweight="bold", pad=15)
        axes[2].set_ylabel("Latency (ms)", fontsize=9)
        axes[2].set_xlabel("", fontsize=9)
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.3f} ms", (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight="bold", color="#334155")
            
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#f1f5f9")
            ax.spines['bottom'].set_color("#cbd5e1")
            ax.tick_params(axis='both', colors='#475569', labelsize=9)
            
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison Table
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom: 15px;'>Model Performance Comparison Table</h4>", unsafe_allow_html=True)
        
        table_html = """
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Model Skenario</th>
                    <th>Tipe Model</th>
                    <th>Pra-proses Citra</th>
                    <th>Akurasi Pengujian</th>
                    <th>Weighted F1-Score</th>
                    <th>Latency per Citra</th>
                    <th>Ukuran Sampel</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in df_metrics.iterrows():
            table_html += f"""
                <tr>
                    <td><b>Skenario {row['scenario_id']}</b></td>
                    <td>{row['model']}</td>
                    <td>{row['restoration'].upper()} + {row['enhancement'].upper()} + {'Segmentasi' if row['segmentation'] else 'No-Seg'}</td>
                    <td><span style="font-weight:700; color:#FF6B4A;">{row['accuracy']*100:.2f}%</span></td>
                    <td>{row['f1_weighted']:.4f}</td>
                    <td>{row['inference_time_ms_per_image']:.3f} ms</td>
                    <td>{int(row['n_test_samples'])} citra</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data metrik hasil pengujian tidak ditemukan di folder results/metrics/.")

else:
    # ----------------------------------------------------
    # ACTIVE INFERENCE VIEW: Uploaded Image Processing
    # ----------------------------------------------------
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("Gagal memuat gambar. Harap unggah gambar yang valid.")
    else:
        h, w = img_bgr.shape[:2]
        
        # 1. Run full DIP pipeline
        img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_ssr = apply_ssr(img_resized)
        img_ssr_rgb = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)
        
        img_clahe = apply_enhancement(img_ssr, "clahe")
        img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        
        img_segmented, mask, obj_ratio, fallback = segment_fruit(img_clahe)
        img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Run predictions in the background
        with st.spinner("Mengekstrak fitur dan menjalankan seluruh model..."):
            try:
                features = extract_features(img_clahe, mask, feature_groups="all", segmented=True)
                features_input = features.reshape(1, -1)
            except Exception:
                features_input = None
                
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
        
        # 2. Layout: Preprocessing Steps (Left 3/5) & Predictions (Right 2/5)
        st.markdown('<div class="section-header">Analysis Output & Comparative Prediction</div>', unsafe_allow_html=True)
        
        col_main_1, col_main_2 = st.columns([3, 2])
        
        with col_main_1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom:15px;'>Digital Image Processing (DIP) Pipeline Stages</h4>", unsafe_allow_html=True)
            
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
            with p_col1:
                st.markdown('<div class="dip-step-card"><div class="dip-step-title">1. Resized</div></div>', unsafe_allow_html=True)
                st.image(img_resized_rgb, use_container_width=True)
                st.caption("224x224")
            with p_col2:
                st.markdown('<div class="dip-step-card"><div class="dip-step-title">2. SSR</div></div>', unsafe_allow_html=True)
                st.image(img_ssr_rgb, use_container_width=True)
                st.caption("CIELAB L SSR")
            with p_col3:
                st.markdown('<div class="dip-step-card"><div class="dip-step-title">3. CLAHE</div></div>', unsafe_allow_html=True)
                st.image(img_clahe_rgb, use_container_width=True)
                st.caption("Adaptive Hist")
            with p_col4:
                st.markdown('<div class="dip-step-card"><div class="dip-step-title">4. Mask</div></div>', unsafe_allow_html=True)
                st.image(mask_rgb, use_container_width=True)
                st.caption(f"Ratio: {obj_ratio:.1%}")
            with p_col5:
                st.markdown('<div class="dip-step-card"><div class="dip-step-title">5. Segmented</div></div>', unsafe_allow_html=True)
                st.image(img_segmented_rgb, use_container_width=True)
                st.caption("Final Output")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_main_2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom:15px;'>Model Inference Results</h4>", unsafe_allow_html=True)
            
            # Predict Heuristic Quality Score
            # If fresh: Quality Score is confidence * 10
            # If rotten: Quality Score is (1 - confidence) * 10
            if cnn_pred_class == "fresh":
                quality_score = cnn_confidence * 10.0
            else:
                quality_score = (1.0 - cnn_confidence) * 10.0
            
            # Clamp quality score between 1 and 10
            quality_score = max(1.0, min(10.0, quality_score))
            
            # Display primary summary
            badge_class = "badge-fresh" if cnn_pred_class == "fresh" else "badge-rotten"
            st.markdown(f"""
            <div style="margin-bottom: 20px; border-bottom: 1px solid #f1f5f9; padding-bottom: 15px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Primary Prediction (MobileNetV2)</div>
                <div style="margin-top: 8px;">
                    <span class="badge {badge_class}">{cnn_pred_class}</span>
                </div>
                <div style="display: flex; gap: 40px; margin-top: 15px;">
                    <div>
                        <div style="font-size: 0.75rem; color:#94a3b8; text-transform:uppercase;">CNN Confidence</div>
                        <div style="font-size: 1.6rem; font-weight:800; color:#0f172a;">{cnn_confidence:.2%}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color:#94a3b8; text-transform:uppercase;">Heuristic Quality</div>
                        <div style="font-size: 1.6rem; font-weight:800; color:#FF6B4A;">{quality_score:.1f} / 10</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparative metrics summary
            st.markdown("<div style='font-size: 0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:10px;'>Model Output Comparison</div>", unsafe_allow_html=True)
            
            comp_table = f"""
            <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <th style="text-align:left; padding:8px 0; color:#475569;">Model</th>
                    <th style="text-align:left; padding:8px 0; color:#475569;">Prediksi</th>
                    <th style="text-align:left; padding:8px 0; color:#475569;">Keyakinan / Nilai</th>
                    <th style="text-align:right; padding:8px 0; color:#475569;">Inference</th>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px 0; font-weight:700; color:#0f172a;">MobileNetV2</td>
                    <td><span style="color:{'#0d9488' if cnn_pred_class=='fresh' else '#e53e3e'}; font-weight:700;">{cnn_pred_class.upper()}</span></td>
                    <td>{cnn_confidence:.1%}</td>
                    <td style="text-align:right; font-weight:500;">{cnn_time:.2f} ms</td>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px 0; font-weight:700; color:#0f172a;">SVM</td>
                    <td><span style="color:{'#0d9488' if svm_pred_class=='fresh' else '#e53e3e'}; font-weight:700;">{svm_pred_class.upper()}</span></td>
                    <td>Score: {svm_decision:.3f}</td>
                    <td style="text-align:right; font-weight:500;">{svm_time:.2f} ms</td>
                </tr>
                <tr style="border-bottom:1px solid #f1f5f9;">
                    <td style="padding:10px 0; font-weight:700; color:#0f172a;">Random Forest</td>
                    <td><span style="color:{'#0d9488' if rf_pred_class=='fresh' else '#e53e3e'}; font-weight:700;">{rf_pred_class.upper()}</span></td>
                    <td>{rf_confidence:.1%}</td>
                    <td style="text-align:right; font-weight:500;">{rf_time:.2f} ms</td>
                </tr>
            </table>
            """
            st.markdown(comp_table, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 3. Grad-CAM Explanation
        if model_cnn is not None:
            st.markdown('<div class="section-header">Deep Learning Explainability (Grad-CAM Activation Map)</div>', unsafe_allow_html=True)
            
            with st.spinner("Menghitung heatmap aktivasi Grad-CAM..."):
                try:
                    heatmap = make_gradcam_heatmap(model_cnn, cnn_input)
                    
                    # Compute overlay BGR -> RGB
                    heatmap_resized = cv2.resize(heatmap, (img_segmented.shape[1], img_segmented.shape[0]))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    
                    img_u8 = to_uint8(img_segmented)
                    overlay = cv2.addWeighted(img_u8, 0.65, heatmap_color, 0.35, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        st.markdown('<div class="saas-card" style="text-align: center;">', unsafe_allow_html=True)
                        st.markdown("<h5>Segmented Input (MobileNetV2 Input)</h5>", unsafe_allow_html=True)
                        st.image(img_segmented_rgb, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with g_col2:
                        st.markdown('<div class="saas-card" style="text-align: center;">', unsafe_allow_html=True)
                        st.markdown("<h5>Grad-CAM Attention Heatmap</h5>", unsafe_allow_html=True)
                        st.image(overlay_rgb, use_container_width=True)
                        st.caption("Peta panas berwarna merah/jingga menunjukkan fitur citra yang menjadi fokus utama neural network dalam membedakan kualitas segar vs busuk.")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Grad-CAM could not be generated: {e}")
