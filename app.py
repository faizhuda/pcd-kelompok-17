import sys
import textwrap
import time
import io
import base64
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

def arr_to_base64(img_rgb):
    # Convert RGB to BGR for cv2 encoding
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import preprocess_from_array, apply_ssr, to_uint8
from src.enhancement import apply_enhancement
from src.segmentation import segment_fruit
from src.features import extract_features
from src.pipeline import process_image, image_to_cnn_input
from src.evaluate import make_gradcam_heatmap

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
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-sizing: border-box;
    }
    .saas-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    .card-title {
        margin-top: 0 !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        border-bottom: 1px solid #f1f5f9 !important;
        padding-bottom: 8px !important;
        min-height: 56px !important;
        display: flex !important;
        align-items: center !important;
        font-size: 1.1rem !important;
        line-height: 1.3 !important;
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
    
    /* DIP Stage Step Cards styling */
    .dip-step-card {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 12px;
        text-align: center;
        border-bottom: 2px solid #FF6B4A;
    }
    .dip-step-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Text layout overrides */
    h1, h2, h3 {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Metrics loader (dynamic - no hardcoding)
# ----------------------------------------------------
@st.cache_data
def load_display_metrics():
    result = {}
    for sid, key in [(5, "svm"), (9, "rf"), (11, "cnn")]:
        path = project_root / "results" / "metrics" / f"scenario_{sid:02d}.csv"
        if path.exists():
            try:
                row = pd.read_csv(path).iloc[0]
                result[key] = {
                    "accuracy": float(row["accuracy"]),
                    "f1": float(row["f1_weighted"]),
                    "inference_ms": float(row["inference_time_ms_per_image"]),
                    "scenario_id": sid,
                }
            except Exception:
                pass
    return result

display_metrics = load_display_metrics()

# ----------------------------------------------------
# Model loaders
# ----------------------------------------------------
@st.cache_resource
def load_cnn_model():
    # S11: CNN raw - best performing CNN (F1=0.987)
    model_path = project_root / "results" / "models" / "mobilenetv2_s11_stage2.keras"
    if not model_path.exists():
        st.sidebar.error(f"CNN model not found: {model_path}")
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load CNN: {e}")
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

model_cnn = load_cnn_model()
model_svm = load_svm_model()
model_rf  = load_rf_model()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-top:15px; margin-bottom:25px;">
        <span style="font-size:3.5rem;">🍎</span>
        <h2 style="margin-top:10px; font-weight:800; font-size:1.4rem; color:#f8fafc; letter-spacing:-0.5px;">Analisis Kualitas Buah</h2>
        <span style="color:#94a3b8; font-size:0.8rem; font-weight:500;">Platform Demo Akademik</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model dalam demo ini")

    cnn_acc = f"{display_metrics['cnn']['accuracy']*100:.2f}%" if "cnn" in display_metrics else "N/A"
    svm_acc = f"{display_metrics['svm']['accuracy']*100:.2f}%" if "svm" in display_metrics else "N/A"
    rf_acc  = f"{display_metrics['rf']['accuracy']*100:.2f}%"  if "rf"  in display_metrics else "N/A"

    st.markdown(f"""
    <div style="background:#1e293b; padding:16px; border-radius:12px; border:1px solid #334155; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">MobileNetV2 (CNN) - S11</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#94a3b8; line-height:1.3;">Gambar mentah, tanpa pra-pemrosesan. Akurasi tertinggi secara keseluruhan.</p>
        <span class="sidebar-badge">{cnn_acc} AKURASI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#1e293b; padding:16px; border-radius:12px; border:1px solid #334155; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">Support Vector Machine - S5</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#94a3b8; line-height:1.3;">SSR + Gamma + Segmentasi + 220 fitur handcrafted.</p>
        <span class="sidebar-badge">{svm_acc} AKURASI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#1e293b; padding:16px; border-radius:12px; border:1px solid #334155; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">Random Forest - S9</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#94a3b8; line-height:1.3;">Pipeline identik S5, pengklasifikasi diganti Random Forest.</p>
        <span class="sidebar-badge">{rf_acc} AKURASI</span>
    </div>
    """, unsafe_allow_html=True)

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
        
        table_html = textwrap.dedent("""\
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
        """)
        for _, row in df_metrics.iterrows():
            rest = str(row.get('restoration', 'ssr')).upper()
            enh = str(row.get('enhancement', 'clahe')).upper()
            seg_val = str(row.get('segmentation', 'True')).strip().lower()
            seg_str = 'Segmentasi' if seg_val in ('true', '1') else 'No-Seg'
            
            try:
                acc_val = float(row.get('accuracy', 0.0)) * 100
                acc_str = f"{acc_val:.2f}%"
            except Exception:
                acc_str = "N/A"
                
            try:
                f1_val = float(row.get('f1_weighted', 0.0))
                f1_str = f"{f1_val:.4f}"
            except Exception:
                f1_str = "N/A"
                
            try:
                lat_val = float(row.get('inference_time_ms_per_image', 0.0))
                lat_str = f"{lat_val:.3f} ms"
            except Exception:
                lat_str = "N/A"
                
            try:
                samples_val = int(row.get('n_test_samples', 0))
                samples_str = f"{samples_val} citra"
            except Exception:
                samples_str = "N/A"
                
            table_html += textwrap.dedent(f"""\
                <tr>
                    <td><b>Skenario {row.get('scenario_id', 'N/A')}</b></td>
                    <td>{row.get('model', 'N/A')}</td>
                    <td>{rest} + {enh} + {seg_str}</td>
                    <td><span style="font-weight:700; color:#FF6B4A;">{acc_str}</span></td>
                    <td>{f1_str}</td>
                    <td>{lat_str}</td>
                    <td>{samples_str}</td>
                </tr>
            """)
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data metrik hasil pengujian tidak ditemukan di folder results/metrics/.")

# ============================================================
# ACTIVE INFERENCE VIEW
# ============================================================
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
                cnn_time  = (time.perf_counter() - t0) * 1000
                cnn_idx   = int(np.argmax(cnn_probs))
                cnn_pred_class  = "fresh" if cnn_idx == 0 else "rotten"
                cnn_confidence  = float(cnn_probs[cnn_idx])

            # SVM S5
            svm_pred_class = "N/A"; svm_time = 0.0; svm_decision = 0.0
            if model_svm is not None and features_input is not None:
                t0 = time.perf_counter()
                svm_idx = model_svm.predict(features_input)[0]
                svm_time = (time.perf_counter() - t0) * 1000
                svm_pred_class = "fresh" if svm_idx == 0 else "rotten"
                try:
                    svm_decision = float(model_svm.decision_function(features_input)[0])
                except Exception:
                    svm_decision = 0.0

            # RF S9
            rf_pred_class = "N/A"; rf_confidence = 0.0; rf_time = 0.0
            if model_rf is not None and features_input is not None:
                t0 = time.perf_counter()
                rf_idx   = model_rf.predict(features_input)[0]
                rf_probs = model_rf.predict_proba(features_input)[0]
                rf_time = (time.perf_counter() - t0) * 1000
                rf_pred_class = "fresh" if rf_idx == 0 else "rotten"
                rf_confidence = rf_probs[rf_idx]
        
        # 2. Layout: Preprocessing Steps (Full Width)
        st.markdown('<div class="section-header" style="margin-bottom: 8px;">1. Digital Image Processing (DIP) Pipeline Stages</div>', unsafe_allow_html=True)
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
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
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">4. Segmented</div></div>', unsafe_allow_html=True)
            st.image(img_segmented_rgb, use_container_width=True)
            st.caption("Final Output")
            
        # 3. Model Inference Comparison (3 Columns cards)
        st.markdown('<div class="section-header" style="margin-top: 0px; padding-top: 0px;">2. Multi-Model Inference Comparison</div>', unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)

        def _badge(pred):
            if pred == "fresh":
                return "badge-fresh", "SEGAR"
            if pred == "rotten":
                return "badge-rotten", "BUSUK"
            return "badge-na", "N/A"

        with m_col1:
            bc, bl = _badge(cnn_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom: 0px;">
                <div>
                    <h4 class="card-title">MobileNetV2 (CNN)</h4>
                    <div style="margin-top: 10px;">
                        <span class="badge {badge_class}">{cnn_pred_class}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 15px;">
                        <div>
                            <div style="font-size: 0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Confidence</div>
                            <div style="font-size: 1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{cnn_confidence:.2%}</div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: auto; padding-top: 15px; font-size: 0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display: flex; justify-content: space-between;">
                    <span>Latency:</span>
                    <b style="color:#334155;">{cnn_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            badge_class = "badge-fresh" if svm_pred_class == "fresh" else "badge-rotten"
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom: 0px;">
                <div>
                    <h4 class="card-title">SVM (Classical ML)</h4>
                    <div style="margin-top: 10px;">
                        <span class="badge {badge_class}">{svm_pred_class}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 15px;">
                        <div>
                            <div style="font-size: 0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Decision Score</div>
                            <div style="font-size: 1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{svm_decision:.3f}</div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: auto; padding-top: 15px; font-size: 0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display: flex; justify-content: space-between;">
                    <span>Latency:</span>
                    <b style="color:#334155;">{svm_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Card 3: RF
        with m_col3:
            bc, bl = _badge(rf_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom: 0px;">
                <div>
                    <h4 class="card-title">Random Forest (Classical ML)</h4>
                    <div style="margin-top: 10px;">
                        <span class="badge {badge_class}">{rf_pred_class}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 15px;">
                        <div>
                            <div style="font-size: 0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Confidence</div>
                            <div style="font-size: 1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{rf_confidence:.2%}</div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: auto; padding-top: 15px; font-size: 0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display: flex; justify-content: space-between;">
                    <span>Latency:</span>
                    <b style="color:#334155;">{rf_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
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
                    st.warning(f"Grad-CAM tidak dapat dihitung: {e}")
