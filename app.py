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
    page_title="Dashboard Analisis Kualitas Buah Berbasis AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Background and Fonts */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #0b0f19 !important;
        color: #cbd5e1 !important;
    }
    
    /* Scrollbars styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #151c2c !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #f8fafc !important;
    }

    /* Equal height columns styling (Desktop) */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        height: auto !important;
    }
    [data-testid="column"] div:has(.saas-card) {
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
        height: 100% !important;
    }

    /* Modern Classy Cards */
    .saas-card {
        background-color: #151c2c;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 24px;
        height: 100% !important;
        flex-grow: 1 !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-sizing: border-box;
    }
    .saas-card:hover {
        transform: translateY(-4px);
        border-color: #FF6B4A;
        box-shadow: 0 12px 30px -10px rgba(255, 107, 74, 0.15);
    }
    .card-title {
        margin-top: 0 !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        border-bottom: 1px solid #1e293b !important;
        padding-bottom: 12px !important;
        min-height: 56px !important;
        display: flex !important;
        align-items: center !important;
        font-size: 1.1rem !important;
        line-height: 1.3 !important;
    }
    
    /* Section Headers with custom glowing underline */
    .section-header {
        font-size: 1.35rem; 
        font-weight: 700; 
        color: #f8fafc;
        margin-top: 1.5rem; 
        margin-bottom: 1.25rem;
        padding-bottom: 8px; 
        border-bottom: 2px solid #1e293b;
        position: relative;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #FF6B4A, #ff8e53);
    }
    
    /* Badges */
    .sidebar-badge {
        background-color: rgba(255, 107, 74, 0.1); 
        color: #FF6B4A; 
        font-weight: 700;
        font-size: 0.75rem; 
        padding: 4px 10px; 
        border-radius: 6px;
        border: 1px solid rgba(255, 107, 74, 0.25); 
        display: inline-block;
    }
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
        background-color: rgba(16, 185, 129, 0.12); 
        color: #10b981; 
        border: 1px solid rgba(16, 185, 129, 0.25); 
    }
    .badge-rotten { 
        background-color: rgba(239, 68, 68, 0.12); 
        color: #ef4444; 
        border: 1px solid rgba(239, 68, 68, 0.25); 
    }
    .badge-na { 
        background-color: rgba(148, 163, 184, 0.1); 
        color: #cbd5e1; 
        border: 1px solid rgba(148, 163, 184, 0.2); 
    }
    
    /* Styled File Uploader container (Full Width and Centered) */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #FF6B4A !important; 
        border-radius: 16px !important; 
        padding: 24px !important;
        background-color: #151c2c !important; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #151c2c !important;
        width: 100% !important;
    }
    div[data-testid="stFileUploader"] label, 
    div[data-testid="stFileUploader"] p, 
    div[data-testid="stFileUploader"] span {
        color: #cbd5e1 !important;
    }
    /* Stretch parent container of file uploader to full width */
    div.element-container:has(div[data-testid="stFileUploader"]) {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Styled Tables */
    .styled-table {
        width: 100%; 
        border-collapse: collapse; 
        margin: 10px 0; 
        font-size: 0.95rem;
        border-radius: 12px; 
        overflow: hidden; 
        border: 1px solid #1e293b; 
        background-color: #151c2c;
    }
    .styled-table th {
        background-color: #1e293b; 
        color: #f8fafc; 
        text-align: left;
        font-weight: 700; 
        padding: 14px 16px; 
        border-bottom: 2px solid #1e293b;
    }
    .styled-table td { 
        padding: 14px 16px; 
        border-bottom: 1px solid #1e293b; 
        color: #cbd5e1; 
    }
    .styled-table tbody tr:nth-of-type(even) { 
        background-color: rgba(30, 41, 59, 0.3); 
    }
    .styled-table tbody tr:hover { 
        background-color: rgba(30, 41, 59, 0.6); 
        transition: background-color 0.2s ease; 
    }
    
    /* Visual Pipeline Containers */
    .pipeline-container {
        display: flex; 
        flex-wrap: wrap; 
        align-items: center; 
        gap: 10px;
        padding: 16px; 
        background-color: #151c2c; 
        border-radius: 12px;
        border: 1px solid #1e293b; 
        margin-bottom: 15px;
    }
    .pipeline-step {
        background-color: #1e293b; 
        border: 1px solid #334155; 
        border-radius: 8px;
        padding: 8px 14px; 
        font-size: 0.85rem; 
        font-weight: 600; 
        color: #cbd5e1;
    }
    .pipeline-arrow { 
        color: #64748b; 
        font-size: 1.1rem; 
        font-weight: bold; 
    }
    
    /* Image processing steps */
    .dip-step-card {
        background-color: #1e293b; 
        border: 1px solid #1e293b;
        border-radius: 8px 8px 0 0; 
        padding: 8px 12px; 
        text-align: center; 
        border-bottom: 2px solid #FF6B4A;
    }
    .dip-step-title { 
        font-size: 0.85rem; 
        font-weight: 700; 
        color: #f8fafc; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
        border-bottom: 2px solid #FF6B4A;
    }
    .preproc-tag {
        display: inline-block; 
        font-size: 0.7rem; 
        font-weight: 600; 
        padding: 2px 8px;
        border-radius: 4px; 
        background-color: #1e293b; 
        color: #94a3b8;
        border: 1px solid #334155; 
        margin-top: 6px;
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #f8fafc !important; 
    }
    
    /* Mobile Responsiveness & Stacking overrides */
    @media (max-width: 768px) {
        /* Reset heights and margins for stacked columns on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 16px !important;
            align-items: stretch !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            margin-bottom: 16px !important;
            display: block !important;
        }
        [data-testid="column"] div:has(.saas-card) {
            display: block !important;
            height: auto !important;
        }
        .saas-card {
            margin-bottom: 16px !important;
            height: auto !important;
            min-height: unset !important;
        }
        div[data-testid="stFileUploader"] {
            padding: 16px !important;
        }
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
    path = project_root / "results" / "models" / "svm_scenario_05.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_rf_model():
    path = project_root / "results" / "models" / "rf_scenario_09.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
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
# MAIN - Hero + KPI
# ----------------------------------------------------
st.markdown("""
<div style="margin-top:10px; margin-bottom:30px;">
    <h1 style="font-size:2.5rem; font-weight:800; color:#f8fafc; margin-bottom:8px; letter-spacing:-1px;">Dashboard Analisis Kualitas Buah</h1>
    <h3 style="font-size:1.15rem; font-weight:400; color:#94a3b8; margin:0; line-height:1.5; max-width:900px;">
        Evaluasi Komparatif Model Machine Learning Klasik dan Deep Learning untuk Klasifikasi Kualitas Buah
    </h3>
</div>
""", unsafe_allow_html=True)

# KPI - dynamic
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
best_f1_val  = display_metrics.get("cnn", {}).get("f1", 0.0)
best_acc_val = display_metrics.get("cnn", {}).get("accuracy", 0.0)
rf_ms        = display_metrics.get("rf",  {}).get("inference_ms", 0.013)

with kpi_col1:
    st.markdown(f"""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Akurasi Terbaik</div>
        <div style="font-size:1.75rem; font-weight:800; color:#FF6B4A; margin-top:6px; letter-spacing:-1px;">{best_acc_val*100:.2f}%</div>
        <div style="font-size:0.8rem; color:#cbd5e1; margin-top:4px; font-weight:500;">MobileNetV2 - S11 (CNN mentah)</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col2:
    st.markdown(f"""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Model Tercepat</div>
        <div style="font-size:1.75rem; font-weight:800; color:#f8fafc; margin-top:6px; letter-spacing:-1px;">Random Forest</div>
        <div style="font-size:0.8rem; color:#cbd5e1; margin-top:4px; font-weight:500;">{rf_ms:.3f} ms / citra</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col3:
    st.markdown("""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Fitur Handcrafted</div>
        <div style="font-size:1.75rem; font-weight:800; color:#f8fafc; margin-top:6px; letter-spacing:-1px;">220 Dimensi</div>
        <div style="font-size:0.8rem; color:#cbd5e1; margin-top:4px; font-weight:500;">Warna HSV + Tekstur GLCM + Bentuk</div>
    </div>
    """, unsafe_allow_html=True)

# Upload widget
st.markdown('<div class="section-header">Unggah & Analisis Gambar</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Unggah gambar buah untuk analisis kualitas...",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Gambar diproses oleh tiga model secara paralel. CNN menggunakan input mentah; SVM & RF menggunakan pipeline lengkap (SSR + Gamma + Segmentasi)."
)

# ============================================================
# LANDING VIEW
# ============================================================
if uploaded_file is None:

    # --- Pipeline diagrams ---
    st.markdown('<div class="section-header">Pipeline Pemrosesan Citra</div>', unsafe_allow_html=True)
    col_p_1, col_p_2 = st.columns(2)
    with col_p_1:
        st.markdown("""
        <div class="saas-card">
            <h4 style="margin-top:0; font-weight:700; color:#f8fafc;">Pipeline ML Klasik - S5 (SVM) & S9 (RF)</h4>
            <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:15px;">Koreksi iluminasi SSR, enhancement Gamma (E*), segmentasi Otsu, ekstraksi 220 fitur handcrafted, lalu SVM atau RF.</p>
            <div class="pipeline-container">
                <span class="pipeline-step">Input</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">SSR + Gamma</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">Segmentasi Otsu</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">220 Fitur</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">SVM / RF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p_2:
        st.markdown("""
        <div class="saas-card">
            <h4 style="margin-top:0; font-weight:700; color:#f8fafc;">Pipeline CNN - S11 (Mentah, Terbaik)</h4>
            <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:15px;">MobileNetV2 dengan transfer learning, tanpa restorasi atau perbaikan citra. Temuan kunci: pra-pemrosesan tidak diperlukan oleh CNN.</p>
            <div class="pipeline-container">
                <span class="pipeline-step">Input</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">Resize 224×224</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">MobileNetV2</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">Fine-tune 20L</span>
                <span class="pipeline-arrow">➔</span>
                <span class="pipeline-step">Prediksi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Performance analytics ---
    st.markdown('<div class="section-header">Analisis Performa (Test Set n=4391)</div>', unsafe_allow_html=True)

    metrics_dir = project_root / "results" / "metrics"

    # Trajectory chart - all 11 scenarios
    scenario_files = sorted(metrics_dir.glob("scenario_0[0-9].csv")) + sorted(metrics_dir.glob("scenario_1[0-9].csv"))
    if scenario_files:
        traj_rows = []
        for f in scenario_files:
            try:
                row = pd.read_csv(f).iloc[0]
                traj_rows.append({"Skenario": f"S{int(row['scenario_id'])}", "F1": float(row["f1_weighted"]), "Model": str(row["model"])})
            except Exception:
                pass

        if traj_rows:
            df_traj = pd.DataFrame(traj_rows)
            st.markdown('<div style="display:none;">', unsafe_allow_html=True)
            plt.style.use("dark_background")
            fig_t, ax_t = plt.subplots(figsize=(13, 4), facecolor='#151c2c')
            ax_t.set_facecolor('#151c2c')
            colors = ["#475569" if m in ("SVM", "RF") else "#FF6B4A" for m in df_traj["Model"]]
            bars = ax_t.bar(df_traj["Skenario"], df_traj["F1"], color=colors, width=0.55)
            for bar, val in zip(bars, df_traj["F1"]):
                ax_t.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#cbd5e1")
            ax_t.set_ylim(0.5, 1.05)
            ax_t.set_ylabel("F1-Score Terbobot", fontsize=9, color="#cbd5e1")
            ax_t.set_xlabel("")
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)
            ax_t.spines["left"].set_color("#1e293b")
            ax_t.spines["bottom"].set_color("#1e293b")
            ax_t.tick_params(colors="#94a3b8", labelsize=9)
            from matplotlib.patches import Patch
            ax_t.legend(handles=[Patch(color="#475569", label="ML Klasik (SVM/RF)"),
                                  Patch(color="#FF6B4A", label="CNN (MobileNetV2)")],
                        fontsize=8, frameon=False, loc="lower right", labelcolor="#cbd5e1")
            plt.tight_layout()
            
            buf_t = io.BytesIO()
            fig_t.savefig(buf_t, format="png", bbox_inches="tight", dpi=150, facecolor=fig_t.get_facecolor())
            buf_t.seek(0)
            img_b64_t = base64.b64encode(buf_t.read()).decode("utf-8")
            plt.close(fig_t)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="saas-card">
                <h4 style="margin-top:0; font-weight:700; color:#f8fafc; margin-bottom:4px;">Trayektori F1-Score: S1 ke S11</h4>
                <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:15px;">Setiap skenario mengubah satu variabel. Grafik menunjukkan kontribusi bersih dari setiap komponen pipeline.</p>
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{img_b64_t}" style="max-width:100%; height:auto; border-radius:8px;" />
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Bar charts - S5 / S9 / S11
    df_list = []
    for sid in (5, 9, 11):
        f = metrics_dir / f"scenario_{sid:02d}.csv"
        if f.exists():
            df_list.append(pd.read_csv(f))

    if df_list:
        df_metrics = pd.concat(df_list, ignore_index=True)
        model_labels = {5: "SVM (S5)", 9: "RF (S9)", 11: "CNN (S11)"}
        df_metrics["label"] = df_metrics["scenario_id"].map(model_labels)

        st.markdown('<div style="display:none;">', unsafe_allow_html=True)
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#151c2c')
        for ax in axes:
            ax.set_facecolor('#151c2c')
        accent = "#FF6B4A"

        sns.barplot(x="label", y="accuracy", data=df_metrics, ax=axes[0], color=accent, alpha=0.9, width=0.45)
        axes[0].set_title("Akurasi Pengujian", fontsize=11, fontweight="bold", pad=15, color="#f8fafc")
        axes[0].set_ylabel("Akurasi", fontsize=9, color="#cbd5e1"); axes[0].set_xlabel("")
        axes[0].set_ylim(0.80, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height()*100:.2f}%",
                (p.get_x() + p.get_width()/2, p.get_height() - 0.05),
                ha="center", va="center", xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white")

        sns.barplot(x="label", y="f1_weighted", data=df_metrics, ax=axes[1], color="#475569", alpha=0.9, width=0.45)
        axes[1].set_title("F1-Score Terbobot", fontsize=11, fontweight="bold", pad=15, color="#f8fafc")
        axes[1].set_ylabel("F1-Score", fontsize=9, color="#cbd5e1"); axes[1].set_xlabel("")
        axes[1].set_ylim(0.80, 1.0)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.4f}",
                (p.get_x() + p.get_width()/2, p.get_height() - 0.05),
                ha="center", va="center", xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white")

        sns.barplot(x="label", y="inference_time_ms_per_image", data=df_metrics, ax=axes[2], color="#1e293b", alpha=0.9, width=0.45)
        axes[2].set_title("Latensi Inferensi (ms/citra)", fontsize=11, fontweight="bold", pad=15, color="#f8fafc")
        axes[2].set_ylabel("Latensi (ms)", fontsize=9, color="#cbd5e1"); axes[2].set_xlabel("")
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.3f} ms",
                (p.get_x() + p.get_width()/2, p.get_height() + 0.1),
                ha="center", va="center", xytext=(0, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#cbd5e1")

        for ax in axes:
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#1e293b"); ax.spines["bottom"].set_color("#1e293b")
            ax.tick_params(colors="#94a3b8", labelsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="saas-card">
            <h4 style="margin-top:0; font-weight:700; color:#f8fafc; margin-bottom:15px;">Perbandingan Model: S5, S9, S11</h4>
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto; border-radius:8px;" />
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison table
        table_html = f"""
        <div class="saas-card">
            <h4 style="margin-top:0; font-weight:700; color:#f8fafc; margin-bottom:15px;">Tabel Perbandingan Performa Model</h4>
            <div style="overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch;">
                <table class="styled-table" style="min-width: 600px; margin: 0;">
                    <thead><tr>
                        <th>Skenario</th><th>Model</th><th>Pra-pemrosesan</th>
                        <th>Akurasi</th><th>F1-Score</th><th>Latensi</th><th>Sampel Uji</th>
                    </tr></thead><tbody>
        """
        for _, row in df_metrics.iterrows():
            sid    = int(row.get("scenario_id", 0))
            rest   = str(row.get("restoration", "")).upper()
            enh    = str(row.get("enhancement", "")).upper()
            seg    = str(row.get("segmentation", "")).strip().lower() in ("true", "1")
            label  = model_labels.get(sid, f"S{sid}")
            prep   = f"{rest} + {enh} + Seg" if seg else ("Mentah" if enh in ("NONE", "") and rest in ("NONE", "") else f"{rest} + {enh}")
            acc_s  = f"{float(row['accuracy'])*100:.2f}%"
            f1_s   = f"{float(row['f1_weighted']):.4f}"
            lat_s  = f"{float(row['inference_time_ms_per_image']):.3f} ms"
            n_s    = f"{int(row['n_test_samples'])} citra"
            table_html += f"""
                <tr>
                    <td><b>{label}</b></td>
                    <td>{row.get('model','N/A')}</td>
                    <td>{prep}</td>
                    <td><span style="font-weight:700; color:#FF6B4A;">{acc_s}</span></td>
                    <td>{f1_s}</td><td>{lat_s}</td><td>{n_s}</td>
                </tr>"""
        table_html += "</tbody></table></div></div>"
        
        # Strip all leading space to prevent markdown code block formatting
        table_html_clean = "\n".join([line.strip() for line in table_html.split("\n")])
        st.markdown(table_html_clean, unsafe_allow_html=True)

    else:
        st.warning("Data metrik tidak ditemukan di folder results/metrics/.")

# ============================================================
# ACTIVE INFERENCE VIEW
# ============================================================
else:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Gagal memuat gambar. Harap unggah gambar yang valid.")
    else:
        # --- Build all pipeline stages ---
        img_resized     = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_ssr     = apply_ssr(img_resized)
        img_ssr_rgb = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2RGB)

        img_gamma     = apply_enhancement(img_ssr, "gamma")   # E* = gamma
        img_gamma_rgb = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)

        img_segmented, mask, obj_ratio, fallback = segment_fruit(img_gamma)
        img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)

        # --- Run all models ---
        with st.spinner("Mengekstrak fitur dan menjalankan seluruh model..."):
            # Features for SVM (S5) and RF (S9): SSR + Gamma + Seg, 220 dim
            try:
                features_classical = extract_features(img_gamma, mask, feature_groups="all", segmented=True)
                features_input = features_classical.reshape(1, -1)
            except Exception:
                features_input = None

            # CNN (S11): raw input - no preprocessing
            cnn_input = image_to_cnn_input(img_resized)

            # MobileNetV2 S11
            cnn_pred_class = "N/A"; cnn_confidence = 0.0; cnn_time = 0.0
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
                rf_time  = (time.perf_counter() - t0) * 1000
                rf_pred_class  = "fresh" if rf_idx == 0 else "rotten"
                rf_confidence  = float(rf_probs[rf_idx])

        # --- DIP Pipeline Stages ---
        st.markdown('<div class="section-header" style="margin-bottom:8px;">1. Tahapan Pipeline DIP (SSR + Gamma + Segmentasi)</div>', unsafe_allow_html=True)
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.markdown(f"""
            <div class="saas-card" style="padding:16px; min-height:300px;">
                <div class="dip-step-title" style="font-size:0.85rem; font-weight:700; color:#f8fafc; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #FF6B4A; padding-bottom:8px; margin-bottom:12px; text-align:center;">1. Hasil Resize</div>
                <div style="text-align:center; margin-bottom:10px;">
                    <img src="data:image/png;base64,{arr_to_base64(img_resized_rgb)}" style="max-width:100%; height:auto; border-radius:8px;" />
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; text-align:center;">224×224 - Masukan CNN (S11)</div>
            </div>
            """, unsafe_allow_html=True)
        with p_col2:
            st.markdown(f"""
            <div class="saas-card" style="padding:16px; min-height:300px;">
                <div class="dip-step-title" style="font-size:0.85rem; font-weight:700; color:#f8fafc; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #FF6B4A; padding-bottom:8px; margin-bottom:12px; text-align:center;">2. SSR</div>
                <div style="text-align:center; margin-bottom:10px;">
                    <img src="data:image/png;base64,{arr_to_base64(img_ssr_rgb)}" style="max-width:100%; height:auto; border-radius:8px;" />
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; text-align:center;">Koreksi iluminasi CIELAB</div>
            </div>
            """, unsafe_allow_html=True)
        with p_col3:
            st.markdown(f"""
            <div class="saas-card" style="padding:16px; min-height:300px;">
                <div class="dip-step-title" style="font-size:0.85rem; font-weight:700; color:#f8fafc; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #FF6B4A; padding-bottom:8px; margin-bottom:12px; text-align:center;">3. Gamma (E*)</div>
                <div style="text-align:center; margin-bottom:10px;">
                    <img src="data:image/png;base64,{arr_to_base64(img_gamma_rgb)}" style="max-width:100%; height:auto; border-radius:8px;" />
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; text-align:center;">Peningkatan Kualitas Terpilih (Nilai F1)</div>
            </div>
            """, unsafe_allow_html=True)
        with p_col4:
            st.markdown(f"""
            <div class="saas-card" style="padding:16px; min-height:300px;">
                <div class="dip-step-title" style="font-size:0.85rem; font-weight:700; color:#f8fafc; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #FF6B4A; padding-bottom:8px; margin-bottom:12px; text-align:center;">4. Hasil Segmentasi</div>
                <div style="text-align:center; margin-bottom:10px;">
                    <img src="data:image/png;base64,{arr_to_base64(img_segmented_rgb)}" style="max-width:100%; height:auto; border-radius:8px;" />
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; text-align:center;">Masukan SVM & RF (Rasio Objek {obj_ratio:.0%})</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Pipeline lengkap (kolom 2–4) digunakan oleh SVM (S5) dan RF (S9). CNN (S11) hanya menggunakan kolom 1 (resize mentah).")

        # --- Multi-model Inference Cards ---
        st.markdown('<div class="section-header" style="margin-top:0;">2. Inferensi Multi-Model</div>', unsafe_allow_html=True)
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
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">MobileNetV2 - S11</h4>
                    <span class="preproc-tag">Masukan: Mentah (Hanya Resize)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Tingkat Keyakinan</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f8fafc; margin-top:4px;">{cnn_confidence:.2%}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#94a3b8; border-top:1px solid #1e293b; display:flex; justify-content:space-between;">
                    <span>Latensi</span><b style="color:#cbd5e1;">{cnn_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            bc, bl = _badge(svm_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">SVM - S5</h4>
                    <span class="preproc-tag">Masukan: SSR + Gamma + Segmentasi (220 dim)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Skor Keputusan</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f8fafc; margin-top:4px;">{svm_decision:.3f}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#94a3b8; border-top:1px solid #1e293b; display:flex; justify-content:space-between;">
                    <span>Latensi</span><b style="color:#cbd5e1;">{svm_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            bc, bl = _badge(rf_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">Random Forest - S9</h4>
                    <span class="preproc-tag">Masukan: SSR + Gamma + Segmentasi (220 dim)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Tingkat Keyakinan</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#f8fafc; margin-top:4px;">{rf_confidence:.2%}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#94a3b8; border-top:1px solid #1e293b; display:flex; justify-content:space-between;">
                    <span>Latensi</span><b style="color:#cbd5e1;">{rf_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


        # --- Grad-CAM (CNN S11 - raw input) ---
        if model_cnn is not None:
            st.markdown('<div class="section-header">3. Penjelasan Deep Learning: Grad-CAM (S11)</div>', unsafe_allow_html=True)
            with st.spinner("Menghitung heatmap aktivasi Grad-CAM..."):
                try:
                    heatmap = make_gradcam_heatmap(model_cnn, cnn_input)
                    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    img_u8  = to_uint8(img_resized)
                    overlay = cv2.addWeighted(img_u8, 0.65, heatmap_color, 0.35, 0)
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        st.markdown(f"""
                        <div class="saas-card" style="text-align:center; padding:16px;">
                            <h5 style="margin-top:0; margin-bottom:12px; color:#f8fafc;">Masukan CNN (Resize Mentah)</h5>
                            <img src="data:image/png;base64,{arr_to_base64(img_resized_rgb)}" style="max-width:100%; height:auto; border-radius:8px;" />
                        </div>
                        """, unsafe_allow_html=True)
                    with g_col2:
                        st.markdown(f"""
                        <div class="saas-card" style="text-align:center; padding:16px;">
                            <h5 style="margin-top:0; margin-bottom:12px; color:#f8fafc;">Heatmap Perhatian Grad-CAM</h5>
                            <img src="data:image/png;base64,{arr_to_base64(overlay_rgb)}" style="max-width:100%; height:auto; border-radius:8px; margin-bottom:8px;" />
                            <div style="font-size:0.75rem; color:#94a3b8; line-height:1.4;">Merah/jingga = area paling berpengaruh terhadap keputusan CNN dalam membedakan Segar vs Busuk.</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Grad-CAM tidak dapat dihitung: {e}")
