import sys
import textwrap
import time
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }
    .saas-card {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 10px 15px -3px rgba(0,0,0,0.03);
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
        box-shadow: 0 12px 20px -8px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.02);
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
    .stSidebar { background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #0f172a;
        margin-top: 1rem; margin-bottom: 1.25rem;
        padding-bottom: 6px; border-bottom: 2px solid #fff0eb;
    }
    .sidebar-badge {
        background-color: #fff0eb; color: #FF6B4A; font-weight: 700;
        font-size: 0.75rem; padding: 4px 10px; border-radius: 6px;
        border: 1px solid #ffe2d9; display: inline-block;
    }
    .badge {
        display: inline-block; padding: 8px 16px; font-size: 0.9rem;
        font-weight: 800; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-fresh { background-color: #e6fffa; color: #0d9488; border: 1px solid #b2f5ea; }
    .badge-rotten { background-color: #fff5f5; color: #e53e3e; border: 1px solid #fed7d7; }
    .badge-na { background-color: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #FF6B4A; border-radius: 16px; padding: 24px;
        background-color: #ffffff; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    }
    .styled-table {
        width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.95rem;
        border-radius: 12px; overflow: hidden; border: 1px solid #f1f5f9; background-color: #ffffff;
    }
    .styled-table th {
        background-color: #f8fafc; color: #475569; text-align: left;
        font-weight: 700; padding: 14px 16px; border-bottom: 2px solid #f1f5f9;
    }
    .styled-table td { padding: 14px 16px; border-bottom: 1px solid #f1f5f9; color: #334155; }
    .styled-table tbody tr:nth-of-type(even) { background-color: #f8fafc; }
    .styled-table tbody tr:hover { background-color: #f1f5f9; transition: background-color 0.2s ease; }
    .pipeline-container {
        display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
        padding: 16px; background-color: #ffffff; border-radius: 12px;
        border: 1px solid #f1f5f9; margin-bottom: 15px;
    }
    .pipeline-step {
        background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 8px 14px; font-size: 0.85rem; font-weight: 600; color: #334155;
    }
    .pipeline-arrow { color: #94a3b8; font-size: 1.1rem; font-weight: bold; }
    .dip-step-card {
        background-color: #ffffff; border: 1px solid #f1f5f9;
        border-radius: 8px 8px 0 0; padding: 8px 12px; text-align: center; border-bottom: 2px solid #FF6B4A;
    }
    .dip-step-title { font-size: 0.85rem; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px; }
    .preproc-tag {
        display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
        border-radius: 4px; background-color: #f1f5f9; color: #475569;
        border: 1px solid #e2e8f0; margin-top: 6px;
    }
    h1, h2, h3 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Metrics loader (dynamic — no hardcoding)
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
    # S11: CNN raw — best performing CNN (F1=0.987)
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
        <h2 style="margin-top:10px; font-weight:800; font-size:1.4rem; color:#0f172a; letter-spacing:-0.5px;">CV Analytics</h2>
        <span style="color:#64748b; font-size:0.8rem; font-weight:500;">Academic Demo Platform</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Models in this demo")

    cnn_acc = f"{display_metrics['cnn']['accuracy']*100:.2f}%" if "cnn" in display_metrics else "N/A"
    svm_acc = f"{display_metrics['svm']['accuracy']*100:.2f}%" if "svm" in display_metrics else "N/A"
    rf_acc  = f"{display_metrics['rf']['accuracy']*100:.2f}%"  if "rf"  in display_metrics else "N/A"

    st.markdown(f"""
    <div style="background:white; padding:16px; border-radius:12px; border:1px solid #f1f5f9; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#0f172a;">MobileNetV2 (CNN) — S11</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#64748b; line-height:1.3;">Raw image, no preprocessing. Highest accuracy overall.</p>
        <span class="sidebar-badge">{cnn_acc} ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:white; padding:16px; border-radius:12px; border:1px solid #f1f5f9; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#0f172a;">Support Vector Machine — S5</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#64748b; line-height:1.3;">SSR + Gamma + Segmentasi + 220 fitur handcrafted.</p>
        <span class="sidebar-badge">{svm_acc} ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:white; padding:16px; border-radius:12px; border:1px solid #f1f5f9; margin-bottom:12px;">
        <h4 style="margin:0 0 4px 0; font-size:0.95rem; font-weight:700; color:#0f172a;">Random Forest — S9</h4>
        <p style="margin:0 0 10px 0; font-size:0.75rem; color:#64748b; line-height:1.3;">Pipeline identik S5, classifier diganti Random Forest.</p>
        <span class="sidebar-badge">{rf_acc} ACCURACY</span>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN — Hero + KPI
# ----------------------------------------------------
st.markdown("""
<div style="margin-top:10px; margin-bottom:30px;">
    <h1 style="font-size:2.5rem; font-weight:800; color:#0f172a; margin-bottom:8px; letter-spacing:-1px;">Fruit Quality Analysis Dashboard</h1>
    <h3 style="font-size:1.15rem; font-weight:400; color:#64748b; margin:0; line-height:1.5; max-width:900px;">
        Comparative Evaluation of Classical Machine Learning and Deep Learning Models for Fruit Quality Classification
    </h3>
</div>
""", unsafe_allow_html=True)

# KPI — dynamic
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
best_f1_val  = display_metrics.get("cnn", {}).get("f1", 0.0)
best_acc_val = display_metrics.get("cnn", {}).get("accuracy", 0.0)
rf_ms        = display_metrics.get("rf",  {}).get("inference_ms", 0.013)

with kpi_col1:
    st.markdown(f"""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Best Accuracy</div>
        <div style="font-size:2.2rem; font-weight:800; color:#FF6B4A; margin-top:6px; letter-spacing:-1px;">{best_acc_val*100:.2f}%</div>
        <div style="font-size:0.8rem; color:#64748b; margin-top:4px; font-weight:500;">MobileNetV2 — S11 (CNN raw)</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col2:
    st.markdown(f"""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Fastest Model</div>
        <div style="font-size:2.2rem; font-weight:800; color:#0f172a; margin-top:6px; letter-spacing:-1px;">Random Forest</div>
        <div style="font-size:0.8rem; color:#64748b; margin-top:4px; font-weight:500;">{rf_ms:.3f} ms / image</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_col3:
    st.markdown("""
    <div class="saas-card">
        <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Handcrafted Features</div>
        <div style="font-size:2.2rem; font-weight:800; color:#0f172a; margin-top:6px; letter-spacing:-1px;">220 Dimensions</div>
        <div style="font-size:0.8rem; color:#64748b; margin-top:4px; font-weight:500;">HSV Colors + GLCM Texture + Shape</div>
    </div>
    """, unsafe_allow_html=True)

# Upload widget
st.markdown('<div class="section-header">Image Upload & Analysis</div>', unsafe_allow_html=True)
col_up_1, col_up_2, col_up_3 = st.columns([1, 4, 1])
with col_up_2:
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
    st.markdown('<div class="section-header">Image Processing Pipelines</div>', unsafe_allow_html=True)
    col_p_1, col_p_2 = st.columns(2)
    with col_p_1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a;'>Classical ML Pipeline — S5 (SVM) & S9 (RF)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-bottom:15px;'>Koreksi iluminasi SSR, enhancement Gamma (E*), segmentasi Otsu, ekstraksi 220 fitur handcrafted, lalu SVM atau RF.</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-container">
            <span class="pipeline-step">Input</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">SSR + Gamma</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Otsu Segment</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">220 Features</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">SVM / RF</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_p_2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a;'>CNN Pipeline — S11 (raw, best)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-bottom:15px;'>MobileNetV2 dengan transfer learning — tanpa restorasi atau enhancement. Temuan kunci: preprocessing tidak diperlukan CNN.</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-container">
            <span class="pipeline-step">Input</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Resize 224×224</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">MobileNetV2</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Fine-tune 20L</span>
            <span class="pipeline-arrow">➔</span>
            <span class="pipeline-step">Prediction</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Performance analytics ---
    st.markdown('<div class="section-header">Performance Analytics (Test Set n=4391)</div>', unsafe_allow_html=True)

    metrics_dir = project_root / "results" / "metrics"

    # Trajectory chart — all 11 scenarios
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
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom:4px;'>F1-Score Trajectory — S1 ke S11</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-bottom:15px;'>Setiap skenario mengubah satu variabel — grafik menunjukkan kontribusi bersih tiap komponen pipeline.</p>", unsafe_allow_html=True)

            fig_t, ax_t = plt.subplots(figsize=(13, 4))
            sns.set_theme(style="white")
            colors = ["#334155" if m in ("SVM", "RF") else "#FF6B4A" for m in df_traj["Model"]]
            bars = ax_t.bar(df_traj["Skenario"], df_traj["F1"], color=colors, width=0.55)
            for bar, val in zip(bars, df_traj["F1"]):
                ax_t.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#334155")
            ax_t.set_ylim(0.5, 1.05)
            ax_t.set_ylabel("F1-Score (weighted)", fontsize=9)
            ax_t.set_xlabel("")
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)
            ax_t.spines["left"].set_color("#f1f5f9")
            ax_t.spines["bottom"].set_color("#cbd5e1")
            ax_t.tick_params(colors="#475569", labelsize=9)
            from matplotlib.patches import Patch
            ax_t.legend(handles=[Patch(color="#334155", label="Classical ML (SVM/RF)"),
                                  Patch(color="#FF6B4A", label="CNN (MobileNetV2)")],
                        fontsize=8, frameon=False, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)
            st.markdown('</div>', unsafe_allow_html=True)

    # Bar charts — S5 / S9 / S11
    df_list = []
    for sid in (5, 9, 11):
        f = metrics_dir / f"scenario_{sid:02d}.csv"
        if f.exists():
            df_list.append(pd.read_csv(f))

    if df_list:
        df_metrics = pd.concat(df_list, ignore_index=True)
        model_labels = {5: "SVM (S5)", 9: "RF (S9)", 11: "CNN (S11)"}
        df_metrics["label"] = df_metrics["scenario_id"].map(model_labels)

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom:15px;'>Model Comparison — S5, S9, S11</h4>", unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        sns.set_theme(style="white")
        accent = "#FF6B4A"

        sns.barplot(x="label", y="accuracy", data=df_metrics, ax=axes[0], color=accent, alpha=0.9, width=0.45)
        axes[0].set_title("Test Accuracy", fontsize=11, fontweight="bold", pad=15)
        axes[0].set_ylabel("Accuracy", fontsize=9); axes[0].set_xlabel("")
        axes[0].set_ylim(0.80, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height()*100:.2f}%",
                (p.get_x() + p.get_width()/2, p.get_height() - 0.05),
                ha="center", va="center", xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white")

        sns.barplot(x="label", y="f1_weighted", data=df_metrics, ax=axes[1], color="#334155", alpha=0.9, width=0.45)
        axes[1].set_title("Weighted F1-Score", fontsize=11, fontweight="bold", pad=15)
        axes[1].set_ylabel("F1-Score", fontsize=9); axes[1].set_xlabel("")
        axes[1].set_ylim(0.80, 1.0)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.4f}",
                (p.get_x() + p.get_width()/2, p.get_height() - 0.05),
                ha="center", va="center", xytext=(0, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white")

        sns.barplot(x="label", y="inference_time_ms_per_image", data=df_metrics, ax=axes[2], color="#cbd5e1", alpha=0.9, width=0.45)
        axes[2].set_title("Inference Latency (ms/image)", fontsize=11, fontweight="bold", pad=15)
        axes[2].set_ylabel("Latency (ms)", fontsize=9); axes[2].set_xlabel("")
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.3f} ms",
                (p.get_x() + p.get_width()/2, p.get_height() + 0.1),
                ha="center", va="center", xytext=(0, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#334155")

        for ax in axes:
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#f1f5f9"); ax.spines["bottom"].set_color("#cbd5e1")
            ax.tick_params(colors="#475569", labelsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Comparison table
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; color:#0f172a; margin-bottom:15px;'>Model Performance Comparison Table</h4>", unsafe_allow_html=True)

        table_html = textwrap.dedent("""\
            <table class="styled-table">
                <thead><tr>
                    <th>Skenario</th><th>Model</th><th>Preprocessing</th>
                    <th>Akurasi</th><th>F1-Score</th><th>Latency</th><th>n Test</th>
                </tr></thead><tbody>
        """)
        for _, row in df_metrics.iterrows():
            sid    = int(row.get("scenario_id", 0))
            rest   = str(row.get("restoration", "")).upper()
            enh    = str(row.get("enhancement", "")).upper()
            seg    = str(row.get("segmentation", "")).strip().lower() in ("true", "1")
            label  = model_labels.get(sid, f"S{sid}")
            prep   = f"{rest} + {enh} + Seg" if seg else ("Raw" if enh in ("NONE", "") and rest in ("NONE", "") else f"{rest} + {enh}")
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
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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

            # CNN (S11): raw input — no preprocessing
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
        st.markdown('<div class="section-header" style="margin-bottom:8px;">1. DIP Pipeline Stages (SSR + Gamma + Segmentasi)</div>', unsafe_allow_html=True)
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">1. Resized</div></div>', unsafe_allow_html=True)
            st.image(img_resized_rgb, use_container_width=True)
            st.caption("224×224 — input CNN (S11)")
        with p_col2:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">2. SSR</div></div>', unsafe_allow_html=True)
            st.image(img_ssr_rgb, use_container_width=True)
            st.caption("Koreksi iluminasi CIELAB")
        with p_col3:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">3. Gamma (E*)</div></div>', unsafe_allow_html=True)
            st.image(img_gamma_rgb, use_container_width=True)
            st.caption("Enhancement terpilih (val F1)")
        with p_col4:
            st.markdown('<div class="dip-step-card"><div class="dip-step-title">4. Segmented</div></div>', unsafe_allow_html=True)
            st.image(img_segmented_rgb, use_container_width=True)
            st.caption(f"Input SVM & RF — obj_ratio {obj_ratio:.0%}")

        st.caption("Pipeline lengkap (kol 2–4) digunakan SVM (S5) dan RF (S9). CNN (S11) hanya menggunakan kol 1 (raw resize).")

        # --- Multi-model Inference Cards ---
        st.markdown('<div class="section-header" style="margin-top:0;">2. Multi-Model Inference</div>', unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)

        def _badge(pred):
            if pred == "fresh":
                return "badge-fresh", "fresh"
            if pred == "rotten":
                return "badge-rotten", "rotten"
            return "badge-na", "N/A"

        with m_col1:
            bc, bl = _badge(cnn_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">MobileNetV2 — S11</h4>
                    <span class="preproc-tag">Input: raw (resize only)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Confidence</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{cnn_confidence:.2%}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between;">
                    <span>Latency</span><b style="color:#334155;">{cnn_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            bc, bl = _badge(svm_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">SVM — S5</h4>
                    <span class="preproc-tag">Input: SSR + Gamma + Seg (220 dim)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Decision Score</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{svm_decision:.3f}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between;">
                    <span>Latency</span><b style="color:#334155;">{svm_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            bc, bl = _badge(rf_pred_class)
            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                <div>
                    <h4 class="card-title">Random Forest — S9</h4>
                    <span class="preproc-tag">Input: SSR + Gamma + Seg (220 dim)</span>
                    <div style="margin-top:12px;"><span class="badge {bc}">{bl}</span></div>
                    <div style="margin-top:14px;">
                        <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Confidence</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#0f172a; margin-top:4px;">{rf_confidence:.2%}</div>
                    </div>
                </div>
                <div style="margin-top:auto; padding-top:15px; font-size:0.75rem; color:#64748b; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between;">
                    <span>Latency</span><b style="color:#334155;">{rf_time:.2f} ms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Grad-CAM (CNN S11 — raw input) ---
        if model_cnn is not None:
            st.markdown('<div class="section-header">3. Deep Learning Explainability — Grad-CAM (S11)</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="saas-card" style="text-align:center;">', unsafe_allow_html=True)
                        st.markdown("<h5>Input CNN (raw resize)</h5>", unsafe_allow_html=True)
                        st.image(img_resized_rgb, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with g_col2:
                        st.markdown('<div class="saas-card" style="text-align:center;">', unsafe_allow_html=True)
                        st.markdown("<h5>Grad-CAM Attention Heatmap</h5>", unsafe_allow_html=True)
                        st.image(overlay_rgb, use_container_width=True)
                        st.caption("Merah/jingga = area paling berpengaruh terhadap keputusan CNN dalam membedakan fresh vs rotten.")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Grad-CAM tidak dapat dihitung: {e}")
