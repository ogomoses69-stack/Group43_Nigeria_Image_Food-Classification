import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naija Food Classifier",
    page_icon="🍲",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f0e0a; }
[data-testid="stAppViewContainer"] { background-color: #0f0e0a; }
[data-testid="stHeader"] { background-color: #0f0e0a; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 900;
    color: #f5c842; line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif; font-weight: 300;
    color: #c4b99a; font-size: 1.05rem;
    margin-bottom: 2rem; letter-spacing: 0.03em;
}
.divider { border: none; border-top: 1px solid #2e2b22; margin: 1.5rem 0; }
.result-card {
    background: linear-gradient(135deg, #1c1a13 0%, #241f10 100%);
    border: 1px solid #3a3420; border-left: 4px solid #f5c842;
    border-radius: 12px; padding: 1.5rem 2rem; margin: 1.5rem 0;
}
.result-label { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: #f5c842; margin: 0; }
.result-conf  { font-size: 0.95rem; color: #8a8070; margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; }
.conf-value   { color: #e8d48b; font-weight: 500; }
.upload-hint  { color: #6b6050; font-size: 0.85rem; text-align: center; margin-top: 0.5rem; }
.model-badge  {
    display: inline-block; background: #1c1a13;
    border: 1px solid #3a3420; border-radius: 20px;
    padding: 0.25rem 0.85rem; font-size: 0.8rem; color: #8a8070; margin-bottom: 1.5rem;
}
[data-testid="stFileUploader"] {
    background: #1c1a13; border: 1px dashed #3a3420; border-radius: 12px; padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)
BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
GDRIVE_FILE_ID = "1sPlNu1DEi6BMBgEJr195CsqcR5_722Fh"

# ── Download model from Google Drive ─────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Downloading model files (first time only)…")
def download_models():
    import gdown, zipfile
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(os.path.join(MODEL_DIR, "class_info.json")):
        return
    url      = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    zip_path = os.path.join(BASE_DIR, "models.zip")
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(MODEL_DIR)
    os.remove(zip_path)

# ── Load class names ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Reading class labels…")
def load_class_names():
    # class_info.json lives inside the models folder after download
    path = os.path.join(MODEL_DIR, "class_info.json")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    return data

# ── Load a specific model ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(fname: str):
    import tensorflow as tf
    path = os.path.join(MODEL_DIR, fname)
    return tf.keras.models.load_model(path, compile=False)

# ── Discover available models (ResNet50 only) ─────────────────────────────────
def discover_models():
    preferred = [
        "ResNet50_model.keras",
        "ResNet50_checkpoint.keras",
    ]
    return [f for f in preferred if os.path.exists(os.path.join(MODEL_DIR, f))]

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image, model_fname: str) -> np.ndarray:
    """
    Apply the correct preprocessing for each architecture.
    ResNet50 expects pixel values in [-1, 1] via resnet50.preprocess_input,
    NOT simple /255 normalisation (which caused wrong predictions).
    """
    import tensorflow as tf
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍲 Naija Food<br>Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a photo — get instant Nigerian dish recognition</div>', unsafe_allow_html=True)

# ── Boot ──────────────────────────────────────────────────────────────────────
try:
    download_models()
    class_names = load_class_names()
    available   = discover_models()
    if not available:
        raise FileNotFoundError("No ResNet50 model files found after download.")
    boot_ok = True
except Exception as e:
    boot_ok    = False
    boot_error = str(e)

if not boot_ok:
    st.error(f"⚠️ Could not load: {boot_error}")
    st.stop()

# ── Model selector ────────────────────────────────────────────────────────────
st.markdown("### 🤖 Choose a Model")
chosen_fname = st.radio(
    "Select the model to use for prediction:",
    available,
    format_func=lambda x: x.replace("_model.keras", " (Main)")
                           .replace("_checkpoint.keras", " (Checkpoint)"),
)

model     = load_model(chosen_fname)
arch_name = chosen_fname.replace("_model.keras", "").replace("_checkpoint.keras", " Checkpoint")

st.markdown(
    f'<div class="model-badge">Model: {arch_name} &nbsp;·&nbsp; {len(class_names)} classes</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### 📸 Upload a Food Image")
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supported: JPG · PNG · WEBP</p>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            preds = model.predict(preprocess(image, chosen_fname), verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_conf  = float(preds[top_idx]) * 100
        top_label = class_names[top_idx].replace("_", " ").title()

        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">{top_label}</p>
            <p class="result-conf">Confidence: <span class="conf-value">{top_conf:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Model Confidence — Top 5 Predictions")

    top5_idx    = np.argsort(preds)[::-1][:5]
    top5_conf   = preds[top5_idx] * 100
    top5_labels = [class_names[i].replace("_", " ").title() for i in top5_idx]
    bar_colors  = ["#f5c842" if i == 0 else "#3a3420" for i in range(5)]

    fig = go.Figure(go.Bar(
        x=top5_conf, y=top5_labels, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in top5_conf],
        textposition="outside",
        textfont=dict(color="#c4b99a", size=12),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0f0e0a", plot_bgcolor="#0f0e0a",
        font=dict(color="#c4b99a", family="DM Sans"),
        xaxis=dict(range=[0, max(top5_conf)*1.22], showgrid=True,
                   gridcolor="#1e1c14", ticksuffix="%",
                   tickfont=dict(color="#6b6050"), zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#c4b99a", size=13)),
        margin=dict(l=10, r=60, t=10, b=10), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Full class distribution"):
        all_idx    = np.argsort(preds)[::-1]
        all_conf   = preds[all_idx] * 100
        all_labels = [class_names[i].replace("_", " ").title() for i in all_idx]
        fig2 = go.Figure(go.Bar(
            x=all_conf, y=all_labels, orientation="h",
            marker=dict(color=["#f5c842" if i==0 else "#2a2820" for i in range(len(all_idx))]),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="#0f0e0a", plot_bgcolor="#0f0e0a",
            font=dict(color="#c4b99a", family="DM Sans"),
            xaxis=dict(showgrid=True, gridcolor="#1e1c14", ticksuffix="%",
                       tickfont=dict(color="#6b6050"), zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c4b99a", size=11)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, len(all_idx)*28),
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0;">
        <div style="font-size: 3.5rem;">🍛</div>
        <div style="font-size: 0.9rem; color: #4a4535; margin-top: 0.5rem;">Waiting for an image…</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color: #3a3420; font-size:0.78rem;">'
    'TechCrush AI/ML Bootcamp · Group 43 · Nigeria Food Classification</p>',
    unsafe_allow_html=True,
)
