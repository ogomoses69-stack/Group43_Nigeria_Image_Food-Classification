import streamlit as st
import gdown
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- FILE NAMES ----------
RESNET_MODEL = "ResNet50_model.keras"
EFFICIENT_MODEL = "EfficientNet80_model.keras"
CLASS_FILE = "class_info.json"

# ---------- GOOGLE DRIVE IDS (REPLACE THESE) ----------
RESNET_ID = "1PUAxOh3Kj_AdKcZ3LUGezlpwZLP1jKTA"
EFFICIENT_ID = "1ex2AEXqfTiMNT3UOlhzxOAt1HJD7n3fe"
CLASS_ID = "1n8clrmDTRI4T6ByV2T-JBTuJw-HUkPtq"

# ---------- DOWNLOAD FUNCTION ----------
def download_file(file_name, file_id):
    if not os.path.exists(file_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)

# ---------- DOWNLOAD FILES ----------
download_file(RESNET_MODEL, RESNET_ID)
download_file(EFFICIENT_MODEL, EFFICIENT_ID)
download_file(CLASS_FILE, CLASS_ID)

# ---------- LOAD MODELS ----------
resnet_model = load_model(RESNET_MODEL)
efficient_model = load_model(EFFICIENT_MODEL)

# ---------- LOAD CLASS LABELS ----------
with open(CLASS_FILE, "r") as f:
    class_info = json.load(f)

# ---------- STREAMLIT UI ----------
st.title("AI Image Classifier")

model_choice = st.selectbox("Choose Model", ["ResNet50", "EfficientNet80"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ---------- PREDICTION ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((224, 224))  # adjust if your model needs different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = resnet_model if model_choice == "ResNet50" else efficient_model

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    label = class_info[str(pred_class)]

    st.success(f"Prediction: {label}")

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naija Food Classifier",
    page_icon="🍲",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f0e0a; }
[data-testid="stAppViewContainer"] { background-color: #0f0e0a; }
[data-testid="stHeader"] { background-color: #0f0e0a; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #f5c842;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #c4b99a;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    letter-spacing: 0.03em;
}
.divider {
    border: none;
    border-top: 1px solid #2e2b22;
    margin: 1.5rem 0;
}
.result-card {
    background: linear-gradient(135deg, #1c1a13 0%, #241f10 100%);
    border: 1px solid #3a3420;
    border-left: 4px solid #f5c842;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5c842;
    margin: 0;
}
.result-conf {
    font-size: 0.95rem;
    color: #8a8070;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.conf-value {
    color: #e8d48b;
    font-weight: 500;
}
.upload-hint {
    color: #6b6050;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 0.5rem;
}
.model-badge {
    display: inline-block;
    background: #1c1a13;
    border: 1px solid #3a3420;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.8rem;
    color: #8a8070;
    margin-bottom: 1.5rem;
}
.stButton > button {
    background: #f5c842 !important;
    color: #0f0e0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: #1c1a13;
    border: 1px dashed #3a3420;
    border-radius: 12px;
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & class info ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_classes(model_dir: str):
    class_info_path = os.path.join(model_dir, "class_info.json")
    with open(class_info_path) as f:
        class_info = json.load(f)
    class_names = class_info["class_names"]

    # Prefer EfficientNetB0; fall back to ResNet50
    for fname in ("EfficientNetB0_model.keras", "ResNet50_model.keras"):
        model_path = os.path.join(model_dir, fname)
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            arch_name = fname.replace("_model.keras", "")
            return model, class_names, arch_name

    raise FileNotFoundError(
        "No model file found. Make sure 'EfficientNetB0_model.keras' or "
        "'ResNet50_model.keras' is inside the 'models/' folder."
    )


IMG_SIZE = (224, 224)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    model, class_names, arch_name = load_model_and_classes(MODEL_DIR)
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍲 Naija Food<br>Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a photo — get instant Nigerian dish recognition</div>', unsafe_allow_html=True)

if model_loaded:
    st.markdown(f'<div class="model-badge">Model: {arch_name} &nbsp;·&nbsp; {len(class_names)} classes</div>', unsafe_allow_html=True)
else:
    st.error(f"⚠️ Could not load model: {load_error}")
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Model selector (optional) ─────────────────────────────────────────────────
with st.expander("⚙️ Switch model"):
    available = [f for f in ("EfficientNetB0_model.keras", "ResNet50_model.keras")
                 if os.path.exists(os.path.join(MODEL_DIR, f))]
    chosen = st.selectbox("Choose architecture", available,
                          index=0,
                          format_func=lambda x: x.replace("_model.keras", ""))

    @st.cache_resource(show_spinner="Loading selected model…")
    def load_specific(fname):
        m = tf.keras.models.load_model(os.path.join(MODEL_DIR, fname))
        return m, fname.replace("_model.keras", "")

    if chosen:
        model, arch_name = load_specific(chosen)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supported: JPG · PNG · WEBP</p>', unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            tensor = preprocess(image)
            preds = model.predict(tensor, verbose=0)[0]          # shape: (num_classes,)

        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx]) * 100
        top_label = class_names[top_idx].replace("_", " ").title()

        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">{top_label}</p>
            <p class="result-conf">Confidence: <span class="conf-value">{top_conf:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Top-5 confidence bar chart ─────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Model Confidence — Top 5 Predictions")

    top5_idx  = np.argsort(preds)[::-1][:5]
    top5_conf = preds[top5_idx] * 100
    top5_labels = [class_names[i].replace("_", " ").title() for i in top5_idx]

    bar_colors = ["#f5c842" if i == 0 else "#3a3420" for i in range(5)]

    fig = go.Figure(go.Bar(
        x=top5_conf,
        y=top5_labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in top5_conf],
        textposition="outside",
        textfont=dict(color="#c4b99a", size=12),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0f0e0a",
        plot_bgcolor="#0f0e0a",
        font=dict(color="#c4b99a", family="DM Sans"),
        xaxis=dict(
            range=[0, max(top5_conf) * 1.22],
            showgrid=True,
            gridcolor="#1e1c14",
            ticksuffix="%",
            tickfont=dict(color="#6b6050"),
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#c4b99a", size=13),
        ),
        margin=dict(l=10, r=60, t=10, b=10),
        height=280,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Full distribution expander ─────────────────────────────────────────
    with st.expander("📊 Full class distribution"):
        all_idx    = np.argsort(preds)[::-1]
        all_conf   = preds[all_idx] * 100
        all_labels = [class_names[i].replace("_", " ").title() for i in all_idx]
        all_colors = ["#f5c842" if i == 0 else "#2a2820" for i in range(len(all_idx))]

        fig2 = go.Figure(go.Bar(
            x=all_conf,
            y=all_labels,
            orientation="h",
            marker=dict(color=all_colors),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="#0f0e0a",
            plot_bgcolor="#0f0e0a",
            font=dict(color="#c4b99a", family="DM Sans"),
            xaxis=dict(
                showgrid=True, gridcolor="#1e1c14",
                ticksuffix="%", tickfont=dict(color="#6b6050"), zeroline=False,
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c4b99a", size=11)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, len(all_idx) * 28),
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #3a3420;">
        <div style="font-size: 3.5rem;">🍛</div>
        <div style="font-size: 0.9rem; color: #4a4535; margin-top: 0.5rem;">
            Waiting for an image…
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color: #3a3420; font-size:0.78rem;">'
    'TechCrush AI/ML Bootcamp · Group 43 · Nigeria Food Classification</p>',
    unsafe_allow_html=True
)
