import os
import json

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ── Page config — MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Naija Food Classifier",
    page_icon="🍲",
    layout="centered",
)

# ── Lazy-load heavy packages ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_tf():
    import tensorflow as tf
    return tf

tf = _import_tf()

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# class_info.json is committed to the repo — no download needed
CLASS_INFO_PATH = os.path.join(os.path.dirname(__file__), "class_info.json")

# Google Drive file IDs — only the large model files live here
DRIVE_FILES = {
    "EfficientNetB0_model.keras": "1ex2AEXqfTiMNT3UOlhzxOAt1HJD7n3fe",
    "ResNet50_model.keras":       "1PUAxOh3Kj_AdKcZ3LUGezlpwZLP1jKTA",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f0e0a; }
[data-testid="stAppViewContainer"] { background-color: #0f0e0a; }
[data-testid="stHeader"]           { background-color: #0f0e0a; }
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
.stButton > button {
    background: #f5c842 !important; color: #0f0e0a !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important; padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stFileUploader"] {
    background: #1c1a13; border: 1px dashed #3a3420; border-radius: 12px; padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Google Drive download ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading files from Google Drive…")
def download_all_files() -> str:
    """
    Download all files listed in DRIVE_FILES into MODEL_DIR.
    Returns MODEL_DIR on success, raises on failure.
    Cached so it only runs once per deployment.
    """
    import gdown

    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, file_id in DRIVE_FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            continue  # already downloaded (e.g. from a previous warm session)
        if file_id.startswith("YOUR_"):
            raise ValueError(
                f"Please replace the placeholder Drive ID for '{filename}' "
                "in the DRIVE_FILES dict at the top of app.py."
            )
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"⬇️ Downloading {filename}…")
        gdown.download(url, dest, quiet=False)
        if not os.path.exists(dest):
            raise RuntimeError(
                f"Download failed for '{filename}'. "
                "Check that the Google Drive file is shared as 'Anyone with the link'."
            )

    return MODEL_DIR


# ── Class-name loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Reading class labels…")
def load_class_names(_unused: str = "") -> list:
    # Reads class_info.json committed in the repo root (next to app.py)
    with open(CLASS_INFO_PATH) as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return data  # plain list


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_by_filename(model_dir: str, fname: str):
    path = os.path.join(model_dir, fname)
    return tf.keras.models.load_model(path)


def discover_models(model_dir: str) -> list:
    preferred = ["EfficientNetB0_model.keras", "ResNet50_model.keras"]
    found = [f for f in preferred if os.path.exists(os.path.join(model_dir, f))]
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".keras") and f not in found:
            found.append(f)
    return found


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Boot sequence ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍲 Naija Food<br>Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a photo — get instant Nigerian dish recognition</div>',
    unsafe_allow_html=True,
)

try:
    model_dir   = download_all_files()
    class_names = load_class_names()
    available   = discover_models(model_dir)
    if not available:
        raise FileNotFoundError("No .keras model files found after download.")
    boot_ok = True
except Exception as exc:
    boot_ok    = False
    boot_error = str(exc)

if not boot_ok:
    st.error(f"⚠️ Could not start: {boot_error}")
    st.markdown("""
    **Common fixes:**
    - Make sure every Google Drive file is shared → *Anyone with the link → Viewer*
    - Replace all `YOUR_..._FILE_ID` placeholders in `DRIVE_FILES` at the top of `app.py`
    - Check that `gdown` is in your `requirements.txt`
    """)
    st.stop()


# ── Model selector ────────────────────────────────────────────────────────────
with st.expander("⚙️ Switch model"):
    chosen_fname = st.selectbox(
        "Choose architecture", available, index=0,
        format_func=lambda x: x.replace("_model.keras", ""),
    )

model     = load_model_by_filename(model_dir, chosen_fname)
arch_name = chosen_fname.replace("_model.keras", "")

st.markdown(
    f'<div class="model-badge">Model: {arch_name} &nbsp;·&nbsp; {len(class_names)} classes</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supported: JPG · PNG · WEBP</p>', unsafe_allow_html=True)


# ── Predict & display ─────────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            tensor = preprocess(image)
            preds  = model.predict(tensor, verbose=0)[0]

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
        xaxis=dict(
            range=[0, max(top5_conf) * 1.22],
            showgrid=True, gridcolor="#1e1c14",
            ticksuffix="%", tickfont=dict(color="#6b6050"), zeroline=False,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#c4b99a", size=13)),
        margin=dict(l=10, r=60, t=10, b=10), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Full class distribution"):
        all_idx    = np.argsort(preds)[::-1]
        all_conf   = preds[all_idx] * 100
        all_labels = [class_names[i].replace("_", " ").title() for i in all_idx]
        all_colors = ["#f5c842" if i == 0 else "#2a2820" for i in range(len(all_idx))]

        fig2 = go.Figure(go.Bar(
            x=all_conf, y=all_labels, orientation="h",
            marker=dict(color=all_colors),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="#0f0e0a", plot_bgcolor="#0f0e0a",
            font=dict(color="#c4b99a", family="DM Sans"),
            xaxis=dict(showgrid=True, gridcolor="#1e1c14", ticksuffix="%",
                       tickfont=dict(color="#6b6050"), zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c4b99a", size=11)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, len(all_idx) * 28),
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0;">
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
    unsafe_allow_html=True,
)
