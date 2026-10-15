# ============================================================
# Retina AI — Diabetic Retinopathy Detection (Streamlit App)
# Clinical dashboard layout with hero header
# ============================================================

import os

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

import streamlit as st
import pandas as pd
import altair as alt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------------------------------------------------------
# Class names
# ------------------------------------------------------------
try:
    from class_names import class_names as CLASS_NAMES
except ImportError:
    CLASS_NAMES = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative_DR"}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild non-proliferative DR — microaneurysms only.",
    2: "Moderate non-proliferative DR — more than microaneurysms, but less than severe.",
    3: "Severe non-proliferative DR — extensive hemorrhages and vascular abnormalities.",
    4: "Proliferative DR — neovascularization present; vision-threatening, urgent referral advised.",
}
STAGE_GUIDANCE = {
    0: ("Routine follow-up", "No retinopathy pattern was detected. Continue regular diabetic eye screening and keep blood glucose, blood pressure, and cholesterol within the targets set by your care team."),
    1: ("Schedule a routine eye visit", "Mild changes were detected. Arrange a non-urgent review with an eye-care professional and continue healthy eating, regular physical activity, and prescribed diabetes care."),
    2: ("Arrange an eye-care review soon", "Moderate changes were detected. Book an appointment with an ophthalmologist or optometrist to discuss the finding and keep diabetes management and follow-up on track."),
    3: ("Contact an eye specialist promptly", "More advanced retinal changes were detected. Seek a prompt ophthalmology appointment, especially if vision changes occur, and follow your diabetes care plan closely."),
    4: ("Seek urgent ophthalmology care", "A vision-threatening pattern was detected. Contact an ophthalmologist urgently. Sudden vision loss, a curtain-like shadow, or new flashes and floaters require immediate medical attention."),
}
RISK_LEVELS = {0: "Low", 1: "Low-Moderate", 2: "Moderate", 3: "High", 4: "Critical"}
RISK_COLORS = {
    "Low": "#0f9d58",
    "Low-Moderate": "#65a30d",
    "Moderate": "#d97706",
    "High": "#ea580c",
    "Critical": "#dc2626",
}
CLASS_CHART_COLORS = ["#0f9d58", "#65a30d", "#d97706", "#ea580c", "#dc2626"]

MODEL_PATH = "best_convnext_model.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THUMB_SIZE = 260  # fixed display size for images — prevents stretching

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Retina AI Clinical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Clinical theme CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: #f4f7fa; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3b4a 0%, #0e5468 100%);
}
section[data-testid="stSidebar"] * { color: #eaf4f6 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #14b8a6; color: #06201c !important; border: none;
    font-weight: 700; border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton button:hover { background: #2dd4bf; }

/* ---------------- Hero banner ---------------- */
.hero-banner {
    position: relative;
    width: 100%;
    box-sizing: border-box;
    background: linear-gradient(120deg, #082f3d 0%, #0b5364 52%, #147d8c 100%);
    border-radius: 18px;
    padding: 56px 32px 48px;
    margin-bottom: 20px;
    text-align: center;
    overflow: hidden;
    box-shadow: 0 14px 32px rgba(8, 47, 61, 0.16);
}
.hero-title {
    position: relative;
    z-index: 2;
    color: #ffffff;
    font-size: clamp(3.8rem, 7vw, 6.4rem);
    line-height: 0.98;
    font-weight: 800;
    letter-spacing: 0;
    margin: 0;
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    text-shadow: 0 3px 18px rgba(0,0,0,0.18);
}
.hero-title span { color: #5eead4; }
.hero-subtitle {
    position: relative;
    z-index: 2;
    color: #cdeef2;
    font-size: 1.12rem;
    margin-top: 16px;
    font-weight: 400;
}
.hero-tagline {
    position: relative;
    z-index: 2;
    display: inline-block;
    margin-top: 16px;
    padding: 6px 18px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    color: #eaf4f6;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-kicker {
    position: relative;
    z-index: 2;
    color: #8ce8df;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* ---------------- Clinical image strip ---------------- */
.clinical-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 0 0 24px;
}
.clinical-image {
    aspect-ratio: 16 / 7;
    border-radius: 12px;
    background-color: #05080b;
    background-size: cover;
    background-position: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(16,24,40,0.1);
    transition: transform 180ms ease, box-shadow 180ms ease;
}
.clinical-image:hover {
    transform: translateY(-4px) scale(1.015);
    background-color: #000000;
    box-shadow: 0 12px 24px rgba(5, 8, 11, 0.3);
}
.clinical-image:hover::after { background: rgba(0, 0, 0, 0.58); }
.clinical-image::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent 25%, rgba(5, 24, 32, 0.78) 100%);
}
.clinical-image span {
    position: absolute;
    z-index: 1;
    bottom: 13px;
    left: 15px;
    color: #ffffff;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.clinical-retina {
    background-image: url('https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?auto=format&fit=crop&w=1000&q=85');
}
.clinical-care {
    background-image: url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=85');
}
.clinical-screening {
    background-image: url('https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&w=800&q=85');
}
@media (max-width: 700px) {
    .hero-banner { padding: 42px 18px 36px; }
    .hero-subtitle { font-size: 0.96rem; }
    .clinical-strip { grid-template-columns: 1fr; }
    .clinical-image { aspect-ratio: 16 / 9; }
}

/* ---------------- Cards ---------------- */
.stat-card {
    background: #ffffff; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.08); border-left: 5px solid #0e5468;
    height: 100%;
}
.stat-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.stat-value { color: #0f172a; font-size: 1.35rem; font-weight: 700; margin-top: 4px; }

.section-card {
    background: #ffffff; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.08); margin-bottom: 18px;
}
.section-title {
    font-size: 0.9rem; font-weight: 700; color: #0e5468;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px;
}
.img-caption { text-align:center; color:#64748b; font-size:0.78rem; margin-top:6px; }

.explainability-note {
    background: #ffffff;
    border-left: 4px solid #0e7490;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 2px 0 18px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.08);
    color: #475569;
    line-height: 1.6;
}
.explainability-note-title {
    color: #0e5468;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.explainability-note p { margin: 0 0 8px; }
.explainability-note ul { margin: 6px 0 0 18px; padding: 0; }
.explainability-note li { margin: 3px 0; }
.heat-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    margin: 10px 0 12px;
    color: #475569;
    font-size: 0.82rem;
}
.heat-legend-item { display: inline-flex; align-items: center; gap: 6px; }
.heat-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.next-step {
    background: #ecfeff;
    border: 1px solid #a5f3fc;
    border-radius: 8px;
    color: #155e75;
    padding: 10px 12px;
    margin-top: 12px;
}
.next-step strong { color: #0e5468; }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Model loading (cached, silent)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("convnext_tiny", pretrained=False)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features, NUM_CLASSES)

    loaded = os.path.exists(MODEL_PATH)
    if loaded:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    def reshape_transform(tensor):
        return tensor.permute(0, 3, 1, 2)

    target_layers = [model.stages[-1].blocks[-1].conv_dw]
    cam_extractor = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    return model, cam_extractor, device, loaded


model, cam_extractor, device, model_loaded = load_model()

infer_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(pil_image: Image.Image):
    pil_image = pil_image.convert("RGB")
    tensor = infer_transform(pil_image).unsqueeze(0).to(device)
    return tensor


def tensor_to_display_array(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = std * img + mean
    return np.clip(img, 0, 1)


def make_thumbnail(pil_image: Image.Image, size=THUMB_SIZE):
    """Square center-cropped thumbnail so portrait/landscape images never stretch."""
    img = pil_image.copy()
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size))
    return img


def run_inference(pil_image: Image.Image):
    input_tensor = preprocess_image(pil_image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    probs_np = probabilities.cpu().numpy()
    prediction_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[prediction_idx] * 100)

    targets = [ClassifierOutputTarget(prediction_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
    rgb_display = tensor_to_display_array(input_tensor)
    overlay = show_cam_on_image(rgb_display, grayscale_cam, use_rgb=True)
    heatmap_img = Image.fromarray(overlay).resize((THUMB_SIZE, THUMB_SIZE))

    return prediction_idx, confidence, probs_np, heatmap_img


# ------------------------------------------------------------
# Sidebar — upload + legend (uses the left side properly)
# ------------------------------------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])
    analyze_clicked = st.button("Analyze Image", use_container_width=True, disabled=uploaded_file is None)

    st.markdown("---")
    st.markdown("**Severity Scale**")
    for i in range(NUM_CLASSES):
        st.markdown(
            f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
            f"background:{CLASS_CHART_COLORS[i]};margin-right:6px;'></span>"
            f"{CLASS_NAMES[i].replace('_',' ')}",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Research demo — not a certified diagnostic device.")
    if not model_loaded:
        st.caption("⚠ Model weights not found in this folder.")

# ------------------------------------------------------------
# Main area — centered hero banner
# ------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
<div class="hero-kicker">Intelligent retinal health</div>
<h1 class="hero-title">Retina <span>AI</span></h1>
<p class="hero-subtitle">Explainable AI for Diabetic Retinopathy Screening</p>
<span class="hero-tagline">ConvNeXt-Tiny &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp; Clinical Decision Support</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="clinical-strip" aria-label="Clinical retinal care imagery">
<div class="clinical-image clinical-retina"><span>Retinal imaging</span></div>
<div class="clinical-image clinical-care"><span>Ophthalmic care</span></div>
<div class="clinical-image clinical-screening"><span>Clinical screening</span></div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a retinal fundus image from the sidebar to begin analysis.")
    st.stop()

pil_image = Image.open(uploaded_file)

if not analyze_clicked:
    col_a, _ = st.columns([1, 2])
    with col_a:
        st.image(make_thumbnail(pil_image), caption="Ready for analysis")
        st.caption('Click "Analyze Image" in the sidebar to run the model.')
    st.stop()

with st.spinner("Analyzing image..."):
    prediction_idx, confidence, probs_np, heatmap_img = run_inference(pil_image)

label = CLASS_NAMES[prediction_idx]
risk = RISK_LEVELS[prediction_idx]
risk_color = RISK_COLORS[risk]

# ------------------------------------------------------------
# Stat row
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="stat-card" style="border-left-color:{risk_color};">
        <div class="stat-label">Predicted Stage</div>
        <div class="stat-value">{label.replace('_',' ')}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Confidence</div>
        <div class="stat-value">{confidence:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="stat-card" style="border-left-color:{risk_color};">
        <div class="stat-label">Risk Level</div>
        <div class="stat-value" style="color:{risk_color};">{risk}</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"<p style='color:#475569; margin:14px 2px 0;'>{CLASS_DESCRIPTIONS[prediction_idx]}</p>", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------
# Images + Chart row
# ------------------------------------------------------------
left, right = st.columns([1, 1.3])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explainability — Grad-CAM</div>', unsafe_allow_html=True)
    img_c1, img_c2 = st.columns(2)
    with img_c1:
        st.image(make_thumbnail(pil_image))
        st.markdown('<p class="img-caption">Original</p>', unsafe_allow_html=True)
    with img_c2:
        st.image(heatmap_img)
        st.markdown('<p class="img-caption">Model Attention</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Class Probability Breakdown</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i].replace("_", " ") for i in range(NUM_CLASSES)],
        "Probability": [round(float(probs_np[i] * 100), 2) for i in range(NUM_CLASSES)],
    })

    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=28)
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("Class:N", sort=None, title=None),
            color=alt.Color("Class:N", scale=alt.Scale(range=CLASS_CHART_COLORS), legend=None),
            tooltip=["Class", "Probability"],
        )
        .properties(height=220)
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="explainability-note">
<div class="explainability-note-title">Why this stage was predicted</div>
<p>The model predicted <strong>{label.replace('_', ' ')}</strong> with <strong>{confidence:.1f}% confidence</strong>. The Grad-CAM overlay highlights the retinal regions that most influenced this decision. These highlighted patterns are the visual evidence the ConvNeXt model connected with the learned features of the <strong>{label.replace('_', ' ')}</strong> stage.</p>
<div class="heat-legend">
<span class="heat-legend-item"><span class="heat-swatch" style="background:#ef4444;"></span>Strongest model attention</span>
<span class="heat-legend-item"><span class="heat-swatch" style="background:#facc15;"></span>High attention</span>
<span class="heat-legend-item"><span class="heat-swatch" style="background:#2563eb;"></span>Lower attention</span>
</div>
<ul>
<li>The original image shows the retinal appearance that was evaluated.</li>
<li>The heat map shows where the model found the most decision-relevant visual evidence.</li>
<li>This supports the prediction, but it does not identify lesions or replace an ophthalmologist's review.</li>
</ul>
<div class="next-step"><strong>{STAGE_GUIDANCE[prediction_idx][0]}:</strong> {STAGE_GUIDANCE[prediction_idx][1]}</div>
</div>
""",
    unsafe_allow_html=True,
)