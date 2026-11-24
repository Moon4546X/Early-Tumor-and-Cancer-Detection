from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
MODELS = ROOT / "Models"
CERVICAL_DATA = DATASET / "kag_risk_factors_cervical_cancer.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Early Cancer Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern CSS — glassmorphism, gradient accents, clean typography
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --ink: #0f172a;
        --ink-strong: #020617;
        --muted: #475569;
        --muted-light: #64748b;
        --bg: #ffffff;
        --card: #ffffff;
        --border: #cbd5e1;
        --border-strong: #94a3b8;
        --primary: #0ea5e9;
        --primary-hover: #0284c7;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --mint: #dcfce7;
        --slate: #f1f5f9;
        --shadow: 0 4px 20px -4px rgba(2, 6, 23, 0.06);
        --shadow-hover: 0 8px 30px -6px rgba(2, 6, 23, 0.1);
        --radius: 12px;
        --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg) !important;
        color: var(--ink) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--ink-strong) !important;
    }

    p, span, div, label {
        color: var(--ink) !important;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Background glow */
    .app-bg {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        background:
            radial-gradient(circle at 10% 20%, rgba(14, 165, 233, 0.10), transparent 45%),
            radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.08), transparent 45%);
    }

    /* Hero */
    .hero {
        padding: 3rem 0 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--primary);
        margin-bottom: 0.75rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 1.1;
        margin: 0.25rem 0 0.75rem;
        letter-spacing: -0.02em;
        color: var(--ink-strong);
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 700px;
        line-height: 1.6;
        margin: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption {
        color: var(--muted) !important;
    }
    .sidebar-brand {
        padding: 1.25rem 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .sidebar-brand .brand-mark {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
        margin-right: 0.75rem;
        vertical-align: middle;
    }
    .sidebar-brand .brand-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        color: var(--ink-strong);
        vertical-align: middle;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1rem 0;
    }
    .sidebar-caption {
        font-size: 0.75rem;
        color: var(--muted);
        padding: 0 1rem 1.25rem;
        line-height: 1.5;
    }

    /* Widget labels - ensure readability */
    .stSelectbox label,
    .stNumberInput label,
    .stFileUploader label,
    .stRadio label,
    .stTextInput label {
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Cards */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }
    .card:hover {
        box-shadow: var(--shadow-hover);
        border-color: var(--border-strong);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        color: var(--ink-strong);
    }
    .card-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 0;
    }

    /* Section headings */
    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: var(--ink-strong);
        margin: 2rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Result box */
    .result {
        border-radius: var(--radius);
        padding: 1.75rem 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result--positive {
        background: #fff7f7;
        border-color: #fca5a5;
        border-left: 5px solid var(--danger);
    }
    .result--negative {
        background: #f0fdf4;
        border-color: #86efac;
        border-left: 5px solid var(--secondary);
    }
    .result h3 {
        margin: 0 0 0.5rem;
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--ink-strong);
    }
    .result p {
        margin: 0 0 1rem;
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 500;
    }
    .result .confidence-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted-light);
        margin-bottom: 0.4rem;
        display: block;
    }

    /* Form elements */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        font-family: 'Inter', sans-serif;
        transition: var(--transition);
        width: 100%;
    }
    .stButton > button:hover {
        background: var(--primary-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px -2px rgba(14, 165, 233, 0.35);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    .stSelectbox > div > div,
    .stNumberInput input {
        border-radius: 10px;
        border: 1px solid var(--border-strong);
        background: var(--card);
        color: var(--ink) !important;
        transition: var(--transition);
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
    }

    /* File uploader */
    .stFileUploader > section {
        border: 2px dashed var(--border-strong);
        border-radius: var(--radius);
        padding: 2.5rem 2rem;
        text-align: center;
        background: var(--slate);
        transition: var(--transition);
    }
    .stFileUploader > section:hover {
        border-color: var(--primary);
        background: #e0f2fe;
    }
    .stFileUploader > section > button {
        background: transparent !important;
        color: var(--primary) !important;
        border: none !important;
        font-weight: 700;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.8rem;
        color: var(--muted-light);
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        line-height: 1.5;
    }

    /* Confidence bar */
    .confidence-track {
        height: 8px;
        background: var(--slate);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .model-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        background: var(--card);
        font-size: 0.85rem;
        color: var(--ink);
        font-weight: 600;
        margin-bottom: 0.5rem;
        transition: var(--transition);
    }
    .model-chip:hover {
        border-color: var(--primary);
        color: var(--primary);
    }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--muted-light);
    }
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data / model loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_cervical_data():
    frame = pd.read_csv(CERVICAL_DATA)
    frame = frame.drop(
        ["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"],
        axis=1,
    )
    frame = frame.replace("?", np.nan).drop_duplicates()
    frame = frame.apply(pd.to_numeric, errors="coerce")

    defaults = {
        "Smokes": 1,
        "Hormonal Contraceptives": 1,
        "IUD": 0,
        "STDs": 1,
        "IUD (years)": 0,
    }
    for column in frame.columns:
        if column == "Biopsy":
            continue
        value = defaults.get(column, frame[column].median())
        frame[column] = frame[column].fillna(value)
    return frame


@st.cache_resource
def load_joblib_model(model_name):
    return joblib.load(MODELS / model_name)


@st.cache_resource
def load_brain_model(model_name):
    import keras
    return keras.models.load_model(MODELS / model_name)


def cervical_training_scaler(features):
    target = load_cervical_data()["Biopsy"]
    train_features, _test_features, _train_target, _test_target = train_test_split(
        features, target, test_size=0.25, random_state=12
    )
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def chip(label):
    st.markdown(f'<span class="model-chip">{label}</span>', unsafe_allow_html=True)


def show_result(title, detail, kind="neutral", probability=None):
    kind = "positive" if kind == "positive" else "negative"
    css_class = f"result--{kind}"
    st.markdown(
        f'<div class="result {css_class}"><h3>{title}</h3><p>{detail}</p></div>',
        unsafe_allow_html=True,
    )
    if probability is not None:
        clamped = max(0.0, min(1.0, float(probability)))
        st.markdown('<span class="confidence-label">Model confidence</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="confidence-track"><div class="confidence-fill" style="width:{clamped*100:.0f}%"></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{clamped:.1%}")


def disclaimer():
    st.markdown(
        '<div class="disclaimer">'
        'This application is for research and education only. Results must not be used '
        'for medical decisions or as a substitute for professional diagnosis.'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Workflow renders
# ---------------------------------------------------------------------------
def render_cervical():
    st.markdown(
        '<div class="section-title">🩺 Cervical risk assessment</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="card-subtitle">Enter the risk-factor values used by the trained cervical models.</p>',
        unsafe_allow_html=True,
    )

    frame = load_cervical_data()
    features = frame.drop(columns=["Biopsy"])
    scaler = cervical_training_scaler(features)

    model_options = {
        "Random forest": "cc_rf_model",
        "SVM": "cc_svm_model",
        "Decision tree": "cc_decisionclassifier",
        "K-nearest neighbors": "cc_knn_model",
        "Voting ensemble": "cc_voting_model",
    }

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sel_col, info_col = st.columns([2, 3])
        with sel_col:
            selected_model = st.selectbox("Model", list(model_options), label_visibility="collapsed")
        with info_col:
            chip(selected_model)

        st.markdown("### Risk factors")
        columns = list(features.columns)
        # Three-column input grid
        for start in range(0, len(columns), 3):
            cols = st.columns(3)
            for slot, column in zip(cols, columns[start : start + 3]):
                with slot:
                    unique_values = set(features[column].dropna().unique())
                    if unique_values and unique_values.issubset({0, 1}):
                        st.selectbox(
                            column,
                            [0, 1],
                            index=int(round(features[column].median())),
                            key=f"cerv_{column}",
                        )
                    else:
                        st.number_input(
                            column,
                            value=float(features[column].median()),
                            min_value=0.0,
                            step=0.1,
                            key=f"cerv_{column}",
                        )

        if st.button("Assess cervical risk", type="primary", use_container_width=True):
            values = {c: st.session_state[f"cerv_{c}"] for c in columns}
            model = load_joblib_model(model_options[selected_model])
            input_frame = pd.DataFrame([values], columns=columns)
            prediction = int(model.predict(scaler.transform(input_frame))[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(scaler.transform(input_frame))[0][prediction])
            kind = "positive" if prediction else "negative"
            title = "Positive biopsy prediction" if prediction else "Negative biopsy prediction"
            detail = (
                "The selected model predicts a positive biopsy result."
                if prediction
                else "The selected model predicts a negative biopsy result."
            )
            show_result(title, detail, kind=kind, probability=probability)

        st.markdown('</div>', unsafe_allow_html=True)

    disclaimer()


def render_oral():
    st.markdown(
        '<div class="section-title">🦷 Oral image classification</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="card-subtitle">Upload an oral image for classification with one of the supplied models.</p>',
        unsafe_allow_html=True,
    )

    model_options = {"Random forest": "oc_rf_model", "SVM": "oc_svm_model", "Decision tree": "oc_dt_model"}

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sel_col, info_col = st.columns([2, 3])
        with sel_col:
            selected_model = st.selectbox("Model", list(model_options), label_visibility="collapsed")
        with info_col:
            chip(selected_model)

        uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded is None:
            st.markdown(
                '<div class="empty-state"><div class="icon">📷</div>'
                '<p>Choose an image to enable prediction</p></div>',
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            if st.button("Classify oral image", type="primary", use_container_width=True):
                model = load_joblib_model(model_options[selected_model])
                pixels = np.asarray(image.resize((64, 64)), dtype=np.float32)
                prediction_input = pixels.reshape(1, -1)
                prediction = int(model.predict(prediction_input)[0])
                probability = None
                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(prediction_input)[0][prediction])
                if prediction == 1:
                    show_result("Cancer detected", "The model classified this image as cancer.", kind="positive", probability=probability)
                else:
                    show_result("No cancer detected", "The model classified this image as non-cancer.", kind="negative", probability=probability)

        st.markdown('</div>', unsafe_allow_html=True)

    disclaimer()


def render_brain():
    st.markdown(
        '<div class="section-title">🧠 Brain-tumor image classification</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="card-subtitle">Upload an MRI image for classification with a supplied CNN model.</p>',
        unsafe_allow_html=True,
    )

    model_options = {"CNN": "bt-cnn.keras", "CNN 2": "bt-cnn2.keras", "MobileNet": "bt-mobilenet.keras"}

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        sel_col, info_col = st.columns([2, 3])
        with sel_col:
            selected_model = st.selectbox("Model", list(model_options), label_visibility="collapsed")
        with info_col:
            chip(selected_model)

        uploaded = st.file_uploader("Upload a JPG or PNG MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded is None:
            st.markdown(
                '<div class="empty-state"><div class="icon">🧠</div>'
                '<p>Choose an MRI image to enable prediction</p></div>',
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded MRI image", use_container_width=True)
            if st.button("Classify brain image", type="primary", use_container_width=True):
                with st.spinner("Analyzing MRI image..."):
                    model = load_brain_model(model_options[selected_model])
                    pixels = np.asarray(image.resize((180, 180)), dtype=np.float32)
                    prediction_input = np.expand_dims(pixels, axis=0)
                    raw_prediction = np.asarray(model.predict(prediction_input, verbose=0))
                    scores = raw_prediction[0]
                    prediction = int(np.argmax(scores))
                classes = ["glioma", "meningioma", "notumor", "pituitary"]
                display_names = {
                    "glioma": "Glioma",
                    "meningioma": "Meningioma",
                    "notumor": "No tumor",
                    "pituitary": "Pituitary tumor",
                }
                predicted_class = classes[prediction]
                probability = float(scores[prediction]) if np.max(scores) <= 1 else None
                kind = "negative" if predicted_class == "notumor" else "positive"
                title = f"Prediction: {display_names[predicted_class]}"
                detail = "The CNN's highest-scoring class is shown above."
                show_result(title, detail, kind=kind, probability=probability)

        st.markdown('</div>', unsafe_allow_html=True)

    disclaimer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Background glow
    st.markdown('<div class="app-bg"></div>', unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div class="hero">'
        '<div class="hero-kicker">🫀 Early Cancer Detection</div>'
        '<h1>Early tumor and cancer prediction</h1>'
        '<p>A research interface for exploring trained machine-learning models across cervical risk, '
        'oral image, and brain-tumor MRI workflows.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="brand-mark">🫀</div>'
            '<div class="brand-name">Early Cancer Detection</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-caption">Choose a workflow to begin.</div>', unsafe_allow_html=True)
        workflow = st.radio(
            "Workflow",
            ["Cervical risk", "Oral image", "Brain-tumor image"],
            label_visibility="collapsed",
        )
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-caption">'
            'For research and education only. Results must not be used for medical decisions.'
            '</div>',
            unsafe_allow_html=True,
        )

    try:
        if workflow == "Cervical risk":
            render_cervical()
        elif workflow == "Oral image":
            render_oral()
        else:
            render_brain()
    except FileNotFoundError as error:
        st.error(f"A required project file is missing: {error.filename}")
    except Exception as error:
        st.error(f"The selected workflow could not run: {error}")
        st.info("Check that the environment was installed from requirements.txt and that the supplied model files are present.")


if __name__ == "__main__":
    main()
