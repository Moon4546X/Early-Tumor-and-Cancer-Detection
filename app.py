from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
MODELS = ROOT / "Models"
CERVICAL_DATA = DATASET / "kag_risk_factors_cervical_cancer.csv"

st.set_page_config(
    page_title="Early Cancer Prediction",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #172322; --muted: #60716d; --mint: #d9f4e6; --green: #176b4d; --coral: #f07858; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .hero { padding: 1.5rem 0 1rem; border-bottom: 1px solid #d9e5df; margin-bottom: 1.5rem; }
    .hero-kicker { color: var(--coral); font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .75rem; }
    .hero h1 { font-size: clamp(2rem, 5vw, 4.3rem); line-height: .98; margin: .35rem 0 .8rem; }
    .hero p { color: var(--muted); max-width: 700px; font-size: 1.05rem; }
    .result { background: var(--mint); border-left: 6px solid var(--green); padding: 1.1rem 1.3rem; margin: 1rem 0; }
    .result h3 { margin: 0 0 .3rem; }
    .note { color: var(--muted); font-size: .9rem; }
    section[data-testid="stSidebar"] { background: #f2f7f3; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def show_result(title, detail, probability=None):
    st.markdown(
        f'<div class="result"><h3>{title}</h3><div>{detail}</div></div>',
        unsafe_allow_html=True,
    )
    if probability is not None:
        # Clamp to [0, 1] — some saved models return decision scores outside this range
        clamped = max(0.0, min(1.0, float(probability)))
        st.progress(clamped, text=f"Model confidence: {clamped:.1%}")


def render_cervical():
    st.header("Cervical risk assessment")
    st.write("Enter the risk-factor values used by the trained cervical models.")
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
    selected_model = st.selectbox("Model", list(model_options))
    values = {}
    columns = list(features.columns)
    for start in range(0, len(columns), 3):
        input_columns = st.columns(3)
        for slot, column in zip(input_columns, columns[start : start + 3]):
            with slot:
                unique_values = set(features[column].dropna().unique())
                if unique_values and unique_values.issubset({0, 1}):
                    values[column] = st.selectbox(
                        column, [0, 1], index=int(round(features[column].median()))
                    )
                else:
                    values[column] = st.number_input(
                        column,
                        value=float(features[column].median()),
                        min_value=0.0,
                        step=0.1,
                    )

    if st.button("Assess cervical risk", type="primary"):
        model = load_joblib_model(model_options[selected_model])
        input_frame = pd.DataFrame([values], columns=columns)
        prediction = int(model.predict(scaler.transform(input_frame))[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(scaler.transform(input_frame))[0][prediction])
        if prediction:
            show_result("Positive biopsy prediction", "The selected model predicts a positive biopsy result.", probability)
        else:
            show_result("Negative biopsy prediction", "The selected model predicts a negative biopsy result.", probability)
        st.caption("This is a research demonstration and is not a medical diagnosis.")


def render_oral():
    st.header("Oral image classification")
    st.write("Upload an oral image for classification with one of the supplied image models.")
    model_options = {"Random forest": "oc_rf_model", "SVM": "oc_svm_model", "Decision tree": "oc_dt_model"}
    selected_model = st.selectbox("Model", list(model_options))
    uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose an image to enable prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=360)
    if st.button("Classify oral image", type="primary"):
        model = load_joblib_model(model_options[selected_model])
        pixels = np.asarray(image.resize((64, 64)), dtype=np.float32)
        prediction_input = pixels.reshape(1, -1)
        prediction = int(model.predict(prediction_input)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(prediction_input)[0][prediction])
        if prediction == 1:
            show_result("Cancer prediction", "The selected model classified this image as cancer.", probability)
        else:
            show_result("Non-cancer prediction", "The selected model classified this image as non-cancer.", probability)
        st.caption("This is a research demonstration and is not a medical diagnosis.")


def render_brain():
    st.header("Brain-tumor image classification")
    st.write("Upload an MRI image for classification with a supplied CNN model.")
    model_options = {"CNN": "bt-cnn.keras", "CNN 2": "bt-cnn2.keras", "MobileNet": "bt-mobilenet.keras"}
    selected_model = st.selectbox("Model", list(model_options))
    uploaded = st.file_uploader("Upload a JPG or PNG MRI image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose an MRI image to enable prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI image", width=360)
    if st.button("Classify brain image", type="primary"):
        with st.spinner("Loading TensorFlow model and classifying image..."):
            model = load_brain_model(model_options[selected_model])
            pixels = np.asarray(image.resize((180, 180)), dtype=np.float32)
            prediction_input = np.expand_dims(pixels, axis=0)
            raw_prediction = np.asarray(model.predict(prediction_input, verbose=0))
            scores = raw_prediction[0]
            prediction = int(np.argmax(scores))
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        probability = float(scores[prediction]) if np.max(scores) <= 1 else None
        show_result(f"Prediction: {classes[prediction]}", "The CNN's highest-scoring class is shown above.", probability)
        st.caption("This is a research demonstration and is not a medical diagnosis.")


def main():
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Machine learning research interface</div>'
        '<h1>Early cancer prediction</h1>'
        '<p>Explore the trained models from this project through one focused local application.</p></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("Prediction area")
    st.sidebar.caption("Choose a workflow to begin.")
    workflow = st.sidebar.radio(
        "Workflow",
        ["Cervical risk", "Oral image", "Brain-tumor image"],
    )
    st.sidebar.divider()
    st.sidebar.caption("For research and education only. Results must not be used for medical decisions.")

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
