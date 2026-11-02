"""
Signature Forgery Detector
--------------------------
Upload a signature image -> the model says Genuine or Forged.

Uses the models saved at the end of final.ipynb:
    mobilenetv2_signature_model.keras   (best: ~94% test accuracy)
    custom_cnn.keras                    (~85% test accuracy)

Run:
    pip install streamlit tensorflow pillow numpy
    streamlit run app.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_FILES = {
    "MobileNetV2 (recommended)": "mobilenetv2_signature_model.keras",
    "Custom CNN": "custom_cnn.keras",
}
THRESHOLD = 0.5  # P(forged) >= 0.5 -> Forged (same as the notebook)


# ---------- preprocessing: identical to the notebook ----------
def load_gray_64x128(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array (64, 128) in [0, 1]."""
    img = img.convert("L").resize((128, 64), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def prep_for_cnn(arr: np.ndarray) -> np.ndarray:
    return arr[np.newaxis, ..., np.newaxis]  # (1, 64, 128, 1)


def prep_for_mobilenet(arr: np.ndarray) -> np.ndarray:
    x = arr[np.newaxis, ..., np.newaxis]                      # (1, 64, 128, 1)
    x = tf.image.resize(x, (128, 128), method="bilinear")     # (1, 128, 128, 1)
    x = tf.image.grayscale_to_rgb(x)                          # (1, 128, 128, 3)
    x = x * 255.0
    return preprocess_input(x).numpy()                        # [-1, 1]


def build_cnn():
    """Same architecture as the notebook's custom CNN."""
    L = keras.layers
    return keras.Sequential([
        L.Input(shape=(64, 128, 1)),
        L.Conv2D(32, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Conv2D(64, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Conv2D(128, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Flatten(),
        L.Dense(128, activation="relu"),
        L.Dropout(0.5),
        L.Dense(1, activation="sigmoid"),
    ], name="custom_cnn")


def build_mobilenet():
    """Same architecture as the notebook's MobileNetV2 classifier."""
    L = keras.layers
    base = keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(128, 128, 3)
    )
    base.trainable = False
    inp = keras.Input(shape=(128, 128, 3))
    x = base(inp, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.5)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="mobilenetv2_signature")


@st.cache_resource
def get_model(path: str):
    try:
        return keras.models.load_model(path)
    except Exception:
        # Keras version mismatch between training and this environment:
        # rebuild the architecture and load only the saved weights.
        model = build_mobilenet() if "mobilenet" in path.lower() else build_cnn()
        model.load_weights(path)
        return model


# ---------- UI ----------
st.set_page_config(page_title="Signature Forgery Detector", page_icon="✍️")
st.title("✍️ Signature Forgery Detector")
st.caption("Upload a signature image and the CNN will tell you if it looks genuine or forged.")

model_name = st.selectbox("Model", list(MODEL_FILES))
model_path = MODEL_FILES[model_name]

if not Path(model_path).exists():
    st.error(
        f"Model file `{model_path}` not found.\n\n"
        "Run the last cells of `final.ipynb` (it saves the models), then put the "
        "`.keras` files in the same folder as this `app.py`."
    )
    st.stop()

uploaded = st.file_uploader("Signature image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded", use_container_width=True)

    arr = load_gray_64x128(image)
    col2.image(arr, caption="What the model sees (128×64 grayscale)", clamp=True,
               use_container_width=True)

    model = get_model(model_path)
    x = prep_for_mobilenet(arr) if "MobileNet" in model_name else prep_for_cnn(arr)
    p_forged = float(model.predict(x, verbose=0).ravel()[0])

    is_forged = p_forged >= THRESHOLD
    confidence = p_forged if is_forged else 1 - p_forged

    if is_forged:
        st.error(f"### 🚨 FORGED  ({confidence:.1%} confidence)")
    else:
        st.success(f"### ✅ GENUINE  ({confidence:.1%} confidence)")

    st.progress(p_forged, text=f"P(forged) = {p_forged:.3f}")

    st.info(
        "⚠️ This model was trained on the CEDAR dataset with an image-level split, so it "
        "works best on CEDAR-style scans (black ink, white background, cropped tightly). "
        "On signatures from new people or phone photos, treat the result as a hint, not proof."
    )