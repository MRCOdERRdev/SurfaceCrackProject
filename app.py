import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/MobileNetV2.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

st.title("🧱 Concrete Crack Detection (Local TF 2.7 Demo)")
st.write("Upload an image and the model will classify crack / no crack.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # IMPORTANT: match training size
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction")
    if prediction > 0.5:
        st.success(f"🟢 No Crack Detected (score: {prediction:.2f})")
    else:
        st.error(f"🔴 Crack Detected (score: {prediction:.2f})")
