import streamlit as st
import numpy as np
import tensorflow as tf
import os
import cv2
import gdown

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1yhTjhQJ9KpVhUUr7Ipqhn8TQ9grEXJMl"
MODEL_PATH = "waste_model.h5"

# -----------------------------
# DOWNLOAD MODEL (RUN ONCE)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Waste Classifier", layout="centered")

st.title("♻️ Waste Classification App")
st.markdown("Upload an image to classify waste type using AI")

st.sidebar.title("About")
st.sidebar.info("AI Waste Classification Project")

st.sidebar.markdown("""
<span style='background-color: #4CAF50; color: white; padding: 5px 15px;
border-radius: 15px; font-weight: bold;'>
Developed by: Susmita Ghosh
</span>
""", unsafe_allow_html=True)

st.sidebar.write("Connect with me on LinkedIn")

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_resized)

    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Result
    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted: {class_names[class_idx]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # Probabilities
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")
