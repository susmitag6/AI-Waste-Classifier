import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import gdown

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1_qg4fZXCiIsDVc3yQW8aJj1SOtVZXnmi"
MODEL_PATH = "waste_classifier.h5"

# -----------------------------
# DOWNLOAD MODEL (RUN ONCE)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/file/d/1_qg4fZXCiIsDVc3yQW8aJj1SOtVZXnmi/view?usp=drive_link={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


# Load model
# model = tf.keras.models.load_model("../models/waste_classifier.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (IMPORTANT)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Page config
st.set_page_config(page_title="Waste Classifier", layout="centered")

# Title
st.title("♻️ Waste Classification App")
st.markdown("Upload an image to classify waste type using AI")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
st.sidebar.title("About")
st.sidebar.info("AI Waste Classification Project")  
  
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show original image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    col1, col2 = st.columns(2)  #Add columns layout
    # Prediction #Add loading spinner
    with st.spinner("Analyzing image..."):
         prediction = model.predict(img_resized)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    st.success(f"♻️ {class_names[class_idx].upper()}")
    # Result section
    st.subheader("🔍 Prediction Result")

    st.success(f"Predicted: {class_names[class_idx]}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    # Show probability for all classes
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
    
