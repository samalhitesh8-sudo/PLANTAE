import streamlit as st
from PIL import Image
import os
from utils.inference import predict_image

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect possible diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Save temporary image
    temp_path = "temp_image.png"
    image.save(temp_path)
    
    # Prediction
    st.write("Predicting...")
    label, confidence = predict_image(temp_path)
    
    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
    
    # Remove temp image
    os.remove(temp_path)
