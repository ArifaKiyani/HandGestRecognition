import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((target_width, target_height))  # Resize as per model's input size
    image = np.array(image) / 255.0  # Normalize if needed
    image = image.reshape(1, target_width, target_height, 1)  # Reshape to match model input
    return image

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = preprocess_image(img)
    prediction = model.predict(img)
    st.write(f"Prediction: {np.argmax(prediction)}")

 
