import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the saved model
model = load_model('handgestpred_cnn.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to match the input shape expected by the model
    image = image.resize((64, 64))  # Adjust based on your model's input size
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app interface
st.title("Hand Gesture Recognition App")

# File uploader to upload an image for prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index
    
    # Display the prediction
    st.write(f"Predicted Hand Gesture: {predicted_class}")

