import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load the trained SVM model
model_path = "model/svm_model.pkl"
svm_model = joblib.load(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (200, 200))  # Resize to match training dimensions
    image = image.reshape(1, -1) / 255  # Flatten and normalize
    return image

# Streamlit UI Configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="centered", page_icon="🧠")

# Title with emoji
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Brain Tumor Detection App 🏥</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #777;'>Upload an MRI scan to check for a brain tumor.</h4>", unsafe_allow_html=True)

# Display logo with updated parameter
# st.image("static/brain_tumor_logo.jpg", width=250, use_container_width=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload MRI Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    #st.image(image, caption="🖼️ Uploaded MRI Scan", width = 400)

    col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns

    with col2:  # Place image in the center column
         st.image(image, caption="🖼️ Uploaded MRI Scan", width=300)  # Set width to 250 for proper display

    


    # Preprocess and predict
    processed_img = preprocess_image(image)
    prediction = svm_model.predict(processed_img)[0]

    # Result mapping
    labels = {0: "🟢 No Tumor Detected", 1: "🔴 Brain Tumor Detected"}
    result = labels[prediction]

    # Display results with styled text
    st.markdown(f"<h3 style='text-align: center; color: #333;'>🩺 **Diagnosis Result:** {result} </h3>", unsafe_allow_html=True)

    # Additional Recommendations if Tumor is Detected
    if prediction == 1:
        st.markdown("---")
        st.markdown("<h3 style='color: #FF5733;'>🚑 Recommended Steps & Lifestyle Changes:</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🥦 Diet & Nutrition:")
            st.markdown("✅ Increase **antioxidant-rich** foods (berries, leafy greens).")
            st.markdown("✅ Eat more **omega-3 fatty acids** (salmon, walnuts).")
            st.markdown("✅ Avoid **processed foods & excessive sugar**.")

        with col2:
            st.markdown("### 🏋️‍♂️ Healthy Lifestyle:")
            st.markdown("✅ Engage in **light exercise & yoga**.")
            st.markdown("✅ **Hydrate well** – drink plenty of water.")
            st.markdown("✅ Manage **stress & get enough sleep**.")

        
        st.markdown("---")

    # Encouraging final message
    st.markdown("### ❤️ Stay Healthy & Consult a Doctor for Professional Advice! 🩺")
