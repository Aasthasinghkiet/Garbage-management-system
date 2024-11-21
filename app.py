import streamlit as st
from dotenv import load_dotenv
import os
import requests  # Used for interacting with the Gemini AI API
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf

# Define a custom DepthwiseConv2D that ignores the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove the 'groups' argument
        super().__init__(*args, **kwargs)

# Load model with custom object
model = load_model("keras_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

# Load the Gemini API key
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

def classify_waste(img):
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_names = open("labels.txt", "r").readlines()
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def generate_carbon_footprint_info(label):
    # Adjust the label extraction based on your specific labeling format
    label = label.split(' ')[1]  # Remove the newline and space
    
    # Prepare the message for the Gemini API (or any custom prompt)
    payload = {
        "prompt": f"Describe the approximate carbon emission or carbon footprint generated from {label} waste. Elaborate in 100 words.",
        "max_tokens": 600,
        "temperature": 0.7,
        "top_p": 1,
        "n": 1
    }
    
    headers = {
        "Authorization": f"Bearer {gemini_api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Make a POST request to the Gemini AI API
        response = requests.post("https://api.gemini.ai/v1/completions", json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['text'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app setup
st.set_page_config(layout='wide')
st.title("Waste Classifier Sustainability App")

# File uploader
input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_container_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1, 1])
            
            # Display images based on classification
            if label == "0 cardboard\n":
                st.success("The image is classified as CARDBOARD.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "1 plastic\n":
                st.success("The image is classified as PLASTIC.")
                with col4:
                    st.image("sdg goals/6 (1).jpg", use_container_width=True)
                    st.image("sdg goals/12.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "2 glass\n":
                st.success("The image is classified as GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
            elif label == "3 metal\n":
                st.success("The image is classified as METAL.")
                with col4:
                    st.image("sdg goals/3.png", use_container_width=True)
                    st.image("sdg goals/6 (1).jpg", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/14.png", use_container_width=True)
            else:
                st.error("The image is not classified as any relevant class.")

        # Generate and display the carbon footprint info
        with col3:
            st.info("Information related to Carbon Emissions")
            result = generate_carbon_footprint_info(label)
            st.success(result)
