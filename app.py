import streamlit as st
from dotenv import load_dotenv
import os
import openai
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

# Load the OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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
    
    # Prepare the message for the new ChatCompletion API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Describe the approximate carbon emission or carbon footprint generated from {label} waste. I just need an approximate number to create awareness. Elaborate in 100 words."}
    ]
    
    try:
        # Use the new ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if needed
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        # Return the assistant's message content
        return response['choices'][0]['message']['content']
    
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
                    st.image("12.png", use_container_width=True)
                    st.image("13.png", use_container_width=True)
                with col5:
                    st.image("14.png", use_container_width=True)
                    st.image("15.png", use_container_width=True)
            elif label == "1 plastic\n":
                st.success("The image is classified as PLASTIC.")
                with col4:
                    st.image("6 (1).jpg", use_container_width=True)
                    st.image("12.png", use_container_width=True)
                with col5:
                    st.image("14.png", use_container_width=True)
                    st.image("15.png", use_container_width=True)
            elif label == "2 glass\n":
                st.success("The image is classified as GLASS.")
                with col4:
                    st.image("12.png", use_container_width=True)
                with col5:
                    st.image("14.png", use_container_width=True)
            elif label == "3 metal\n":
                st.success("The image is classified as METAL.")
                with col4:
                    st.image("3.png", use_container_width=True)
                    st.image("6 (1).jpg", use_container_width=True)
                with col5:
                    st.image("12.png", use_container_width=True)
                    st.image("14.png", use_container_width=True)
            else:
                st.error("The image is not classified as any relevant class.")

        # Generate and display the carbon footprint info
        with col3:
            st.info("Information related to Carbon Emissions")
            result = generate_carbon_footprint_info(label)
            st.success(result)
