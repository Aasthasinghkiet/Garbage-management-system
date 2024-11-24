import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from keras.layers import DepthwiseConv2D
import tensorflow as tf
from groq import Groq

# Define a custom DepthwiseConv2D that ignores the 'groups' argument
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove the 'groups' argument
        super().__init__(*args, **kwargs)

# Load model with custom object and Input layer
model = load_model(
    "keras_model.h5",
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
    compile=False
)

# Initialize the GROQ client
groq_client = Groq(api_key="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")

# Load environment variables
load_dotenv()

def classify_waste(img):
    np.set_printoptions(suppress=True)

    # Prepare the image for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_names = open("labels.txt", "r").readlines()
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def generate_carbon_footprint_info(label):
    messages = [
        {"role": "system", "content": "You are an expert in sustainability and carbon emissions."},
        {"role": "user", "content": f"Describe the approximate carbon emission or carbon footprint generated from {label} waste. Elaborate in 100 words."}
    ]

    try:
        # Call the GROQ API
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True
        )

        # Collect the response
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""
        
        return response_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    st.title("Waste Classifier Sustainability App")

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

                if label == "0 cardboard":
                    st.success("The image is classified as CARDBOARD.")
                    with col4:
                        st.image("sdg goals/12.png", use_container_width=True)
                        st.image("sdg goals/13.png", use_container_width=True)
                    with col5:
                        st.image("sdg goals/13.png", use_container_width=True)
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
                        st.image("sdg goals/12.png", use_container_width=True)
                elif label == "3 metal\n":
                    st.success("The image is classified as METAL.")
                    with col4:
                        st.image("sdg goals/3.png", use_container_width=True)
                        st.image("sdg goals/6 (1).jpg", use_container_width=True)
                else:
                    st.error("The image is not classified as any relevant class.")

            with col3:
                st.info("Information related to Carbon Emissions")
                result = generate_carbon_footprint_info(label)
                st.success(result)
