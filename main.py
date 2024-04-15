import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
#st.set_option("deprecation.showfileUploaderEncoding", False)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Retina_model.h5")
    return model

model = load_model()
st.header("Retinopathy for Diabetics")
file = st.file_uploader("Upload an image of your retina", type=["jpg", "png"])

def preprocess_image(image):
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gaussian = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    resized = cv2.resize(gaussian, (224, 224))
    return resized

def prediction(image, model):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image / 255.0
    predict = model.predict(processed_image)
    pre_index = np.argmax(predict, axis=1)
    return pre_index

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = prediction(image, model)
    if predictions == 1:
        st.success('Diabetic Retinopathy Not Detected')
    else:
        st.success('Diabetic Retinopathy Detected')
