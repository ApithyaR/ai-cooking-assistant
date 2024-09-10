import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("model.h5")
    image = Image.open(test_image)
    image = image.resize((299, 299))  
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0 
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max prediction

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project Page
# elif app_mode == "About Project":
#     st.header("About Project")
#     st.subheader("About Dataset")
#     st.text("This dataset contains images of the following food items:")
#     st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
#     st.code("Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy beans, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
#     st.subheader("Content")
#     st.text("This dataset contains three folders:")
#     st.text("1. train (100 images each)")
#     st.text("2. test (10 images each)")
#     st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")
            
            result_index = model_prediction(test_image)
            
            # Read the labels from the 'labels.txt' file
            with open("labels.txt", "r") as f:
                labels = f.read().splitlines()  # Read all lines and remove newline characters
            
            # Ensure result_index is within the bounds of the labels list
            if result_index < len(labels):
                st.success(f"Model is predicting: {labels[result_index]}")
            else:
                st.error("Prediction index is out of range. Please check the model or labels file.")
