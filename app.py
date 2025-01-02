import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model(r'E:\portfolio\mnist\mnsit_classifier.h5')

st.title('MNIST Classifier')
st.write('Please upload an image of a digit (28X28) to predict its labe') 

uploaded_file = st.file_uploader("Choose an image",type = ["png","jpg","jpeg"])

if uploaded_file is not None:
    img  = Image.open(uploaded_file).convert('L').resize((28,28))
    st.image(img,caption='Image uploaded',use_column_width = True)

    #converting the image to array for model input
    img_ary  = np.array(img)/255.0 
    img_ary = img_ary.reshape(1,28,28,1)

    prediction = model.predict(img_ary)
    prediction_label = np.argmax(prediction)

    #displaying the prediction
    st.write(f"Predicted label: {prediction_label}")
    


