import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('model.h5')

st.title("고양이 vs 개 분류기 🐱🐶")
uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).resize((100,100))
    st.image(img, caption='업로드한 이미지', use_container_width=True)
    
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1, 100, 100, 3)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.write("이 이미지는 **🐶 개** 입니다.")
    else:
        st.write("이 이미지는 **🐱 고양이** 입니다.")
