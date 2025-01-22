import streamlit as st
from fastai.vision.all import *
import pathlib 
import plotly.express as px  
import platform

plt = platform.system()
if plt=='Linux': 
    pathlib.WindowsPath = pathlib.PosixPath
st.title('Transportni klassifikasiya qiluvchi model')

#rasimni joylash 
file = st.file_uploader('Rasimi joylsh', type=['png','svg', 'gif', 'jpeg'])
if file:
    #PIL convert
    img = PILImage.create(file)

    st.image(file)

    # model
    model = load_learner('transport_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Extimollik: {probs[pred_id]*100: .1f}%')

    #ploting
fig = px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)
