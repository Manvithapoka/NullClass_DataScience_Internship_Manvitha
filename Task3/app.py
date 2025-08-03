import streamlit as st
from voice_model import predict_voice_info

st.set_page_config(page_title="Voice Age & Emotion Detection")

st.title("🎤 Age and Emotion Detection from Voice")
st.write("Upload a male voice recording to begin.")

uploaded_file = st.file_uploader("Upload .mp3 or .wav", type=["mp3", "wav"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing...")
    result = predict_voice_info("temp_audio.mp3")
    st.success(result)