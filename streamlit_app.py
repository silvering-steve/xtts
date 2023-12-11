import numpy as np
import torch
from TTS.api import TTS

import os
import tempfile

import streamlit as st


@st.cache_resource
def load_model():
    xtts = TTS(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    ).to("cuda")

    return xtts


def on_click():
    st.session_state.enable = True


def main(model):
    # Set the main page
    st.header("Multi Language Text To Speech", divider="gray")

    if 'enable' not in st.session_state:
        st.session_state.enable = False

    text = st.text_area(
        label="Input Text",
        placeholder="Input text here..."
    )

    col1_1, col1_2 = st.columns(2)

    with col1_1:
        lang = st.selectbox(
            label="Language",
            options=("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi")
        )

    col2_1, col2_2 = st.columns(2)

    file = st.file_uploader(
        label="Audio Reference",
    )
    if file:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
            temp.write(file.getvalue())
            temp.seek(0)

            if st.session_state.enable:
                with col2_1:
                    st.write("Reference")
                    st.audio(temp.name)

            if st.button("Convert", type="primary", on_click=on_click):
                with st.spinner():
                    model.tts_to_file(
                        text=text,
                        language=lang,
                        speaker_wav=temp.name,
                        file_path="temp.wav"
                    )

                    with col2_2:
                        st.write("Output")
                        st.audio("temp.wav")


if __name__ == '__main__':
    st.set_page_config(
        page_title="XTTS",
    )

    main(load_model())

