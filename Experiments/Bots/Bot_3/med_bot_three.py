import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import base64


# Initialize OpenAI API
def setup_openai(api_key):
    return openai.OpenAI(api_key=api_key)


# FUnctuo to transcribe audio to text


def trabscribe_audio(client, audio_file_path):

    with open(audio_file_path, "rb") as file:
        audio_transcribed_data = client.audio.transcriptions.create(
            model="whisper-1", file=file
        )
        return audio_transcribed_data.text


# Take response from OpenAI


def get_openai_response(client, text):
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", messages=messages
    )
    return response.choices[0].message.content


# Function to convert text to audio
def text_to_audio(client, text, audio_file_path):
    response = client.audio.speech.create(model="tts-1", voice="echo", input=text)
    response.stream_to_file(audio_file_path)


# text cards functions
def text_card(text, title="Response"):
    card_html = f"""
    <style>
        .card {{
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            transition: 0.3s;
            border-radius: 5px;
            padding: 20px;
        }}
        .card:hover {{
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
        }}
        .container {{
            padding: 2px 16px;
        }}
    </style>
    <div class="card">
        <div class="container">
            <h4><b>{title}</b></h4>
            <p>{text}</p>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


# Auto play audio
def auto_play_audio(audio_file):
    with open(audio_file, "rb") as file:
        audio_bytes = file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio src="data:audio/wav;base64,{base64_audio}" controls autoplay>'
    st.markdown(audio_html, unsafe_allow_html=True)


def main():
    st.sidebar.title("OpenAI API Key Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    st.title("Med Bot 3")
    st.write("Welcome to Med Bot 3! Please record your query below.")

    # Chekc if API key is provided
    if api_key:
        client = setup_openai(api_key)
        recorded_audio = audio_recorder()

        # Check if audio is recorded
        if recorded_audio:
            audio_file = "audio.wav"
            with open(audio_file, "wb") as file:
                file.write(recorded_audio)

            # st.write("Audio recorded successfully!")
            # st.audio(recorded_audio, format="audio/wav")

            # Transcribe audio to text
            text = trabscribe_audio(client, audio_file)
            # st.write("Transcribed Text: ", text)
            text_card(text, "Transcribed Text")

            ai_response = get_openai_response(client, text)
            response_audio_file = "response_audio.wav"
            text_to_audio(client, ai_response, response_audio_file)
            # st.audio(response_audio_file)
            auto_play_audio(response_audio_file)

            # st.write("AI Response: ", ai_response)
            text_card(ai_response, "AI Response")


if __name__ == "__main__":
    main()
