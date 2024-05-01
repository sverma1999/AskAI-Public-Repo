from dotenv import load_dotenv, find_dotenv
import os
import requests

# You can access the audio with IPython.display for example
from IPython.display import Audio

# Streamlit for the frontend
import streamlit as st


# from transformers import pipeline
# from langchain import PromtTemplate, HuggingFaceHub, LLMChain

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")


# # llm
def generate_response(prompt):

    API_URL = (
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payload = {
        "CONTEXT:": """
        You are an AI Medical Chatbot Assistant, provide comprehensive and informative responses to your inquiries.
        """,
        "inputs": prompt,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# In case you want to use OpenAI API as your llm instead of mistral=====================================>>>>>>>>>>>>

# # Keep in mind, you would need to add OpenAI API key to the .env file
# from langchain import PromptTemplate, LLMChain, OpenAI

# def get_diagnosis(question):
#     # Create a prompt template
#     template = """
#     You are a nurse at a hospotal.
#     You can generate a diagnosis and treatment for a patient based on the given information, keep your response more than 60 words, but less than 100 words.
#     CONTEXT: {question}
#     RESPONSE:
#     """
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     response_llm = LLMChain(
#         llm=OpenAI(model_name="gpt-3.5-turbo", tempreture=2),
#         prompt=prompt,
#         verbose=True,
#     )

#     response = response_llm.predict(question=question)
#     # print(response)
#     return response


# text to speech
def text_to_speech(message):

    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payload = {
        "inputs": message,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open("output_1.flac", "wb") as f:
        f.write(response.content)


# Main function
def main():
    st.set_page_config(page_title="AskAI", page_icon="🧠")
    st.header("AskAI - Health Diagnosis Chatbot")

    enter_question = st.text_input("Enter your question here:")

    if enter_question != "":
        output = generate_response(enter_question)
        print(output)
        generated_text = output[0]["generated_text"]
        text_after_first_delimiter = generated_text.split("\n\n", 1)[1]
        print()
        print(text_after_first_delimiter)
        # st.write(text_after_first_delimiter)
        text_to_speech(text_after_first_delimiter)
        with st.expander("Your question"):
            st.write(enter_question)

        with st.expander("What AskAI thinks:"):
            st.write(text_after_first_delimiter)

        st.audio("output_1.flac")


if __name__ == "__main__":
    main()
