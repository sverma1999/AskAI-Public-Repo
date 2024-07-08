# import assemblyai as aai
from elevenlabs import generate, stream

# from openai import OpenAI
from dotenv import load_dotenv
import os

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
import asyncio
import shutil

import subprocess
import requests
import time

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


load_dotenv()


# NeMo Guardrails ==================================================IMPORTANT==================================================

from nemoguardrails import RailsConfig, LLMRails


# define user express greeting
#   "hello"
#   "hi"
#   "hey"
#   "my name"

# define bot express greeting
#   "Hello there, I am Eliza! How are you feeling today?"

# define bot personal greeting
#     "Hello $username, nice to see you again! How are you feeling today?"

# define flow hello
#     user express greeting
#     if $username
#         bot personal greeting
#     else
#         bot express greeting

new_colang_content = """
define user non_medical_chat
    "What's the weather like?"
    "Tell me a joke."
    "What's your favorite color?"
    "Will I die alone?"

define bot stay_on_topic
    "I can only help with health-related questions. Please let me know if you have any health concerns."

define flow non_medical
    user non_medical_chat
    bot stay_on_topic
"""

yaml_content = """
models:
- type: main
  engine: openai
  model: babbage-002
"""

# config = RailsConfig.from_content(
#     yaml_content=yaml_content, colang_content=new_colang_content
# )
# rails = LLMRails(config=config)


# ==================================================Start of Chatbot==================================================


class Transcriber:

    def __init__(self):
        self.reset()

    # Reset the transcript list by emptying it.
    def reset(self):
        self.transcript_chunks = []

    # Add the chunk of text to the full transcript list.
    def add_chunk(self, chunk):
        self.transcript_chunks.append(chunk)

    # Convert transcript list
    def get_full_transcript(self):
        return " ".join(self.transcript_chunks)


transcriber = Transcriber()


async def get_transcript(callback):
    # this event will signal completion of the transcription.
    transcription_complete = asyncio.Event()

    try:
        # Client config setup which is logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        deepgram_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):

            # prepare the sentence for the transcription
            sentence = result.channel.alternatives[0].transcript

            # If speech is not finished, then add new chunk of sentence to the transcription list.
            if not result.speech_final:
                transcriber.add_chunk(sentence)
            else:
                # if speech is still continueing, then add the sentence chunk to the transcription.
                transcriber.add_chunk(sentence)

                # This is the final part of the current sentence
                full_sentence = transcriber.get_full_transcript()

                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcriber.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await deepgram_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(deepgram_connection.send)
        microphone.start()

        # Wait for the transcription to complete instead of looping indefinitely
        await transcription_complete.wait()

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await deepgram_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


# We can use any Language Model from the langchain library to process the text and generate responses.
class LanguageModel:

    def __init__(self):
        # self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(
        #     temperature=0,
        #     model_name="gpt-4-0125-preview",
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-0125",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # The memory object is used to store the chat history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Load the system prompt from a file, so LLM dont go off track.
        with open("system_prompt_bot2.txt", "r") as file:
            system_prompt = file.read().strip()

        # Create a prompt template for the chat by combining the system prompt, chat history and human text.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

        # Create a conversation chain object to handle the chat.
        self.conversation = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory
        )

    def process(self, text):
        # Add user message to memory
        self.memory.chat_memory.add_user_message(text)

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response["text"])

        # Print the response and the time taken to generate it.
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response["text"]

    def is_relevant(self, text):
        healthcare_keywords = [
            "appointment",
            "doctor",
            "health",
            "prescription",
            "hospital",
            "clinic",
            "medicine",
            "nurse",
            "care",
            "emergency",
            "pain",
            "fever",
            "cold",
            "flu",
            "sick",
            "injury",
            "headache",
            "stomach",
            "cough",
            "checkup",
            "test",
            "allergy",
            "symptoms",
            "treatment",
            "vaccine",
            "insurance",
            "coverage",
            "depression",
            "anxiety",
            "therapy",
            "counseling",
            "wellness",
            "illness",
            "recovery",
            "surgery",
            "referral",
            "specialist",
            "screening",
            "diagnosis",
            "chronic",
            "acute",
            "medication",
            "physical",
            "consultation",
            "policy",
            "health plan",
            "condition",
            "symptom management",
            "urgent care",
            "wound",
            "fracture",
            "bacteria",
            "virus",
            "infection",
            "transmission",
            "contagious",
            "spread",
            "vaccination",
            "immunization",
            "disease",
            "disorder",
            "vomiting",
            "diarrhea",
            "rash",
            "itch",
            "burn",
            "cut",
            "bruise",
            "inflammation",
            "swelling",
            "bleeding",
            "painful",
            "sore",
            "itchy",
            "red",
            "yellow",
        ]
        return any(keyword in text.lower() for keyword in healthcare_keywords)


class TextToSpeech:

    def __init__(self):
        # ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
        # aai.settings.api_key = ASSEMBLYAI_API_KEY
        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_api_key = ELEVENLABS_API_KEY

        print("Initializing AI Assistant...")
        # print("self.openai_client: ", self.openai_client)
        print("self.elevenlabs_api_key: ", self.elevenlabs_api_key)

        # self.transcriber = None

        # # Prompt
        # self.full_transcript = [
        #     {
        #         "role": "system",
        #         "content": "You are a doctor at a walk-in health clinic in Canada. Be resourceful and efficient.",
        #     },
        # ]

    # This function generates audio from the text
    def generate_audio(self, text):
        # "text" is the text that came as the response from OpenAI and needs to be converted to audio

        # First we add the AI response to the full transcript
        # self.full_transcript.append({"role": "assistant", "content": text})
        # print(f"\nAI Doc: {text}")

        # Then we generate audio from the text using ElevenLabs API
        audio_stream = generate(
            api_key=self.elevenlabs_api_key, text=text, voice="Rachel", stream=True
        )

        stream(audio_stream)


# Conversaiton Manager
class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModel()

        self.initial_greeting_sent = False

        # NeMo Guardrails
        self.rails = LLMRails(
            config=RailsConfig.from_content(
                yaml_content=yaml_content, colang_content=new_colang_content
            )
        )

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:

            if not self.initial_greeting_sent:
                tts = TextToSpeech()
                greeting = "Hello there, I am Eliza! How are you feeling today?"
                tts.generate_audio(greeting)
                self.initial_greeting_sent = True
                continue

            await get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            # llm_response = self.llm.process(self.transcription_response)

            if self.llm.is_relevant(self.transcription_response):
                llm_response = self.llm.process(self.transcription_response)
                print(f"Inside relevant: {llm_response}")
            else:
                # Use Guardrails for non-relevant messages with predefined responses
                messages = [{"role": "user", "content": self.transcription_response}]
                guardrail_response = await self.rails.generate_async(messages=messages)
                llm_response = guardrail_response["content"]
                print(f"Inside Non-relevant: {llm_response}")

            tts = TextToSpeech()
            tts.generate_audio(llm_response)
            # tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
