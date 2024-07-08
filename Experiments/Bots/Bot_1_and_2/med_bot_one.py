import asyncio
from dotenv import load_dotenv

import shutil

import subprocess
import requests
import time
import os

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()


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
        with open("system_prompt.txt", "r") as file:
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


# This class will be used to convert the text to speech
class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    # MODEL_NAME = "aura-helios-en"  # Example model name, change as needed
    MODEL_NAME = "aura-asteria-en"

    # Check if the library is installed
    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    # This function will stream the audio to the ffplay player
    def speak(self, text):
        # Check if ffplay is installed
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        # Set the Deepgram URL with the desired model
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        # these are the commands to play the audio
        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]

        # Send the request to Deepgram to get the audio. We are using the subprocess library to pipe the audio to the ffplay player.
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        # Send the request to Deepgram and stream the audio to the player
        with requests.post(
            DEEPGRAM_URL, stream=True, headers=headers, json=payload
        ) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if (
                        first_byte_time is None
                    ):  # Check if this is the first chunk received
                        first_byte_time = (
                            time.time()
                        )  # Record the time when the first byte is received
                        ttfb = int(
                            (first_byte_time - start_time) * 1000
                        )  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        # Close the player process after the audio is finished
        if player_process.stdin:
            player_process.stdin.close()
        # Wait for the player process to finish
        player_process.wait()


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModel()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break

            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
    # asyncio.run(get_transcript())
