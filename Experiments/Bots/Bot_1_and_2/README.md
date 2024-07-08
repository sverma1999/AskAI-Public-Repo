# These two demos showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.


### Prerequisites:
- You will need to sign up for a Deepgram account and get an API key.
- You will need to sign up for an OpenAI account and get an API key.

### Setup:
- Create virtual environment:
```
conda create -n bot_1_and_2 python=3.8 -y
conda activate bot_1_and_2
```

- Create .env file in the Bot_1_and_2 directory.
- Add the following to the .env file:
```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```


## Quick bot-one demo using Deepgram and OpenAI

This demo is set up to use [Deepgram](www.deepgram.com) for the audio service and [OpenAI](https://openai.com) the LLM.

This demo utilizes streaming for sst and tts to speed things up. sst means that the bot can start processing the audio before the user is done speaking. tts means that the bot can start speaking before it has finished processing the response.


### Run the following command to start the bot:
```
cd Bot_1_and_2
pip install -r requirements.txt
python med_bot_one.py
```

## Quick bot-two demo using Elevenlabs and OpenAI

This demo is set up to use [Elevenlabs](www.elevenlabs.com) for the audio service and [OpenAI](https://openai.com) the LLM.


### Run the following command to start the bot:
```
cd Bot_1_and_2
pip install -r requirements.txt
python med_bot_two.py
```

