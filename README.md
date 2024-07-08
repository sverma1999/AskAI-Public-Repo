# AskAI-Health-Diagnosis-Companion
Utilizing advanced machine learning techniques including NLP, CNN, and large language and vision models, to offer immediate assistance and personalized healthcare guidance, with the goal of addressing accessibility and wait time challenges in the healthcare sector.

```diff
! In Progress
```


```diff
This repository is intended to showcase sample code of the project for public presentation.
The main repository is private (for confidentiality), but feel free to request the project architecture.
```




# Go Inside Experiments/Bots

## Bot_1_and_3
This folder contains two bots to have live speech to speech conversation with human.
Bot 1 uses Deepgram for speech to text and text to speech conversion.
Bot 2 uses Deepgram for speech to text and ElevenLabs for text to speech conversion.

## Bot_3
This bot uses streamlit for the user interface and transcribes the speech to text using audio_recorder and base64 libraries.


## NemoGuardrails

practice.ipynb contains the code for the NemoGuardrails trails.

### Environment Setup

```bash
conda create -n nemoGuardrails python=3.10 -y
conda activate nemoGuardrails
pip install -r requirements.txt
```


