from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from datetime import date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from .langchain_pydantic_model_generator import (
    create_pydantic_model_from_config,
    print_pydantic_model,
    print_pydantic_instance,
)
import json
from dotenv import load_dotenv
import os
from gtts import gTTS
from io import BytesIO
import base64
import ffmpeg

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Function to generate audio using gTTS
def generate_audio(text: str) -> BytesIO:
    audio_file = BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# def generate_audio_base64(text: str) -> str:
#     audio_file = BytesIO()
#     tts = gTTS(text=text, lang='en')
#     tts.write_to_fp(audio_file)
#     audio_file.seek(0)
#     audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
#     return audio_base64

def generate_audio_base64(text: str, playback_rate: float = 1.0) -> str:
    # Step 1: Generate the audio file using gTTS
    audio_file = BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Step 2: Use ffmpeg to process the audio and adjust speed (in memory)
    input_audio = audio_file.read()
    input_stream = BytesIO(input_audio)
    output_stream = BytesIO()

    # Use ffmpeg to process the input audio and write the result to the output stream
    process = (
        ffmpeg.input("pipe:0")
        .filter("atempo", playback_rate)
        .output("pipe:1", format="mp3")
        .run(input=input_stream.read(), capture_stdout=True, capture_stderr=True)
    )

    # Step 3: Encode the processed audio to base64
    processed_audio = process[0]
    audio_base64 = base64.b64encode(processed_audio).decode("utf-8")

    return audio_base64