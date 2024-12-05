import os
import logging
from deepgram.utils import verboselogs
from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import base64
import os
from io import BytesIO

class DeepgramTTS:
    def __init__(self, api_key: str):
        """
        Initialize the DeepgramTTS client with the provided API key.
        """
        self.client = DeepgramClient(api_key)

    def text_to_speech_base64(self, text: str, model: str = "aura-asteria-en") -> str:
        """
        Convert text to speech and return the audio as a Base64-encoded string.

        Args:
            text (str): The text to be converted to speech.
            model (str): The Deepgram model to be used for TTS. Default is "aura-asteria-en".

        Returns:
            str: The Base64-encoded audio content.
        """
        try:
            # Configure TTS options
            options = SpeakOptions(model=model)

            # Generate the audio file using Deepgram TTS
            audio_file_name = "temp_audio.mp3"
            self.client.speak.rest.v("1").save(audio_file_name, {"text": text}, options)

            # Read the audio file content
            with open(audio_file_name, "rb") as audio_file:
                audio_content = audio_file.read()

            # Encode the audio content to Base64
            base64_audio = base64.b64encode(audio_content).decode("utf-8")

            # Clean up temporary file
            os.remove(audio_file_name)

            return base64_audio
        except Exception as e:
            raise RuntimeError(f"Failed to convert text to speech: {e}")

    def text_to_speech(self, text: str, model: str = "aura-asteria-en") -> BytesIO:
        """
        Convert text to speech and return the audio in BytesIO.

        Args:
            text (str): The text to be converted to speech.
            model (str): The Deepgram model to be used for TTS. Default is "aura-asteria-en".

        Returns:
            BytesIO: The audio content in BytesIO format.
        """
        try:
            # Configure TTS options
            options = SpeakOptions(model=model)

            # Generate the audio file using Deepgram TTS
            audio_file_name = "temp_audio.mp3"
            self.client.speak.rest.v("1").save(audio_file_name, {"text": text}, options)

            # Read the audio file content into BytesIO
            audio_buffer = BytesIO()
            with open(audio_file_name, "rb") as audio_file:
                audio_buffer.write(audio_file.read())

            # Ensure the buffer pointer is at the beginning
            audio_buffer.seek(0)

            # Clean up temporary file
            os.remove(audio_file_name)

            return audio_buffer

        except Exception as e:
            raise RuntimeError(f"Failed to convert text to speech: {e}")
