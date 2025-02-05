import os
import logging
from deepgram.utils import verboselogs
from deepgram import (
    DeepgramClient,
    SpeakOptions,
    PrerecordedOptions,
    FileSource,
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

    async def speech_to_text(self, audio_file, language: str = "en") -> str:
        """
        Convert an uploaded audio file to text using Deepgram's STT API.

        Args:
            audio_file (UploadFile): The uploaded audio file from FastAPI.
            language (str): The language code for transcription (default is "en" for English).

        Returns:
            str: The transcribed text from the audio file.
        """
        try:
            # Read audio file as binary
            file_content = await audio_file.read()

            # Prepare payload following Deepgram's example
            payload = {"buffer": file_content}

            # Configure Deepgram options
            options = PrerecordedOptions(
                model="nova-2-general",
                smart_format=True,
                detect_language=True,  # Enable automatic language detection
            )

            # Remove await since Deepgram's transcribe_file returns a synchronous response
            response = self.client.listen.rest.v("1").transcribe_file(payload, options)

            # Get the transcript from the response
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")
