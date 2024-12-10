from groq import Groq
from dotenv import load_dotenv
import os

groq_api_key = os.getenv("GROQ_API_KEY")

class GroqTranscriptionService:
    def __init__(self, model: str = "whisper-large-v3-turbo", language: str = "en"):
        self.client = Groq(api_key=groq_api_key)  # Initialize the Groq client
        self.model = model
        self.language = language

    def transcribe_audio(self, file: bytes, filename: str) -> str:
        """
        Transcribes the audio file using Groq's Whisper model.
        :param file: The audio file in bytes.
        :param filename: The name of the file.
        :return: Transcribed text.
        """
        transcription = self.client.audio.transcriptions.create(
            file=(filename, file),  # Provide the file content
            model=self.model,  # Use the specified model
            response_format="json",  # Return JSON
            language=self.language,  # Language of transcription
            temperature=0.0,  # Optional temperature setting
        )
        return transcription.text
