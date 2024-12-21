from dotenv import load_dotenv
import os
from gtts import gTTS
from io import BytesIO
import base64
import ffmpeg
import tempfile
import pycountry
from google.cloud import texttospeech

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

# def generate_audio_base64(text: str, playback_rate: float = 1.0) -> str:
#     # Step 1: Generate the audio file using gTTS
#     audio_file = BytesIO()
#     tts = gTTS(text=text, lang='en')
#     tts.write_to_fp(audio_file)
#     audio_file.seek(0)

#     # Step 2: Use ffmpeg to process the audio and adjust speed (in memory)
#     input_audio = audio_file.read()
#     input_stream = BytesIO(input_audio)
#     output_stream = BytesIO()

#     # Use ffmpeg to process the input audio and write the result to the output stream
#     process = (
#         ffmpeg.input("pipe:0")
#         .filter("atempo", playback_rate)
#         .output("pipe:1", format="mp3")
#         .run(input=input_stream.read(), capture_stdout=True, capture_stderr=True)
#     )

#     # Step 3: Encode the processed audio to base64
#     processed_audio = process[0]
#     audio_base64 = base64.b64encode(processed_audio).decode("utf-8")

#     return audio_base64

def generate_audio_base64(text: str, playback_rate: float = 1.0,) -> str:
    try:
        # Step 1: Generate the audio file using gTTS
        audio_file = BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(audio_file)
        audio_file.seek(0)

        # Step 2: Use ffmpeg to process the audio and adjust speed (in memory)
        input_audio = audio_file.read()
        input_stream = BytesIO(input_audio)
        output_stream = BytesIO()

        try:
            # Use ffmpeg to process the input audio and write the result to the output stream
            process = (
                ffmpeg.input("pipe:0")
                .filter("atempo", playback_rate)
                .output("pipe:1", format="wav")
                .run(input=input_stream.read(), capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            # Capture FFmpeg-specific errors
            error_message = e.stderr.decode("utf-8") if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg error: {error_message}")

        # Step 3: Encode the processed audio to base64
        processed_audio = process[0]
        audio_base64 = base64.b64encode(processed_audio).decode("utf-8")

        return audio_base64

    except Exception as e:
        # General error handling
        print(f"An error occurred: {e}")
        raise  # Reraise the exception after logging

def generate_audio_base64_file(text: str, playback_rate: float = 1.0,  language: str = 'english') -> str:
    try:
        # Step 1: Convert language name to code dynamically
        language_code = get_language_code(language)

        # Step 1: Generate gTTS output to a temporary file
        temp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_input.close()  # Close to avoid permission errors on Windows
            temp_output.close()

            # Generate gTTS audio and save to temp_input file
            gTTS(text=text, lang=language_code).save(temp_input.name)

            # Step 2: Adjust playback rate using FFmpeg
            if playback_rate != 1.0:
                (
                    ffmpeg.input(temp_input.name)
                    .filter("atempo", str(playback_rate))
                    .output(temp_output.name, format="wav")
                    .overwrite_output()
                    .run(quiet=True)
                )
                output_file = temp_output.name
            else:
                output_file = temp_input.name  # Use input directly if no speed adjustment

            # Step 3: Read the processed WAV file and encode to Base64
            with open(output_file, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            return audio_base64

        finally:
            # Step 4: Cleanup temporary files
            os.unlink(temp_input.name)  # Delete temp input file
            os.unlink(temp_output.name)  # Delete temp output file

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def get_language_code(language_name: str) -> str:
    """
    Convert a language name to its ISO 639-1 language code or gTTS-specific code.
    Supports variants for Chinese.
    """
    try:
        language_name_lower = language_name.lower()
        
        # Special handling for Chinese
        if "chinese" in language_name_lower:
            if "simplified" in language_name_lower or "cn" in language_name_lower:
                return "zh-cn"  # Simplified Chinese
            elif "traditional" in language_name_lower or "tw" in language_name_lower:
                return "zh-tw"  # Traditional Chinese
            else:
                return "zh"  # Default to zh if no variant specified
        
        # General lookup for other languages
        language = pycountry.languages.lookup(language_name)
        return language.alpha_2  # Return ISO 639-1 code
    except LookupError:
        raise ValueError(f"Unsupported language: {language_name}. Please provide a valid language name.")
    
def generate_audio_base64_file_gg(text: str, playback_rate: float = 1.0, language: str = 'english') -> str:
    try:
        # Step 1: Convert language name to Google Cloud language code dynamically
        language_code = get_language_code_gg(language)

        # Step 2: Generate Google Cloud TTS output to a temporary file
        temp_input = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_input.close()  # Close to avoid permission errors on Windows
            temp_output.close()

            # Create a Text-to-Speech client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request with Standard voice (cheapest)
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
                name=f"{language_code}-Standard-C"  # Use standard voice
            )

            # Select the audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Write the response to the temporary input file
            with open(temp_input.name, "wb") as f:
                f.write(response.audio_content)

            # Step 3: Adjust playback rate using FFmpeg if necessary
            if playback_rate != 1.0:
                (
                    ffmpeg.input(temp_input.name)
                    .filter("atempo", str(playback_rate))
                    .output(temp_output.name, format="wav")
                    .overwrite_output()
                    .run(quiet=True)
                )
                output_file = temp_output.name
            else:
                output_file = temp_input.name  # Use input directly if no speed adjustment

            # Step 4: Read the processed WAV file and encode to Base64
            with open(output_file, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            return audio_base64

        finally:
            # Step 5: Cleanup temporary files
            os.unlink(temp_input.name)  # Delete temp input file
            os.unlink(temp_output.name)  # Delete temp output file

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def get_language_code_gg(language_name: str) -> str:
    """
    Convert a language name to its Google Cloud language code.
    Special handling for variants such as Simplified and Traditional Chinese.
    """
    language_name_lower = language_name.lower()

    # Special handling for Chinese
    if "chinese" in language_name_lower:
        if "simplified" in language_name_lower or "cn" in language_name_lower:
            return "zh-CN"  # Simplified Chinese
        elif "traditional" in language_name_lower or "tw" in language_name_lower:
            return "zh-TW"  # Traditional Chinese
        else:
            return "zh"  # Default to Chinese if no variant specified

    # Mapping common language names to Google Cloud language codes
    language_map = {
        "english": "en-US",
        "spanish": "es-ES",
        "french": "fr-FR",
        "german": "de-DE",
        "japanese": "ja-JP",
        "korean": "ko-KR",
        "dutch": "nl-NL",
        "chinese": "cmn-CN",
        "hindi": "hi-IN",
    }

    if language_name_lower in language_map:
        return language_map[language_name_lower]

    raise ValueError(f"Unsupported language: {language_name}. Please provide a valid language name.")

async def generate_audio_base64_stream(
    llm_stream, playback_rate: float = 1.0, language: str = "english"
):
    """
    Streams LLM response into Google Cloud TTS and encodes audio as Base64.
    Args:
        llm_stream: Async generator yielding LLM response chunks.
        playback_rate: Playback speed adjustment (default is 1.0).
        language: Language for TTS synthesis.
    Returns:
        Async generator yielding Base64-encoded audio chunks.
    """
    try:
        # Step 1: Convert language name to Google Cloud language code
        language_code = get_language_code_gg(language)

        # Create a Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Configure TTS parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            name=f"{language_code}-Standard-C",  # Use standard voice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        async for llm_chunk in llm_stream:
            text_chunk = llm_chunk.get("text", "").strip()
            if not text_chunk:
                continue  # Skip empty chunks

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text_chunk)

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Step 2: Adjust playback rate if necessary
            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                temp_output.close()

                # Write the audio content to a temp file
                with open(temp_output.name, "wb") as f:
                    f.write(response.audio_content)

                if playback_rate != 1.0:
                    adjusted_output = tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    )
                    adjusted_output.close()

                    # Adjust playback rate with FFmpeg
                    (
                        ffmpeg.input(temp_output.name)
                        .filter("atempo", str(playback_rate))
                        .output(adjusted_output.name, format="wav")
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    os.unlink(temp_output.name)  # Delete the original temp file
                    temp_output.name = adjusted_output.name

                # Step 3: Read the processed audio and encode to Base64
                with open(temp_output.name, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

                yield audio_base64

            finally:
                # Cleanup temporary files
                if os.path.exists(temp_output.name):
                    os.unlink(temp_output.name)

    except Exception as e:
        print(f"An error occurred during streaming TTS: {e}")
        raise