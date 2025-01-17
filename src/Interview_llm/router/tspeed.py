from datetime import date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
import json
from dotenv import load_dotenv
import os
from gtts import gTTS
from io import BytesIO
import base64
import ffmpeg
import tempfile
import pycountry
import time
import torchaudio
import tempfile
import torch
from TTS.api import TTS
import numpy as np

text = """
  Technology has transformed the way we connect, learn, and create. From instant communication to limitless access to knowledge, the digital age has opened new horizons for innovation and collaboration. As we navigate this ever-evolving landscape, it's essential to strike a balance between embracing progress and preserving the human touch that keeps us grounded. Together, we can harness the power of technology to build a more inclusive and sustainable future.
"""

# ====================================== Private functions ==============================================
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

def generate_audio_base64_file(
    text: str, playback_rate: float = 1.0, language: str = "english"
) -> str:
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
                output_file = (
                    temp_input.name
                )  # Use input directly if no speed adjustment

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
        raise ValueError(
            f"Unsupported language: {language_name}. Please provide a valid language name."
        )


## ================= Coqui TTS =================
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Initialize TTS
# xtts_v2
# coqtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# vi
coqtts = TTS("tts_models/en/ljspeech/vits").to(device)

# List speakers
print(coqtts.speakers)
## ================= Coqui TTS =================
def generate_audio_base64_mixed(text: str, playback_rate=1.0) -> str:
    """
    Generate Base64-encoded WAV audio from input text using FFmpeg.

    Args:
        text (str): Text to synthesize into speech.

    Returns:
        str: Base64-encoded audio in WAV format.
    """
    try:
        # raw_audio = coqtts.tts(
        # text=text,
        # language="en",
        # speaker="Craig Gutsy",
        # )

        raw_audio = coqtts.tts(
            text=text,
            # speaker="VCTK_p225",
            # language="en",
        )
        

        # Convert raw audio (float32) to 16-bit PCM
        # int16_audio = (np.array(raw_audio) * 32767).astype(np.int16)
        # audio_bytes = int16_audio.tobytes()

        # # Prepare input and output buffers for FFmpeg
        # input_buffer = BytesIO(audio_bytes)
        # output_buffer = BytesIO()

        # # Ensure the buffer is at the beginning
        # input_buffer.seek(0)

        # # Use FFmpeg to adjust playback rate and output WAV format
        # process = (
        #     ffmpeg.input(
        #         "pipe:0", format="s16le", ac=1, ar="22050"
        #     )  # 16-bit PCM, mono, 22.05 kHz
        #     .filter("atempo", playback_rate)
        #     .output("pipe:1", format="wav")
        #     .run(input=input_buffer.read(), capture_stdout=True, capture_stderr=True)
        # )

        # # Write FFmpeg's output to a buffer
        # output_buffer.write(process[0])
        # output_buffer.seek(0)

        # # Encode the audio to Base64
        # audio_base64 = base64.b64encode(output_buffer.read()).decode("utf-8")

        # return audio_base64
        return None


    except ffmpeg.Error as e:
        error_message = e.stderr.decode("utf-8") if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg processing error: {error_message}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


# ====================================== Main function ==============================================
if __name__ == "__main__":
    start_time_total = time.time()  # Start total time tracking

    ## Test generate_audio_base64_file see which is faster. ByteIO or File***
    start_time_tts_file = time.time()
    response_audio = generate_audio_base64_file(
        text, playback_rate=1.15, language="english"
    )
    tts_duration_file = time.time() - start_time_tts_file
    print(f"Time taken for Text-to-Speech (TTS) with File: {tts_duration_file:.4f} seconds")

 ## Test generate_audio_base64_file see which is faster. ByteIO or File***
    start_time_tts = time.time()
    response_audio = generate_audio_base64(
        text, playback_rate=1.15
    )
    tts_duration = time.time() - start_time_tts
    print(f"Time taken for Text-to-Speech (TTS): {tts_duration:.4f} seconds")

 ## Test generate_audio_base64_file see which is faster. ByteIO or File***
    start_time_tts_coqui = time.time()
    response_audio = generate_audio_base64_mixed(
        text, playback_rate=1.15
    )
    tts_duration_coqui  = time.time() - start_time_tts_coqui 
    print(f"Time taken for Text-to-Speech (TTS): {tts_duration_coqui:.4f} seconds")

    total_duration = time.time() - start_time_total  # Calculate total time taken
    print(f"Total time taken: {total_duration:.4f} seconds")