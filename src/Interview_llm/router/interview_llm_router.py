from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from uuid import uuid4
from ..util.session_histories_util import get_session_history, session_histories
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from ..util.interview_llm_util import (
    generate_audio_base64_file,
    generate_audio_base64_file_gg,
)
import whisper
from langchain_core.messages import BaseMessage

# Groq
from ..util.groq_llm_util import GroqTranscriptionService
from langchain_groq import ChatGroq

# Deepgram
from ..util.deepgram_util import DeepgramTTS

from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import speech
import httpx

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========================= Flags =========================
# STT
STT_isGroq = False
STT_isOpenWhisper = False #True
STT_isOpenWhisper_en = False #True
STT_isIOpenAI = True

# LLM
LLM_isGroq = False
LLM_isOpenAI = False
LLM_isGoogleGenerativeAI = True

# TTS
TTS_isDeepgram = False
TTS_isGTTS = False
TTS_isGGCloud = True
# ========================= Flags =========================
router = APIRouter(
    prefix="/api/llm-interview",
)


# Define Pydantic model for requestDTO for session start
class InterviewSessionStartRequestDto(BaseModel):
    questions: str
    language: str


class InterviewStartRequestDto(BaseModel):
    memory_type: Optional[str] = "chat"
    context: Dict


class InterviewInput(BaseModel):
    session_id: str
    context: Dict


# Define Pydantic model for structured output
class InterviewOutput(BaseModel):
    interviewer_output: str = Field(
        ...,
        description="The interviewer output with the most current LLM response to it. The question should be in the output when the interviewer is asking a question.",
    )
    question: Optional[str] = Field(
        None,
        description="The current question being asked by the interviewer. This is just to store the question for reference.",
    )
    answer: Optional[str] = Field(
        None,
        description="The interviewee's answer to the current question. Do not paraphrase or summarize the answer. This is just to store the answer for reference.",
    )
    is_done: bool = Field(
        ..., description="Indicates if the interview process is complete"
    )
    summary: Optional[str] = Field(
        None,
        description="A summary of all answers provided by the interviewee at the end of the interview",
    )
    chat_history: Optional[str] = Field(
        None,
        description="The full chat history of the interview. Interview output and user input.",
    )


class InterviewStartResponseDto(BaseModel):
    session_id: str
    response: InterviewOutput
    response_audio: Optional[str] = None
    chat_history: Optional[List[BaseMessage]] = None


## =========================== Define the Prompt =========================
## LAMA 3.1
LLAMA_system_prompt = """
You are an interviewer conducting a structured interview. Your task is to engage the interviewee in a professional and polite manner, asking one question at a time and waiting for their response before proceeding. Follow these steps:

Language: Conduct the interview in the specified language: **{language}**

Greeting: Start by greeting the interviewee politely "Good morning! Thank you for joining the interview today. I hope you're doing well. Let's get started." + the first question .

Do not ask any other questions. Only the provided interview questions should be asked.

Always ask the question in the interviewer_output field and wait for the interviewee's response before proceeding.

Interview Questions: Begin the interview by asking the provided questions:

{interview_questions}

Ask each question one at a time. After receiving an answer, provide brief and constructive feedback (if appropriate) before moving to the next question.
If the interviewees does not provide an appropriate response after 3 tries, proceed on with the interview.

Do not ask any follow up questions. Only ask the interview questions provided.

Do not ask extra questions or provide additional information. Only ask the interview questions provided. 

Always end your response with the question to keep the conversation flowing.

If only one question is provided, ask that question and wait for the response.

Wrapping Up: Once all the questions have been answered:

Ask the interviewee if they have anything to add.
Summarize the interview by restating the key points discussed and confirm with the interviewee if all the information provided is accurate.
Ending the Interview:

If the interviewee confirms that the information is accurate, thank them for their participation and end the interview.
Only set is_done to True once the interview is fully completed, and no further input from the interviewee is expected.

Please only return in response JSON valid format. Below is the schema for the JSON output.
1. **Ensure proper JSON syntax**:
   - Use double quotes (`"`) for all keys and string values.
   - Do not include extra characters, comments, or formatting outside the JSON.
{format_instructions}
"""

OPENAI_system_prompt = """
You are an interviewer conducting a structured interview. Your task is to engage the interviewee in a professional and polite manner, asking one question at a time and waiting for their response before proceeding. Follow these steps:

Language: Conduct the interview in the specified language: **{language}**

Greeting: Start by greeting the interviewee politely.

Interview Questions: Begin the interview by asking the provided questions:

{interview_questions}

Ask each question one at a time. After receiving an answer, provide brief and constructive feedback (if appropriate) before moving to the next question.
If the interviewees does not provide an appropriate response after 3 tries, proceed on with the interview.

If only one question is provided, ask that question and wait for the response.

Wrapping Up: Once all the questions have been answered:

Ask the interviewee if they have anything to add.
Summarize the interview by restating the key points discussed and confirm with the interviewee if all the information provided is accurate.
Ending the Interview:

If the interviewee confirms that the information is accurate, thank them for their participation and end the interview.
Only set is_done to True once the interview is fully completed, and no further input from the interviewee is expected.
{format_instructions}
"""

# # ========================= Define the prompt template ======================
# if LLM_isGroq:
#     system_prompt = LLAMA_system_prompt
# elif LLM_isOpenAI:
#     system_prompt = OPENAI_system_prompt
# elif LLM_isGoogleGenerativeAI:
#     system_prompt = LLAMA_system_prompt


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}"),
#     ]
# )
# # ========================= Initialize the LLM Model =========================
# # Initialize the parser
# if LLM_isGroq:
#     prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",  # Specify the desired model
#         temperature=0.3,  # Set the temperature as needed
#     )
# elif LLM_isOpenAI:
#     parser = PydanticOutputParser(pydantic_object=InterviewOutput)
#     prompt = prompt.partial(format_instructions=parser.get_format_instructions())
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# elif LLM_isGoogleGenerativeAI:
#     prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)

# # Gemini Test only
# # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# # Chain the prompt and the LLM model
# chain = prompt | llm

# # Wrap the chain with RunnableWithMessageHistory
# chain_with_history = RunnableWithMessageHistory(
#     runnable=chain,
#     get_session_history=get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
# )

## If the language is English, use Groq, otherwise use Google Generative AI
chain_with_history = None


def buildChainWithLang(language):
    if LLM_isGroq:
        system_prompt = LLAMA_system_prompt
    elif LLM_isOpenAI:
        system_prompt = OPENAI_system_prompt
    elif LLM_isGoogleGenerativeAI:
        system_prompt = LLAMA_system_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    # ========================= Initialize the LLM Model =========================
    # Initialize the parser
    if language == "english":
        prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Specify the desired model
            temperature=0.3,  # Set the temperature as needed
        )
        print("Using Groq LLAMA")
    else:
        prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        print("Using Google Generative AI")

    # Gemini Test only
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Chain the prompt and the LLM model
    chain = prompt | llm

    # Wrap the chain with RunnableWithMessageHistory
    return RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


## =========================== Load TTS Model =========================
# Load whisper
# whisper_model = whisper.load_model("tiny.en")
whisper_model = whisper.load_model("tiny")
whisper_model_en = whisper.load_model("tiny.en")

# Load Groq Transcription Service
groq_transcription_client = GroqTranscriptionService()

## =========================== Load STT Model =========================
# Deep gram
deepgram_tts = DeepgramTTS(api_key=deepgram_api_key)


## =========================== Private method =========================
async def transcript_audio(audio_file):
    # Read the audio file content from `UploadFile` and save it to a temporary file
    audio_content = await audio_file.read()  # Read the audio data as bytes

    # Transcribe the audio data using Whisper
    with open("temp_audio.wav", "wb") as temp_audio_file:
        temp_audio_file.write(audio_content)

    # Run Whisper transcription on the saved file
    transcription_result = whisper_model.transcribe("temp_audio.wav")
    transcription_text = transcription_result["text"]
    # print("Transcription:", transcription_text)
    return transcription_text

async def transcript_audio_lang(audio_file, language):
    # Read the audio file content from `UploadFile` and save it to a temporary file
    audio_content = await audio_file.read()  # Read the audio data as bytes

    # Transcribe the audio data using Whisper
    with open("temp_audio.wav", "wb") as temp_audio_file:
        temp_audio_file.write(audio_content)

    # Run Whisper transcription on the saved file
    if language == "english":
        print("Transcribing in English")
        transcription_result = whisper_model_en.transcribe("temp_audio.wav")
    else:
        print("Transcribing in foreign language")
        transcription_result = whisper_model.transcribe("temp_audio.wav")
    transcription_text = transcription_result["text"]
    return transcription_text


def parse_llm_output(raw_content: str):
    try:
        # Clean the raw content by removing backticks and formatting
        if raw_content.startswith("```json"):
            raw_content = raw_content.strip("```json\n").strip("\n```")

        # Parse the JSON
        parsed_data = json.loads(raw_content)

        # Convert to Pydantic model
        return InterviewOutput(**parsed_data)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        raise ValueError("Failed to parse LLM output as JSON.")


def clean_chat_history(chat_history):
    """
    Cleans the content of all messages in the chat history by stripping unwanted formatting.

    Args:
        chat_history (list): A list of HumanMessage or AIMessage objects.

    Returns:
        list: A list of cleaned HumanMessage or AIMessage objects.
    """

    def clean_content(raw_content: str) -> str:
        if raw_content.startswith("```json"):
            return raw_content.lstrip("```json\n").rstrip("\n```")
        return raw_content

    cleaned_history = []
    for message in chat_history:
        if hasattr(message, "content"):
            message.content = clean_content(message.content)
        cleaned_history.append(message)

    return cleaned_history


async def transcript_audio_gg(audio_file: UploadFile):
    """
    Asynchronously transcribes an uploaded audio file using Google Speech-to-Text API.
    Args:
        audio_file (UploadFile): The audio file uploaded via FastAPI.
    Returns:
        str: Transcribed text from the audio.
    """
    # Read the audio file content from `UploadFile`
    audio_content = await audio_file.read()

    # Initialize the Google Cloud Speech client
    client = speech.SpeechClient()

    # Configure recognition
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Adjust based on your file format
        language_code="en-US",
    )

    # Perform synchronous recognition
    response = client.recognize(config=config, audio=audio)

    # Combine all transcripts into a single string
    transcription_text = " ".join(
        result.alternatives[0].transcript for result in response.results
    )

    return transcription_text


chains = {}


@router.post("/start-interview")
async def start_interview(interview_start_request_dto: InterviewStartRequestDto):
    session_id = str(uuid4())

    language = interview_start_request_dto.context["language"]

    # Build the chain with the specified language
    chain_with_history = buildChainWithLang(language)
    chains[session_id] = chain_with_history

    # Initialize the session with the specified memory type
    # get_session_history(session_id, memory_type=interview_start_request_dto.memory_type)

    # Start the interview with an empty input (LLM will handle the flow)
    result = chain_with_history.invoke(
        {
            "input": "Let's start the interview",
            **interview_start_request_dto.context,
        },
        config={"configurable": {"session_id": session_id}},
    )
    # print("Result:", result)
    # print("Result type:", type(result))
    # print("Result content:", result.content)

    if LLM_isGroq or LLM_isGoogleGenerativeAI:
        interview_output = parse_llm_output(result.content)
    elif LLM_isOpenAI:
        interview_output = InterviewOutput(**json.loads(result.content))

    # Text-to-Speech Step
    if TTS_isDeepgram:
        response_audio = deepgram_tts.text_to_speech_base64(
            interview_output.interviewer_output
        )
    elif TTS_isGTTS:
        response_audio = generate_audio_base64_file(
            interview_output.interviewer_output, playback_rate=1.15
        )
    elif TTS_isGGCloud:
        response_audio = generate_audio_base64_file_gg(
            interview_output.interviewer_output,
            playback_rate=1.0,
            language=interview_start_request_dto.context["language"],
        )

    # View Chat History
    chat_history = session_histories[session_id].messages
    if LLM_isGroq or LLM_isGoogleGenerativeAI:
        clean_chat_history(chat_history)
    # print("Chat History:", chat_history)

    return InterviewStartResponseDto(
        session_id=session_id,
        response=interview_output,
        response_audio=response_audio,
        chat_history=chat_history,
    )


@router.post("/interview")
async def interview(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    context: str = Form(...),
):
    start_time_total = time.time()  # Start total time tracking

    # Parse JSON `context` string to a Python dictionary
    start_time_parse = time.time()
    context_dict = json.loads(context)
    parse_duration = time.time() - start_time_parse
    print(f"Time taken for parsing input context: {parse_duration:.4f} seconds")

    # Create `InterviewInput` model manually with parsed data
    interview_input = InterviewInput(session_id=session_id, context=context_dict)

    # Speech-to-Text Step
    start_time_stt = time.time()
    if STT_isGroq:
        file_content = await audio_file.read()
        transcription_text = groq_transcription_client.transcribe_audio(
            file_content, audio_file.filename
        )
    elif STT_isOpenWhisper:
        transcription_text = await transcript_audio(audio_file)
    stt_duration = time.time() - start_time_stt
    print(f"Time taken for Speech-to-Text (STT): {stt_duration:.4f} seconds")

    interview_input.context["input"] = transcription_text

    # Check if session exists
    if interview_input.session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # LLM Call Step
    start_time_llm = time.time()
    chain_with_history = chains[interview_input.session_id]  # This is new
    result = chain_with_history.invoke(
        {**interview_input.context},
        config={"configurable": {"session_id": interview_input.session_id}},
    )
    llm_duration = time.time() - start_time_llm
    print(f"Time taken for LLM call: {llm_duration:.4f} seconds")

    if LLM_isGroq or LLM_isGoogleGenerativeAI:
        interview_output = parse_llm_output(result.content)
    elif LLM_isOpenAI:
        interview_output = InterviewOutput(**json.loads(result.content))

    # Text-to-Speech Step
    start_time_tts = time.time()
    if TTS_isDeepgram:
        response_audio = deepgram_tts.text_to_speech_base64(
            interview_output.interviewer_output
        )
    elif TTS_isGTTS:
        response_audio = generate_audio_base64_file(
            interview_output.interviewer_output, playback_rate=1.15
        )
    elif TTS_isGGCloud:
        response_audio = generate_audio_base64_file_gg(
            interview_output.interviewer_output,
            playback_rate=1.0,
            language=interview_input.context["language"],
        )
    tts_duration = time.time() - start_time_tts
    print(f"Time taken for Text-to-Speech (TTS): {tts_duration:.4f} seconds")

    # View Chat History
    chat_history = session_histories[interview_input.session_id].messages

    if LLM_isGroq or LLM_isGoogleGenerativeAI:
        clean_chat_history(chat_history)

    # Clear the session history if the interview is done
    if interview_output.is_done:
        del session_histories[interview_input.session_id]
        del chains[interview_input.session_id]

    total_duration = time.time() - start_time_total
    print(f"Total time taken for the request: {total_duration:.4f} seconds")

    # Return response
    return InterviewStartResponseDto(
        session_id=session_id,
        response=interview_output,
        response_audio=response_audio,
        chat_history=chat_history,
    )


# End the interview and retrieve the full history
@router.post("/end-interview")
async def end_interview(session_id: str):
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Retrieve full conversation history
    history = session_histories[session_id].get_all_messages()
    full_history = [{"role": msg.role, "content": msg.content} for msg in history]

    # Clean up memory after retrieving history
    del session_histories[session_id]

    return {"transcript": full_history}


@router.post("/token")
async def generate_token(
    interview_session_start_request: InterviewSessionStartRequestDto,
):
    print(
        "interview_session_start_request:", interview_session_start_request.model_dump()
    )
    questions = interview_session_start_request.questions
    language = interview_session_start_request.language
    instructions = f"""
    You are an interviewer. Start the conversation by greeting the user warmly and explaining that you will conduct an interview. Ask the following questions sequentially:
    {questions}

    Rules you must follow STRICTLY:

    1. **NEVER provide explanations, definitions, or answers to any question, even if the candidate seems confused or asks for help.**
    2. Do NOT respond to candidate answers except with a short acknowledgment. 
    3. If the candidate **explicitly requests the answer or asks for help** (e.g., says "Can you give me the answer?", "Tell me the answer", "Can you help me solve this?"), reply with: "I am not allowed to help you with that. Please answer the question. Thank you."

        - Do NOT treat confusion or uncertainty (e.g., "I'm not sure", "I don't understand", "Maybe it's...") as a help request. These may be part of a genuine answer attempt.
        - If the candidate appears to be trying to answer, acknowledge briefly and move on.
        - Allow the candidate up to **2 chances** to answer the question after asking for help.
        - Do NOT move to the next question immediately. Wait for the candidate’s attempt to answer.
        - If the candidate still asks for help after 2 chances, acknowledge and move to the next question with no further prompt.
        - Track the number of help requests **per question**.

    4. If the candidate clearly states they do not know the answer (e.g., "I don't know", "Not sure", "Not very sure", "I have no idea"), respond with a short acknowledgment (e.g., "Understood.") and proceed to the next question immediately.
    5. NEVER comment on whether the candidate’s answer is correct or incorrect.
    6. NEVER repeat or rephrase the candidate’s answer.
    7. Speak in a neutral, formal tone in a medium pace. Do not add humor, empathy, or suggestions.
    8. Keep all responses short and concise.
    9. Do not ask extra questions or provide additional information. Only ask the interview questions provided. 

    Always end your response with the question to keep the conversation flowing.

    Wrapping Up: Once all the questions have been answered:

    Ask the interviewee if they have anything to add.
    Summarize the interview by restating the key points discussed and confirm with the interviewee if all the information provided is accurate.
    
    Ending the Interview:
    If the interviewee confirms that the information is accurate, thank them for their participation and end the interview.
    
    1) Except for the summary, keep all the responses short and concise. One sentence is enough.
    2) Conduct the interview in this language: *{language}*.
    3) The last sentence is crucial to end the interview. Make sure to use it when the interview is complete.
    4) Finally, please response "**<-//->**". It is a marker to end the interview application. This should not be in the same response as the summary.
    """
    url = "https://api.openai.com/v1/realtime/sessions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini-realtime-preview",
        "voice": "shimmer",
        "modalities": ["audio", "text"],
        # "input_audio_transcription": {"model": "whisper-1"},
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "instructions": instructions,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()
        pass

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate token: {str(e)}"
        )

async def transcript_audio_openAI(audio_file: UploadFile) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    temp_files = []  # Keep track of all created files for cleanup
    
    try:
        # Read the file content
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise ValueError("Uploaded file is empty or unreadable.")
        
        ## NOT THAT FILE COMING FROM Frontend is wav format. SO EVERYTHING HAS TO BE WAV 
        ## FOR THE NEW MODELS TO WORK!        
        
        # Try with whisper-1 model
        files = {"file": ("recording.wav", audio_bytes , "audio/wav")}

        data = {
            "model": "gpt-4o-mini-transcribe",  # Try with different model
            "response_format": "json"
        }
        
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                print(f"Response text: {response.text}")
                raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise
    
    finally:
        # Clean up all temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
                except Exception as cleanup_error:
                    print(f"Failed to delete temporary file {file_path}: {str(cleanup_error)}")


@router.post("/transcribe-audio")
async def interview(
    audio_file: UploadFile = File(...),
    language: str = Form(...),
    order: Optional[int] = Form(None),
):
    # Speech-to-Text Step
    start_time_stt = time.time()
    if STT_isGroq:
        file_content = await audio_file.read()
        transcription_text = groq_transcription_client.transcribe_audio(
            file_content, audio_file.filename
        )
    elif STT_isOpenWhisper and STT_isOpenWhisper_en:
        transcription_text = await transcript_audio_lang(audio_file, language)
    elif STT_isOpenWhisper:
        transcription_text = await transcript_audio(audio_file)
    elif STT_isIOpenAI:
        transcription_text = await transcript_audio_openAI(audio_file)

    stt_duration = time.time() - start_time_stt
    print(f"Time taken for Speech-to-Text (STT): {stt_duration:.4f} seconds")

    # Trim text front and back for empty spaces
    transcription_text = transcription_text.strip()

    # Return response
    return {"transcript": transcription_text, "order": order}

