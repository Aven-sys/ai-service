from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from common.util import pydantic_util
from common.util.langchain_pydantic_model_generator import (
    print_pydantic_instance,
    print_model_fields,
)
from ..util.langchain_pydantic_model_generator import (
    create_pydantic_model_from_config,
    print_model_fields,
)
from ..service import interview_llm_service

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from langchain_openai import ChatOpenAI
from uuid import uuid4
from ..util.session_histories_util import get_session_history, session_histories
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from ..util.interview_llm_util import generate_audio_base64, generate_audio_base64_file, generate_audio_base64_file_gg
import whisper
from langchain_core.messages import BaseMessage

# Groq
from ..util.groq_llm_util import GroqTranscriptionService
from langchain_groq import ChatGroq

# Deepgram
from ..util.deepgram_util import DeepgramTTS

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

# ========================= Flags =========================
isUseGroq = False
isDeepgram = False
# ========================= Flags =========================

router = APIRouter(
    prefix="/api/llm-interview",
)


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


# System prompt that includes all questions and instructions
# system_prompt = """
# You are an interviewer conducting a structured interview. Ask one question at a time, then wait for the interviewee's response before proceeding to the next question.
# Greet the interviewee first politely. Then start the interview by asking the following questions (there may be 1 or more questions):

# {interview_questions}

# After each answer, provide brief feedback if appropriate, then stop. Wait for the interviewee’s next input before asking the next question. Note that there may only be one question. In that case, just ask that question and wait for the interviewee’s response.
# Once the interview is completed, ask the interviewee’s if he/she has anything to add.
# If not, summarize the interview and asked the interviewee’s if the information is correct. If the interviewee’s agree all information provided are correct, end the interview and thank the interviewee’s for participating in the interview and end the interview. Only set is_done to True when the interview is completed and you will not expect a response from the interviewee’s.

# {format_instructions}
# """

system_prompt = """
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

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=InterviewOutput)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Initialize the LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Combine the prompt, LLM, and parser into a chain
# chain = prompt | llm | parser
chain = prompt | llm

# Wrap the chain with RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Load whisper
# whisper_model = whisper.load_model("tiny.en")
whisper_model = whisper.load_model("tiny")

# Load Groq Transcription Service
groq_transcription_client = GroqTranscriptionService()

# Deep gram
deepgram_tts = DeepgramTTS(api_key=deepgram_api_key)


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


@router.post("/start-interview")
async def start_interview(interview_start_request_dto: InterviewStartRequestDto):
    session_id = str(uuid4())

    # Initialize the session with the specified memory type
    get_session_history(session_id, memory_type=interview_start_request_dto.memory_type)

    # Start the interview with an empty input (LLM will handle the flow)
    result = chain_with_history.invoke(
        {
            "input": "Let's start the interview",
            **interview_start_request_dto.context,
        },
        config={"configurable": {"session_id": session_id}},
    )
    # print("Result:", result)

    interview_output = InterviewOutput(**json.loads(result.content))

    # if isDeepgram:
    #     response_audio = deepgram_tts.text_to_speech_base64(
    #         interview_output.interviewer_output
    #     )
    # else:
    #     response_audio = generate_audio_base64(
    #         interview_output.interviewer_output, playback_rate=1.15
    #     )

    response_audio = generate_audio_base64_file_gg(
        interview_output.interviewer_output,
        playback_rate=1.15,
        language=interview_start_request_dto.context["language"],
    )

    # View Chat History
    chat_history = session_histories[session_id].messages

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
    if isUseGroq:
        file_content = await audio_file.read()
        transcription_text = groq_transcription_client.transcribe_audio(
            file_content, audio_file.filename
        )
    else:
        transcription_text = await transcript_audio(audio_file)
    stt_duration = time.time() - start_time_stt
    print(f"Time taken for Speech-to-Text (STT): {stt_duration:.4f} seconds")

    interview_input.context["input"] = transcription_text

    # Check if session exists
    if interview_input.session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # LLM Call Step
    start_time_llm = time.time()
    result = chain_with_history.invoke(
        {**interview_input.context},
        config={"configurable": {"session_id": interview_input.session_id}},
    )
    llm_duration = time.time() - start_time_llm
    print(f"Time taken for LLM call: {llm_duration:.4f} seconds")

    interview_output = InterviewOutput(**json.loads(result.content))

    # Text-to-Speech Step
    # start_time_tts = time.time()
    # if isDeepgram:
    #     response_audio = deepgram_tts.text_to_speech_base64(
    #         interview_output.interviewer_output
    #     )
    # else:
    #     response_audio = generate_audio_base64(
    #         interview_output.interviewer_output, playback_rate=1.15
    #     )
    # tts_duration = time.time() - start_time_tts
    # print(f"Time taken for Text-to-Speech (TTS): {tts_duration:.4f} seconds")

    ## Test generate_audio_base64_file see which is faster. ByteIO or File***
    start_time_tts_file = time.time()
    response_audio = generate_audio_base64_file_gg(
        interview_output.interviewer_output,
        playback_rate=1.15,
        language=interview_input.context["language"],
    )
    tts_duration_file = time.time() - start_time_tts_file
    print(
        f"Time taken for Text-to-Speech (TTS) with File: {tts_duration_file:.4f} seconds"
    )

    # View Chat History
    chat_history = session_histories[interview_input.session_id].messages

    # Clear the session history if the interview is done
    if interview_output.is_done:
        del session_histories[interview_input.session_id]

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
