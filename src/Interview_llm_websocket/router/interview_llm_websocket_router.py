from fastapi import (
    APIRouter,
    HTTPException,
    File,
    UploadFile,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from uuid import uuid4
from ..util.session_histories_util import get_session_history, session_histories
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from ..util.interview_llm_util import (
    generate_audio_base64,
    generate_audio,
    generate_audio_bytesio_gg,
)
import whisper
from langchain_core.messages import BaseMessage
import base64
import struct

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
from google.cloud import speech

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

# ========================= Flags =========================
# STT
STT_isGroq = True
STT_isOpenWhisper = False

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
    prefix="/api/llm-interview-websocket",
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
    type: str


## =========================== Define the Prompt =========================
## LAMA 3.1
LLAMA_system_prompt = """
You are an interviewer conducting a structured interview. Your task is to engage the interviewee in a professional and polite manner, asking one question at a time and waiting for their response before proceeding. Follow these steps:

Conduct the interview strictly in the specified language: **{language}**.

At every stage of the interview, ensure that your responses are written in **{language}** only.

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

# ========================= Define the prompt template ======================
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
if LLM_isGroq:
    prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Specify the desired model
        temperature=0.3,  # Set the temperature as needed
    )
elif LLM_isOpenAI:
    parser = PydanticOutputParser(pydantic_object=InterviewOutput)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
elif LLM_isGoogleGenerativeAI:
    prompt = prompt.partial(format_instructions=InterviewOutput.model_json_schema())
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Gemini Test only
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Chain the prompt and the LLM model
chain = prompt | llm

# Wrap the chain with RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

## =========================== Load TTS Model =========================
# Load whisper
whisper_model = whisper.load_model("tiny")

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


async def transcribe_audio(audio_data):
    """Convert audio to text using Whisper."""
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_data)
    result = whisper_model.transcribe("temp_audio.wav")
    return result["text"]


async def generate_response(interview_input: InterviewInput):
    """Generate a response using OpenAI GPT."""
    # Invoke the chain with user response and session history
    result = chain_with_history.invoke(
        {**interview_input.context},
        config={"configurable": {"session_id": interview_input.session_id}},
    )
    return result


def combine_audio_and_json(audio_bytes: bytes, json_metadata: dict) -> bytes:
    # Serialize JSON metadata to a byte string
    json_bytes = json.dumps(json_metadata).encode("utf-8")

    # Get the length of the JSON bytes as a 4-byte little-endian integer
    json_length = struct.pack("<I", len(json_bytes))  # Little-endian 4 bytes

    # Combine JSON length, JSON bytes, and audio bytes
    combined_bytes = json_length + json_bytes + audio_bytes
    return combined_bytes


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


# Set Temp socket connection to store the session_id
connections = {}


@router.websocket("/ws/interview-chatbot")
async def interview_chatbot(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    try:
        while True:
            # Receive the message (it can be text or bytes)
            message = await websocket.receive()

            if "text" in message:  # JSON or text message
                if message["text"] == "RESPONSE_END":
                    print("End of response received.")
                    # print ("1 - Connections: ", connections)

                    audio_data = None
                    # Get the audio Buffer array from connections[websocket]
                    if websocket in connections:
                        if "audio_buffer" in connections[websocket]:
                            audio_data = bytes(connections[websocket]["audio_buffer"])

                    # Transcription logic
                    if STT_isGroq:
                        transcription_text = groq_transcription_client.transcribe_audio(
                            audio_data, "audio.wav"  # Filename is optional
                        )
                    elif STT_isOpenWhisper:
                        transcription_text = await transcribe_audio(audio_data)

                    # Get session ID and context from the WebSocket connection
                    session_id = connections[websocket]["session_id"]
                    context = connections[websocket]["context"]

                    context["input"] = transcription_text

                    # Check if session Id is in session_histories
                    if session_id not in session_histories:
                        raise HTTPException(status_code=404, detail="Session not found")

                    result = chain_with_history.invoke(
                        {**context},
                        config={"configurable": {"session_id": session_id}},
                    )

                    if LLM_isGroq or LLM_isGoogleGenerativeAI:
                        interview_output = parse_llm_output(result.content)
                    elif LLM_isOpenAI:
                        interview_output = InterviewOutput(**json.loads(result.content))

                    # Generate TTS audio
                    if TTS_isDeepgram:
                        audio_data = deepgram_tts.text_to_speech(
                            interview_output.interviewer_output
                        )
                    elif TTS_isGGCloud:
                        audio_data = generate_audio_bytesio_gg(
                            interview_output.interviewer_output,
                            playback_rate=1.0,
                            language=context["language"],
                        )
                    elif TTS_isGTTS:
                        audio_data = generate_audio(interview_output.interviewer_output)

                    chat_history = session_histories[session_id].messages
                    if LLM_isGroq or LLM_isGoogleGenerativeAI:
                        clean_chat_history(chat_history)

                    outResponse = InterviewStartResponseDto(
                        session_id=session_id,
                        response=interview_output,
                        # response_audio=response_audio,
                        chat_history=chat_history,
                        type="interview_start_response",
                    )

                    # COmbined and sent the JSON and audio in bytes
                    combined_bytes = combine_audio_and_json(
                        audio_data.read(), outResponse.model_dump()
                    )
                    await websocket.send_bytes(combined_bytes)

                    # delete the audio_buffer from connections[websocket]
                    if websocket in connections:
                        if "audio_buffer" in connections[websocket]:
                            del connections[websocket]["audio_buffer"]
                    print("2 - Connections: ", connections)
                    continue

                try:
                    # Parse the text as JSON
                    data = json.loads(message["text"])

                    # Validate the structure: "type" and "data" must exist
                    if "type" in data and "data" in data:
                        if data["type"] == "start_interview":
                            interview_start_request_dto = InterviewStartRequestDto(
                                **data["data"]
                            )
                            session_id = str(uuid4())

                            # Associate the session ID with the WebSocket connection so that session ID do not need to be sent in every message
                            connections[websocket] = {
                                "session_id": session_id,
                                "context": interview_start_request_dto.context,
                            }

                            result = chain_with_history.invoke(
                                {
                                    "input": "Let's start the interview",
                                    **interview_start_request_dto.context,
                                },
                                config={"configurable": {"session_id": session_id}},
                            )
                        print("Result: ", result.content)

                        if LLM_isGroq or LLM_isGoogleGenerativeAI:
                            interview_output = parse_llm_output(result.content)
                        elif LLM_isOpenAI:
                            interview_output = InterviewOutput(
                                **json.loads(result.content)
                            )

                        # Generate TTS audio
                        if TTS_isDeepgram:
                            audio_data = deepgram_tts.text_to_speech(
                                interview_output.interviewer_output
                            )
                        elif TTS_isGGCloud:
                            audio_data = generate_audio_bytesio_gg(
                                interview_output.interviewer_output,
                                playback_rate=1.0,
                                language=interview_start_request_dto.context["language"],
                            )
                        elif TTS_isGTTS:
                            audio_data = generate_audio(
                                interview_output.interviewer_output
                            )

                        chat_history = session_histories[session_id].messages
                        if LLM_isGroq or LLM_isGoogleGenerativeAI:
                            clean_chat_history(chat_history)

                        outResponse = InterviewStartResponseDto(
                            session_id=session_id,
                            response=interview_output,
                            # response_audio=response_audio,
                            chat_history=chat_history,
                            type="interview_start_response",
                        )

                        # Send the response to the client
                        # await websocket.send_text(outResponse.model_dump_json())

                        # Send binary audio data
                        # await websocket.send_bytes(audio_data.read())

                        # COmbined and sent the JSON and audio in bytes
                        combined_bytes = combine_audio_and_json(
                            audio_data.read(), outResponse.model_dump()
                        )
                        await websocket.send_bytes(combined_bytes)

                    else:
                        await websocket.send_text(
                            "Invalid JSON structure: Must contain 'type' and 'data' fields."
                        )
                except json.JSONDecodeError:
                    print("Invalid JSON received.")
                    await websocket.send_text("Invalid JSON format.")
            elif "bytes" in message:  # Binary data
                binary_data = message["bytes"]
                print(f"Received binary data: {len(binary_data)} bytes")

                audio_chunk = message["bytes"]
                print(f"Received audio chunk: {len(audio_chunk)} bytes")

                # check if connections[websocket]["audio_buffer"] exists, if not create it
                if "audio_buffer" not in connections[websocket]:
                    connections[websocket]["audio_buffer"] = bytearray()

                # Append chunk to buffer
                connections[websocket]["audio_buffer"].extend(audio_chunk)

                # audio_file_path = "received_audio.wav"

                # # Save the binary audio data to a file
                # with open(audio_file_path, "wb") as f:
                #     f.write(binary_data)

                # # Transcription logic
                # if isUseGroq:
                #     transcription_text = groq_transcription_client.transcribe_audio(
                #         binary_data, "audio.wav"  # Filename is optional
                #     )
                # else:
                #     transcription_text = await transcribe_audio(binary_data)

                # # Get session ID and context from the WebSocket connection
                # session_id = connections[websocket]["session_id"]
                # context = connections[websocket]["context"]

                # context["input"] = transcription_text

                # # Check if session Id is in session_histories
                # if session_id not in session_histories:
                #     raise HTTPException(status_code=404, detail="Session not found")

                # result = chain_with_history.invoke(
                #     {**context},
                #     config={"configurable": {"session_id": session_id}},
                # )
                # interview_output = InterviewOutput(**json.loads(result.content))
                # chat_history = session_histories[session_id].messages

                #                 # Generate TTS audio
                # if isDeepgram:
                #     audio_data = deepgram_tts.text_to_speech(
                #         interview_output.interviewer_output
                #     )
                # else:
                #     audio_data = generate_audio(interview_output.interviewer_output)

                # outResponse = InterviewStartResponseDto(
                #     session_id=session_id,
                #     response=interview_output,
                #     # response_audio=response_audio,
                #     chat_history=chat_history,
                #     type="interview_start_response",
                # )

                # # Send the response to the client
                # # await websocket.send_text(outResponse.model_dump_json())

                # # # Send binary audio data
                # # await websocket.send_bytes(audio_data.read())
                # # # Acknowledge receipt
                # # await websocket.send_text("Binary data received successfully.")

                # # COmbined and sent the JSON and audio in bytes
                # combined_bytes = combine_audio_and_json(
                #     audio_data.read(), outResponse.model_dump()
                # )
                # await websocket.send_bytes(combined_bytes)
            else:
                print("Unknown message type received.")
                await websocket.send_text("Unknown message type.")

    except WebSocketDisconnect:
        print("WebSocket connection closed.")

        # Clean up session data
        if websocket in connections:
            session_data = connections.pop(websocket, None)
            if session_data:
                session_id = session_data["session_id"]

                # Remove session history
                session_histories.pop(session_id, None)

        print(f"Cleaned up session for WebSocket: {websocket}")

    except Exception as e:
        print(f"Error occurred: {e}")

        print("WebSocket connection closed.")

        # Clean up session data
        if websocket in connections:
            session_data = connections.pop(websocket, None)
            if session_data:
                session_id = session_data["session_id"]

                # Remove session history
                session_histories.pop(session_id, None)

        print(f"Cleaned up session for WebSocket: {websocket}")
        print("Connections:", connections)
        print("Session Histories:", session_histories)
