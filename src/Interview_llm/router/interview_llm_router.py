from fastapi import APIRouter, HTTPException
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

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from uuid import uuid4
from ..util.session_histories_util import get_session_history, session_histories
from langchain_core.runnables.history import RunnableWithMessageHistory

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm-interview",
)

class InterviewInput(BaseModel):
    session_id: str 
    user_input: str 

# Define Pydantic model for structured output
class InterviewOutput(BaseModel):
    full_output: str = Field(
        ..., description="LLm response"
    )
    question: Optional[str] = Field(
        None, description="The current question being asked by the interviewer"
    )
    answer: Optional[str] = Field(
        None, description="The interviewee's answer to the current question"
    )
    is_done: bool = Field(
        ..., description="Indicates if the interview process is complete"
    )
    summary: Optional[str] = Field(
        None,
        description="A summary of all answers provided by the interviewee at the end of the interview",
    )


# System prompt that includes all questions and instructions
system_prompt = """
You are an interviewer conducting a structured interview. Ask one question at a time, then wait for the interviewee's response before proceeding to the next question. 
Greet the interviewer first and then ask the following questions:
1. How old are you?
2. What is your favorite movie?
3. What is your favorite food?
4. What is your highest education level?
5. What is your favorite animal?

After each answer, provide brief feedback if appropriate, then stop. Wait for the interviewee’s next input before asking the next question.

{format_instructions}
"""

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=InterviewOutput)

# Initalize the chat prompt template
prompt_template = PromptTemplate(
    input_variables=["history"],
    template=system_prompt + "\n{history}",
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Combine the prompt, LLM, and parser into a chain
chain = prompt_template | llm | parser

# Wrap the chain with RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="history",
    history_messages_key="history",
)


@router.post("/start-interview")
async def start_interview(memory_type: str = "chat"):
    session_id = str(uuid4())

    # Initialize the session with the specified memory type
    get_session_history(session_id, memory_type=memory_type)

    # Start the interview with an empty input (LLM will handle the flow)
    result = chain_with_history.invoke(
        {}, config={"configurable": {"session_id": session_id}}
    )

    return {"session_id": session_id, **result.dict()}

@router.post("/interview")
async def interview(interview_input: InterviewInput):
    # Check if session exists
    if interview_input.session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Retrieve and print the chat history to verify it's being retained
    history = session_histories[interview_input.session_id]
    print("Current chat history:")
    
    
    # Invoke the chain with user response and session history
    result = chain_with_history.invoke(
        {"history": interview_input.user_input},
        config={"configurable": {"session_id": interview_input.session_id}}
    )
    
    # Determine if interview is done based on `is_done` in result
    if result.is_done:
        summary = "Summary of the interview:\n" + result.summary if result.summary else "No summary available."
        return {"response": result.full_output, "is_done": True, "summary": summary}
    
    return {"response": result.full_output, "is_done": False}

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
